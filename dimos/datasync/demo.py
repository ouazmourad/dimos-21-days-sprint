#!/usr/bin/env python3
"""
DimOS DataSync — Multi-Sensor Time Synchronization & ML Data Export

  LIVE:     .venv/bin/python -m dimos.datasync.demo --live --duration 10
  OFFLINE:  .venv/bin/python -m dimos.datasync.demo --session data/sessions/my_run
"""

import argparse
import signal
import time

import plotext as plt
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from dimos.datasync.export import DataFrameExporter
from dimos.datasync.session import Session
from dimos.datasync.sync import SyncPolicy, SyncTransformer

console = Console()

CYAN = "bold cyan"
GREEN = "bold green"
YELLOW = "bold yellow"
DIM = "dim"


def _header() -> None:
    console.print()
    console.print(
        Panel(
            "[bold cyan]DimOS DataSync[/]\n"
            "[dim]Multi-Sensor Time Synchronization & ML Data Export[/]",
            border_style="cyan",
            padding=(1, 4),
        )
    )
    console.print()


def record_live(duration: float, session_id: str, max_rate_hz: float) -> Session:
    from dimos.datasync.recorder import StandaloneRecorder

    console.print(f"  [cyan]SESSION[/]   {session_id}")
    console.print(f"  [cyan]MAX RATE[/]  {max_rate_hz} Hz per topic")
    console.print()

    recorder = StandaloneRecorder(
        session_id=session_id,
        robot_type="auto",
        tags=["demo"],
        max_rate_hz=max_rate_hz,
    )
    recorder.start()

    interrupted = False

    def _on_sigint(sig, frame):
        nonlocal interrupted
        interrupted = True

    old_handler = signal.signal(signal.SIGINT, _on_sigint)

    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[cyan]RECORDING[/]"),
        BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("[dim]{task.fields[msgs]} msgs[/]"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("recording", total=duration, msgs=0)
        t0 = time.time()
        while time.time() - t0 < duration and not interrupted:
            elapsed = time.time() - t0
            progress.update(task, completed=elapsed, msgs=recorder.msg_count)
            time.sleep(0.2)
        progress.update(task, completed=duration, msgs=recorder.msg_count)

    signal.signal(signal.SIGINT, old_handler)
    recorder.stop()

    session = Session.open(recorder.session.base_dir)

    # Summary table
    tbl = Table(show_header=False, border_style="green", padding=(0, 2))
    tbl.add_column(style="green")
    tbl.add_column()
    tbl.add_row("Messages", f"[bold]{recorder.msg_count}[/]")
    tbl.add_row("Topics", ", ".join(session.topic_keys) or "[dim]none[/]")
    tbl.add_row("Saved to", f"[dim]{session.base_dir}[/]")
    console.print(Panel(tbl, title="[green]Recording Complete[/]", border_style="green"))
    console.print()

    return session


def sync_and_export(session: Session, target_hz: float) -> None:
    topics = session.topic_keys
    if not topics:
        console.print("[red]No topics recorded.[/] Is the simulation running?")
        return

    # Filter out topics with too few samples to sync meaningfully
    MIN_SAMPLES = 10
    viable_topics = []
    for key in topics:
        store = session.get_store(key)
        if len(store) >= MIN_SAMPLES:
            viable_topics.append(key)
        else:
            console.print(f"  [dim]Skipping {key} ({len(store)} samples)[/]")

    if not viable_topics:
        console.print("[red]No topics with enough data to sync.[/]")
        return

    # --- Sync ---
    sync = SyncTransformer.from_session(session, target_hz=target_hz, policy=SyncPolicy.HOLD, topic_keys=viable_topics)

    tbl = Table(title=f"[cyan]Sensor Streams[/]  (syncing to [bold]{target_hz} Hz[/] grid)",
                border_style="cyan", show_lines=False)
    tbl.add_column("Topic", style="bold")
    tbl.add_column("Samples", justify="right")
    tbl.add_column("Duration", justify="right")
    tbl.add_column("Native Hz", justify="right")
    tbl.add_column("Status", justify="center")

    for key, store in sync.stores.items():
        n = len(store)
        dur = store.duration()
        hz = n / dur if dur > 0 else 0
        status = "[green]OK[/]" if n > 0 else "[red]EMPTY[/]"
        tbl.add_row(key, str(n), f"{dur:.1f}s", f"~{hz:.0f}", status)

    console.print(tbl)
    console.print()

    # --- Export ---
    with console.status("[cyan]Synchronizing & exporting...[/]", spinner="dots"):
        df = DataFrameExporter(sync).to_dataframe()

    if df.empty:
        console.print("[red]DataFrame is empty.[/] Sensors may not overlap in time.")
        return

    duration = df.index[-1] - df.index[0]

    # DataFrame summary
    info = Table(show_header=False, border_style="cyan", padding=(0, 2))
    info.add_column(style="cyan")
    info.add_column()
    info.add_row("Rows", f"[bold]{df.shape[0]}[/]")
    info.add_row("Columns", f"[bold]{df.shape[1]}[/]")
    info.add_row("Duration", f"{duration:.1f}s")
    info.add_row("Grid", f"{target_hz} Hz")
    console.print(Panel(info, title="[cyan]Synchronized DataFrame[/]", border_style="cyan"))
    console.print()

    # Column listing
    col_tbl = Table(title="[cyan]Columns[/]", border_style="dim", show_lines=False)
    col_tbl.add_column("Column", style="bold")
    col_tbl.add_column("Type", style="dim")
    col_tbl.add_column("Min", justify="right")
    col_tbl.add_column("Max", justify="right")
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype.startswith("float") or dtype.startswith("int"):
            col_tbl.add_row(col, dtype, f"{df[col].min():.4f}", f"{df[col].max():.4f}")
        else:
            col_tbl.add_row(col, dtype, str(df[col].iloc[0])[:20], "")
    console.print(col_tbl)
    console.print()

    # --- Terminal plot ---
    # Prioritize interesting columns: odom position/orientation, then other varying numerics
    # Skip constant columns (like image width/height) and timestamp columns
    priority_prefixes = ["odom.", "drone_odom.", "imu."]
    skip_suffixes = [".width", ".height", ".channels", ".timestamp", ".ts", ".frame_id"]

    def _is_interesting(col: str) -> bool:
        if any(col.endswith(s) for s in skip_suffixes):
            return False
        if str(df[col].dtype) not in ("float64", "float32", "int64"):
            return False
        # Skip constant columns
        if df[col].nunique() <= 1:
            return False
        return True

    # Pick priority columns first, then fill with other interesting ones
    plot_cols = []
    for prefix in priority_prefixes:
        for col in df.columns:
            if col.startswith(prefix) and _is_interesting(col) and col not in plot_cols:
                plot_cols.append(col)
    for col in df.columns:
        if _is_interesting(col) and col not in plot_cols:
            plot_cols.append(col)
    plot_cols = plot_cols[:4]

    if plot_cols:
        ts = [t - df.index[0] for t in df.index]  # relative time
        plt.clear_figure()
        plt.theme("dark")
        plt.title("Sensor Data (synchronized)")
        plt.xlabel("Time (s)")
        for col in plot_cols:
            values = df[col].ffill().fillna(0).tolist()
            plt.plot(ts, values, label=col)
        plt.plotsize(80, 20)
        plt.show()
        console.print()

    # --- Sample data ---
    sample_tbl = Table(title="[cyan]Sample Data (first 5 rows)[/]", border_style="dim")
    sample_tbl.add_column("Time (s)", style="dim", justify="right")
    for col in df.columns[:8]:
        sample_tbl.add_column(col.split(".")[-1], justify="right")
    if len(df.columns) > 8:
        sample_tbl.add_column("...", style="dim")

    for ts_val in df.index[:5]:
        row_data = [f"{ts_val - df.index[0]:.2f}"]
        for col in df.columns[:8]:
            val = df.loc[ts_val, col]
            if isinstance(val, float):
                row_data.append(f"{val:.4f}")
            else:
                row_data.append(str(val)[:10])
        if len(df.columns) > 8:
            row_data.append("")
        sample_tbl.add_row(*row_data)
    console.print(sample_tbl)
    console.print()

    # --- Export to file ---
    out_path = session.base_dir / "export.parquet"
    df.to_parquet(str(out_path))
    size_kb = out_path.stat().st_size / 1024

    console.print(Panel(
        f"[green]Parquet[/]  {out_path}\n"
        f"[green]Size[/]     {size_kb:.1f} KB\n"
        f"[green]Rows[/]     {df.shape[0]}  x  [green]Cols[/]  {df.shape[1]}",
        title="[green]Export Complete[/]",
        border_style="green",
    ))
    console.print()

    # --- Policy comparison ---
    pol_tbl = Table(title="[cyan]Sync Policy Comparison[/]", border_style="cyan")
    pol_tbl.add_column("Policy", style="bold")
    pol_tbl.add_column("Rows", justify="right")
    pol_tbl.add_column("Description", style="dim")
    descs = {
        SyncPolicy.HOLD: "Carry forward last known value",
        SyncPolicy.ASOF: "Carry forward with staleness limit",
        SyncPolicy.DROP: "Only emit when all topics have data",
    }
    for policy in SyncPolicy:
        s = SyncTransformer.from_session(session, target_hz=target_hz, policy=policy, topic_keys=viable_topics)
        rows = sum(1 for _ in s.iterate_synced())
        pol_tbl.add_row(policy.value.upper(), str(rows), descs[policy])
    console.print(pol_tbl)
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="DimOS DataSync Demo")
    parser.add_argument("--live", action="store_true", help="Record from live LCM topics")
    parser.add_argument("--session", type=str, help="Open an existing session directory")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds")
    parser.add_argument("--session-id", type=str, default=None, help="Session ID")
    parser.add_argument("--target-hz", type=float, default=10.0, help="Sync grid frequency")
    parser.add_argument("--max-rate-hz", type=float, default=50.0, help="Max recording rate per topic")
    args = parser.parse_args()

    _header()

    if args.live:
        sid = args.session_id or f"demo_{int(time.time())}"
        session = record_live(args.duration, sid, args.max_rate_hz)
        sync_and_export(session, args.target_hz)

    elif args.session:
        session = Session.open(args.session)
        console.print(f"  [cyan]SESSION[/]  {session.session_id}")
        console.print(f"  [cyan]ROBOT[/]    {session.meta.robot_type}")
        console.print(f"  [cyan]TAGS[/]     {session.meta.tags}")
        console.print()
        sync_and_export(session, args.target_hz)

    else:
        console.print("[yellow]Usage:[/]")
        console.print("  [bold]Live recording[/] from a running simulation:")
        console.print("    [cyan].venv/bin/python -m dimos.datasync.demo --live --duration 10[/]")
        console.print()
        console.print("  [bold]Replay[/] an existing session:")
        console.print("    [cyan].venv/bin/python -m dimos.datasync.demo --session data/sessions/my_run[/]")
        console.print()
        console.print("[dim]Start a simulation first:[/]")
        console.print("  [dim]Drone:[/]  .venv/bin/python dimos/robot/drone/mujoco_sim.py")
        console.print("  [dim]Go2:[/]    .venv/bin/dimos --simulation --viewer none run unitree-go2-nightwatch")


if __name__ == "__main__":
    main()
