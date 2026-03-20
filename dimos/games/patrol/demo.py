"""Collaborative Patrol Demo — futuristic live radio feed display.

CI=1 .venv/bin/python dimos/games/patrol/demo.py
"""

import itertools
import os
import subprocess
import sys
import textwrap
import threading
import time

# ── ANSI ───────────────────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
WHITE = "\033[37m"
BR_BLACK = "\033[90m"
BR_CYAN = "\033[96m"
BR_GREEN = "\033[92m"
BR_YELLOW = "\033[93m"
BR_MAGENTA = "\033[95m"
BG_YELLOW = "\033[43m"
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"
BG_RED = "\033[41m"
BLACK = "\033[30m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR_LINE = "\033[2K\r"

W = 70


def hline(char="─", color=BR_BLACK):
    return f"{color}{char * W}{RESET}"


def timed(t0, text):
    elapsed = time.time() - t0
    print(f"  {BR_BLACK}[{elapsed:5.1f}s]{RESET}  {text}")


def slow_type(text, delay=0.03):
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def spinner(msg, done_event, t0):
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    for f in itertools.cycle(frames):
        if done_event.is_set():
            break
        elapsed = time.time() - t0
        sys.stdout.write(f"{CLEAR_LINE}  {BOLD}{CYAN}{f}{RESET} {msg} {DIM}[{elapsed:.0f}s]{RESET}")
        sys.stdout.flush()
        time.sleep(0.08)
    sys.stdout.write(CLEAR_LINE)
    sys.stdout.flush()


def progress_bar(pct, width=30, color=CYAN):
    filled = int(width * pct)
    return f"  {BOLD}{color}{'█' * filled}{BR_BLACK}{'░' * (width - filled)}{RESET} {pct * 100:3.0f}%"


# ── Radio display helpers ──────────────────────────────────────────

_print_lock = threading.Lock()
_msg_counter = {"a": 0, "c": 0}


def _radio_panel(sender: str, color: str, bg: str, icon: str, message: str, t0: float) -> None:
    """Print a styled radio message panel."""
    elapsed = time.time() - t0
    ts = f"{BR_BLACK}[{elapsed:5.1f}s]{RESET}"

    with _print_lock:
        print()
        print(f"  {ts}  {BOLD}{bg}{BLACK} {icon} {sender} {RESET}  {DIM}radio transmission{RESET}")
        # Wrap long messages
        for line in textwrap.wrap(message, W - 12):
            print(f"           {color}│{RESET} {line}")
        print(f"           {color}╰{'─' * (W - 15)}{RESET}")


def _action_line(sender: str, color: str, action: str, t0: float) -> None:
    """Print a compact action line (move, turn, observe)."""
    elapsed = time.time() - t0
    with _print_lock:
        print(f"  {BR_BLACK}[{elapsed:5.1f}s]{RESET}  {color}▸ {sender}{RESET}  {DIM}{action}{RESET}")


def _subscribe_radio(topic: str, sender: str, color: str, bg: str, icon: str, t0: float) -> None:
    """Subscribe to a radio LCM stream and display messages."""
    from dimos.core.transport import pLCMTransport

    transport = pLCMTransport(topic)

    def on_msg(msg: str) -> None:
        # Filter: only show broadcast content, not internal logs
        if msg and len(msg) > 5:
            _radio_panel(sender, color, bg, icon, msg, t0)

    transport.subscribe(on_msg)


def _subscribe_actions(topic: str, sender: str, color: str, t0: float) -> None:
    """Subscribe to agent output stream to show tool calls."""
    from dimos.core.transport import pLCMTransport
    import pickle

    transport = pLCMTransport(topic)

    def on_msg(msg) -> None:
        try:
            # Agent publishes BaseMessage objects
            content = msg.content if hasattr(msg, "content") else str(msg)
            tool_calls = getattr(msg, "tool_calls", None) or (
                msg.additional_kwargs.get("tool_calls") if hasattr(msg, "additional_kwargs") else None
            )

            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})

                    if name == "broadcast":
                        pass  # Handled by radio subscriber
                    elif name == "describe_surroundings":
                        _action_line(sender, color, "👁  observing surroundings...", t0)
                    elif name == "move_forward":
                        dur = args.get("duration", 5)
                        _action_line(sender, color, f"🐾 moving forward ({dur}s)", t0)
                    elif name.startswith("turn_"):
                        direction = "↰ left" if "left" in name else "↱ right"
                        dur = args.get("duration", 2)
                        _action_line(sender, color, f"{direction} ({dur}s)", t0)
                    elif name == "request_help":
                        desc = args.get("description", "")
                        with _print_lock:
                            print(f"\n  {BOLD}{BG_RED}{WHITE} ⚠ EMERGENCY — {sender} {RESET}  {desc}")
                    elif name == "stop_moving":
                        _action_line(sender, color, "⏹  stopped", t0)
        except Exception:
            pass

    transport.subscribe(on_msg)


# ── Preflight ──────────────────────────────────────────────────────

def _preflight():
    for name in ["openclaw-gateway", "snap-store", "gnome-software"]:
        try:
            subprocess.run(["pkill", "-f", name], capture_output=True, timeout=3)
        except Exception:
            pass

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    gb = int(line.split()[1]) / 1024 / 1024
                    if gb < 8:
                        print(f"  {YELLOW}⚠ {gb:.1f} GB RAM — close browsers for best results{RESET}")
                    else:
                        print(f"  {BR_GREEN}✓ RAM: {gb:.1f} GB available{RESET}")
                    break
    except Exception:
        pass


# ── Main ───────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(HIDE_CURSOR, end="")
    try:
        _run(t0)
    finally:
        print(SHOW_CURSOR, end="")


def _run(t0):
    _preflight()

    # ── Title ──
    print()
    print(f"  {BOLD}{CYAN}{'━' * W}{RESET}")
    print()
    slow_type(f"  {BOLD}{BR_CYAN}C O L L A B O R A T I V E   P A T R O L{RESET}", 0.03)
    print(f"  {DIM}Autonomous robots  ·  Radio comms  ·  GPT-4o vision  ·  DIMOS{RESET}")
    print()
    print(f"  {BOLD}{CYAN}{'━' * W}{RESET}")
    print()

    print(f"  {BOLD}AGENTS{RESET}")
    print(f"  {BR_BLACK}{'─' * 40}{RESET}")
    print(f"  {BOLD}{BG_YELLOW}{BLACK} α {RESET} {YELLOW}Alpha{RESET}    {DIM}patrol zone A — office west{RESET}")
    print(f"  {BOLD}{BG_MAGENTA}{WHITE} χ {RESET} {MAGENTA}Charlie{RESET}  {DIM}patrol zone C — office east{RESET}")
    print(f"  {BOLD}{BG_CYAN}{BLACK} ⇄ {RESET} {CYAN}Radio{RESET}    {DIM}cross-robot messaging via LCM{RESET}")
    print()
    time.sleep(1)

    # ── Build ──
    print(hline("─"))
    timed(t0, f"{BOLD}Composing patrol blueprint...{RESET}")

    from dimos.games.patrol.blueprint import build_patrol
    from dimos.games.patrol.coordinator import PatrolCoordinator

    game = build_patrol()
    timed(t0, f"{GREEN}✓ Blueprint{RESET}  {DIM}13 modules · 2 sims · 2 agents{RESET}")

    timed(t0, f"{BOLD}Launching simulations...{RESET}")
    done = threading.Event()
    spin = threading.Thread(target=spinner, args=("Spawning MuJoCo + Agents", done, t0), daemon=True)
    spin.start()

    coordinator = game.build()

    done.set()
    spin.join()
    timed(t0, f"{BR_GREEN}✓ All modules online{RESET}")

    # ── Warmup ──
    warmup = 15
    print()
    for i in range(warmup + 1):
        pct = i / warmup
        sys.stdout.write(f"{CLEAR_LINE}{progress_bar(pct)}  {DIM}camera warmup{RESET}")
        sys.stdout.flush()
        if i < warmup:
            time.sleep(1)
    print()
    print()

    # ── Subscribe to radio + agent streams BEFORE starting mission ──
    _subscribe_radio("/radio_a_out", "ALPHA", YELLOW, BG_YELLOW, "α", t0)
    _subscribe_radio("/radio_c_out", "CHARLIE", MAGENTA, BG_MAGENTA, "χ", t0)
    _subscribe_actions("/a_agent", "Alpha", YELLOW, t0)
    _subscribe_actions("/c_agent", "Charlie", MAGENTA, t0)

    # ── Mission start ──
    patrol_ctrl = coordinator.get_instance(PatrolCoordinator)
    if patrol_ctrl is None:
        timed(t0, f"{RED}FATAL: PatrolCoordinator not found{RESET}")
        coordinator.stop()
        sys.exit(1)

    print(f"  {BOLD}{CYAN}┌{'─' * (W - 4)}┐{RESET}")
    print(f"  {BOLD}{CYAN}│{RESET}  {BOLD}MISSION{RESET}: Patrol office · Observe · Report via radio       {BOLD}{CYAN}│{RESET}")
    print(f"  {BOLD}{CYAN}└{'─' * (W - 4)}┘{RESET}")
    print()

    patrol_ctrl.start_mission("patrol office, observe surroundings, report via radio")

    from dimos.core.transport import pLCMTransport

    time.sleep(3)
    pLCMTransport("/a_human_input").publish(
        "Mission started. You are Alpha. Start moving NOW: "
        "call move_forward(5) immediately, then turn_right(2), "
        "then describe_surroundings, then broadcast what you see to Charlie. "
        "Keep this patrol loop going non-stop."
    )
    timed(t0, f"{YELLOW}▶ Alpha{RESET} deployed to zone A")

    time.sleep(5)
    pLCMTransport("/c_human_input").publish(
        "Mission started. You are Charlie. Start moving NOW: "
        "call move_forward(5) immediately, then turn_left(2), "
        "then describe_surroundings, then broadcast what you see to Alpha. "
        "Keep this patrol loop going non-stop."
    )
    timed(t0, f"{MAGENTA}▶ Charlie{RESET} deployed to zone C")

    print()
    print(f"  {BOLD}{CYAN}{'━' * W}{RESET}")
    print(f"  {BOLD}  LIVE RADIO FEED{RESET}  {DIM}— watching robot communications{RESET}")
    print(f"  {BOLD}{CYAN}{'━' * W}{RESET}")

    # ── Patrol phase — just wait while radio messages print live ──
    mission_duration = 80
    start_patrol = time.time()

    while time.time() - start_patrol < mission_duration:
        time.sleep(1)

    # ── End ──
    print()
    print(f"  {BOLD}{CYAN}{'━' * W}{RESET}")
    timed(t0, f"{BOLD}Mission complete. Shutting down...{RESET}")

    done2 = threading.Event()
    spin2 = threading.Thread(target=spinner, args=("Stopping processes", done2, t0), daemon=True)
    spin2.start()

    coordinator.stop()

    done2.set()
    spin2.join()

    elapsed = time.time() - t0
    timed(t0, f"{BR_GREEN}✓ All systems offline{RESET}")
    print()
    print(f"  {BOLD}{CYAN}{'━' * W}{RESET}")
    print(f"  {BOLD}{CYAN}COLLABORATIVE PATROL{RESET}  {DIM}completed in {elapsed:.0f}s{RESET}")
    print(f"  {DIM}13 modules · 2 sims · 2 agents · radio bridge · DIMOS{RESET}")
    print(f"  {BOLD}{CYAN}{'━' * W}{RESET}")
    print()


if __name__ == "__main__":
    main()
