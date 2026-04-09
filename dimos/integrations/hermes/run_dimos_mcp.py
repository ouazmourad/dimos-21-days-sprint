"""Launch DIMOS MCP server with a Go2 sim, ready for Hermes to connect.

Usage:
    # Light variant (default — fits 16 GB laptops):
    CI=1 .venv/bin/python -m dimos.integrations.hermes.run_dimos_mcp

    # Full variant (perception + nav; needs >= 32 GB):
    CI=1 .venv/bin/python -m dimos.integrations.hermes.run_dimos_mcp --full

After launching, point Hermes at http://127.0.0.1:9990/mcp via cli-config.yaml.
The script blocks — Ctrl-C to shut down. The MuJoCo viewer window will
appear once the sim is ready (~60s for first run, faster after assets cache).
"""

import atexit
import fcntl
import os
import signal
import sys
import time

from dimos.core.global_config import global_config
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_LOCK_PATH = "/tmp/dimos_mcp_hermes.lock"
_MIN_RAM_GB_LITE = 3.0
_MIN_RAM_GB_FULL = 8.0


def _check_available_ram(min_gb: float) -> None:
    """Abort early if available RAM is below the minimum threshold."""
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                meminfo[parts[0].rstrip(":")] = int(parts[1]) * 1024  # kB -> bytes
        available_gb = meminfo.get("MemAvailable", 0) / (1024**3)
    except (OSError, KeyError, ValueError):
        return  # non-Linux or can't read — skip check
    if available_gb < min_gb:
        print(f"\n  ERROR: Only {available_gb:.1f} GB RAM available, need {min_gb:.0f} GB minimum.")
        print("  Close other applications or use the lite variant.\n")
        sys.exit(1)
    print(f"  RAM check: {available_gb:.1f} GB available (need {min_gb:.0f} GB) — OK")


def _acquire_lockfile() -> "int | None":
    """Prevent multiple DIMOS MuJoCo instances from launching at once.

    Returns the fd on success. Exits if another instance holds the lock.
    """
    try:
        fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_WRONLY, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, f"{os.getpid()}\n".encode())
    except OSError:
        print("\n  ERROR: Another DIMOS MCP server is already running.")
        print(f"  Lock file: {_LOCK_PATH}")
        print("  Stop it first, or delete the lock file if stale.\n")
        sys.exit(1)
    return fd


def _release_lockfile(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        os.unlink(_LOCK_PATH)
    except OSError:
        pass


def main() -> None:
    full = "--full" in sys.argv

    # Guard: prevent accidental multi-instance launches that OOM the machine
    min_ram = _MIN_RAM_GB_FULL if full else _MIN_RAM_GB_LITE
    _check_available_ram(min_ram)
    lock_fd = _acquire_lockfile()
    atexit.register(_release_lockfile, lock_fd)

    # Performance tier: low — fits 16 GB laptops.
    global_config.simulation = True
    global_config.performance_tier = "low"
    global_config.resolve_performance_tier()

    # Override after resolve so these don't get clobbered
    global_config.mujoco_steps_per_frame = 7  # Go2 RL policy needs >= 7
    global_config.mujoco_video_width = 160
    global_config.mujoco_video_height = 120
    global_config.mujoco_video_fps = 5
    global_config.mujoco_lidar_fps = 1
    global_config.mujoco_shadows = False
    global_config.mujoco_reflections = False
    global_config.mujoco_shadowsize = 0

    # MCP host/port — match the example Hermes config
    global_config.mcp_host = "127.0.0.1"
    global_config.mcp_port = 9990

    print()
    print("=" * 64)
    print("  DIMOS <-> Hermes  |  MCP Bridge for Go2 Quadruped")
    print("=" * 64)
    print()
    print(f"  Variant:   {'FULL (perception + nav)' if full else 'LITE (movement only)'}")
    print(f"  Endpoint:  http://{global_config.mcp_host}:{global_config.mcp_port}/mcp")
    print(f"  Robot:     Unitree Go2 quadruped (MuJoCo simulation)")
    print(f"  Perf:      low (160x120, 5 fps, no shadows)")
    print()
    print("  Connect Hermes by adding to ~/.hermes/config.yaml:")
    print()
    print("    mcp_servers:")
    print("      dimos:")
    print(f"        url: http://{global_config.mcp_host}:{global_config.mcp_port}/mcp")
    print("        timeout: 60")
    print("        connect_timeout: 30")
    print()
    print("  Required env vars:")
    print("    OPENAI_API_KEY  — for the speak skill (TTS via OpenAI)")
    print("                      Optional. If absent, speak() will fail but")
    print("                      movement and navigation still work.")
    print()
    print("=" * 64)
    print()

    if full:
        from dimos.integrations.hermes.blueprint import hermes_dimos_go2_full as bp
    else:
        from dimos.integrations.hermes.blueprint import hermes_dimos_go2_lite as bp

    print("Building blueprint...")
    coordinator = bp.build()
    print("All modules online. MCP server is now accepting connections.")
    print("Press Ctrl-C to shut down.")
    print()

    stop = False

    def _on_signal(signum, frame):
        nonlocal stop
        print()
        print("Shutting down...")
        stop = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        while not stop:
            time.sleep(1)
    finally:
        coordinator.stop()
        print("Done.")


if __name__ == "__main__":
    main()
