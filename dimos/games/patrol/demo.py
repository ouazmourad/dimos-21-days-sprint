"""Collaborative Patrol Demo — 80-second demo after launch.

Prerequisites:
    sudo ip link set lo multicast on
    sudo ip route add 224.0.0.0/4 dev lo 2>/dev/null
    sudo sysctl -w net.core.rmem_max=67108864
    sudo sysctl -w net.core.rmem_default=67108864

    CI=1 .venv/bin/python dimos/games/patrol/demo.py
"""

import itertools
import os
import subprocess
import sys
import threading
import time

# ANSI
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BR_BLACK = "\033[90m"
BR_CYAN = "\033[96m"
BR_GREEN = "\033[92m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR_LINE = "\033[2K\r"

W = 62


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
                        print(f"  {YELLOW}WARNING: {gb:.1f} GB RAM. Close browsers for best results.{RESET}")
                    else:
                        print(f"  {BR_GREEN}RAM: {gb:.1f} GB available{RESET}")
                    break
    except Exception:
        pass


def main():
    t0 = time.time()
    print(HIDE_CURSOR, end="")
    try:
        _run(t0)
    finally:
        print(SHOW_CURSOR, end="")


def _run(t0):
    _preflight()

    # ── Title (0-3s) ──
    print()
    print(hline("═", CYAN))
    print()
    slow_type(f"  {BOLD}{BR_CYAN}C O L L A B O R A T I V E   P A T R O L{RESET}", 0.03)
    print(f"  {DIM}Two autonomous robots  //  Radio communication  //  DIMOS{RESET}")
    print()
    print(hline("═", CYAN))
    print()

    print(f"  {BOLD}SYSTEM{RESET}")
    print(f"  {BR_BLACK}{'─' * 40}{RESET}")
    print(f"  {YELLOW}Alpha{RESET}    Patrol robot — zone A")
    print(f"  {MAGENTA}Charlie{RESET}  Patrol robot — zone C")
    print(f"  {GREEN}Radio{RESET}    Cross-robot messaging via LCM streams")
    print(f"  {CYAN}VLM{RESET}      GPT-4o vision for scene observation")
    print()
    time.sleep(1)

    # ── Build & Launch ──
    print(hline("─", BR_BLACK))
    timed(t0, f"{BOLD}Initializing patrol modules...{RESET}")

    from dimos.games.patrol.blueprint import build_patrol
    from dimos.games.patrol.coordinator import PatrolCoordinator

    game = build_patrol()
    timed(t0, f"{GREEN}Blueprint composed{RESET}  {DIM}13 modules | 2 MuJoCo sims | 2 Agents{RESET}")

    timed(t0, f"{BOLD}Launching simulations...{RESET}")
    done = threading.Event()
    spin = threading.Thread(target=spinner, args=("Spawning MuJoCo + Agents", done, t0), daemon=True)
    spin.start()

    coordinator = game.build()

    done.set()
    spin.join()
    timed(t0, f"{BR_GREEN}All modules online{RESET}")

    # ── Camera warmup (15s) ──
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

    # ── Start mission ──
    patrol_ctrl = coordinator.get_instance(PatrolCoordinator)
    if patrol_ctrl is None:
        timed(t0, f"{RED}FATAL: PatrolCoordinator not found{RESET}")
        coordinator.stop()
        sys.exit(1)

    print(f"  {BOLD}{CYAN}┌─ MISSION ─────────────────────────────────────────────┐{RESET}")
    print(f"  {BOLD}{CYAN}│{RESET} Patrol office. Observe and report via radio.         {BOLD}{CYAN}│{RESET}")
    print(f"  {BOLD}{CYAN}│{RESET} Duration: 80 seconds.                                {BOLD}{CYAN}│{RESET}")
    print(f"  {BOLD}{CYAN}└───────────────────────────────────────────────────────┘{RESET}")
    print()

    patrol_ctrl.start_mission("patrol office, observe surroundings, report via radio")

    # ── Inject patrol commands with staggered timing ──
    from dimos.core.transport import pLCMTransport

    time.sleep(3)

    # Alpha starts first — move immediately then observe
    pLCMTransport("/a_human_input").publish(
        "Mission started. You are Alpha. Start moving NOW: "
        "call move_forward(5) immediately, then turn_right(2), "
        "then describe_surroundings, then broadcast what you see to Charlie. "
        "Keep this patrol loop going non-stop."
    )
    timed(t0, f"{YELLOW}Alpha{RESET} patrol command injected")

    time.sleep(5)

    # Charlie starts after Alpha is already moving
    pLCMTransport("/c_human_input").publish(
        "Mission started. You are Charlie. Start moving NOW: "
        "call move_forward(5) immediately, then turn_left(2), "
        "then describe_surroundings, then broadcast what you see to Alpha. "
        "Keep this patrol loop going non-stop."
    )
    timed(t0, f"{MAGENTA}Charlie{RESET} patrol command injected")

    # ── Patrol phase (80s) ──
    mission_duration = 80
    start_patrol = time.time()

    while time.time() - start_patrol < mission_duration:
        remaining = int(mission_duration - (time.time() - start_patrol))
        if remaining > 0 and remaining % 20 == 0:
            timed(t0, f"{DIM}{remaining}s remaining...{RESET}")
        time.sleep(5)

    # ── End mission ──
    print()
    print(hline("─", BR_BLACK))
    patrol_ctrl.end_mission()

    log = patrol_ctrl.get_mission_log()
    if log:
        print()
        print(f"  {BOLD}MISSION LOG{RESET}  {DIM}({len(log)} entries){RESET}")
        print(f"  {BR_BLACK}{'─' * 40}{RESET}")
        for entry in log[-10:]:
            src = entry["source"]
            color = YELLOW if src == "Alpha" else (MAGENTA if src == "Charlie" else CYAN)
            msg = entry["message"][:80]
            print(f"  {color}{src:>12}{RESET}  {DIM}{msg}{RESET}")
        if len(log) > 10:
            print(f"  {DIM}... and {len(log) - 10} more{RESET}")

    # ── Shutdown ──
    print()
    timed(t0, "Shutting down...")
    done2 = threading.Event()
    spin2 = threading.Thread(target=spinner, args=("Stopping processes", done2, t0), daemon=True)
    spin2.start()

    coordinator.stop()

    done2.set()
    spin2.join()

    elapsed = time.time() - t0
    timed(t0, f"{BR_GREEN}All systems offline{RESET}")
    print()
    print(hline("═", CYAN))
    print(f"  {BOLD}{CYAN}COLLABORATIVE PATROL{RESET}  {DIM}completed in {elapsed:.0f}s{RESET}")
    print(f"  {DIM}13 modules | 2 sims | 2 agents | radio bridge | DIMOS{RESET}")
    print(hline("═", CYAN))
    print()


if __name__ == "__main__":
    main()
