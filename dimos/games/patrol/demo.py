"""Collaborative Patrol Demo — live radio conversation between two robots.

CI=1 .venv/bin/python dimos/games/patrol/demo.py
"""

import itertools
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
BG_YELLOW = "\033[43m"
BG_MAGENTA = "\033[45m"
BG_RED = "\033[41m"
BLACK = "\033[30m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR_LINE = "\033[2K\r"

W = 72

_print_lock = threading.Lock()
_msg_num = [0]


def hline(char="━", color=CYAN):
    return f"  {BOLD}{color}{char * W}{RESET}"


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


# ── Radio display ─────────────────────────────────────────────────

def _radio_msg(sender, color, bg, icon, message, t0):
    """Print a radio message as a chat bubble."""
    elapsed = time.time() - t0
    _msg_num[0] += 1

    with _print_lock:
        print()
        # Sender badge on left for Alpha, right for Charlie
        if sender == "ALPHA":
            print(f"  {BOLD}{bg}{BLACK} {icon} {sender} {RESET}  {BR_BLACK}{time.strftime('%H:%M:%S')}{RESET}")
            for line in textwrap.wrap(message, W - 6):
                print(f"  {color}┃{RESET} {line}")
            print(f"  {color}┗{'━' * 50}{RESET}")
        else:
            pad = W - len(sender) - 8
            print(f"  {' ' * pad}{BOLD}{bg}{WHITE} {icon} {sender} {RESET}  {BR_BLACK}{time.strftime('%H:%M:%S')}{RESET}")
            for line in textwrap.wrap(message, W - 6):
                rpad = W - len(line) - 4
                print(f"  {' ' * max(rpad, 2)}{line} {color}┃{RESET}")
            print(f"  {' ' * (W - 52)}{color}{'━' * 50}┛{RESET}")


def _subscribe_radio(topic, sender, color, bg, icon, t0):
    from dimos.core.transport import pLCMTransport
    transport = pLCMTransport(topic)

    def on_msg(msg):
        if msg and len(msg) > 5:
            _radio_msg(sender, color, bg, icon, msg, t0)

    transport.subscribe(on_msg)


# ── Preflight ─────────────────────────────────────────────────────

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
                    color = BR_GREEN if gb >= 8 else YELLOW
                    print(f"  {color}{'✓' if gb >= 8 else '⚠'} RAM: {gb:.1f} GB{RESET}")
                    break
    except Exception:
        pass


# ── Main ──────────────────────────────────────────────────────────

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
    print(hline())
    print()
    print(f"  {BOLD}{BR_CYAN}  C O L L A B O R A T I V E   P A T R O L{RESET}")
    print(f"  {DIM}  Two autonomous robots talking to each other in real-time{RESET}")
    print()
    print(hline())
    print()

    print(f"  {BOLD}{BG_YELLOW}{BLACK} α {RESET} {BOLD}{YELLOW}Alpha{RESET}    {DIM}patrol robot — office west side{RESET}")
    print(f"  {BOLD}{BG_MAGENTA}{WHITE} χ {RESET} {BOLD}{MAGENTA}Charlie{RESET}  {DIM}patrol robot — office east side{RESET}")
    print()
    time.sleep(1)

    # ── Build ──
    print(f"  {BR_BLACK}{'─' * W}{RESET}")

    from dimos.games.patrol.blueprint import build_patrol
    from dimos.games.patrol.coordinator import PatrolCoordinator

    game = build_patrol()

    done = threading.Event()
    spin = threading.Thread(target=spinner, args=("Launching simulations", done, t0), daemon=True)
    spin.start()

    coordinator = game.build()

    done.set()
    spin.join()
    print(f"  {BR_GREEN}✓ 13 modules online · 2 MuJoCo sims · 2 LLM agents{RESET}")

    # ── Warmup ──
    warmup = 15
    print()
    for i in range(warmup + 1):
        pct = i / warmup
        sys.stdout.write(f"{CLEAR_LINE}{progress_bar(pct)}  {DIM}waiting for cameras{RESET}")
        sys.stdout.flush()
        if i < warmup:
            time.sleep(1)
    print()

    # ── Subscribe to radio streams ONLY ──
    _subscribe_radio("/radio_a_out", "ALPHA", YELLOW, BG_YELLOW, "α", t0)
    _subscribe_radio("/radio_c_out", "CHARLIE", MAGENTA, BG_MAGENTA, "χ", t0)

    # ── Start mission ──
    patrol_ctrl = coordinator.get_instance(PatrolCoordinator)
    if patrol_ctrl is None:
        print(f"  {RED}FATAL: PatrolCoordinator not found{RESET}")
        coordinator.stop()
        sys.exit(1)

    patrol_ctrl.start_mission("patrol office, observe surroundings, report via radio")

    from dimos.core.transport import pLCMTransport

    time.sleep(3)
    pLCMTransport("/a_human_input").publish(
        "Mission started. You are Alpha. Start moving NOW: "
        "call move_forward(5) immediately, then turn_right(2), "
        "then describe_surroundings, then broadcast what you see to Charlie. "
        "Keep this patrol loop going non-stop."
    )

    time.sleep(5)
    pLCMTransport("/c_human_input").publish(
        "Mission started. You are Charlie. Start moving NOW: "
        "call move_forward(5) immediately, then turn_left(2), "
        "then describe_surroundings, then broadcast what you see to Alpha. "
        "Keep this patrol loop going non-stop."
    )

    # ── Live radio feed header ──
    print()
    print(hline())
    print(f"  {BOLD}  LIVE RADIO FEED{RESET}")
    print(hline())

    # ── Just watch the conversation ──
    mission_duration = 80
    start_patrol = time.time()

    while time.time() - start_patrol < mission_duration:
        time.sleep(1)

    # ── End ──
    print()
    print(hline())
    print(f"  {BOLD}  MISSION COMPLETE{RESET}  {DIM}· {_msg_num[0]} radio messages exchanged{RESET}")
    print(hline())
    print()

    done2 = threading.Event()
    spin2 = threading.Thread(target=spinner, args=("Shutting down", done2, t0), daemon=True)
    spin2.start()

    coordinator.stop()

    done2.set()
    spin2.join()

    elapsed = time.time() - t0
    print(f"  {BR_GREEN}✓ All systems offline{RESET}  {DIM}· {elapsed:.0f}s total{RESET}")
    print()


if __name__ == "__main__":
    main()
