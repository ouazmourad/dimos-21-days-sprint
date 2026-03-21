"""Robot Escape Room Demo — live puzzle-solving display.

CI=1 .venv/bin/python dimos/games/escape_room/demo.py
"""

import itertools
import subprocess
import sys
import textwrap
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
WHITE = "\033[37m"
BR_BLACK = "\033[90m"
BR_CYAN = "\033[96m"
BR_GREEN = "\033[92m"
BR_YELLOW = "\033[93m"
BG_YELLOW = "\033[43m"
BG_MAGENTA = "\033[45m"
BG_GREEN = "\033[42m"
BG_RED = "\033[41m"
BG_CYAN = "\033[46m"
BLACK = "\033[30m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR_LINE = "\033[2K\r"

W = 72
_print_lock = threading.Lock()
_clues_found = [0]
_game_won = [False]


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


def _radio_msg(sender, color, bg, icon, message, t0):
    with _print_lock:
        print()
        if sender == "TRAPPED":
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


def _subscribe_game_events(t0):
    """Subscribe to game events (clue found, game won)."""
    from dimos.core.transport import pLCMTransport
    transport = pLCMTransport("/game_event")

    def on_event(msg):
        if not msg:
            return
        with _print_lock:
            if msg.startswith("CLUE_FOUND:"):
                parts = msg.split(":")
                found = int(parts[1])
                name = parts[2]
                _clues_found[0] = found
                print()
                print(f"  {BOLD}{BG_GREEN}{BLACK} ✓ CLUE {found}/3 FOUND {RESET}  {GREEN}{name}{RESET}")
                boxes = f"{'■' * found}{'□' * (3 - found)}"
                print(f"  {GREEN}  Progress: [{boxes}]{RESET}")
                print()
            elif msg.startswith("GAME_WON:"):
                secs = msg.split(":")[1]
                _game_won[0] = True
                print()
                print(f"  {BOLD}{BG_GREEN}{BLACK}{'=' * 50}{RESET}")
                print(f"  {BOLD}{BG_GREEN}{BLACK}   🎉  ESCAPED IN {secs} SECONDS!  🎉   {RESET}")
                print(f"  {BOLD}{BG_GREEN}{BLACK}{'=' * 50}{RESET}")
                print()

    transport.subscribe(on_event)


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
                    color = BR_GREEN if gb >= 6 else YELLOW
                    print(f"  {color}{'✓' if gb >= 6 else '⚠'} RAM: {gb:.1f} GB{RESET}")
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

    # ── Title ──
    print()
    print(hline())
    print()
    print(f"  {BOLD}{BR_CYAN}  R O B O T   E S C A P E   R O O M{RESET}")
    print(f"  {DIM}  One robot trapped  ·  One guide  ·  3 clues to find{RESET}")
    print()
    print(hline())
    print()

    print(f"  {BOLD}{BG_YELLOW}{BLACK} 🔒 {RESET} {BOLD}{YELLOW}Trapped{RESET}  {DIM}robot in the room — must find 3 clues{RESET}")
    print(f"  {BOLD}{BG_MAGENTA}{WHITE} 🗝 {RESET} {BOLD}{MAGENTA}Guide{RESET}    {DIM}gives hints via radio — cannot see the room{RESET}")
    print(f"  {DIM}  Progress: [□□□] — find all 3 to escape{RESET}")
    print()
    time.sleep(1)

    # ── Build ──
    print(f"  {BR_BLACK}{'─' * W}{RESET}")

    from dimos.games.escape_room.blueprint import build_escape_room
    from dimos.games.escape_room.game_master import GameMaster

    game = build_escape_room()

    done = threading.Event()
    spin = threading.Thread(target=spinner, args=("Launching simulation", done, t0), daemon=True)
    spin.start()

    coordinator = game.build()

    done.set()
    spin.join()
    print(f"  {BR_GREEN}✓ Modules online · 1 MuJoCo sim · 2 LLM agents{RESET}")

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

    # ── Subscribe to radio + game events ──
    _subscribe_radio("/radio_trapped_out", "TRAPPED", YELLOW, BG_YELLOW, "🔒", t0)
    _subscribe_radio("/radio_guide_out", "GUIDE", MAGENTA, BG_MAGENTA, "🗝", t0)
    _subscribe_game_events(t0)

    # ── Start game ──
    print()
    print(hline())
    print(f"  {BOLD}  ESCAPE ROOM — LIVE{RESET}")
    print(hline())

    from dimos.core.transport import pLCMTransport

    time.sleep(3)

    # Kick off the Guide — it will call start_game and broadcast hints
    pLCMTransport("/guide_input").publish(
        "The escape room is ready. Call start_game to begin, then "
        "broadcast the first hint to the Trapped robot."
    )

    time.sleep(3)

    # Kick off the Trapped robot — tell it to start searching
    pLCMTransport("/trapped_input").publish(
        "You are in the escape room. The Guide will give you hints via radio. "
        "Start by calling describe_surroundings to see where you are, then "
        "move around and search for clues. When you find something matching "
        "the hint, broadcast a detailed description to the Guide."
    )

    # ── Wait for game to complete or timeout ──
    game_timeout = 180  # 3 minutes max
    start_game = time.time()

    while time.time() - start_game < game_timeout:
        if _game_won[0]:
            time.sleep(3)  # let the celebration print
            break
        time.sleep(1)

    if not _game_won[0]:
        with _print_lock:
            print()
            print(f"  {BOLD}{RED}  TIME'S UP — {_clues_found[0]}/3 clues found{RESET}")
            print()

    # ── Shutdown ──
    print(hline())

    done2 = threading.Event()
    spin2 = threading.Thread(target=spinner, args=("Shutting down", done2, t0), daemon=True)
    spin2.start()

    coordinator.stop()

    done2.set()
    spin2.join()

    elapsed = time.time() - t0
    print(f"  {BR_GREEN}✓ Done{RESET}  {DIM}· {elapsed:.0f}s total · {_clues_found[0]}/3 clues{RESET}")
    print()


if __name__ == "__main__":
    main()
