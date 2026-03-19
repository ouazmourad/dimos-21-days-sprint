"""Robot Telephone Demo Script — 100-second video recording.

This script runs a timed demo of the Robot Telephone game optimized for
screen recording. Futuristic terminal UI with animated spinners,
progress bars, and styled output panels.

Prerequisites:
    # 1. Apply LCM system settings (one-time, needs sudo):
    sudo ip link set lo multicast on
    sudo ip route add 224.0.0.0/4 dev lo
    sudo sysctl -w net.core.rmem_max=67108864
    sudo sysctl -w net.core.rmem_default=67108864

    # 2. Set your OpenAI API key:
    export OPENAI_API_KEY="sk-..."

    # 3. Run the demo:
    CI=1 python dimos/games/telephone/demo.py
"""

import itertools
import sys
import threading
import time
import textwrap

# ── ANSI escape codes ──────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
BLINK = "\033[5m"
RESET = "\033[0m"

# Foreground
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

# Bright foreground
BR_BLACK = "\033[90m"
BR_RED = "\033[91m"
BR_GREEN = "\033[92m"
BR_YELLOW = "\033[93m"
BR_BLUE = "\033[94m"
BR_MAGENTA = "\033[95m"
BR_CYAN = "\033[96m"
BR_WHITE = "\033[97m"

# Background
BG_BLACK = "\033[40m"
BG_CYAN = "\033[46m"
BG_WHITE = "\033[47m"

# Cursor
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR_LINE = "\033[2K\r"

W = 62  # panel width


# ── Drawing helpers ────────────────────────────────────────────────

def box_top(title: str = "", color: str = CYAN) -> str:
    if title:
        pad = W - 4 - len(title)
        return f"{BOLD}{color}┌─ {title} {'─' * pad}┐{RESET}"
    return f"{BOLD}{color}┌{'─' * (W - 2)}┐{RESET}"


def box_mid(text: str = "", color: str = CYAN, text_color: str = "") -> str:
    tc = text_color or RESET
    inner = W - 4
    if not text:
        return f"{BOLD}{color}│{RESET}{' ' * (W - 2)}{BOLD}{color}│{RESET}"
    lines = textwrap.wrap(text, inner) or [""]
    out = []
    for line in lines:
        pad = inner - len(line)
        out.append(f"{BOLD}{color}│{RESET} {tc}{line}{RESET}{' ' * pad} {BOLD}{color}│{RESET}")
    return "\n".join(out)


def box_bot(color: str = CYAN) -> str:
    return f"{BOLD}{color}└{'─' * (W - 2)}┘{RESET}"


def hline(char: str = "─", color: str = BR_BLACK) -> str:
    return f"{color}{char * W}{RESET}"


def tag(label: str, color: str, bg: str = "") -> str:
    return f"{BOLD}{bg}{color} {label} {RESET}"


def progress_bar(pct: float, width: int = 30, color: str = CYAN) -> str:
    filled = int(width * pct)
    bar = f"{BOLD}{color}{'█' * filled}{BR_BLACK}{'░' * (width - filled)}{RESET}"
    return f"  {bar} {pct * 100:3.0f}%"


def spinner(msg: str, done_event: threading.Event, t0: float) -> None:
    """Animated spinner that runs until done_event is set."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    for f in itertools.cycle(frames):
        if done_event.is_set():
            break
        elapsed = time.time() - t0
        sys.stdout.write(f"{CLEAR_LINE}  {BOLD}{CYAN}{f}{RESET} {msg} {DIM}[{elapsed:.0f}s]{RESET}")
        sys.stdout.flush()
        time.sleep(0.08)
    sys.stdout.write(f"{CLEAR_LINE}")
    sys.stdout.flush()


def timed(t0: float, text: str) -> None:
    elapsed = time.time() - t0
    print(f"  {BR_BLACK}[{elapsed:5.1f}s]{RESET}  {text}")


def slow_type(text: str, delay: float = 0.015) -> None:
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def panel_quote(speaker: str, color: str, icon: str, text: str) -> None:
    print(f"\n  {BOLD}{color}{icon} {speaker}{RESET}")
    for line in textwrap.wrap(text, W - 8):
        print(f"  {color}│{RESET} {DIM}{line}{RESET}")
    print(f"  {color}╰{'─' * 20}{RESET}")


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print(HIDE_CURSOR, end="")

    try:
        _run(t0)
    finally:
        print(SHOW_CURSOR, end="")


def _run(t0: float) -> None:

    # ═══════════════════════════════════════════════════════════════
    #  SCENE 1 — TITLE
    # ═══════════════════════════════════════════════════════════════
    print()
    print(hline("═", CYAN))
    print()
    slow_type(f"  {BOLD}{BR_CYAN}R O B O T   T E L E P H O N E{RESET}", 0.04)
    print(f"  {DIM}Multi-Robot Communication Experiment  //  DIMOS Framework{RESET}")
    print()
    print(hline("═", CYAN))
    print()
    time.sleep(0.8)

    # Architecture
    print(f"  {BOLD}{WHITE}SYSTEM ARCHITECTURE{RESET}")
    print(f"  {BR_BLACK}{'─' * 40}{RESET}")
    print(f"  {tag('A', BLACK, f'{BG_WHITE}')} {YELLOW}Describer{RESET}   {DIM}VLM sees scene, describes object{RESET}")
    print(f"  {tag('B', BLACK, f'{BG_WHITE}')} {GREEN}Relay{RESET}       {DIM}Reinterprets description in own words{RESET}")
    print(f"  {tag('C', BLACK, f'{BG_WHITE}')} {MAGENTA}Seeker{RESET}      {DIM}Receives clue, searches & verifies{RESET}")
    print(f"  {tag('J', BLACK, f'{BG_WHITE}')} {CYAN}Judge{RESET}       {DIM}GPT-4o scores the telephone drift{RESET}")
    print()
    print(f"  {DIM}Pipeline:  A ──text──▶ B ──text──▶ C ──verdict──▶ Judge{RESET}")
    print()
    time.sleep(2)

    # ═══════════════════════════════════════════════════════════════
    #  SCENE 2 — BUILD & LAUNCH
    # ═══════════════════════════════════════════════════════════════
    print(hline("─", BR_BLACK))
    timed(t0, f"{BOLD}Initializing DIMOS modules...{RESET}")

    from dimos.games.telephone.blueprint import build_telephone_game
    from dimos.games.telephone.controller import GameController

    game = build_telephone_game(vlm_model="gpt-4o")
    timed(t0, f"{GREEN}Blueprint composed{RESET}  {DIM}8 modules | 2 MuJoCo sims | 3 VLM agents{RESET}")

    # Launch with spinner
    timed(t0, f"{BOLD}Launching simulation...{RESET}")
    done = threading.Event()
    spin = threading.Thread(target=spinner, args=("Spawning MuJoCo processes", done, t0), daemon=True)
    spin.start()

    coordinator = game.build()

    done.set()
    spin.join()
    timed(t0, f"{BR_GREEN}All modules online{RESET}  {DIM}streams connected{RESET}")

    # Camera warmup with progress bar
    warmup = 20
    print()
    for i in range(warmup + 1):
        pct = i / warmup
        sys.stdout.write(f"{CLEAR_LINE}{progress_bar(pct, 30, CYAN)}  {DIM}camera stream warmup{RESET}")
        sys.stdout.flush()
        if i < warmup:
            time.sleep(1)
    print()
    print()

    # ═══════════════════════════════════════════════════════════════
    #  SCENE 3 — ROUND START
    # ═══════════════════════════════════════════════════════════════
    game_ctrl = coordinator.get_instance(GameController)
    if game_ctrl is None:
        timed(t0, f"{RED}FATAL: GameController not found{RESET}")
        coordinator.stop()
        sys.exit(1)

    print(box_top("ROUND 1", CYAN))
    print(box_mid("Target: the most interesting object", CYAN, BR_CYAN))
    print(box_bot(CYAN))
    print()

    game_ctrl.start_round("the most interesting object")

    timed(t0, f"{YELLOW}Robot A{RESET} scanning environment...")

    # ═══════════════════════════════════════════════════════════════
    #  SCENE 4 — WATCH THE CHAIN
    # ═══════════════════════════════════════════════════════════════
    last_a = ""
    last_b = ""
    last_c = ""
    scored = False

    for i in range(90):
        time.sleep(1)
        history = game_ctrl.get_history()

        if history:
            h = history[-1]

            if h.get("a") and h["a"] != last_a:
                last_a = h["a"]
                panel_quote("ROBOT A  //  Describer", YELLOW, "◉", last_a)

            if h.get("b") and h["b"] != last_b:
                last_b = h["b"]
                panel_quote("ROBOT B  //  Relay", GREEN, "◉", last_b)

            if h.get("c") and h["c"] != last_c:
                last_c = h["c"]
                panel_quote("ROBOT C  //  Seeker", MAGENTA, "◉", last_c)

            if h.get("score") and not scored:
                scored = True
                score = h["score"]
                print()
                print(box_top("RESULTS", BR_CYAN))

                if isinstance(score, dict) and "a_accuracy" in score:
                    def score_bar(label: str, val: int) -> str:
                        bar = f"{'▓' * val}{'░' * (10 - val)}"
                        color = BR_GREEN if val >= 7 else (YELLOW if val >= 4 else RED)
                        return f"  {label:<26s} {color}{bar}{RESET}  {BOLD}{val}/10{RESET}"

                    print(box_mid())
                    print(score_bar("  Description accuracy", score.get("a_accuracy", 0)))
                    print(score_bar("  Relay fidelity", score.get("b_fidelity", 0)))
                    print(score_bar("  Seek success", score.get("c_success", 0)))
                    print(score_bar("  Telephone drift", score.get("telephone_drift", 0)))
                    print()
                    overall = score.get("overall", 0)
                    oc = BR_GREEN if overall >= 7 else (YELLOW if overall >= 4 else RED)
                    print(f"  {BOLD}  OVERALL{' ' * 19}{oc}{'█' * overall}{'░' * (10 - overall)}{RESET}  {BOLD}{oc}{overall}/10{RESET}")
                    print(box_mid())

                    if score.get("commentary"):
                        print(box_mid(score["commentary"], BR_CYAN, DIM))
                else:
                    print(box_mid(str(score), BR_CYAN))

                print(box_bot(BR_CYAN))
                break

        if i % 15 == 0 and i > 0 and not scored:
            timed(t0, f"{DIM}processing... ({i}s){RESET}")

    if not scored:
        timed(t0, f"{RED}Timeout — chain did not complete{RESET}")

    # ═══════════════════════════════════════════════════════════════
    #  SCENE 5 — SHUTDOWN
    # ═══════════════════════════════════════════════════════════════
    print()
    print(hline("─", BR_BLACK))
    timed(t0, f"Shutting down simulation...")

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
    print(f"  {BOLD}{CYAN}ROBOT TELEPHONE{RESET}  {DIM}completed in {elapsed:.0f}s{RESET}")
    print(f"  {DIM}8 modules | 2 sims | 3 VLMs | 1 judge | DIMOS Framework{RESET}")
    print(hline("═", CYAN))
    print()


if __name__ == "__main__":
    main()
