"""Robot Telephone Demo Script — 70-second video recording.

This script runs a timed demo of the Robot Telephone game optimized for
screen recording. It prints clear, colorful terminal output suitable
for a demo video.

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

    # 4. (Optional) Record with asciinema:
    CI=1 asciinema rec -c "python dimos/games/telephone/demo.py" demo.cast
"""

import sys
import time

# ANSI color codes
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
DIM = "\033[2m"
RESET = "\033[0m"

BANNER = f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗
║                                                          ║
║            🤖  ROBOT TELEPHONE  🤖                       ║
║       Multi-Robot Communication Experiment               ║
║                                                          ║
║   3 robots play telephone in MuJoCo simulation           ║
║   A describes → B relays → C seeks → Judge scores        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝{RESET}
"""


def slow_print(text: str, delay: float = 0.02) -> None:
    """Print text character by character for dramatic effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def timed_print(t0: float, text: str) -> None:
    """Print with elapsed timestamp."""
    elapsed = time.time() - t0
    print(f"{DIM}[{elapsed:5.1f}s]{RESET} {text}")


def main() -> None:
    t0 = time.time()

    # ===== 0-5s: Title & Setup =====
    print(BANNER)
    time.sleep(1)

    slow_print(f"{BOLD}Architecture:{RESET}")
    print(f"  {YELLOW}Robot A{RESET} (Describer)  — VLM sees an object, describes it")
    print(f"  {GREEN}Robot B{RESET} (Relay)      — Receives description, reinterprets")
    print(f"  {MAGENTA}Robot C{RESET} (Seeker)     — Finds the object from relayed clues")
    print(f"  {CYAN}Judge{RESET}               — GPT-4o scores the telephone drift")
    print()
    time.sleep(2)

    # ===== 5-25s: Build & Initialize =====
    timed_print(t0, f"{BOLD}Initializing DIMOS modules...{RESET}")

    from dimos.games.telephone.blueprint import build_telephone_game
    from dimos.games.telephone.controller import GameController

    game = build_telephone_game(vlm_model="gpt-4o")
    timed_print(t0, "Blueprint composed: 8 modules, 3 MuJoCo processes")

    timed_print(t0, f"{BOLD}Launching simulation...{RESET}")
    coordinator = game.build()

    timed_print(t0, f"{GREEN}All modules started!{RESET}")
    timed_print(t0, "Waiting for camera streams to stabilize...")
    time.sleep(20)

    # ===== 25-30s: Start Round =====
    game_ctrl = coordinator.get_instance(GameController)
    if game_ctrl is None:
        timed_print(t0, f"{RED}ERROR: GameController not found{RESET}")
        coordinator.stop()
        sys.exit(1)

    print()
    print(f"{BOLD}{CYAN}{'═' * 58}{RESET}")
    print(f"{BOLD}{CYAN}  ROUND 1 — Target: the most interesting object{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 58}{RESET}")
    print()

    game_ctrl.start_round("the most interesting object")

    # ===== 30-55s: Watch the chain =====
    timed_print(t0, f"{YELLOW}[Robot A]{RESET} Looking at the scene...")

    last_a = ""
    last_b = ""
    last_c = ""
    scored = False

    for i in range(60):
        time.sleep(1)
        history = game_ctrl.get_history()

        if history:
            h = history[-1]
            # Print each step as it arrives
            if h.get("a") and h["a"] != last_a:
                last_a = h["a"]
                print()
                timed_print(t0, f"{BOLD}{YELLOW}[Robot A] Describes:{RESET}")
                print(f"  \"{last_a}\"")

            if h.get("b") and h["b"] != last_b:
                last_b = h["b"]
                print()
                timed_print(t0, f"{BOLD}{GREEN}[Robot B] Relays:{RESET}")
                print(f"  \"{last_b}\"")

            if h.get("c") and h["c"] != last_c:
                last_c = h["c"]
                print()
                timed_print(t0, f"{BOLD}{MAGENTA}[Robot C] Concludes:{RESET}")
                print(f"  \"{last_c}\"")

            if h.get("score") and not scored:
                scored = True
                score = h["score"]
                print()
                print(f"{BOLD}{CYAN}{'═' * 58}{RESET}")
                print(f"{BOLD}{CYAN}  RESULTS{RESET}")
                print(f"{BOLD}{CYAN}{'═' * 58}{RESET}")

                if isinstance(score, dict) and "a_accuracy" in score:
                    print(f"  Description accuracy:  {score.get('a_accuracy', '?')}/10")
                    print(f"  Relay fidelity:        {score.get('b_fidelity', '?')}/10")
                    print(f"  Seek success:          {score.get('c_success', '?')}/10")
                    print(f"  Telephone drift:       {score.get('telephone_drift', '?')}/10")
                    print(f"  {BOLD}Overall:               {score.get('overall', '?')}/10{RESET}")
                    if score.get("commentary"):
                        print(f"\n  {DIM}{score['commentary']}{RESET}")
                else:
                    print(f"  {score}")

                print(f"{BOLD}{CYAN}{'═' * 58}{RESET}")
                break

        if i % 10 == 0 and i > 0 and not scored:
            timed_print(t0, f"{DIM}  Processing... ({i}s){RESET}")

    if not scored:
        timed_print(t0, f"{RED}Timeout — chain did not complete{RESET}")

    # ===== 55-70s: Wrap up =====
    print()
    timed_print(t0, f"{BOLD}Shutting down simulation...{RESET}")
    coordinator.stop()
    timed_print(t0, f"{GREEN}Done!{RESET}")

    elapsed = time.time() - t0
    print(f"\n{DIM}Total runtime: {elapsed:.0f}s{RESET}")


if __name__ == "__main__":
    main()
