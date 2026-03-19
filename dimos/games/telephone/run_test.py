"""Test script for Robot Telephone game.

Builds the game, starts a round, and prints the transcript.
"""

import sys
import time

from dimos.games.telephone.blueprint import build_telephone_game
from dimos.games.telephone.controller import GameController
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def main() -> None:
    print("=== Robot Telephone Test ===")
    print("Building blueprint...")
    game = build_telephone_game(vlm_model="gpt-4o")

    print("Building coordinator...")
    coordinator = game.build()

    print("All modules started! Waiting for camera streams to stabilize...")
    # MuJoCo subprocesses need time to start rendering and for LCM
    # transport to begin delivering frames to VLMAgents.
    time.sleep(15)
    print("Streams should be flowing now.")

    # Get the GameController proxy via its type
    game_ctrl = coordinator.get_instance(GameController)
    if game_ctrl is None:
        print("ERROR: GameController not found")
        sys.exit(1)

    print("\n=== Starting Round 1 ===")
    result = game_ctrl.start_round("the most interesting object")
    print(f"Controller says: {result}")

    # Wait for the telephone chain to complete (A -> B -> C -> score)
    print("Waiting for telephone chain to complete (up to 120s)...")
    for i in range(120):
        time.sleep(1)
        history = game_ctrl.get_history()
        if history and history[-1].get("score"):
            print("\n=== ROUND COMPLETE ===")
            h = history[-1]
            print(f"  [A] Described: {h['a']}")
            print(f"  [B] Relayed:   {h['b']}")
            print(f"  [C] Found:     {h['c']}")
            print(f"  Score: {h['score']}")
            break
        if i % 10 == 0 and i > 0:
            print(f"  ... still waiting ({i}s)")
    else:
        print("Timeout waiting for round to complete")
        history = game_ctrl.get_history()
        if history:
            print(f"  Partial: {history[-1]}")

    print("\nShutting down...")
    coordinator.stop()
    print("Done!")


if __name__ == "__main__":
    main()
