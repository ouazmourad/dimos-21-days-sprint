"""Test script for Robot Telephone game.

Builds the game, starts a round, and prints the transcript.
Supports --single mode to test with just 1 MuJoCo process.
"""

import sys
import time

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

SINGLE_MODE = "--single" in sys.argv


def test_single_robot() -> None:
    """Test just the VLM + describer with 1 MuJoCo process."""
    import copy

    from dimos.agents.vlm_agent import VLMAgent
    from dimos.core.blueprints import autoconnect
    from dimos.core.global_config import global_config
    from dimos.robot.unitree.g1.sim import g1_sim_connection

    print("=== Single Robot VLM Test ===")

    # Force low performance tier
    global_config.simulation = True
    global_config.performance_tier = "low"
    global_config.resolve_performance_tier()

    sim = g1_sim_connection()
    vlm = VLMAgent.blueprint(model="gpt-4o")

    game = autoconnect(sim, vlm)
    print("Building coordinator...")
    coordinator = game.build()

    print("Waiting 15s for camera stream...")
    time.sleep(15)

    # Get the VLMAgent and test visual_query directly
    vlm_instance = coordinator.get_instance(VLMAgent)
    if vlm_instance is None:
        print("ERROR: VLMAgent not found")
        coordinator.stop()
        sys.exit(1)

    print("Querying VLM with camera image...")
    result = vlm_instance.query("Describe what you see in detail. What objects are visible?")
    print(f"\nVLM Response: {result}")

    print("\nShutting down...")
    coordinator.stop()
    print("Done!")


def test_full_game() -> None:
    """Full 3-robot telephone game."""
    from dimos.games.telephone.blueprint import build_telephone_game
    from dimos.games.telephone.controller import GameController

    print("=== Robot Telephone Full Test ===")
    print("Building blueprint...")
    game = build_telephone_game(vlm_model="gpt-4o")

    print("Building coordinator...")
    coordinator = game.build()

    print("Waiting 20s for camera streams to stabilize...")
    time.sleep(20)

    game_ctrl = coordinator.get_instance(GameController)
    if game_ctrl is None:
        print("ERROR: GameController not found")
        coordinator.stop()
        sys.exit(1)

    print("\n=== Starting Round 1 ===")
    result = game_ctrl.start_round("the most interesting object")
    print(f"Controller says: {result}")

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
    if SINGLE_MODE:
        test_single_robot()
    else:
        test_full_game()
