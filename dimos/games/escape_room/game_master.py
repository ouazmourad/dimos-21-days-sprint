"""GameMaster — orchestrates the escape room puzzle sequence.

Tracks which clues have been found, validates discoveries via VLM,
and announces progress. The Guide agent receives hints from here.
"""

import time
from typing import Any

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.games.escape_room.puzzles import PUZZLES, Puzzle, check_discovery
from dimos.utils.logging_config import setup_logger
from reactivex.disposable import Disposable

logger = setup_logger()


class GameMaster(Module):
    """Orchestrates the escape room: tracks puzzle progress, validates
    clue discoveries, and feeds hints to the Guide."""

    game_event: Out[str]  # announcements (clue found, game won, etc.)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._current_puzzle = 0
        self._found: list[bool] = [False] * len(PUZZLES)
        self._start_time = 0.0
        self._game_active = False

    @rpc
    def start(self) -> None:
        super().start()

    @skill
    def start_game(self) -> str:
        """Start the escape room game. The Trapped robot must find 3 clues.

        Returns:
            The first hint for the Guide to relay.
        """
        self._start_time = time.time()
        self._game_active = True
        self._current_puzzle = 0
        self._found = [False] * len(PUZZLES)

        logger.info("\n" + "=" * 60)
        logger.info("  ESCAPE ROOM STARTED")
        logger.info(f"  {len(PUZZLES)} clues to find")
        logger.info("=" * 60 + "\n")

        self.game_event.publish("GAME_START")

        puzzle = PUZZLES[0]
        return (
            f"Escape room started! There are {len(PUZZLES)} clues to find. "
            f"Here is the first hint to give the Trapped robot: {puzzle.hint}"
        )

    @skill
    def submit_discovery(self, description: str) -> str:
        """Submit what the Trapped robot found. The GameMaster checks
        if it matches the current clue.

        Args:
            description: What the robot sees (from describe_surroundings).

        Returns:
            Whether the clue was correct and the next hint (if any).
        """
        if not self._game_active:
            return "No game in progress. Call start_game first."

        if self._current_puzzle >= len(PUZZLES):
            return "All clues already found! The game is complete."

        puzzle = PUZZLES[self._current_puzzle]

        if check_discovery(description, puzzle):
            self._found[self._current_puzzle] = True
            self._current_puzzle += 1
            elapsed = time.time() - self._start_time

            found_count = sum(self._found)
            total = len(PUZZLES)

            logger.info(f"\n{'=' * 60}")
            logger.info(f"  CLUE {found_count}/{total} FOUND: {puzzle.name}")
            logger.info(f"  Time: {elapsed:.0f}s")
            logger.info(f"{'=' * 60}\n")

            self.game_event.publish(f"CLUE_FOUND:{found_count}:{puzzle.name}")

            if self._current_puzzle >= len(PUZZLES):
                self._game_active = False
                self.game_event.publish(f"GAME_WON:{elapsed:.0f}")
                logger.info(f"\n{'*' * 60}")
                logger.info(f"  ESCAPE ROOM COMPLETE! Time: {elapsed:.0f}s")
                logger.info(f"{'*' * 60}\n")
                return (
                    f"CORRECT! That was {puzzle.name}! "
                    f"All {total} clues found in {elapsed:.0f} seconds! "
                    f"The Trapped robot has ESCAPED! Tell them they did it!"
                )

            next_puzzle = PUZZLES[self._current_puzzle]
            return (
                f"CORRECT! That was {puzzle.name}! "
                f"({found_count}/{total} found). "
                f"Here is the next hint: {next_puzzle.hint}"
            )

        return (
            f"Not quite — that doesn't match the current clue. "
            f"Remind the Trapped robot: {puzzle.hint}"
        )

    @skill
    def get_current_hint(self) -> str:
        """Get the hint for the current puzzle.

        Returns:
            The current hint text.
        """
        if not self._game_active:
            return "No game in progress."
        if self._current_puzzle >= len(PUZZLES):
            return "All clues found!"

        puzzle = PUZZLES[self._current_puzzle]
        found = sum(self._found)
        return f"Clue {found + 1}/{len(PUZZLES)}: {puzzle.hint}"

    @rpc
    def get_progress(self) -> dict[str, Any]:
        """Get game progress."""
        elapsed = time.time() - self._start_time if self._game_active else 0
        return {
            "active": self._game_active,
            "found": sum(self._found),
            "total": len(PUZZLES),
            "elapsed": elapsed,
            "current_puzzle": PUZZLES[self._current_puzzle].name if self._current_puzzle < len(PUZZLES) else "DONE",
        }


game_master = GameMaster.blueprint

__all__ = ["GameMaster", "game_master"]
