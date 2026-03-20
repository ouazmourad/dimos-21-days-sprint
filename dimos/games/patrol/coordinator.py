"""PatrolCoordinator — mission lifecycle and status tracking.

Manages start/stop of patrol missions and collects status updates
from both robots for logging.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import Out
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass
class MissionEntry:
    timestamp: float
    source: str
    message: str


class PatrolCoordinator(Module):
    """Orchestrates patrol missions for two robots."""

    mission_command: Out[str]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._active = False
        self._mission_log: list[MissionEntry] = []
        self._start_time = 0.0

    @rpc
    def start(self) -> None:
        super().start()

    def _log(self, source: str, message: str) -> None:
        entry = MissionEntry(timestamp=time.time(), source=source, message=message)
        self._mission_log.append(entry)
        logger.info(f"[PATROL] [{source}] {message}")

    @skill
    def start_mission(self, objective: str = "patrol and report observations") -> str:
        """Start a patrol mission. Both robots will begin patrolling
        their assigned zones and sharing observations via radio.

        Args:
            objective: The mission objective description.

        Returns:
            Status message.
        """
        self._active = True
        self._start_time = time.time()
        self._mission_log.clear()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  MISSION START  |  Objective: {objective}")
        logger.info(f"{'=' * 60}\n")

        self.mission_command.publish(f"PATROL_START:{objective}")
        self._log("Coordinator", f"Mission started: {objective}")

        return f"Mission started. Objective: {objective}. Both robots are now patrolling."

    @skill
    def end_mission(self) -> str:
        """End the current patrol mission.

        Returns:
            Mission summary.
        """
        if not self._active:
            return "No active mission."

        self._active = False
        elapsed = time.time() - self._start_time
        self.mission_command.publish("PATROL_STOP")
        self._log("Coordinator", f"Mission ended after {elapsed:.0f}s")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  MISSION END  |  Duration: {elapsed:.0f}s  |  Log entries: {len(self._mission_log)}")
        logger.info(f"{'=' * 60}\n")

        return f"Mission ended. Duration: {elapsed:.0f}s. {len(self._mission_log)} log entries."

    @rpc
    def get_mission_log(self) -> list[dict[str, Any]]:
        """Get the full mission log."""
        return [
            {
                "time": e.timestamp,
                "source": e.source,
                "message": e.message,
            }
            for e in self._mission_log
        ]


patrol_coordinator = PatrolCoordinator.blueprint

__all__ = ["PatrolCoordinator", "patrol_coordinator"]
