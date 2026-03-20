"""SimpleNavSkill — basic movement commands for patrol robots.

Avoids loading Qwen model (which NavigationSkillContainer requires),
keeping VRAM usage minimal. The Agent can command forward/turn/stop
via these skills.
"""

import threading
import time
from typing import Any

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class SimpleNavSkill(Module):
    """Basic movement skills for patrol robots — forward, turn, stop."""

    cmd_vel: Out[Twist]
    odom: In[PoseStamped]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._moving = False
        self._stop_timer: threading.Timer | None = None

    def _send_velocity(self, linear_x: float, angular_z: float, duration: float) -> None:
        twist = Twist(
            linear=Vector3(linear_x, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, angular_z),
        )
        self.cmd_vel.publish(twist)
        self._moving = True

        if self._stop_timer:
            self._stop_timer.cancel()

        def _auto_stop():
            self._send_stop()

        self._stop_timer = threading.Timer(duration, _auto_stop)
        self._stop_timer.daemon = True
        self._stop_timer.start()

    def _send_stop(self) -> None:
        twist = Twist(
            linear=Vector3(0.0, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 0.0),
        )
        self.cmd_vel.publish(twist)
        self._moving = False

    @skill
    def move_forward(self, duration: float = 2.0) -> str:
        """Walk forward for a specified duration in seconds.

        Args:
            duration: How long to walk forward (default 2 seconds).

        Returns:
            Status message.
        """
        self._send_velocity(0.3, 0.0, duration)
        logger.info(f"[NAV] Moving forward for {duration}s")
        return f"Moving forward for {duration} seconds."

    @skill
    def move_backward(self, duration: float = 2.0) -> str:
        """Walk backward for a specified duration in seconds.

        Args:
            duration: How long to walk backward (default 2 seconds).

        Returns:
            Status message.
        """
        self._send_velocity(-0.2, 0.0, duration)
        logger.info(f"[NAV] Moving backward for {duration}s")
        return f"Moving backward for {duration} seconds."

    @skill
    def turn_left(self, duration: float = 1.5) -> str:
        """Turn left (counterclockwise) for a specified duration.

        Args:
            duration: How long to turn (default 1.5 seconds, roughly 90 degrees).

        Returns:
            Status message.
        """
        self._send_velocity(0.0, 0.5, duration)
        logger.info(f"[NAV] Turning left for {duration}s")
        return f"Turning left for {duration} seconds."

    @skill
    def turn_right(self, duration: float = 1.5) -> str:
        """Turn right (clockwise) for a specified duration.

        Args:
            duration: How long to turn (default 1.5 seconds, roughly 90 degrees).

        Returns:
            Status message.
        """
        self._send_velocity(0.0, -0.5, duration)
        logger.info(f"[NAV] Turning right for {duration}s")
        return f"Turning right for {duration} seconds."

    @skill
    def stop_moving(self) -> str:
        """Stop all movement immediately.

        Returns:
            Confirmation.
        """
        if self._stop_timer:
            self._stop_timer.cancel()
        self._send_stop()
        logger.info("[NAV] Stopped")
        return "Stopped."


simple_nav_skill = SimpleNavSkill.blueprint

__all__ = ["SimpleNavSkill", "simple_nav_skill"]
