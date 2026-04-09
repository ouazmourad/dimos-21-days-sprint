"""DirectMoveSkill — bypasses the planner and writes Twists straight
to the robot's cmd_vel stream.

The default Unitree skill container's `relative_move` uses
`NavigationInterface.set_goal` which requires a planner with a
costmap — too heavy for the Hermes lite blueprint. This module
exposes simple movement skills that the LLM can call directly:

  - move_forward(seconds=2.0, speed=0.4)
  - move_backward(seconds=2.0, speed=0.3)
  - turn_left(seconds=2.0, speed=0.7)
  - turn_right(seconds=2.0, speed=0.7)
  - stop()

Each skill publishes a Twist on `cmd_vel`. MujocoConnection (or any
real Unitree connection) consumes that stream and applies it.
"""

import threading
import time
from typing import Any

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs import Twist, Vector3
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class DirectMoveSkill(Module):
    """Direct cmd_vel movement skills — no planner required.

    This module ONLY publishes — it does not subscribe to anything,
    so it has no upstream dependencies and works with any connection
    module that consumes a `cmd_vel` stream (MujocoConnection,
    GO2Connection, G1SimConnection, ...).
    """

    cmd_vel: Out[Twist]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stop_timer: threading.Timer | None = None
        self._moving: bool = False

    @rpc
    def start(self) -> None:
        super().start()

    def _send_velocity(self, linear_x: float, angular_z: float, duration: float) -> None:
        """Continuously publish Twist at 10Hz for the entire duration.

        A single publish can be missed by LCM or overridden by the
        RL policy. Continuous publishing ensures the command buffer
        in shared memory stays populated.
        """
        twist = Twist(
            linear=Vector3(linear_x, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, angular_z),
        )
        stop_twist = Twist(
            linear=Vector3(0.0, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 0.0),
        )

        if self._stop_timer:
            self._stop_timer.cancel()
        self._moving = True

        def _publish_loop() -> None:
            end_time = time.time() + duration
            while time.time() < end_time and self._moving:
                self.cmd_vel.publish(twist)
                time.sleep(0.1)  # 10 Hz
            self.cmd_vel.publish(stop_twist)
            self._moving = False

        t = threading.Thread(target=_publish_loop, daemon=True)
        t.start()

    @skill
    def move_forward(self, seconds: float = 2.0, speed: float = 0.4) -> str:
        """Walk the robot forward by sending a positive linear-x velocity.

        Use this whenever the user asks the robot to walk, go, advance,
        or move forward. This is a direct velocity command — no path
        planning, no map, no obstacles. The robot walks for `seconds`
        seconds at `speed` m/s, then stops automatically.

        Args:
            seconds: How long to walk forward (seconds). Default 2.0.
            speed: Forward speed in m/s. Default 0.4. Safe range 0.1 to 0.8.

        Returns:
            Status string confirming the command was issued.
        """
        speed = max(0.0, min(0.8, float(speed)))
        seconds = max(0.1, float(seconds))
        self._send_velocity(speed, 0.0, seconds)
        msg = f"Walking forward for {seconds:.1f}s at {speed:.2f} m/s"
        logger.info(f"[DirectMoveSkill] {msg}")
        return msg

    @skill
    def move_backward(self, seconds: float = 2.0, speed: float = 0.3) -> str:
        """Walk the robot backward (negative linear x) for `seconds` seconds.

        Args:
            seconds: How long to walk backward.
            speed: Backward speed in m/s (positive number). Default 0.3.
        """
        speed = max(0.0, min(0.5, float(speed)))
        seconds = max(0.1, float(seconds))
        self._send_velocity(-speed, 0.0, seconds)
        msg = f"Walking backward for {seconds:.1f}s at {speed:.2f} m/s"
        logger.info(f"[DirectMoveSkill] {msg}")
        return msg

    @skill
    def turn_left(self, seconds: float = 2.0, speed: float = 0.7) -> str:
        """Rotate the robot to the left (positive angular z) in place.

        Args:
            seconds: How long to turn.
            speed: Angular speed in rad/s. Default 0.7.
        """
        speed = max(0.0, min(1.5, float(speed)))
        seconds = max(0.1, float(seconds))
        self._send_velocity(0.0, speed, seconds)
        msg = f"Turning left for {seconds:.1f}s at {speed:.2f} rad/s"
        logger.info(f"[DirectMoveSkill] {msg}")
        return msg

    @skill
    def turn_right(self, seconds: float = 2.0, speed: float = 0.7) -> str:
        """Rotate the robot to the right (negative angular z) in place.

        Args:
            seconds: How long to turn.
            speed: Angular speed in rad/s. Default 0.7.
        """
        speed = max(0.0, min(1.5, float(speed)))
        seconds = max(0.1, float(seconds))
        self._send_velocity(0.0, -speed, seconds)
        msg = f"Turning right for {seconds:.1f}s at {speed:.2f} rad/s"
        logger.info(f"[DirectMoveSkill] {msg}")
        return msg

    @skill
    def stop(self) -> str:
        """Immediately stop the robot — zero linear and angular velocity."""
        self._moving = False
        self.cmd_vel.publish(
            Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0))
        )
        logger.info("[DirectMoveSkill] Stopped")
        return "Stopped"


direct_move_skill = DirectMoveSkill.blueprint

__all__ = ["DirectMoveSkill", "direct_move_skill"]
