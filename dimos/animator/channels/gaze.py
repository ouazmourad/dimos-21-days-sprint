"""Gaze channel — where the robot is "looking".

On the Go2 there is no head joint. Gaze is implemented through two
virtual joints:

* ``trunk_yaw``   — rotate the body around z (left/right look)
* ``trunk_pitch`` — tilt the body around y (nose up/down)

The channel takes a stream of target bearings (yaw, pitch) in
radians relative to the body's default heading, plus the current
``Personality``. It returns a smoothed target snapshot each step.

Personality bias rules:
* curiosity → faster reorientation toward novelty
* confidence → longer dwell on each target (sticky)
* softness → smoother (lower bandwidth) transitions
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dimos.animator.personality import Personality


@dataclass
class GazeTarget:
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0


class GazeChannel:
    """First-order low-pass tracker of a gaze target."""

    def __init__(self, base_time_constant_s: float = 0.6) -> None:
        # Time constant of the low-pass filter. Larger = slower / softer.
        self._tau = base_time_constant_s
        self._state = GazeTarget()
        # Dwell tracking: how long we've been pointed at the current target.
        self._target = GazeTarget()
        self._dwell_remaining_s = 0.0

    def set_target(self, target: GazeTarget, personality: Personality) -> None:
        """Update the target. ``personality`` controls dwell + filter."""
        self._target = target
        # Confident / curious characters dwell longer on a chosen target
        # before allowing it to be overridden by something else.
        self._dwell_remaining_s = 0.4 * personality.dwell_scale()

    def step(self, dt: float, personality: Personality) -> GazeTarget:
        # Effective time constant adjusts with softness + energy.
        tau = self._tau * personality.settle_scale() / max(0.1, personality.speed_scale())
        # Discretised low-pass: state += (target - state) * dt / tau
        alpha = min(1.0, dt / max(0.01, tau))
        self._state = GazeTarget(
            yaw_rad=self._state.yaw_rad + (self._target.yaw_rad - self._state.yaw_rad) * alpha,
            pitch_rad=self._state.pitch_rad + (self._target.pitch_rad - self._state.pitch_rad) * alpha,
        )
        self._dwell_remaining_s = max(0.0, self._dwell_remaining_s - dt)
        return self._state

    def can_redirect(self) -> bool:
        """True if dwell expired and we accept a new target now."""
        return self._dwell_remaining_s <= 0.0

    @staticmethod
    def bearing_to(target_xy: tuple[float, float], default_xy: tuple[float, float] = (1.0, 0.0)) -> float:
        """Helper: angle from the robot's default heading to a target."""
        dx = target_xy[0] - default_xy[0]
        dy = target_xy[1] - default_xy[1]
        return math.atan2(dy, dx)
