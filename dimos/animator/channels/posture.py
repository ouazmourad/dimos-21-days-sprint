"""Posture channel — body lean, height, weight shift.

Virtual joints on the Go2:

* ``trunk_z``    — uniform leg extension (squat ↔ standing tall)
* ``trunk_x``    — symmetric forward/back body shift (approach / recoil)
* ``trunk_roll`` — side-to-side lean (curious tilt / nervous shift)

Personality bias rules:
* confidence → taller standing height, slower height changes
* softness   → larger forward weight shifts (rounder curves)
* energy     → faster settles
"""

from __future__ import annotations

from dataclasses import dataclass

from dimos.animator.personality import Personality


@dataclass
class PostureTarget:
    """All values are virtual-joint offsets relative to the default pose."""

    z_offset_rad: float = 0.0       # +: stand taller, -: squat
    x_offset_rad: float = 0.0       # +: lean forward, -: lean back
    roll_offset_rad: float = 0.0    # +: lean right, -: lean left


class PostureChannel:
    """Critically-damped second-order target tracker.

    Posture changes should feel weighty, not snappy — they're how the
    body's mass moves. Using a damped spring rather than a low-pass
    makes 'lean forward then recover' look like an actual physical
    motion rather than a slew.
    """

    def __init__(self, base_natural_freq_hz: float = 1.2) -> None:
        self._wn = 2 * 3.14159265 * base_natural_freq_hz
        self._state = PostureTarget()
        self._velocity = PostureTarget()
        self._target = PostureTarget()

    def set_target(self, target: PostureTarget) -> None:
        self._target = target

    def step(self, dt: float, personality: Personality) -> PostureTarget:
        # Personality reshapes the spring stiffness.
        wn = self._wn * personality.speed_scale() / max(0.5, personality.settle_scale())
        zeta = 1.0   # critically damped — no overshoot for posture

        for axis in ("z_offset_rad", "x_offset_rad", "roll_offset_rad"):
            x = getattr(self._state, axis)
            v = getattr(self._velocity, axis)
            target_x = getattr(self._target, axis)
            # x_ddot = wn^2 * (target - x) - 2*zeta*wn*v
            accel = wn * wn * (target_x - x) - 2 * zeta * wn * v
            v += accel * dt
            x += v * dt
            setattr(self._velocity, axis, v)
            setattr(self._state, axis, x)

        # Confidence bias: nudge resting height up by a small amount.
        # Acts as a static offset, doesn't fight the damped tracker.
        z_with_bias = self._state.z_offset_rad + 0.02 * personality.confidence

        return PostureTarget(
            z_offset_rad=z_with_bias,
            x_offset_rad=self._state.x_offset_rad,
            roll_offset_rad=self._state.roll_offset_rad,
        )
