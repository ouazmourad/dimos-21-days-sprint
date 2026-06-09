"""Kinematic trot gait for the wooden Go2 — authored locomotion.

This is the "periodic motion" type from the Disney bipedal-character
paper: a cyclic phase drives a repeating gait. Here it's authored
(kinematic) rather than RL-tracked — the legs cycle through a trot and
the body translates across the floor by setting joint angles + the
free-joint base directly. No physics; this is animation.

v2 quality upgrades:

* **Foot-speed sync** — the body's forward speed is derived from the
  stance feet's backward sweep speed (stride x leg length x cadence),
  so feet visually plant instead of skating across the floor.
* **Start/stop ramps** — toggling the gait ramps a weight 0→1 (and
  back) over ~0.5 s; leg angles blend between the standing pose and
  the gait, and the base speed scales with the same weight. The robot
  eases into a trot and settles back to a stand, never pops.
* **Steering** — a turn command curves the path (differential hip
  splay + heading integration by the caller).

A trot moves diagonal leg pairs together (FR+RL, then FL+RR).
Personality modulates cadence (energy), step height (playfulness),
and body bob.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dimos.animator.personality import Personality

# Diagonal trot phase offsets: FR+RL step together, FL+RR a half-cycle later.
_LEG_PHASE = {
    "FR": 0.0,
    "RL": 0.0,
    "FL": 0.5,
    "RR": 0.5,
}

# Effective leg length (hip pivot to foot) used for foot-speed sync.
_LEG_LEN_M = 0.30
# Mean |sin| over the stance half-cycle (foot speed varies sinusoidally).
_STANCE_SPEED_FACTOR = 0.64


@dataclass
class GaitOutput:
    """One tick of gait output."""

    leg_angles: dict[str, float]   # absolute joint targets (rad), pre-blended
    base_dx: float                 # forward translation this tick (m)
    base_dz: float                 # height offset from the body bob (m)
    base_dyaw: float               # heading change this tick (rad)
    weight: float                  # gait blend weight in [0, 1]


class GaitGenerator:
    """Produces a kinematic trot, parameterised by personality."""

    def __init__(
        self,
        default_pose: dict[str, float],
        base_freq_hz: float = 1.6,
        step_height: float = 0.45,
        stride: float = 0.35,
        ramp_s: float = 0.5,
    ) -> None:
        self._default = dict(default_pose)
        self._base_freq = base_freq_hz
        self._step_height = step_height
        self._stride = stride
        self._ramp_rate = 1.0 / max(0.05, ramp_s)
        self._phase = 0.0
        self._active = False
        self._weight = 0.0

    # -- control --------------------------------------------------------

    def set_active(self, active: bool) -> None:
        """Start (ramp in) or stop (ramp out + settle) the gait."""
        self._active = active

    @property
    def is_moving(self) -> bool:
        """True while the gait has any visible influence."""
        return self._weight > 1e-3

    def reset(self) -> None:
        self._phase = 0.0
        self._weight = 0.0
        self._active = False

    # -- per tick --------------------------------------------------------

    def step(self, dt: float, personality: Personality, turn: float = 0.0) -> GaitOutput:
        """Advance the gait by ``dt``. ``turn`` in [-1, 1] steers."""
        # Ramp the blend weight toward the active state.
        target_w = 1.0 if self._active else 0.0
        if self._weight < target_w:
            self._weight = min(target_w, self._weight + self._ramp_rate * dt)
        elif self._weight > target_w:
            self._weight = max(target_w, self._weight - self._ramp_rate * dt)
        w = self._weight

        if w <= 0.0:
            # Fully stopped: hold the standing pose exactly.
            return GaitOutput(
                leg_angles=dict(self._default),
                base_dx=0.0, base_dz=0.0, base_dyaw=0.0, weight=0.0,
            )

        # Personality modulation.
        freq = self._base_freq * (1.0 + 0.4 * personality.energy)
        step_h = self._step_height * (1.0 + 0.3 * personality.playfulness
                                       + 0.2 * personality.energy)
        stride = self._stride * (1.0 + 0.25 * personality.energy)

        self._phase = (self._phase + freq * dt) % 1.0

        leg_angles: dict[str, float] = {}
        for leg, offset in _LEG_PHASE.items():
            ph = (self._phase + offset) % 1.0
            fore_aft = stride * math.sin(2 * math.pi * ph)
            lift = step_h * max(0.0, math.sin(2 * math.pi * ph))

            thigh = self._default[f"{leg}_thigh_joint"] + fore_aft * w
            calf = self._default[f"{leg}_calf_joint"] - lift * w
            hip = self._default[f"{leg}_hip_joint"]
            if turn != 0.0:
                side = 1.0 if leg in ("FL", "RL") else -1.0
                hip += 0.15 * turn * side * w

            leg_angles[f"{leg}_hip_joint"] = hip
            leg_angles[f"{leg}_thigh_joint"] = thigh
            leg_angles[f"{leg}_calf_joint"] = calf

        # Body bob, scaled by the blend weight.
        bob = 0.02 * (1.0 + 0.5 * personality.playfulness)
        base_dz = -bob * abs(math.sin(2 * math.pi * self._phase)) * w

        # Foot-speed sync: forward speed = how fast the stance feet sweep
        # backwards. Peak foot speed = stride·2π·freq·L; the body should
        # move at the stance-average of that, scaled by the blend weight.
        speed = stride * 2 * math.pi * freq * _LEG_LEN_M * _STANCE_SPEED_FACTOR
        base_dx = speed * w * dt
        base_dyaw = 0.6 * turn * w * dt

        return GaitOutput(
            leg_angles=leg_angles,
            base_dx=base_dx,
            base_dz=base_dz,
            base_dyaw=base_dyaw,
            weight=w,
        )
