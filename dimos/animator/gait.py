"""Kinematic trot gait for the wooden Go2 — authored locomotion.

This is the "periodic motion" type from the Disney bipedal-character
paper: a cyclic phase drives a repeating gait. Here it's authored
(kinematic) rather than RL-tracked — the legs cycle through a
trot and the body translates across the floor, all by setting joint
angles + the free-joint base directly. No physics; this is animation.

A trot moves diagonal leg pairs together (FR+RL, then FL+RR), which
reads clearly as "walking" on a four-legged toy. Personality modulates
step frequency (energy), step height + stride (playfulness / energy),
and a small body bob.

The gait owns the 12 leg joints while it's active; the animation
engine keeps driving the expressive head (neck / eyes / brows) on top.
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


@dataclass
class GaitOutput:
    """One tick of gait: leg joint angle offsets + base motion."""

    leg_angles: dict[str, float]   # absolute joint targets (rad)
    base_dx: float                 # forward translation this tick (m)
    base_dz: float                 # height offset from the body bob (m)
    base_dyaw: float               # heading change this tick (rad)


class GaitGenerator:
    """Produces a kinematic trot, parameterised by personality."""

    def __init__(
        self,
        default_pose: dict[str, float],
        base_freq_hz: float = 1.6,
        step_height: float = 0.45,
        stride: float = 0.35,
        speed: float = 0.45,
    ) -> None:
        self._default = dict(default_pose)
        self._base_freq = base_freq_hz
        self._step_height = step_height
        self._stride = stride
        self._speed = speed
        self._phase = 0.0

    def reset(self) -> None:
        self._phase = 0.0

    def step(self, dt: float, personality: Personality, turn: float = 0.0) -> GaitOutput:
        """Advance the gait by ``dt``. ``turn`` in [-1, 1] steers."""
        # Energy speeds the cadence; playful/energetic exaggerate the step.
        freq = self._base_freq * (1.0 + 0.4 * personality.energy)
        step_h = self._step_height * (1.0 + 0.3 * personality.playfulness
                                       + 0.2 * personality.energy)
        stride = self._stride * (1.0 + 0.25 * personality.energy)

        self._phase = (self._phase + freq * dt) % 1.0

        leg_angles: dict[str, float] = {}
        for leg, offset in _LEG_PHASE.items():
            ph = (self._phase + offset) % 1.0
            # Fore-aft swing: thigh swings forward at mid-swing, back in stance.
            #   sin(2π·ph): +1 at ph=0.25 (forward), -1 at ph=0.75 (back).
            fore_aft = stride * math.sin(2 * math.pi * ph)
            # Foot lift: flex the calf during the swing half (ph in [0, 0.5)).
            lift = step_h * max(0.0, math.sin(2 * math.pi * ph))

            thigh = self._default[f"{leg}_thigh_joint"] + fore_aft
            calf = self._default[f"{leg}_calf_joint"] - lift
            hip = self._default[f"{leg}_hip_joint"]
            # Steering: splay the hips of the inside/outside legs.
            if turn != 0.0:
                side = 1.0 if leg in ("FL", "RL") else -1.0
                hip += 0.15 * turn * side

            leg_angles[f"{leg}_hip_joint"] = hip
            leg_angles[f"{leg}_thigh_joint"] = thigh
            leg_angles[f"{leg}_calf_joint"] = calf

        # Body bob: dips twice per cycle (once per diagonal plant).
        bob = 0.02 * (1.0 + 0.5 * personality.playfulness)
        base_dz = -bob * abs(math.sin(2 * math.pi * self._phase))

        base_dx = self._speed * (1.0 + 0.4 * personality.energy) * dt
        base_dyaw = 0.6 * turn * dt

        return GaitOutput(
            leg_angles=leg_angles,
            base_dx=base_dx,
            base_dz=base_dz,
            base_dyaw=base_dyaw,
        )
