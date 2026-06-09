"""Secondary-motion channel — ears and tail with spring dynamics.

Classic animation principle: appendages lag the primary motion and
overshoot (follow-through / overlapping action). The ears and tail are
under-damped spring-dampers whose *setpoints* come from mood, and which
get kicked by head motion so they flop naturally when the head moves.

Mood mapping:
* ears  — perk forward when curious/alert (negative angle), hang back
          when shy / low-energy (positive). Head-velocity kicks add flop.
* tail  — pitch carriage: high/up when confident, tucked when timid.
          yaw wag: oscillates when excitement is high. Excitement =
          playfulness + energy baseline, spiked by gestures (the
          orchestrator calls ``excite()`` when a paw wave fires) and
          boosted while walking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dimos.animator.personality import Personality


@dataclass
class SecondaryState:
    ear_l: float = 0.0
    ear_r: float = 0.0
    tail_yaw: float = 0.0
    tail_pitch: float = 0.0


class _Spring:
    """Under-damped second-order tracker — the floppy-appendage core."""

    def __init__(self, freq_hz: float = 2.2, zeta: float = 0.28) -> None:
        self._wn = 2 * math.pi * freq_hz
        self._zeta = zeta
        self.x = 0.0
        self.v = 0.0

    def step(self, dt: float, setpoint: float, kick: float = 0.0) -> float:
        a = self._wn ** 2 * (setpoint - self.x) - 2 * self._zeta * self._wn * self.v + kick
        self.v += a * dt
        self.x += self.v * dt
        return self.x


class SecondaryMotionChannel:
    def __init__(self) -> None:
        self._ear_l = _Spring(freq_hz=2.4, zeta=0.3)
        self._ear_r = _Spring(freq_hz=2.4, zeta=0.3)
        self._tail_yaw = _Spring(freq_hz=2.8, zeta=0.35)
        self._tail_pitch = _Spring(freq_hz=1.8, zeta=0.5)
        self._excite = 0.0
        self._walking = False
        self._wag_phase = 0.0
        self._last_neck = (0.0, 0.0)

    def excite(self, amount: float = 0.9) -> None:
        """Spike excitement (e.g. when a happy gesture fires)."""
        self._excite = min(1.0, self._excite + amount)

    def set_walking(self, walking: bool) -> None:
        self._walking = walking

    def step(
        self,
        dt: float,
        personality: Personality,
        neck_yaw: float,
        neck_pitch: float,
    ) -> SecondaryState:
        # Excitement: baseline from personality, spikes decay over ~2 s,
        # walking adds a sustained boost (trotting dogs wag).
        baseline = max(0.0, 0.35 * personality.playfulness + 0.2 * personality.energy)
        self._excite = max(0.0, self._excite - dt / 2.0)
        excitement = min(1.0, baseline + self._excite + (0.35 if self._walking else 0.0))

        # Head-motion kick: ears flop opposite to neck acceleration.
        dvy = (neck_yaw - self._last_neck[0]) / max(dt, 1e-4)
        dvp = (neck_pitch - self._last_neck[1]) / max(dt, 1e-4)
        self._last_neck = (neck_yaw, neck_pitch)
        ear_kick = -2.2 * dvp  # pitch motion flops ears fore/aft

        # Ear setpoint: perked (-0.25) when curious/confident, drooped
        # (+0.9) when shy or drained.
        alert = 0.5 * personality.curiosity + 0.3 * personality.confidence \
            + 0.3 * excitement
        ear_set = 0.35 - 0.6 * alert
        ear_set = max(-0.3, min(1.0, ear_set))
        # Slight asymmetry for charm (left ear a touch lazier).
        ear_l = self._ear_l.step(dt, ear_set + 0.05, kick=ear_kick)
        ear_r = self._ear_r.step(dt, ear_set, kick=ear_kick * 0.9)

        # Tail carriage: up (-0.25) when confident, tucked (+0.8) when timid.
        carriage = 0.25 - 0.55 * personality.confidence
        if personality.confidence < -0.3:
            carriage += 0.25  # extra tuck for properly shy characters
        tail_pitch = self._tail_pitch.step(dt, carriage)

        # Tail wag: oscillating yaw setpoint, gated by excitement.
        wag_freq = 2.2 + 2.0 * excitement
        self._wag_phase = (self._wag_phase + wag_freq * dt) % 1.0
        wag_amp = 0.55 * excitement
        wag_set = wag_amp * math.sin(2 * math.pi * self._wag_phase)
        # The spring adds natural lag + overshoot to the wag.
        tail_yaw = self._tail_yaw.step(dt, wag_set, kick=-1.2 * dvy)

        return SecondaryState(
            ear_l=ear_l, ear_r=ear_r,
            tail_yaw=tail_yaw, tail_pitch=tail_pitch,
        )
