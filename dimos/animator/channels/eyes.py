"""Eyes channel — saccades, fixation, micro-saccades.

The single most lifelike motion cue: eyes move FIRST in fast jumps
(saccades), the head follows slowly, and the eyes counter-rotate to
hold fixation while the head catches up (the vestibulo-ocular reflex).

The orchestrator feeds this channel the *gaze error* — the difference
between where the character wants to look and where the neck currently
points. The eyes snap toward that error (clamped to the eyeball range);
as the neck catches up the error shrinks and the eyes recentre on
their own. That cascade is what reads as "it saw something".

When idle (small error), the channel emits occasional micro-saccades —
tiny random fixation jumps every ~1–2 s — which keep the eyes alive
instead of glassy.

Personality bias:
* curiosity → more frequent micro-saccades
* energy    → faster saccade velocity
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from dimos.animator.personality import Personality

_EYE_YAW_MAX = 0.55
_EYE_PITCH_MAX = 0.45


@dataclass
class EyesState:
    yaw: float = 0.0
    pitch: float = 0.0


class EyesChannel:
    def __init__(self, seed: int | None = None) -> None:
        self._yaw = 0.0
        self._pitch = 0.0
        self._rng = random.Random(seed)
        self._micro_timer = 0.0
        self._micro_next = 1.2
        self._micro_offset = (0.0, 0.0)
        self._micro_hold = 0.0

    def step(
        self,
        dt: float,
        personality: Personality,
        gaze_error_yaw: float,
        gaze_error_pitch: float,
    ) -> EyesState:
        # Saccade target: jump toward the (clamped) gaze error.
        ty = max(-_EYE_YAW_MAX, min(_EYE_YAW_MAX, gaze_error_yaw))
        tp = max(-_EYE_PITCH_MAX, min(_EYE_PITCH_MAX, gaze_error_pitch))

        # Micro-saccades while fixating (error small).
        if abs(ty) < 0.06 and abs(tp) < 0.06:
            self._micro_timer += dt
            interval = self._micro_next / (1.0 + 0.6 * max(0.0, personality.curiosity))
            if self._micro_hold > 0.0:
                self._micro_hold -= dt
                if self._micro_hold <= 0.0:
                    self._micro_offset = (0.0, 0.0)
            elif self._micro_timer >= interval:
                self._micro_timer = 0.0
                self._micro_next = self._rng.uniform(0.8, 2.2)
                self._micro_offset = (
                    self._rng.uniform(-0.05, 0.05),
                    self._rng.uniform(-0.03, 0.03),
                )
                self._micro_hold = self._rng.uniform(0.12, 0.3)
            ty += self._micro_offset[0]
            tp += self._micro_offset[1]
        else:
            self._micro_offset = (0.0, 0.0)
            self._micro_hold = 0.0
            self._micro_timer = 0.0

        # Saccadic slew: very fast (eyes are the fastest joint in any
        # body — human saccades hit 500°/s). 12 rad/s base.
        rate = 12.0 * (1.0 + 0.3 * personality.energy)
        max_step = rate * dt
        dy = max(-max_step, min(max_step, ty - self._yaw))
        dp = max(-max_step, min(max_step, tp - self._pitch))
        self._yaw += dy
        self._pitch += dp

        return EyesState(yaw=self._yaw, pitch=self._pitch)
