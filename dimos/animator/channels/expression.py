"""Expression channel — eye openness, brow angle, blinks.

This is the channel that only exists because option B gave the sim
robot an articulated face. It maps the personality (and a transient
"surprise" cue derived from how fast the gaze target is moving) into:

* ``eye_openness`` in [0, 1]   — 0 = lids closed, 1 = wide open
* ``brow_raise``   in [-1, 1]  — -1 = inner-down (focused), +1 = raised (curious)

plus an automatic blink every few seconds. The retargeter turns these
into eyelid + brow joint angles.

Personality bias rules:
* curiosity  → wider eyes, raised brows, more frequent blinks
* confidence → wider eyes, neutral brows
* energy     → faster blinks
* softness   → gentler (slower) openness changes
"""

from __future__ import annotations

from dataclasses import dataclass

from dimos.animator.personality import Personality


@dataclass
class ExpressionState:
    eye_openness: float = 0.7   # [0, 1]
    brow_raise: float = 0.0     # [-1, 1]
    head_pitch: float = 0.0     # rad; + = chin up (proud), - = chin down (shy)


class ExpressionChannel:
    def __init__(self) -> None:
        self._openness = 0.7
        self._blink_timer_s = 0.0
        self._blink_active_s = 0.0
        self._surprise = 0.0
        # Director overrides (the show's "sleep" beat etc). ``None`` =
        # personality drives the value as usual.
        self.override_openness: float | None = None
        self.override_brow: float | None = None

    def _baseline_openness(self, p: Personality) -> float:
        if self.override_openness is not None:
            return self.override_openness
        v = 0.5 + 0.45 * p.curiosity + 0.22 * p.confidence + 0.12 * p.energy
        return max(0.12, min(1.0, v))

    def _baseline_brow(self, p: Personality) -> float:
        if self.override_brow is not None:
            return self.override_brow
        # Curious / playful raise the brows; low confidence pulls the inner
        # corners up into a worried tilt (negative), high confidence flat.
        v = 0.6 * p.curiosity + 0.25 * p.playfulness
        if p.confidence < 0:
            v += 0.5 * p.confidence  # worried droop
        return max(-1.0, min(1.0, v))

    def _blink_interval(self, p: Personality) -> float:
        base = 4.0 - 1.5 * p.curiosity - 1.0 * p.energy
        return max(1.2, base)

    def notice_surprise(self, gaze_delta: float) -> None:
        """Inject a transient eye-widening when the gaze jumps.

        Called by the orchestrator with the magnitude of the gaze
        target change this tick. A big jump = the robot just noticed
        something = eyes pop open briefly.
        """
        self._surprise = min(1.0, self._surprise + 3.0 * abs(gaze_delta))

    def step(self, dt: float, personality: Personality) -> ExpressionState:
        baseline = self._baseline_openness(personality)

        # Surprise decays over ~0.6 s.
        self._surprise = max(0.0, self._surprise - dt / 0.6)
        target = min(1.0, baseline + 0.5 * self._surprise)

        # Blink bookkeeping.
        self._blink_timer_s += dt
        if self._blink_active_s > 0.0:
            self._blink_active_s -= dt
            target = 0.0  # lids shut during a blink
        elif self._blink_timer_s >= self._blink_interval(personality):
            self._blink_timer_s = 0.0
            self._blink_active_s = 0.12  # blink lasts ~120 ms

        # Smooth openness toward target. Softer personalities ease slowly.
        tau = 0.12 * personality.settle_scale()
        alpha = min(1.0, dt / max(0.01, tau))
        self._openness += (target - self._openness) * alpha

        # Chin carriage: confident characters lift the chin, timid /
        # low-confidence ones drop it. Small range so it reads as posture,
        # not a full head-tilt.
        head_pitch = 0.22 * personality.confidence

        return ExpressionState(
            eye_openness=max(0.0, min(1.0, self._openness)),
            brow_raise=self._baseline_brow(personality),
            head_pitch=head_pitch,
        )
