"""Gesture channel — discrete, scripted movements.

A Go2 has no arms, so the only physical gesture surface is "lift one
paw off the ground briefly". v1 exposes one virtual joint:

* ``paw_lift_fl`` — front-left foot lift amount in radians (calf flex)

Gestures are short, time-keyframed animations (start, peak, settle).
A gesture is queued via :meth:`trigger`; ``step`` emits the current
amplitude.

Personality bias rules:
* energy      → faster peak, shorter total duration
* amplitude   → controlled by ``confidence``
* anticipation→ controlled by ``softness`` (longer wind-up)
"""

from __future__ import annotations

from dataclasses import dataclass


from dimos.animator.personality import Personality


@dataclass
class GestureState:
    paw_lift_fl_offset_rad: float = 0.0


class GestureChannel:
    def __init__(self) -> None:
        self._active: str | None = None
        self._t_s: float = 0.0
        self._duration_s: float = 0.0
        self._peak_amp_rad: float = 0.0

    def trigger(self, name: str, personality: Personality) -> None:
        """Start a gesture. ``name`` is a key like 'paw_wave' or 'paw_acknowledge'.

        Only one gesture can be active at a time; triggering during an
        active gesture restarts.
        """
        if name == "paw_wave":
            base_dur = 1.0
            base_amp = 0.4
        elif name == "paw_acknowledge":
            base_dur = 0.6
            base_amp = 0.25
        else:
            base_dur = 0.8
            base_amp = 0.3

        self._active = name
        self._t_s = 0.0
        self._duration_s = base_dur / max(0.1, personality.speed_scale())
        # Confident characters lift higher; shy characters barely.
        amp_scale = personality.amplitude_scale()
        # ``confidence < 0`` shouldn't go negative-amplitude.
        self._peak_amp_rad = max(0.0, base_amp * amp_scale)

    def step(self, dt: float, personality: Personality) -> GestureState:
        if self._active is None:
            return GestureState(paw_lift_fl_offset_rad=0.0)

        self._t_s += dt
        if self._t_s >= self._duration_s:
            self._active = None
            return GestureState(paw_lift_fl_offset_rad=0.0)

        # Three-phase envelope: rise (anticipation) → peak → settle.
        # Softer personalities get a longer anticipation phase.
        antic = 0.2 + 0.2 * personality.softness
        antic = min(0.5, max(0.05, antic))
        peak_dur = 0.3
        t_norm = self._t_s / self._duration_s

        if t_norm < antic:
            # Anticipation: small dip below zero (optional charm bias)
            phase = t_norm / antic
            amp = self._peak_amp_rad * (-0.15 * personality.playfulness) * phase
        elif t_norm < antic + peak_dur:
            phase = (t_norm - antic) / peak_dur
            # Quadratic rise to the peak
            amp = self._peak_amp_rad * (1.0 - (1.0 - phase) ** 2)
        else:
            phase = (t_norm - antic - peak_dur) / max(0.05, 1.0 - antic - peak_dur)
            # Smooth decay back to zero
            amp = self._peak_amp_rad * (1.0 - phase) ** 2

        return GestureState(paw_lift_fl_offset_rad=amp)

    @property
    def is_active(self) -> bool:
        return self._active is not None
