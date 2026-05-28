"""Breathing channel — subtle, continuous body-height modulation.

This is the channel that prevents a stationary robot from looking
"frozen". The amplitude is small (a few centimetres of leg
extension), the frequency is in the human breathing range (0.2–0.5
Hz), and personality biases both.

* energy      → faster breathing
* softness    → larger amplitude
* confidence  → lower amplitude (controlled, calm)
* playfulness → small asymmetry (uneven inhale/exhale)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dimos.animator.personality import Personality


@dataclass
class BreathingState:
    z_breath_offset_rad: float = 0.0


class BreathingChannel:
    def __init__(self, base_freq_hz: float = 0.3, base_amp_rad: float = 0.025) -> None:
        self._base_freq = base_freq_hz
        self._base_amp = base_amp_rad
        self._phase_rad: float = 0.0

    def step(self, dt: float, personality: Personality) -> BreathingState:
        # Frequency: energy speeds up, confidence slows.
        freq = self._base_freq * (1.0 + 0.5 * personality.energy
                                   - 0.2 * personality.confidence)
        freq = max(0.05, freq)
        # Amplitude: softness adds, confidence subtracts.
        amp = self._base_amp * (1.0 + 0.6 * personality.softness
                                 - 0.4 * personality.confidence)
        amp = max(0.005, amp)
        # Asymmetric waveform when playful: shaped sine with skew.
        skew = 0.5 + 0.4 * personality.playfulness
        skew = min(0.9, max(0.1, skew))

        self._phase_rad += 2 * math.pi * freq * dt
        if self._phase_rad > 2 * math.pi:
            self._phase_rad -= 2 * math.pi

        # Skewed-sine: inhale portion gets ``skew``, exhale gets ``1 - skew``.
        phase_norm = self._phase_rad / (2 * math.pi)
        if phase_norm < skew:
            local = phase_norm / skew
            wave = math.sin(local * math.pi)
        else:
            local = (phase_norm - skew) / (1.0 - skew)
            wave = -math.sin(local * math.pi)

        return BreathingState(z_breath_offset_rad=amp * wave)
