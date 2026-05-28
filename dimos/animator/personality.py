"""Personality profile — five scalar dimensions that bias every channel.

Values live in ``[-1, 1]`` with ``0`` as the neutral baseline. Each
channel module reads the current ``Personality`` and adjusts its
output accordingly. The contract intentionally has very few knobs:
the proposal's failure mode list flags that artist craft can't be
reduced to scalar parameters, so v1 stays minimal and we put the
nuance in the per-intent timing curves rather than the personality
schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Personality:
    """Five biases applied across all expressive channels.

    Each value is in ``[-1, 1]``. Out-of-range values are clamped on
    load so the rest of the stack never has to defend against them.
    """

    curiosity: float = 0.0
    energy: float = 0.0
    confidence: float = 0.0
    softness: float = 0.0
    playfulness: float = 0.0

    def __post_init__(self) -> None:
        # Frozen dataclass — use object.__setattr__ to clamp in place.
        for name in ("curiosity", "energy", "confidence", "softness", "playfulness"):
            value = getattr(self, name)
            if value > 1.0:
                object.__setattr__(self, name, 1.0)
            elif value < -1.0:
                object.__setattr__(self, name, -1.0)

    @classmethod
    def from_yaml(cls, path: Path | str) -> Personality:
        data = yaml.safe_load(Path(path).read_text())
        return cls(**{k: float(data.get(k, 0.0)) for k in (
            "curiosity", "energy", "confidence", "softness", "playfulness",
        )})

    def speed_scale(self) -> float:
        """How fast motion should be. >1 = faster, <1 = slower."""
        return 1.0 + 0.5 * self.energy

    def amplitude_scale(self) -> float:
        """How big motion should be. >1 = bigger, <1 = smaller."""
        # Confident characters take up more space; shy ones less.
        return 1.0 + 0.4 * self.confidence + 0.2 * self.energy

    def dwell_scale(self) -> float:
        """How long postures should be held. >1 = longer, <1 = shorter."""
        return 1.0 + 0.6 * self.confidence + 0.3 * self.curiosity

    def settle_scale(self) -> float:
        """How slowly the body settles into the final pose."""
        # Soft characters ease in slowly; high-energy ones snap.
        return 1.0 + 0.6 * self.softness - 0.3 * self.energy

    def initiation_delay(self) -> float:
        """Seconds of delay before a behavior actually starts.

        Shy / low-confidence characters hesitate. Confident ones move
        immediately. Clamped to ``[0, 1.5]``.
        """
        delay = 0.6 - 0.5 * self.confidence + 0.3 * self.softness
        return max(0.0, min(1.5, delay))
