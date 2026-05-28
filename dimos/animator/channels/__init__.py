"""Expressive channels — gaze, posture, gesture, breathing.

Each channel reads the current ``Personality`` and the latest scene
state, then emits a ``ChannelState`` (per-virtual-joint targets in
radians). The ``BehaviorMixer`` combines them into a single
``MixedMotion`` which the retargeter turns into real joint commands.

For v1 each channel is a plain Python class with a ``step(dt, ...)``
method, not a DimOS ``Module``. That keeps the prove-the-architecture
demo lean and testable in a single process; wrapping each one in a
``Module`` with ``In``/``Out`` ports is a v1.1 refinement.
"""

from dimos.animator.channels.breathing import BreathingChannel
from dimos.animator.channels.gaze import GazeChannel
from dimos.animator.channels.gesture import GestureChannel
from dimos.animator.channels.posture import PostureChannel

__all__ = [
    "BreathingChannel",
    "GazeChannel",
    "GestureChannel",
    "PostureChannel",
]
