"""curious_head_tilt — a small, head-only "huh?" beat.

Tilts the body around its roll axis (no head, remember) and adds a
tiny gaze pitch shift. Brief: ~1.2 s total. Useful as a reaction to
an unfamiliar sound or object, or as filler between other intents.
"""

from __future__ import annotations

from collections.abc import Iterator

from dimos.animator.channels.gaze import GazeTarget
from dimos.animator.channels.posture import PostureTarget
from dimos.animator.intents.notice_guest import IntentTick
from dimos.animator.personality import Personality


def curious_head_tilt(
    direction: int = 1,
    personality: Personality | None = None,
    tick_dt: float = 1.0 / 50.0,
) -> Iterator[IntentTick]:
    """Yield IntentTicks for one curious-tilt beat.

    Args:
        direction: ``+1`` for right tilt, ``-1`` for left.
        personality: bias amplitude + timing.
        tick_dt: caller's tick period.
    """
    personality = personality or Personality()
    speed = max(0.2, personality.speed_scale())
    # Amplitude is dialed by curiosity, capped at the rig's roll limit.
    base_roll = 0.10 * direction * (1.0 + 0.5 * personality.curiosity)
    base_pitch = 0.06 * (1.0 + 0.3 * personality.curiosity)

    timeline = [
        ("enter",  0.35 / speed),
        ("hold",   0.4 / speed * personality.dwell_scale()),
        ("exit",   0.4 / speed * personality.settle_scale()),
    ]

    for phase, phase_dur in timeline:
        steps = max(1, int(round(phase_dur / tick_dt)))
        for step_i in range(steps):
            phase_t = step_i / max(1, steps - 1) if steps > 1 else 1.0
            tick = IntentTick()

            if phase == "enter":
                # Quadratic ease-in
                ease = phase_t * phase_t
                tick.posture_target = PostureTarget(roll_offset_rad=base_roll * ease)
                tick.gaze_target = GazeTarget(pitch_rad=base_pitch * ease)
            elif phase == "hold":
                tick.posture_target = PostureTarget(roll_offset_rad=base_roll)
                tick.gaze_target = GazeTarget(pitch_rad=base_pitch)
            else:
                # Smooth release back to neutral.
                ease = 1.0 - (1.0 - phase_t) ** 2
                release = 1.0 - ease
                tick.posture_target = PostureTarget(roll_offset_rad=base_roll * release)
                tick.gaze_target = GazeTarget(pitch_rad=base_pitch * release)

            yield tick

    yield IntentTick(finished=True)
