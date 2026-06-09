"""proud_chest_lift — body extends taller, nose tips up, then settles.

Used after task completion. Reads as "chest out, look up" on a body
that has a torso (it would). On the Go2 it reads as "stand tall,
look upward". Confident personalities hold the pose longer; shy ones
barely lift at all.
"""

from __future__ import annotations

from collections.abc import Iterator

from dimos.animator.channels.gaze import GazeTarget
from dimos.animator.channels.posture import PostureTarget
from dimos.animator.intents.notice_guest import IntentTick
from dimos.animator.personality import Personality


def proud_chest_lift(
    personality: Personality | None = None,
    tick_dt: float = 1.0 / 50.0,
) -> Iterator[IntentTick]:
    personality = personality or Personality()
    speed = max(0.2, personality.speed_scale())
    # Big chest puff + chin lift (rig z limit now 0.18, gaze pitch 0.45).
    # If confidence ≤ -0.5 the beat stays small — by design.
    height_lift = max(0.02, 0.15 + 0.05 * personality.confidence)
    pitch_up = max(0.03, 0.22 + 0.08 * personality.confidence)

    timeline = [
        ("inhale", 0.4 / speed),
        ("hold",   0.7 / speed * personality.dwell_scale()),
        ("exhale", 0.7 / speed * personality.settle_scale()),
    ]

    for phase, phase_dur in timeline:
        steps = max(1, int(round(phase_dur / tick_dt)))
        for step_i in range(steps):
            phase_t = step_i / max(1, steps - 1) if steps > 1 else 1.0
            tick = IntentTick()

            if phase == "inhale":
                # Smoothstep into the pose.
                ease = phase_t * phase_t * (3 - 2 * phase_t)
                tick.posture_target = PostureTarget(z_offset_rad=height_lift * ease)
                tick.gaze_target = GazeTarget(pitch_rad=pitch_up * ease)
            elif phase == "hold":
                tick.posture_target = PostureTarget(z_offset_rad=height_lift)
                tick.gaze_target = GazeTarget(pitch_rad=pitch_up)
            else:
                # Exhale: relax slowly.
                release = 1.0 - phase_t
                tick.posture_target = PostureTarget(z_offset_rad=height_lift * release)
                tick.gaze_target = GazeTarget(pitch_rad=pitch_up * release)

            yield tick

    yield IntentTick(finished=True)
