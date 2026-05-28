"""notice_guest — the robot orients toward a person, holds, acknowledges.

Beat structure (default timing):
  0.0–0.6 s   initiation delay (shy / low-confidence stretches this)
  0.6–1.0 s   orient body yaw toward target
  1.0–1.4 s   slight nose-up tilt (look at face)
  1.4–2.0 s   hold (dwell duration biased by confidence + curiosity)
  2.0–2.6 s   small paw acknowledge (only if confidence > 0)
  2.6–3.2 s   settle back toward neutral

The intent doesn't move the robot through space — it expresses noticing.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

from dimos.animator.channels.gaze import GazeTarget
from dimos.animator.channels.posture import PostureTarget
from dimos.animator.personality import Personality


@dataclass
class IntentTick:
    """One tick of intent execution — what each channel should target *now*."""

    gaze_target: GazeTarget | None = None
    posture_target: PostureTarget | None = None
    fire_gesture: str | None = None
    finished: bool = False


def notice_guest(
    target_yaw_rad: float,
    target_pitch_rad: float,
    personality: Personality,
    tick_dt: float = 1.0 / 50.0,
) -> Iterator[IntentTick]:
    """Yield one IntentTick per simulation/control tick.

    Args:
        target_yaw_rad: bearing to the person, in the body frame.
        target_pitch_rad: pitch to the face / head, in the body frame.
        personality: bias for timing + amplitude.
        tick_dt: caller's control period (must match the orchestrator).

    Yields:
        A finite sequence of IntentTick. Last one has ``finished=True``.
    """
    # Stretch the whole timeline by speed_scale (>1 = faster).
    speed = max(0.2, personality.speed_scale())
    delay = personality.initiation_delay()

    timeline = [
        ("delay",        delay),
        ("orient",       0.4 / speed),
        ("look",         0.4 / speed),
        ("hold",         0.6 / speed * personality.dwell_scale()),
        ("acknowledge",  0.6 / speed),
        ("settle",       0.6 / speed),
    ]

    # The acknowledge phase only fires for confident-ish personalities.
    should_acknowledge = personality.confidence > -0.3

    t = 0.0
    for phase_name, phase_dur in timeline:
        steps = max(1, int(round(phase_dur / tick_dt)))
        for step_i in range(steps):
            phase_t = step_i / max(1, steps - 1) if steps > 1 else 1.0
            tick = IntentTick()

            if phase_name == "delay":
                # Do nothing. Channels keep their idle behavior (breathing).
                pass
            elif phase_name == "orient":
                # Set the gaze yaw target. Channel low-pass-filters it.
                tick.gaze_target = GazeTarget(
                    yaw_rad=target_yaw_rad * phase_t,
                    pitch_rad=0.0,
                )
            elif phase_name == "look":
                tick.gaze_target = GazeTarget(
                    yaw_rad=target_yaw_rad,
                    pitch_rad=target_pitch_rad * phase_t,
                )
                # Lean very slightly forward as we look up — engagement.
                lean = 0.02 * (1.0 + personality.curiosity)
                tick.posture_target = PostureTarget(x_offset_rad=lean)
            elif phase_name == "hold":
                tick.gaze_target = GazeTarget(
                    yaw_rad=target_yaw_rad,
                    pitch_rad=target_pitch_rad,
                )
                # Hold the lean.
                lean = 0.02 * (1.0 + personality.curiosity)
                tick.posture_target = PostureTarget(x_offset_rad=lean)
            elif phase_name == "acknowledge":
                # Fire on first tick of phase; no need to re-fire.
                if step_i == 0 and should_acknowledge:
                    tick.fire_gesture = "paw_acknowledge"
                # Hold gaze + posture during the gesture.
                tick.gaze_target = GazeTarget(
                    yaw_rad=target_yaw_rad,
                    pitch_rad=target_pitch_rad,
                )
            elif phase_name == "settle":
                # Smoothly release lean and re-center gaze pitch.
                release = 1.0 - phase_t
                tick.gaze_target = GazeTarget(
                    yaw_rad=target_yaw_rad * release,
                    pitch_rad=target_pitch_rad * 0.3 * release,
                )
                tick.posture_target = PostureTarget(
                    x_offset_rad=0.02 * personality.curiosity * release,
                )

            t += tick_dt
            yield tick

    yield IntentTick(finished=True)
