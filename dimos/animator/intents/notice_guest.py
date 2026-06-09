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

    # A cartoon double-take: a quick first glance, a beat looking away
    # (the "wait, what?"), then a big snap back with a forward lean and a
    # paw raise. Much more characterful than a single smooth orient.
    timeline = [
        ("delay",        delay),
        ("glance",       0.25 / speed),   # quick first look
        ("away",         0.30 / speed),   # double-take: look away again
        ("snap",         0.22 / speed),   # snap back, fast + big
        ("lean_in",      0.35 / speed),   # lean in, look up at the face
        ("hold",         0.7 / speed * personality.dwell_scale()),
        ("acknowledge",  0.7 / speed),
        ("settle",       0.7 / speed),
    ]

    # Big lean scales with curiosity; confident characters lean more boldly.
    lean_amp = 0.12 * (1.0 + 0.5 * personality.curiosity + 0.3 * personality.confidence)
    should_acknowledge = personality.confidence > -0.4

    for phase_name, phase_dur in timeline:
        steps = max(1, int(round(phase_dur / tick_dt)))
        for step_i in range(steps):
            phase_t = step_i / max(1, steps - 1) if steps > 1 else 1.0
            tick = IntentTick()

            if phase_name == "delay":
                pass
            elif phase_name == "glance":
                # Quick partial look toward the target.
                tick.gaze_target = GazeTarget(yaw_rad=target_yaw_rad * 0.5 * phase_t)
            elif phase_name == "away":
                # Double-take: drift back toward neutral (look away).
                tick.gaze_target = GazeTarget(yaw_rad=target_yaw_rad * 0.5 * (1.0 - phase_t))
            elif phase_name == "snap":
                # Snap to the target, overshooting slightly for a "pop".
                overshoot = 1.0 + 0.15 * math.sin(math.pi * phase_t)
                tick.gaze_target = GazeTarget(
                    yaw_rad=target_yaw_rad * overshoot,
                    pitch_rad=target_pitch_rad * 0.6 * phase_t,
                )
            elif phase_name == "lean_in":
                tick.gaze_target = GazeTarget(yaw_rad=target_yaw_rad, pitch_rad=target_pitch_rad)
                tick.posture_target = PostureTarget(x_offset_rad=lean_amp * phase_t)
            elif phase_name == "hold":
                tick.gaze_target = GazeTarget(yaw_rad=target_yaw_rad, pitch_rad=target_pitch_rad)
                tick.posture_target = PostureTarget(x_offset_rad=lean_amp)
            elif phase_name == "acknowledge":
                if step_i == 0 and should_acknowledge:
                    tick.fire_gesture = "paw_wave"   # bigger than acknowledge
                tick.gaze_target = GazeTarget(yaw_rad=target_yaw_rad, pitch_rad=target_pitch_rad)
                tick.posture_target = PostureTarget(x_offset_rad=lean_amp)
            elif phase_name == "settle":
                release = 1.0 - phase_t
                tick.gaze_target = GazeTarget(
                    yaw_rad=target_yaw_rad * release,
                    pitch_rad=target_pitch_rad * 0.3 * release,
                )
                tick.posture_target = PostureTarget(x_offset_rad=lean_amp * release)

            yield tick

    yield IntentTick(finished=True)
