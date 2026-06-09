"""search_room — a slow scan sweep across multiple gaze targets.

Sweeps the body yaw across a range, pausing at each "look" point.
Used when the robot is told to "look around" or after losing track
of a person. Curious personalities sweep faster and with more pauses;
calm ones sweep slowly with fewer holds.
"""

from __future__ import annotations

from collections.abc import Iterator

from dimos.animator.channels.gaze import GazeTarget
from dimos.animator.channels.posture import PostureTarget
from dimos.animator.intents.notice_guest import IntentTick
from dimos.animator.personality import Personality


def search_room(
    yaw_range_rad: float = 0.8,
    n_stops: int = 5,
    personality: Personality | None = None,
    tick_dt: float = 1.0 / 50.0,
) -> Iterator[IntentTick]:
    personality = personality or Personality()
    speed = max(0.2, personality.speed_scale())
    # Curious adds more stops; calm reduces them (clamped to 2+).
    actual_stops = max(2, int(round(n_stops * (1.0 + 0.5 * personality.curiosity))))
    dwell = 0.35 / speed * personality.dwell_scale()
    travel = 0.25 / speed * personality.settle_scale()

    # Stops alternate left/right, starting toward the +yaw side.
    half = yaw_range_rad
    stops: list[float] = []
    for i in range(actual_stops):
        # Alternate sign, with shrinking magnitude on each pass.
        sign = 1 if i % 2 == 0 else -1
        magnitude = half * (1.0 - i / (actual_stops + 2))
        stops.append(sign * magnitude)
    # Always return to center at the end.
    stops.append(0.0)

    prev_yaw = 0.0
    for target_yaw in stops:
        # Travel phase: ramp from prev_yaw to target_yaw.
        travel_steps = max(1, int(round(travel / tick_dt)))
        for step_i in range(travel_steps):
            phase_t = step_i / max(1, travel_steps - 1) if travel_steps > 1 else 1.0
            ease = phase_t * phase_t * (3 - 2 * phase_t)
            yaw = prev_yaw + (target_yaw - prev_yaw) * ease
            tick = IntentTick()
            tick.gaze_target = GazeTarget(yaw_rad=yaw)
            yield tick

        # Dwell phase: hold at target.
        dwell_steps = max(1, int(round(dwell / tick_dt)))
        for _ in range(dwell_steps):
            tick = IntentTick()
            tick.gaze_target = GazeTarget(yaw_rad=target_yaw)
            yield tick

        prev_yaw = target_yaw

    yield IntentTick(finished=True)
