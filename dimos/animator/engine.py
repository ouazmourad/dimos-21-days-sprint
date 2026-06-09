"""Animation engine — layered composition of background + triggered motion.

Models the architecture from Disney Research's "Design and Control of a
Bipedal Robotic Character" (arXiv:2501.05204), specifically its animation
engine:

  Layer 1 — Background: a continuous, always-on ambient animation
            (idle breathing, slow gaze drift, blinks). Never stops.
  Layer 2 — Triggered: artist-authored intent clips that blend *over* the
            background with a linear-ramping blend weight α. α ramps
            0→1 over T_ramp at the start of a clip and 1→0 over T_ramp
            before it ends, so intents ease in and out instead of
            hard-cutting.
  Output  — A first-order-hold + low-pass filter on the final joint
            command stream (their 37.5 Hz-cutoff smoothing stage),
            which removes the high-frequency steps a blend can introduce.

Blend math (matches the paper):
    config_blend = interp(config_bg, config_trig, α)
with component-wise linear interpolation for scalars / positions. (We
have no free quaternions in the Go2 face rig, so slerp isn't needed
here — but the blend points are where it would go.)
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

from dimos.animator.channels.gaze import GazeTarget
from dimos.animator.channels.posture import PostureTarget
from dimos.animator.intents.notice_guest import IntentTick
from dimos.animator.orchestrator import OrchestratorTick, PerformanceOrchestrator
from dimos.animator.personality import Personality


@dataclass
class Frame:
    """The blendable 'configuration' produced by a layer each tick.

    This is the reference the channels are told to aim at — the analog
    of the paper's configuration vector c_t that gets blended before the
    tracking controller.
    """

    gaze_target: GazeTarget = field(default_factory=GazeTarget)
    posture_target: PostureTarget = field(default_factory=PostureTarget)
    fire_gesture: str | None = None


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def blend_frames(bg: Frame, trig: Frame, alpha: float) -> Frame:
    """Blend a triggered frame over a background frame by weight α∈[0,1].

    α=0 → pure background, α=1 → pure triggered. Component-wise linear
    interpolation (the paper's ``interp`` for positions / joint angles).
    The gesture trigger is owned by whichever layer dominates (α≥0.5),
    since a gesture is a discrete event that can't be partially fired.
    """
    a = max(0.0, min(1.0, alpha))
    gaze = GazeTarget(
        yaw_rad=_lerp(bg.gaze_target.yaw_rad, trig.gaze_target.yaw_rad, a),
        pitch_rad=_lerp(bg.gaze_target.pitch_rad, trig.gaze_target.pitch_rad, a),
    )
    posture = PostureTarget(
        z_offset_rad=_lerp(bg.posture_target.z_offset_rad, trig.posture_target.z_offset_rad, a),
        x_offset_rad=_lerp(bg.posture_target.x_offset_rad, trig.posture_target.x_offset_rad, a),
        roll_offset_rad=_lerp(bg.posture_target.roll_offset_rad, trig.posture_target.roll_offset_rad, a),
    )
    # Gesture: triggered layer owns it once it's the dominant layer.
    gesture = trig.fire_gesture if a >= 0.5 else None
    return Frame(gaze_target=gaze, posture_target=posture, fire_gesture=gesture)


class BackgroundIdle:
    """Always-on ambient layer: slow gaze drift + occasional look-arounds.

    Breathing and blinks are produced by the breathing / expression
    channels regardless, so the background only needs to supply the
    gaze + posture references that keep an idle robot 'alive'.
    """

    def __init__(self, drift_period_s: float = 6.0, drift_amp_rad: float = 0.12) -> None:
        self._period = drift_period_s
        self._amp = drift_amp_rad
        self._t = 0.0

    def tick(self, dt: float, personality: Personality) -> Frame:
        self._t += dt
        # Curious characters drift more and faster; calm ones barely.
        amp = self._amp * (1.0 + 0.6 * personality.curiosity)
        period = self._period / max(0.4, personality.speed_scale())
        yaw = amp * math.sin(2 * math.pi * self._t / period)
        # A gentle secondary pitch drift, out of phase, for liveliness.
        pitch = 0.4 * amp * math.sin(2 * math.pi * self._t / (period * 1.7))
        return Frame(gaze_target=GazeTarget(yaw_rad=yaw, pitch_rad=pitch))


@dataclass
class _ActiveClip:
    """A triggered intent being played, with its ramp bookkeeping."""

    ticks: list[IntentTick]
    index: int = 0
    ramp_ticks: int = 0          # ramp-in / ramp-out length in ticks


class AnimationEngine:
    """Composes a background layer with ramped triggered intents.

    Usage:
        eng = AnimationEngine(orchestrator, personality)
        eng.trigger(notice_guest(...))     # queue an intent
        while running:
            tick = eng.tick()              # always returns a valid command
    """

    def __init__(
        self,
        orchestrator: PerformanceOrchestrator,
        personality: Personality,
        ramp_s: float = 0.35,            # paper's T_alpha
        lowpass_cutoff_hz: float = 37.5,  # paper's low-pass cutoff
    ) -> None:
        self._orch = orchestrator
        self._personality = personality
        self._dt = orchestrator.tick_dt
        self._background = BackgroundIdle()
        self._ramp_ticks = max(1, int(round(ramp_s / self._dt)))
        self._active: _ActiveClip | None = None

        # First-order low-pass on the output joint vector. The EMA factor
        # is derived from the cutoff the paper uses (RC = 1/(2π fc)).
        rc = 1.0 / (2 * math.pi * lowpass_cutoff_hz)
        self._lp_alpha = self._dt / (rc + self._dt)
        self._filtered: dict[str, float] | None = None

    # -- triggering ----------------------------------------------------

    def trigger(self, intent: Iterator[IntentTick]) -> None:
        """Queue an episodic intent to blend over the background."""
        ticks = [t for t in intent if not t.finished]
        self._active = _ActiveClip(ticks=ticks, ramp_ticks=self._ramp_ticks)

    @property
    def is_playing(self) -> bool:
        return self._active is not None

    def _alpha_for(self, clip: _ActiveClip) -> float:
        """Linear ramp: 0→1 over the first ramp_ticks, 1→0 over the last."""
        n = len(clip.ticks)
        i = clip.index
        r = clip.ramp_ticks
        if n <= 2 * r:
            # Short clip: triangular ramp peaking at the midpoint.
            half = max(1, n // 2)
            return min(i, n - 1 - i) / half if half else 1.0
        if i < r:
            return i / r
        if i >= n - r:
            return max(0.0, (n - 1 - i) / r)
        return 1.0

    # -- per-tick ------------------------------------------------------

    def tick(self) -> OrchestratorTick:
        bg_frame = self._background.tick(self._dt, self._personality)

        if self._active is not None:
            clip = self._active
            trig_tick = clip.ticks[clip.index]
            trig_frame = Frame(
                gaze_target=trig_tick.gaze_target or GazeTarget(),
                posture_target=trig_tick.posture_target or PostureTarget(),
                fire_gesture=trig_tick.fire_gesture,
            )
            alpha = self._alpha_for(clip)
            frame = blend_frames(bg_frame, trig_frame, alpha)
            clip.index += 1
            if clip.index >= len(clip.ticks):
                self._active = None
        else:
            frame = bg_frame

        # Feed the blended reference into the channel/mix/retarget stack.
        snapshot = self._orch._step_channels(  # noqa: SLF001 — same package
            frame.gaze_target, frame.posture_target, frame.fire_gesture,
        )
        mixed = self._orch._mixer.mix(snapshot)            # noqa: SLF001
        command = self._orch._retargeter.retarget(mixed)   # noqa: SLF001

        # Output low-pass (first-order). Removes high-frequency steps the
        # blend transitions can introduce, matching the paper's smoothing.
        if self._filtered is None:
            self._filtered = dict(command.angles)
        else:
            for k, v in command.angles.items():
                prev = self._filtered.get(k, v)
                self._filtered[k] = prev + (v - prev) * self._lp_alpha
        command.angles = dict(self._filtered)

        return OrchestratorTick(command=command, snapshot=snapshot,
                                finished=not self.is_playing)
