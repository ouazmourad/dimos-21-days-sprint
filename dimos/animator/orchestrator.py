"""Performance orchestrator — runs one intent end-to-end.

Holds the four channels, the mixer, the retargeter, and the
personality. Each call to ``run_intent`` drains an intent iterator
and yields the resulting joint command stream at the configured
tick rate. Callers (the driver script, the blueprint module, tests)
consume the stream however they want.

For v1 this is single-process and synchronous. v1.1 wraps it in a
DimOS ``Module`` with ``In``/``Out`` ports so perception can drive
the gaze/posture targets directly without going through the intent
iterator.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from dimos.animator.channels import (
    BreathingChannel,
    GazeChannel,
    GestureChannel,
    PostureChannel,
)
from dimos.animator.channels.gaze import GazeTarget
from dimos.animator.channels.posture import PostureTarget
from dimos.animator.intents.notice_guest import IntentTick
from dimos.animator.mixer import BehaviorMixer, ChannelState
from dimos.animator.personality import Personality
from dimos.animator.retargeter import Go2Retargeter, JointCommand
from dimos.animator.rig import CharacterRig


@dataclass
class OrchestratorTick:
    """One tick of orchestration output."""

    command: JointCommand
    snapshot: ChannelState
    finished: bool = False


class PerformanceOrchestrator:
    """Runs intents on a rig with a personality."""

    def __init__(
        self,
        rig: CharacterRig,
        personality: Personality,
        tick_hz: float = 50.0,
    ) -> None:
        self._rig = rig
        self._personality = personality
        self._tick_dt = 1.0 / tick_hz
        self._gaze = GazeChannel()
        self._posture = PostureChannel()
        self._gesture = GestureChannel()
        self._breathing = BreathingChannel()
        self._mixer = BehaviorMixer(rig)
        self._retargeter = Go2Retargeter(rig)

    @property
    def tick_dt(self) -> float:
        return self._tick_dt

    @property
    def personality(self) -> Personality:
        return self._personality

    def set_personality(self, p: Personality) -> None:
        """Hot-swap the personality. Channels react immediately."""
        self._personality = p

    def _step_channels(
        self,
        gaze_target: GazeTarget | None,
        posture_target: PostureTarget | None,
        fire_gesture: str | None,
    ) -> ChannelState:
        # Push intent-supplied targets into the channels first.
        if gaze_target is not None:
            self._gaze.set_target(gaze_target, self._personality)
        if posture_target is not None:
            self._posture.set_target(posture_target)
        if fire_gesture is not None:
            self._gesture.trigger(fire_gesture, self._personality)

        # Then step every channel forward by ``tick_dt``.
        gaze_state = self._gaze.step(self._tick_dt, self._personality)
        posture_state = self._posture.step(self._tick_dt, self._personality)
        gesture_state = self._gesture.step(self._tick_dt, self._personality)
        breath_state = self._breathing.step(self._tick_dt, self._personality)

        return ChannelState(
            gaze=gaze_state,
            posture=posture_state,
            gesture=gesture_state,
            breathing=breath_state,
        )

    def run_intent(self, intent: Iterable[IntentTick]) -> Iterator[OrchestratorTick]:
        """Drain an intent iterator, yielding one joint command per tick."""
        for intent_tick in intent:
            if intent_tick.finished:
                # One last tick at the current state so the consumer always
                # sees a clean "finished" signal alongside a valid command.
                snapshot = self._step_channels(None, None, None)
                mixed = self._mixer.mix(snapshot)
                command = self._retargeter.retarget(mixed)
                yield OrchestratorTick(command=command, snapshot=snapshot, finished=True)
                return

            snapshot = self._step_channels(
                intent_tick.gaze_target,
                intent_tick.posture_target,
                intent_tick.fire_gesture,
            )
            mixed = self._mixer.mix(snapshot)
            command = self._retargeter.retarget(mixed)
            yield OrchestratorTick(command=command, snapshot=snapshot, finished=False)

    def idle_tick(self) -> OrchestratorTick:
        """Run channels for one tick with no intent target.

        Use this between intents so breathing keeps cycling and the
        gaze/posture channels drift back to whatever they were last
        commanded toward.
        """
        snapshot = self._step_channels(None, None, None)
        mixed = self._mixer.mix(snapshot)
        command = self._retargeter.retarget(mixed)
        return OrchestratorTick(command=command, snapshot=snapshot, finished=False)
