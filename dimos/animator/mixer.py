"""Behavior mixer — combines channel outputs into one motion snapshot.

The four channels publish into a common namespace of virtual joints.
The mixer's job is to:

1. Collect the latest snapshot from each channel.
2. Apply per-joint clamping defined by the rig.
3. Detect conflicts (two channels trying to drive the same virtual
   joint) and resolve via the priority rules below.
4. Hand the result to the retargeter as a single ``MixedMotion``.

Priority for resolving conflicts on the *same* virtual joint:
    gesture > posture > gaze > breathing
Rationale: a gesture is a deliberate beat that the designer scripted;
posture is the body's framing; gaze is reactive; breathing is ambient.
A gesture should never be cancelled because gaze decided to look
sideways at the same moment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dimos.animator.channels.breathing import BreathingState
from dimos.animator.channels.gaze import GazeTarget
from dimos.animator.channels.gesture import GestureState
from dimos.animator.channels.posture import PostureTarget
from dimos.animator.rig import CharacterRig

# Priority order — last write wins among the channels listed here in
# order of *ascending* priority.  i.e. gesture overrides everything
# above it.
_PRIORITY_ORDER = ("breathing", "gaze", "posture", "gesture")


@dataclass
class ChannelState:
    """Latest snapshot from every channel."""

    gaze: GazeTarget = field(default_factory=GazeTarget)
    posture: PostureTarget = field(default_factory=PostureTarget)
    gesture: GestureState = field(default_factory=GestureState)
    breathing: BreathingState = field(default_factory=BreathingState)


@dataclass
class MixedMotion:
    """Per-virtual-joint offsets, in radians, relative to the default pose.

    Whatever isn't set here is implicitly zero (i.e. holds default).
    """

    trunk_yaw: float = 0.0
    trunk_pitch: float = 0.0
    trunk_roll: float = 0.0
    trunk_z: float = 0.0
    trunk_x: float = 0.0
    trunk_z_breath: float = 0.0
    paw_lift_fl: float = 0.0


class BehaviorMixer:
    """Stateless mixer parameterised by the rig's clamp values."""

    def __init__(self, rig: CharacterRig) -> None:
        self._role_by_joint = {r.joint: r for r in rig.roles}

    def _clamp(self, joint: str, value: float) -> float:
        role = self._role_by_joint.get(joint)
        return role.clamp(value) if role is not None else value

    def mix(self, snapshot: ChannelState) -> MixedMotion:
        """Combine the channel snapshot into a single joint-offset bundle.

        Conflict resolution follows ``_PRIORITY_ORDER``: each virtual
        joint is assigned by the highest-priority channel that drives
        it. For v1 each channel drives a disjoint set of joints, so
        priority only matters when we eventually add cross-channel
        joints (e.g. gesture overriding posture for a recoil beat).
        """
        # Default-pose-relative offsets, populated channel-by-channel.
        out: dict[str, float] = {}

        # Walk channels in priority order; later writes win.
        for channel_name in _PRIORITY_ORDER:
            if channel_name == "breathing":
                out["trunk_z_breath"] = snapshot.breathing.z_breath_offset_rad
            elif channel_name == "gaze":
                out["trunk_yaw"] = snapshot.gaze.yaw_rad
                out["trunk_pitch"] = snapshot.gaze.pitch_rad
            elif channel_name == "posture":
                out["trunk_z"] = snapshot.posture.z_offset_rad
                out["trunk_x"] = snapshot.posture.x_offset_rad
                out["trunk_roll"] = snapshot.posture.roll_offset_rad
            elif channel_name == "gesture":
                # Gesture only drives paw_lift_fl in v1.
                out["paw_lift_fl"] = snapshot.gesture.paw_lift_fl_offset_rad

        # Per-joint clamp from rig metadata.
        return MixedMotion(
            trunk_yaw=self._clamp("trunk_yaw", out.get("trunk_yaw", 0.0)),
            trunk_pitch=self._clamp("trunk_pitch", out.get("trunk_pitch", 0.0)),
            trunk_roll=self._clamp("trunk_roll", out.get("trunk_roll", 0.0)),
            trunk_z=self._clamp("trunk_z", out.get("trunk_z", 0.0)),
            trunk_x=self._clamp("trunk_x", out.get("trunk_x", 0.0)),
            trunk_z_breath=self._clamp("trunk_z_breath", out.get("trunk_z_breath", 0.0)),
            paw_lift_fl=self._clamp("paw_lift_fl", out.get("paw_lift_fl", 0.0)),
        )
