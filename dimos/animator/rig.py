"""Character rig — semantic body schema on top of a robot's URDF/MJCF.

A standard robot description tells us *what joints exist*. A character
rig says *what each joint is for* in performance terms. The rig is the
contract between intent (gaze, posture, gesture, breathing) and the
actual motors that have to move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

ChannelName = Literal["gaze", "posture", "gesture", "breathing"]
CHANNEL_NAMES: tuple[ChannelName, ...] = ("gaze", "posture", "gesture", "breathing")


@dataclass(frozen=True)
class JointRole:
    """A single joint's role in the character rig.

    Real joints are referenced by their MJCF/URDF name. Virtual joints
    (``trunk_yaw``, ``trunk_pitch``, ``trunk_z`` for the Go2) are
    composite expressive axes that the retargeter implements by
    coordinating multiple real joints.
    """

    joint: str
    channel: ChannelName
    sign: float = 1.0
    max_offset_rad: float = 0.3
    speed_limit_rad_s: float = 2.0
    virtual: bool = False

    def clamp(self, value: float) -> float:
        """Clamp a desired offset to ``[-max_offset, max_offset]``."""
        if value > self.max_offset_rad:
            return self.max_offset_rad
        if value < -self.max_offset_rad:
            return -self.max_offset_rad
        return value


@dataclass
class CharacterRig:
    """Semantic body schema for one robot."""

    robot: str
    default_pose: dict[str, float]
    roles: list[JointRole] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path | str) -> CharacterRig:
        path = Path(path)
        data = yaml.safe_load(path.read_text())
        roles = [JointRole(**r) for r in data.get("roles", [])]
        return cls(
            robot=data["robot"],
            default_pose=dict(data.get("default_pose", {})),
            roles=roles,
        )

    def joints_for_channel(self, channel: ChannelName) -> list[JointRole]:
        return [r for r in self.roles if r.channel == channel]

    def has_channel(self, channel: ChannelName) -> bool:
        return any(r.channel == channel for r in self.roles)

    def virtual_joints(self) -> list[JointRole]:
        return [r for r in self.roles if r.virtual]

    def real_joints(self) -> list[JointRole]:
        return [r for r in self.roles if not r.virtual]
