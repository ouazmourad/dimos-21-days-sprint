"""Retargeter — virtual joints → real Go1/Go2 joint commands.

The Go2 has 12 actuated joints (4 legs × hip / thigh / calf). It has
no head, no neck pitch joint, no arms. Every expressive channel
output therefore has to be expressed by coordinating these 12 joints.

Coordinate convention follows the menagerie Go1 XML:

* ``+x`` is forward, ``+y`` is left, ``+z`` is up.
* Hip joint     (FR_hip / FL_hip / RR_hip / RL_hip):
    abduction — rotates the leg out from the body around x.
    Default magnitude ``0.1``; sign alternates left/right.
* Thigh joint   (FR_thigh / FL_thigh / RR_thigh / RL_thigh):
    pitch — rotates the upper leg around y.
    Default ``0.9`` rad (the home-pose squat).
* Calf joint    (FR_calf / FL_calf / RR_calf / RL_calf):
    knee bend — rotates the lower leg around y.
    Default ``-1.8`` rad.

The mapping from virtual joints to per-leg offsets:

* ``trunk_yaw``    — push two diagonally-opposite hips outward.
  This rotates the *standing posture* without moving the feet — the
  body twists in place. Front-right + rear-left out for +yaw.
* ``trunk_pitch``  — change front vs. rear leg height.
  Lift the front by reducing thigh angle on the front legs.
  Drop the rear with the opposite. Net: nose up.
* ``trunk_roll``   — change left vs. right leg height.
* ``trunk_z``      — uniform thigh + calf bend across all legs.
  More squat → smaller robot. Less squat → taller.
* ``trunk_z_breath`` — same as trunk_z but smaller, additive.
* ``trunk_x``      — shift mass forward/back by leaning the legs.
  Pitch the thighs forward on all four legs. Doesn't translate
  the body but changes its silhouette.
* ``paw_lift_fl``  — flex the FL thigh and calf together to pull
  the foot off the ground. Visually a "wave".
"""

from __future__ import annotations

from dataclasses import dataclass

from dimos.animator.mixer import MixedMotion
from dimos.animator.rig import CharacterRig

GO1_JOINT_ORDER: tuple[str, ...] = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)


@dataclass
class JointCommand:
    """Per-joint absolute target in radians, ready for the controller."""

    angles: dict[str, float]

    def as_vector(self, order: tuple[str, ...] = GO1_JOINT_ORDER) -> list[float]:
        return [self.angles[j] for j in order]


class Go2Retargeter:
    """Maps ``MixedMotion`` virtual joints to absolute Go1/Go2 joint targets.

    The retargeter is deterministic and stateless — it's a coordinate
    transform. Per-joint clamping is the rig's responsibility (already
    applied by the mixer); the retargeter only does geometry.
    """

    def __init__(self, rig: CharacterRig) -> None:
        if not rig.default_pose:
            raise ValueError("rig must define a default_pose")
        self._defaults = dict(rig.default_pose)

    def retarget(self, motion: MixedMotion) -> JointCommand:
        # Start from default pose; we'll add per-joint offsets below.
        out: dict[str, float] = dict(self._defaults)

        # ---- trunk_yaw: differential hip abduction ----
        # +yaw → robot rotates body left (nose right? convention-dependent).
        # Push FR hip outward (negative direction for right side) and
        # RL hip inward, etc. Net moment around z, no foot translation.
        yaw = motion.trunk_yaw
        out["FR_hip_joint"] += -yaw * 0.3
        out["FL_hip_joint"] += yaw * 0.3
        out["RR_hip_joint"] += yaw * 0.3
        out["RL_hip_joint"] += -yaw * 0.3

        # ---- trunk_pitch: front legs taller, rear legs shorter for nose-up ----
        # Positive pitch = nose up. Reduce front thigh flex (closer to 0)
        # → straighter front legs → body rotates forward-up at the front.
        pitch = motion.trunk_pitch
        out["FR_thigh_joint"] += -pitch * 0.5
        out["FL_thigh_joint"] += -pitch * 0.5
        out["RR_thigh_joint"] += pitch * 0.5
        out["RL_thigh_joint"] += pitch * 0.5

        # ---- trunk_roll: left vs. right leg height ----
        # Positive roll = leans right. Pull right legs shorter.
        roll = motion.trunk_roll
        out["FR_thigh_joint"] += roll * 0.4
        out["RR_thigh_joint"] += roll * 0.4
        out["FL_thigh_joint"] += -roll * 0.4
        out["RL_thigh_joint"] += -roll * 0.4

        # ---- trunk_z: overall height. More positive = taller (less squat) ----
        # In the default pose, thigh ≈ 0.9, calf ≈ -1.8. Reducing thigh and
        # making calf less negative both extend the leg. Apply symmetrically.
        z = motion.trunk_z + motion.trunk_z_breath
        for prefix in ("FR_", "FL_", "RR_", "RL_"):
            out[prefix + "thigh_joint"] += -z * 0.6
            out[prefix + "calf_joint"] += z * 1.2

        # ---- trunk_x: forward/back body shift via thigh pitch ----
        # Positive x shifts the body forward by pitching all thighs forward.
        x = motion.trunk_x
        for prefix in ("FR_", "FL_", "RR_", "RL_"):
            out[prefix + "thigh_joint"] += x * 0.3

        # ---- paw_lift_fl: front-left foot lift ----
        # Flex thigh and calf together to pull the foot off the ground.
        paw = motion.paw_lift_fl
        out["FL_thigh_joint"] += -paw * 1.0   # thigh flexes up
        out["FL_calf_joint"] += -paw * 1.5    # calf flexes inward

        # ---- articulated head (only if the model has these joints) ----
        # Emitted unconditionally; the consumer writes only the joints that
        # exist in its model. See dimos/animator/sim_head.py.
        out["neck_yaw"] = motion.neck_yaw
        out["neck_pitch"] = motion.neck_pitch

        # Eyelids: openness 1.0 → lids up (open, small negative angle);
        # openness 0.0 → lids swept down over the eye (~1.4 rad).
        lid_angle = 1.4 - 1.7 * motion.eye_openness
        out["lid_l"] = lid_angle
        out["lid_r"] = lid_angle

        # Brows: brow_raise +1 → raised (positive pitch), -1 → lowered.
        brow_angle = 0.45 * motion.brow_raise
        out["brow_l"] = brow_angle
        out["brow_r"] = brow_angle

        return JointCommand(angles=out)
