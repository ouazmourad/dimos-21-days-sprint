"""Go2LocomotionEnv -- quadruped locomotion matching Go1OnnxController obs."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from dimos.simulation.gym.base_env import DimOSMuJoCoEnv
from dimos.simulation.mujoco.model import get_assets
from dimos.utils.data import get_data


def _build_go1_xml() -> str:
    """Compose a minimal training scene including the Go1 robot."""
    scene_path = str(get_data("mujoco_sim")) + "/scene_empty.xml"
    with open(scene_path) as f:
        scene_xml = f.read()
    root = ET.fromstring(scene_xml)
    root.set("model", "go1_training")
    root.insert(0, ET.Element("include", file="unitree_go1.xml"))
    return ET.tostring(root, encoding="unicode")


class Go2LocomotionEnv(DimOSMuJoCoEnv):
    """Quadruped locomotion with observation layout matching Go1OnnxController.

    Observation (48): linvel(3), gyro(3), gravity(3), joint_pos(12),
                      joint_vel(12), last_action(12), command(3)
    Action (12):      joint position offsets in [-1, 1], applied as
                      ctrl = action * 0.5 + default_angles
    """

    _ACTION_SCALE = 0.5

    def __init__(
        self,
        episode_length: int = 1000,
        render_mode: str | None = None,
    ) -> None:
        super().__init__(
            sim_dt=0.005,
            ctrl_dt=0.02,
            episode_length=episode_length,
            render_mode=render_mode,
        )

        xml_string = _build_go1_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string, assets=get_assets())
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._default_angles = np.array(
            self.model.keyframe("home").qpos[7:], dtype=np.float32,
        )
        self._default_height = float(self.model.keyframe("home").qpos[2])

        self._body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk",
        )
        self._imu_site_id = self.model.site("imu").id

        # Velocity command [vx, vy, yaw_rate] — starts at zero (standing)
        self._command = np.zeros(3, dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)
        self._finish_init()

    # -- action space --------------------------------------------------

    def _action_bounds(self):
        n = len(self._default_angles)
        return -np.ones(n, dtype=np.float32), np.ones(n, dtype=np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        self.data.ctrl[:] = action * self._ACTION_SCALE + self._default_angles

    # -- reset ---------------------------------------------------------

    def _reset_noise(self, np_random: np.random.Generator) -> None:
        nq_joint = len(self._default_angles)
        # Very small noise — let the robot start near a stable pose
        self.data.qpos[7:] += np_random.uniform(-0.02, 0.02, size=nq_joint)
        self.data.qvel[6:] += np_random.uniform(-0.05, 0.05, size=nq_joint)

        # Curriculum: 70% chance of zero command (standing), 30% small velocity
        if np_random.random() < 0.7:
            self._command = np.zeros(3, dtype=np.float32)
        else:
            self._command = np.array([
                np_random.uniform(-0.3, 0.3),
                np_random.uniform(-0.15, 0.15),
                np_random.uniform(-0.2, 0.2),
            ], dtype=np.float32)

    # -- observation (matches Go1OnnxController.get_obs) ---------------

    def _get_obs(self) -> np.ndarray:
        linvel = self.data.sensor("local_linvel").data.copy()
        gyro = self.data.sensor("gyro").data.copy()
        imu_xmat = self.data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0.0, 0.0, -1.0])

        joint_angles = self.data.qpos[7:] - self._default_angles
        joint_velocities = self.data.qvel[6:]

        last_action = self._last_action if self._last_action is not None else np.zeros(12)

        # Command amplified x2 to match existing controller
        command = self._command.copy()
        command[0] *= 2.0
        command[1] *= 2.0

        return np.concatenate([
            linvel,
            gyro,
            gravity,
            joint_angles,
            joint_velocities,
            last_action,
            command,
        ]).astype(np.float32)

    # -- reward --------------------------------------------------------

    def _get_reward(self, action: np.ndarray) -> float:
        body_height = float(self.data.xpos[self._body_id][2])
        imu_xmat = self.data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity_body = imu_xmat.T @ np.array([0.0, 0.0, -1.0])

        r = 0.0

        # 1. SURVIVAL — dominant signal: large bonus each step alive
        r += 2.0

        # 2. HEIGHT — exponential kernel, peaks at default standing height
        height_error = body_height - self._default_height
        r += 1.5 * np.exp(-20.0 * height_error ** 2)

        # 3. UPRIGHT — reward for gravity vector aligned with body z-axis
        # gravity_body[2] = -1 when perfectly upright, +1 when inverted
        upright = (-gravity_body[2] + 1.0) / 2.0  # normalize to [0, 1]
        r += 1.0 * upright

        # 4. JOINT REGULARIZATION — stay near default pose
        joint_deviation = self.data.qpos[7:] - self._default_angles
        r -= 0.02 * float(np.sum(joint_deviation ** 2))

        # 5. VELOCITY TRACKING — only meaningful once standing is learned
        linvel = self.data.sensor("local_linvel").data
        vel_error = np.array([
            linvel[0] - self._command[0],
            linvel[1] - self._command[1],
        ])
        r += 0.5 * np.exp(-4.0 * float(np.sum(vel_error ** 2)))

        # 6. ANGULAR VELOCITY PENALTY — penalize wobbling/spinning
        gyro = self.data.sensor("gyro").data
        r -= 0.01 * float(np.sum(gyro ** 2))

        # 7. ENERGY — very small, don't discourage exploration
        r -= 0.0005 * float(np.sum(action ** 2))

        # 8. SMOOTHNESS — small action rate penalty
        if self._last_action is not None:
            r -= 0.005 * float(np.sum((action - self._last_action) ** 2))

        return r

    # -- termination ---------------------------------------------------

    def _is_terminated(self) -> bool:
        body_height = float(self.data.xpos[self._body_id][2])
        if body_height < 0.13:
            return True

        # Also terminate if body is heavily tilted (> 70 degrees)
        imu_xmat = self.data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity_body = imu_xmat.T @ np.array([0.0, 0.0, -1.0])
        if gravity_body[2] > -0.35:  # cos(70°) ≈ 0.34
            return True

        return False
