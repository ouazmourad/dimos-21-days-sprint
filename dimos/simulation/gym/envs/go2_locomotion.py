"""Go2LocomotionEnv -- quadruped locomotion matching Go1OnnxController obs."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from dimos.simulation.gym.base_env import DimOSMuJoCoEnv
from dimos.simulation.gym.rewards import (
    reward_alive,
    reward_energy_penalty,
    reward_smoothness,
    reward_upright,
    reward_velocity_tracking,
)
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

        self._body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk",
        )
        self._imu_site_id = self.model.site("imu").id

        # Velocity command [vx, vy, yaw_rate], randomised each episode
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
        # Joint noise
        nq_joint = len(self._default_angles)
        self.data.qpos[7:] += np_random.uniform(-0.05, 0.05, size=nq_joint)
        self.data.qvel[6:] += np_random.uniform(-0.1, 0.1, size=nq_joint)

        # Randomise velocity command
        self._command = np.array([
            np_random.uniform(-1.0, 1.0),   # vx
            np_random.uniform(-0.5, 0.5),   # vy
            np_random.uniform(-0.5, 0.5),   # yaw_rate
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

        # Command amplified ×2 to match existing controller
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
        linvel = self.data.sensor("local_linvel").data
        imu_xmat = self.data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity_body = imu_xmat.T @ np.array([0.0, 0.0, -1.0])

        actual_vel = np.array([linvel[0], linvel[1], 0.0])
        target_vel = np.array([self._command[0], self._command[1], 0.0])
        body_height = float(self.data.xpos[self._body_id][2])

        r = 0.0
        r += reward_velocity_tracking(actual_vel[:2], target_vel[:2], scale=2.0)
        r += reward_upright(gravity_body) * 0.5
        r += reward_alive(body_height, 0.15)
        r += reward_energy_penalty(action, scale=0.005)
        if self._last_action is not None:
            r += reward_smoothness(action, self._last_action, scale=0.01)
        return r

    # -- termination ---------------------------------------------------

    def _is_terminated(self) -> bool:
        body_height = float(self.data.xpos[self._body_id][2])
        return body_height < 0.15
