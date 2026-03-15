"""G1StandingEnv -- humanoid standing balance matching G1OnnxController obs."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from dimos.simulation.gym.base_env import DimOSMuJoCoEnv
from dimos.simulation.gym.rewards import (
    reward_alive,
    reward_altitude_tracking,
    reward_energy_penalty,
    reward_smoothness,
    reward_upright,
)
from dimos.simulation.mujoco.model import get_assets
from dimos.utils.data import get_data


def _build_g1_xml() -> str:
    """Compose a minimal training scene including the G1 robot."""
    scene_path = str(get_data("mujoco_sim")) + "/scene_empty.xml"
    with open(scene_path) as f:
        scene_xml = f.read()
    root = ET.fromstring(scene_xml)
    root.set("model", "g1_training")
    root.insert(0, ET.Element("include", file="unitree_g1.xml"))
    return ET.tostring(root, encoding="unicode")


class G1StandingEnv(DimOSMuJoCoEnv):
    """Humanoid must stand and maintain balance.

    Observation (103): linvel(3), gyro(3), gravity(3), command(3),
                       joint_pos(29), joint_vel(29), last_action(29), phase(4)
    Action (29):       joint position offsets in [-1, 1], applied as
                       ctrl = action * 0.5 + default_angles
    """

    _ACTION_SCALE = 0.5
    _GAIT_FREQ = 1.5  # Hz (matches G1OnnxController)

    def __init__(
        self,
        target_height: float = 0.755,
        episode_length: int = 1000,
        render_mode: str | None = None,
    ) -> None:
        super().__init__(
            sim_dt=0.002,
            ctrl_dt=0.02,
            episode_length=episode_length,
            render_mode=render_mode,
        )
        self._target_height = target_height

        xml_string = _build_g1_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string, assets=get_assets())
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._default_angles = np.array(
            self.model.keyframe("home").qpos[7:], dtype=np.float32,
        )

        self._pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis",
        )
        self._imu_site_id = self.model.site("imu_in_pelvis").id

        # Gait phase (matches G1OnnxController)
        self._phase = np.array([0.0, np.pi])
        self._phase_dt = 2 * np.pi * self._GAIT_FREQ * self._ctrl_dt

        # Standing command (zero velocity)
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
        self.data.qpos[7:] += np_random.uniform(-0.02, 0.02, size=nq_joint)
        self.data.qvel[6:] += np_random.uniform(-0.05, 0.05, size=nq_joint)
        self._phase = np.array([0.0, np.pi])

    # -- observation (matches G1OnnxController.get_obs) ----------------

    def _get_obs(self) -> np.ndarray:
        linvel = self.data.sensor("local_linvel_pelvis").data.copy()
        gyro = self.data.sensor("gyro_pelvis").data.copy()
        imu_xmat = self.data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0.0, 0.0, -1.0])

        joint_angles = self.data.qpos[7:] - self._default_angles
        joint_velocities = self.data.qvel[6:]

        last_action = self._last_action if self._last_action is not None else np.zeros(29)

        phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)])

        # Command (with drift compensation matching existing controller)
        command = self._command.copy()

        return np.concatenate([
            linvel,
            gyro,
            gravity,
            command,
            joint_angles,
            joint_velocities,
            last_action,
            phase,
        ]).astype(np.float32)

    # -- reward --------------------------------------------------------

    def _get_reward(self, action: np.ndarray) -> float:
        imu_xmat = self.data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity_body = imu_xmat.T @ np.array([0.0, 0.0, -1.0])
        pelvis_height = float(self.data.xpos[self._pelvis_id][2])

        r = 0.0
        r += reward_upright(gravity_body) * 1.0
        r += reward_alive(pelvis_height, 0.4)
        r += reward_altitude_tracking(pelvis_height, self._target_height, scale=2.0)
        r += reward_energy_penalty(action, scale=0.003)
        if self._last_action is not None:
            r += reward_smoothness(action, self._last_action, scale=0.01)
        return r

    # -- termination ---------------------------------------------------

    def _is_terminated(self) -> bool:
        pelvis_height = float(self.data.xpos[self._pelvis_id][2])
        return pelvis_height < 0.4

    # -- step override for phase update --------------------------------

    def step(self, action: np.ndarray):  # type: ignore[override]
        result = super().step(action)
        # Advance gait phase (matches G1OnnxController._post_control_update)
        self._phase = np.fmod(self._phase + self._phase_dt + np.pi, 2 * np.pi) - np.pi
        return result
