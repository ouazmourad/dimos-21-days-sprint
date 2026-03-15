"""DroneVelocityEnv -- track velocity commands with a Skydio X2 quadrotor."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from dimos.simulation.gym.base_env import DimOSMuJoCoEnv
from dimos.simulation.gym.envs.drone_hover import load_scene_with_assets
from dimos.simulation.gym.rewards import (
    reward_altitude_tracking,
    reward_alive,
    reward_energy_penalty,
    reward_smoothness,
    reward_upright,
    reward_velocity_tracking,
)

# Same physics-only drone XML as DroneHoverEnv.
_DRONE_XML = """\
<mujoco model="drone_velocity_training">
  <compiler autolimits="true"/>
  <option timestep="0.01" density="1.225" viscosity="1.8e-5"/>

  <default>
    <default class="x2">
      <geom mass="0"/>
      <motor ctrlrange="0 13"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 0 10" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="50 50 0.1" type="plane"/>

    <body name="x2" pos="0 0 0.1" childclass="x2">
      <freejoint/>
      <site name="imu" pos="0 0 .02"/>
      <geom type="box" size=".06 .027 .02" pos=".04 0 .02"/>
      <geom type="box" size=".06 .027 .02" pos=".04 0 .06"/>
      <geom type="box" size=".05 .027 .02" pos="-.07 0 .065"/>
      <geom type="box" size=".023 .017 .01" pos="-.137 .008 .065" quat="1 0 0 1"/>
      <geom name="rotor1" type="ellipsoid" size=".13 .13 .01" pos="-.14 -.18 .05" mass=".25"/>
      <geom name="rotor2" type="ellipsoid" size=".13 .13 .01" pos="-.14 .18 .05" mass=".25"/>
      <geom name="rotor3" type="ellipsoid" size=".13 .13 .01" pos=".14 .18 .08" mass=".25"/>
      <geom name="rotor4" type="ellipsoid" size=".13 .13 .01" pos=".14 -.18 .08" mass=".25"/>
      <geom type="ellipsoid" size=".16 .04 .02" pos="0 0 0.02" mass=".325"
            contype="0" conaffinity="0"/>
      <site name="thrust1" pos="-.14 -.18 .05"/>
      <site name="thrust2" pos="-.14 .18 .05"/>
      <site name="thrust3" pos=".14 .18 .08"/>
      <site name="thrust4" pos=".14 -.18 .08"/>
    </body>
  </worldbody>

  <actuator>
    <motor class="x2" name="thrust1" site="thrust1" gear="0 0 1 0 0 -.0201"/>
    <motor class="x2" name="thrust2" site="thrust2" gear="0 0 1 0 0  .0201"/>
    <motor class="x2" name="thrust3" site="thrust3" gear="0 0 1 0 0  .0201"/>
    <motor class="x2" name="thrust4" site="thrust4" gear="0 0 1 0 0 -.0201"/>
  </actuator>

  <sensor>
    <gyro name="body_gyro" site="imu"/>
    <accelerometer name="body_linacc" site="imu"/>
    <framequat name="body_quat" objtype="site" objname="imu"/>
  </sensor>

  <keyframe>
    <key name="hover" qpos="0 0 3 1 0 0 0"
         ctrl="3.2495625 3.2495625 3.2495625 3.2495625"/>
  </keyframe>
</mujoco>
"""


class DroneVelocityEnv(DimOSMuJoCoEnv):
    """Quadrotor must track random velocity commands (vx, vy, vz, yaw_rate).

    Observation (17): body angles(2), body velocities(3), angular rates(3),
                      target velocity command(4), motor thrusts(4), altitude(1)
    Action (4):       motor thrusts in [0, 13] N
    """

    def __init__(
        self,
        target_altitude: float = 3.0,
        cmd_change_interval: tuple[int, int] = (100, 300),
        episode_length: int = 1000,
        render_mode: str | None = None,
        xml_path: str | None = None,
        asset_files: list[str] | None = None,
    ) -> None:
        super().__init__(
            sim_dt=0.01,
            ctrl_dt=0.02,
            episode_length=episode_length,
            render_mode=render_mode,
        )
        self._target_altitude = target_altitude
        self._cmd_change_interval = cmd_change_interval

        if xml_path is not None:
            xml_string, assets = load_scene_with_assets(xml_path, asset_files)
            self.model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
        else:
            self.model = mujoco.MjModel.from_xml_string(_DRONE_XML)
        self.data = mujoco.MjData(self.model)
        self._body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "x2",
        )

        # Velocity command: [vx, vy, vz, yaw_rate]
        self._cmd = np.zeros(4, dtype=np.float32)
        self._next_cmd_change = 0

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self._finish_init()

    def _keyframe_id(self) -> int:
        return 0

    def _reset_noise(self, np_random: np.random.Generator) -> None:
        self.data.qpos[:3] += np_random.uniform(-0.2, 0.2, size=3)
        self.data.qpos[2] = max(self.data.qpos[2], 0.5)
        self.data.qvel[:3] += np_random.uniform(-0.1, 0.1, size=3)
        self._randomise_command()

    def _randomise_command(self) -> None:
        lo, hi = self._cmd_change_interval
        self._next_cmd_change = self._step_count + self.np_random.integers(lo, hi)
        self._cmd = np.array([
            self.np_random.uniform(-1.0, 1.0),   # vx
            self.np_random.uniform(-1.0, 1.0),   # vy
            self.np_random.uniform(-0.5, 0.5),   # vz
            self.np_random.uniform(-0.5, 0.5),   # yaw_rate
        ], dtype=np.float32)

    # -- observation ---------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        quat = self.data.xquat[self._body_id]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)

        gz = R[2, :]
        pitch = math.atan2(-gz[0], gz[2])
        roll = math.atan2(gz[1], gz[2])

        vel = self.data.cvel[self._body_id]
        v_world = vel[3:]
        v_body = R.T @ v_world
        gyro = self.data.sensor("body_gyro").data.copy()

        alt = float(self.data.xpos[self._body_id][2])
        thrusts = self.data.ctrl[:4] / 13.0

        return np.concatenate([
            [pitch, roll],
            v_body,
            gyro,
            self._cmd,
            thrusts,
            [alt],
        ]).astype(np.float32)

    # -- reward --------------------------------------------------------

    def _get_reward(self, action: np.ndarray) -> float:
        quat = self.data.xquat[self._body_id]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)

        vel = self.data.cvel[self._body_id]
        v_world = vel[3:]
        v_body = R.T @ v_world
        omega_body = R.T @ vel[:3]

        # Velocity tracking in body frame
        actual_vel = np.array([v_body[0], v_body[1], v_world[2], omega_body[2]])
        target_vel = self._cmd

        alt = float(self.data.xpos[self._body_id][2])

        r = 0.0
        r += reward_velocity_tracking(actual_vel, target_vel, scale=1.0)
        r += reward_altitude_tracking(alt, self._target_altitude, scale=0.5)
        r += reward_upright(R[2, :]) * 0.3
        r += reward_alive(alt, 0.3)
        r += reward_energy_penalty(action, scale=0.003)
        if self._last_action is not None:
            r += reward_smoothness(action, self._last_action, scale=0.005)
        return r

    # -- termination ---------------------------------------------------

    def _is_terminated(self) -> bool:
        alt = float(self.data.xpos[self._body_id][2])
        if alt < 0.3:
            return True
        quat = self.data.xquat[self._body_id]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        if R[2, 2] < math.cos(math.radians(60)):
            return True
        return False

    # -- step override for command randomisation -----------------------

    def step(self, action: np.ndarray):  # type: ignore[override]
        obs, reward, terminated, truncated, info = super().step(action)
        if self._step_count >= self._next_cmd_change:
            self._randomise_command()
        return obs, reward, terminated, truncated, info
