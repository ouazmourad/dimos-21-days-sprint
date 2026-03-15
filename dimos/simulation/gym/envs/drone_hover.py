"""DroneHoverEnv -- hover a Skydio X2 quadrotor at a target altitude."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from dimos.simulation.gym.base_env import DimOSMuJoCoEnv
from dimos.simulation.gym.rewards import (
    reward_alive,
    reward_altitude_tracking,
    reward_energy_penalty,
    reward_upright,
)


def load_scene_with_assets(
    xml_path: str,
    asset_files: list[str] | None = None,
) -> tuple[str, dict[str, bytes]]:
    """Load a MuJoCo scene XML and external asset files into a string + dict.

    Strips ``assetdir`` from the compiler so all files resolve via the dict.
    """
    with open(xml_path) as f:
        xml = f.read()

    root = ET.fromstring(xml)
    compiler = root.find("compiler")
    if compiler is not None and "assetdir" in compiler.attrib:
        del compiler.attrib["assetdir"]
    xml_string = ET.tostring(root, encoding="unicode")

    assets: dict[str, bytes] = {}
    for path in asset_files or []:
        with open(path, "rb") as f:
            assets[os.path.basename(path)] = f.read()

    return xml_string, assets

# Minimal physics-only drone XML (no visual meshes required).
_DRONE_XML = """\
<mujoco model="drone_training">
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


class DroneHoverEnv(DimOSMuJoCoEnv):
    """Quadrotor must hover at a target altitude.

    Observation (13): pitch, roll, vertical_vel, horizontal_vel(2),
                      angular_rates(3), altitude_error, motor_thrusts(4)
    Action (4):       motor thrusts in [0, 13] N
    """

    def __init__(
        self,
        target_altitude: float = 3.0,
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

        if xml_path is not None:
            xml_string, assets = load_scene_with_assets(xml_path, asset_files)
            self.model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
        else:
            self.model = mujoco.MjModel.from_xml_string(_DRONE_XML)
        self.data = mujoco.MjData(self.model)

        self._body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "x2",
        )

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self._finish_init()

    # -- keyframe ------------------------------------------------------

    def _keyframe_id(self) -> int:
        return 0  # "hover"

    def _reset_noise(self, np_random: np.random.Generator) -> None:
        # Small perturbation to position and velocity
        self.data.qpos[:3] += np_random.uniform(-0.2, 0.2, size=3)
        self.data.qpos[2] = max(self.data.qpos[2], 0.5)  # don't start on ground
        self.data.qvel[:3] += np_random.uniform(-0.1, 0.1, size=3)

    # -- observation ---------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        quat = self.data.xquat[self._body_id]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)

        # Body-frame pitch / roll from gravity direction
        gz = R[2, :]
        pitch = math.atan2(-gz[0], gz[2])
        roll = math.atan2(gz[1], gz[2])

        # World velocity -> body velocity
        vel = self.data.cvel[self._body_id]
        v_world = vel[3:]
        v_body = R.T @ v_world

        # Angular rates from gyro
        gyro = self.data.sensor("body_gyro").data.copy()

        # Altitude error
        alt = self.data.xpos[self._body_id][2]
        alt_error = self._target_altitude - alt

        # Normalised motor thrusts
        thrusts = self.data.ctrl[:4] / 13.0

        return np.concatenate([
            [pitch, roll],
            [v_world[2]],
            v_body[:2],
            gyro,
            [alt_error],
            thrusts,
        ]).astype(np.float32)

    # -- reward --------------------------------------------------------

    def _get_reward(self, action: np.ndarray) -> float:
        alt = float(self.data.xpos[self._body_id][2])
        quat = self.data.xquat[self._body_id]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        up = R[2, :]

        r = 0.0
        r += reward_altitude_tracking(alt, self._target_altitude, scale=2.0)
        r += reward_upright(up) * 0.5
        r += reward_alive(alt, 0.3)
        r += reward_energy_penalty(action, scale=0.005)
        return r

    # -- termination ---------------------------------------------------

    def _is_terminated(self) -> bool:
        alt = float(self.data.xpos[self._body_id][2])
        if alt < 0.3:
            return True

        # Excessive tilt (> 60 deg)
        quat = self.data.xquat[self._body_id]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        cos_tilt = R[2, 2]  # dot of body-z with world-z
        if cos_tilt < math.cos(math.radians(60)):
            return True

        return False
