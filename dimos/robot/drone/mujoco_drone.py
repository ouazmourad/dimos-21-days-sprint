#!/usr/bin/env python3
"""MuJoCo-based drone simulation module for DimOS.

Provides MuJocoDroneController (PD flight controller) and MuJocoDroneModule
(DimOS Module with same stream interface as DroneConnectionModule).

The Skydio X2 quadrotor has 4 motors in X configuration:
    Motor 1: rear-left   (CCW prop, gear torque -0.0201)
    Motor 2: rear-right  (CW prop,  gear torque +0.0201)
    Motor 3: front-right (CW prop,  gear torque +0.0201)
    Motor 4: front-left  (CCW prop, gear torque -0.0201)

Hover thrust per motor: 3.2495625 N (from keyframe).
"""

import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.mapping.types import LatLon
from dimos.msgs.geometry_msgs import PoseStamped, Quaternion, Twist, Vector3
from dimos.msgs.sensor_msgs import Image
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# GPS origin: 454 Natoma St, San Francisco
GPS_ORIGIN_LAT = 37.780967
GPS_ORIGIN_LON = -122.406883
EARTH_R = 6_371_000.0


def _local_to_gps(x: float, y: float) -> tuple[float, float]:
    """Convert local XY (meters) to GPS lat/lon."""
    lat = GPS_ORIGIN_LAT + math.degrees(x / EARTH_R)
    lon = GPS_ORIGIN_LON + math.degrees(
        -y / (EARTH_R * math.cos(math.radians(GPS_ORIGIN_LAT)))
    )
    return lat, lon


def _find_skydio_model(city: bool = False) -> str:
    """Locate the Skydio X2 MuJoCo XML.

    If city=True, use the city_scene.xml bundled with this package
    (and copy mesh assets from menagerie if needed).
    """
    import importlib
    import shutil

    drone_dir = Path(__file__).parent
    city_scene = drone_dir / "city_scene.xml"

    if city and city_scene.exists():
        assets_dst = drone_dir / "assets"
        if not assets_dst.exists():
            spec = importlib.util.find_spec("mujoco_playground")
            if spec and spec.origin:
                src = (
                    Path(spec.origin).parent
                    / "external_deps"
                    / "mujoco_menagerie"
                    / "skydio_x2"
                    / "assets"
                )
                if src.exists():
                    shutil.copytree(str(src), str(assets_dst))
        return str(city_scene)

    # Fallback: menagerie scene
    for pkg in ("mujoco_playground", "mujoco_menagerie"):
        spec = importlib.util.find_spec(pkg)
        if spec and spec.origin:
            pkg_dir = Path(spec.origin).parent
            if pkg == "mujoco_playground":
                candidate = (
                    pkg_dir
                    / "external_deps"
                    / "mujoco_menagerie"
                    / "skydio_x2"
                    / "scene.xml"
                )
            else:
                candidate = pkg_dir / "skydio_x2" / "scene.xml"
            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(
        "Skydio X2 model not found. Install mujoco_menagerie or mujoco_playground."
    )


# ---------------------------------------------------------------------------
# Flight controller
# ---------------------------------------------------------------------------


class MuJocoDroneController:
    """PD attitude controller: velocity commands → 4 motor thrusts.

    Control pipeline (every physics step):
      1. Read body quaternion, world velocity, body angular rates
      2. Vertical: PD on vertical velocity error → collective thrust
      3. Horizontal: velocity error → desired tilt angle (clamped)
      4. Attitude: PD on (desired − actual) roll/pitch → differential thrust
      5. Yaw: heading-lock (angle PD) or rate-tracking (rate P)
      6. Motor mixing → clipped to [0, 13] per motor
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x2")

        # Equilibrium thrust per motor (from keyframe)
        self.hover_thrust = 3.2495625

        # --- Gains (tuned for 0.01 s timestep) ---
        self.kp_vxy = 1.2       # velocity → target tilt (rad per m/s)
        self.kp_vz = 3.0        # vertical velocity P
        self.kp_att = 8.0       # attitude angle P
        self.kd_att = 4.0       # attitude angular-rate D
        self.kp_yaw_rate = 1.5  # yaw rate tracking P
        self.kp_yaw_angle = 3.0 # yaw heading-lock P
        self.kd_yaw = 1.5       # yaw rate damping D

        # Velocity setpoints (body frame)
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_vz = 0.0
        self.cmd_yaw_rate = 0.0

        # Yaw heading lock: capture current heading from physics state.
        # Requires mj_forward() to have been called so xquat is valid.
        quat = self.data.xquat[self.body_id]
        self._target_yaw = math.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        )
        self._yaw_locked = True

    # -- command interface ----------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        self.cmd_vx = vx
        self.cmd_vy = vy
        self.cmd_vz = vz
        self.cmd_yaw_rate = yaw_rate

        if yaw_rate != 0.0:
            self._yaw_locked = False
        else:
            if not self._yaw_locked:
                # Capture current heading as new target
                q = self.data.xquat[self.body_id]
                self._target_yaw = math.atan2(
                    2.0 * (q[0] * q[3] + q[1] * q[2]),
                    1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
                )
                self._yaw_locked = True

    # -- control law ----------------------------------------------------------

    def compute_control(self) -> np.ndarray:
        quat = self.data.xquat[self.body_id].copy()     # (w, x, y, z)
        vel = self.data.cvel[self.body_id]               # (ang3, lin3) world
        omega_world = vel[:3]
        v_world = vel[3:]

        # Body-frame transform
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        v_body = R.T @ v_world
        omega_body = R.T @ omega_world

        w, qx, qy, qz = quat
        roll = math.atan2(2 * (w * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        pitch = math.asin(float(np.clip(2 * (w * qy - qz * qx), -1, 1)))

        # 1. Collective thrust (vertical velocity PD)
        vz_err = self.cmd_vz - v_world[2]
        base = self.hover_thrust + self.kp_vz * vz_err

        # 2. Desired tilt from horizontal velocity error
        vx_err = self.cmd_vx - v_body[0]
        target_pitch = float(np.clip(self.kp_vxy * vx_err, -0.4, 0.4))

        vy_err = self.cmd_vy - v_body[1]
        target_roll = float(np.clip(self.kp_vxy * vy_err, -0.4, 0.4))

        # 3. Attitude PD
        pitch_cmd = self.kp_att * (target_pitch - pitch) - self.kd_att * omega_body[1]
        roll_cmd = self.kp_att * (target_roll - roll) - self.kd_att * omega_body[0]

        # 4. Yaw controller
        if self._yaw_locked:
            cur_yaw = math.atan2(
                2 * (w * qz + qx * qy), 1 - 2 * (qy**2 + qz**2)
            )
            angle_err = math.atan2(
                math.sin(self._target_yaw - cur_yaw),
                math.cos(self._target_yaw - cur_yaw),
            )
            yaw_cmd = self.kp_yaw_angle * angle_err - self.kd_yaw * omega_body[2]
        else:
            yaw_cmd = self.kp_yaw_rate * (self.cmd_yaw_rate - omega_body[2])

        # 5. Motor mixing (X-config)
        t1 = base + pitch_cmd + roll_cmd - yaw_cmd   # rear-left   CCW
        t2 = base + pitch_cmd - roll_cmd + yaw_cmd   # rear-right  CW
        t3 = base - pitch_cmd - roll_cmd + yaw_cmd   # front-right CW
        t4 = base - pitch_cmd + roll_cmd - yaw_cmd   # front-left  CCW

        return np.clip(np.array([t1, t2, t3, t4]), 0.0, 13.0)


# ---------------------------------------------------------------------------
# DimOS Module (drop-in for DroneConnectionModule)
# ---------------------------------------------------------------------------


class MuJocoDroneModule(Module):
    """MuJoCo quadrotor simulation with the same stream/skill interface
    as DroneConnectionModule."""

    # Inputs
    movecmd: In[Vector3]
    movecmd_twist: In[Twist]
    gps_goal: In[LatLon]
    tracking_status: In[Any]

    # Outputs
    odom: Out[PoseStamped]
    gps_location: Out[LatLon]
    status: Out[Any]
    telemetry: Out[Any]
    video: Out[Image]
    follow_object_cmd: Out[Any]

    # Parameters
    render_width: int
    render_height: int
    sim_rate: float

    def __init__(
        self,
        render_width: int = 640,
        render_height: int = 480,
        sim_rate: float = 500.0,
        camera_name: str = "track",
        headless: bool = False,
    ) -> None:
        self.render_width = render_width
        self.render_height = render_height
        self.sim_rate = sim_rate
        self.camera_name = camera_name
        self.headless = headless

        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._controller: MuJocoDroneController | None = None
        self._renderer: mujoco.Renderer | None = None

        self._running = False
        self._armed = False
        self._mode = "STABILIZE"
        self._sim_thread: threading.Thread | None = None

        super().__init__()

    @rpc
    def start(self) -> None:
        super().start()

        scene_path = _find_skydio_model()
        logger.info(f"Loading MuJoCo drone from {scene_path}")
        self._model = mujoco.MjModel.from_xml_path(scene_path)
        self._data = mujoco.MjData(self._model)

        mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        mujoco.mj_forward(self._model, self._data)

        self._controller = MuJocoDroneController(self._model, self._data)

        if self.movecmd.transport:
            self._disposables.add(self.movecmd.subscribe(self._on_move))
        if self.movecmd_twist.transport:
            self._disposables.add(self.movecmd_twist.subscribe(self._on_move_twist))
        if self.gps_goal.transport:
            self._disposables.add(self.gps_goal.subscribe(self._on_gps_goal))

        self._running = True
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()
        logger.info("MuJoCo drone simulation started")

    # -- simulation loop ------------------------------------------------------

    def _sim_loop(self) -> None:
        assert self._model and self._data and self._controller

        if not self.headless:
            self._renderer = mujoco.Renderer(
                self._model, self.render_height, self.render_width
            )

        step_count = 0
        publish_every = int(self.sim_rate / 30)
        render_every = int(self.sim_rate / 15)

        while self._running:
            if self._armed and self._mode == "GUIDED":
                self._data.ctrl[:] = self._controller.compute_control()
            elif self._armed:
                self._data.ctrl[:] = self._controller.hover_thrust
            else:
                self._data.ctrl[:] = 0.0

            mujoco.mj_step(self._model, self._data)
            step_count += 1

            if step_count % publish_every == 0:
                self._publish_state()
            if not self.headless and step_count % render_every == 0:
                self._publish_camera()

            # Real-time pacing
            sim_t = self._data.time
            wall_t = time.monotonic()
            if not hasattr(self, "_wall0"):
                self._wall0 = wall_t
                self._sim0 = sim_t
            dt = (self._wall0 + (sim_t - self._sim0)) - wall_t
            if dt > 0.001:
                time.sleep(dt)

    # -- state publishing -----------------------------------------------------

    def _publish_state(self) -> None:
        m, d = self._model, self._data
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "x2")
        pos = d.xpos[bid].copy()
        q = d.xquat[bid].copy()
        vel = d.cvel[bid]

        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        now = time.time()

        self.odom.publish(
            PoseStamped(
                position=Vector3(x, y, z),
                orientation=Quaternion(float(q[1]), float(q[2]), float(q[3]), float(q[0])),
                frame_id="world",
                ts=now,
            )
        )

        lat, lon = _local_to_gps(x, y)
        self.gps_location.publish(LatLon(lat=lat, lon=lon))

        siny = 2.0 * (q[0] * q[3] + q[1] * q[2])
        cosy = 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
        yaw_rad = math.atan2(siny, cosy)
        heading = math.degrees(yaw_rad) % 360

        self.status.publish(
            {
                "armed": self._armed,
                "mode": self._mode,
                "battery_voltage": 16.4,
                "battery_remaining": 95,
                "altitude": z,
                "heading": heading,
                "vx": float(vel[3]),
                "vy": float(vel[4]),
                "vz": float(vel[5]),
                "lat": lat,
                "lon": lon,
                "ts": now,
                "simulator": "mujoco",
            }
        )

        roll_val = math.atan2(
            2 * (q[0] * q[1] + q[2] * q[3]),
            1 - 2 * (q[1] ** 2 + q[2] ** 2),
        )
        pitch_val = math.asin(float(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1, 1)))

        self.telemetry.publish(
            {
                "GLOBAL_POSITION_INT": {
                    "lat": lat,
                    "lon": lon,
                    "alt": z,
                    "relative_alt": z,
                    "vx": float(vel[3]),
                    "vy": float(vel[4]),
                    "vz": float(vel[5]),
                    "hdg": heading,
                },
                "ATTITUDE": {"roll": roll_val, "pitch": pitch_val, "yaw": yaw_rad},
                "timestamp": now,
            }
        )

    def _publish_camera(self) -> None:
        if not self._renderer or not self._model or not self._data:
            return
        self._renderer.update_scene(self._data, camera=self.camera_name)
        pixels = self._renderer.render()
        self.video.publish(
            Image(
                data=pixels.tobytes(),
                width=self.render_width,
                height=self.render_height,
                encoding="rgb8",
                step=self.render_width * 3,
            )
        )

    # -- input handlers -------------------------------------------------------

    def _on_move(self, v: Vector3) -> None:
        if self._controller:
            self._controller.set_velocity(float(v.x), float(v.y), float(v.z), 0.0)

    def _on_move_twist(self, msg: Twist) -> None:
        if self._controller:
            self._controller.set_velocity(
                float(msg.linear.x),
                float(msg.linear.y),
                float(msg.linear.z),
                float(msg.angular.z),
            )

    def _on_gps_goal(self, cmd: LatLon) -> None:
        if not self._data or not self._controller:
            return
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "x2")
        pos = self._data.xpos[bid]
        dx = math.radians(cmd.lat - GPS_ORIGIN_LAT) * EARTH_R
        dy = -(
            math.radians(cmd.lon - GPS_ORIGIN_LON)
            * EARTH_R
            * math.cos(math.radians(GPS_ORIGIN_LAT))
        )
        ex, ey = dx - pos[0], dy - pos[1]
        dist = math.sqrt(ex**2 + ey**2)
        if dist > 0.5:
            s = min(2.0, dist)
            self._controller.set_velocity(s * ex / dist, s * ey / dist, 0.0, 0.0)
        else:
            self._controller.set_velocity(0.0, 0.0, 0.0, 0.0)

    # -- skills (same as DroneConnectionModule) -------------------------------

    @skill
    def move(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, duration: float = 0.0) -> str:
        """Send velocity command. x=forward, y=right, z=up (m/s)."""
        if self._controller:
            self._controller.set_velocity(x, y, z, 0.0)
            if duration > 0:
                threading.Timer(duration, lambda: self._controller.set_velocity(0, 0, 0, 0)).start()
            return f"Moving: vx={x}, vy={y}, vz={z} for {duration}s"
        return "Failed: Simulation not running"

    @skill
    def move_with_yaw(self, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
                      yaw_rate: float = 0.0, duration: float = 2.0) -> str:
        """Move with velocity and yaw. Positive yaw_rate = turn right."""
        if self._controller:
            self._controller.set_velocity(vx, vy, vz, yaw_rate)
            if duration > 0:
                threading.Timer(duration, lambda: self._controller.set_velocity(0, 0, 0, 0)).start()
            return f"Moving: vx={vx}, vy={vy}, vz={vz}, yaw={yaw_rate} for {duration}s"
        return "Failed: Simulation not running"

    @skill
    def takeoff(self, altitude: float = 3.0) -> str:
        """Arm and takeoff to altitude."""
        self._armed = True
        self._mode = "GUIDED"
        if self._controller:
            self._controller.set_velocity(0, 0, 1.0, 0)

            def _check():
                if self._data is not None:
                    bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "x2")
                    if self._data.xpos[bid][2] >= altitude * 0.9:
                        self._controller.set_velocity(0, 0, 0, 0)
                        return
                if self._running:
                    threading.Timer(0.2, _check).start()

            threading.Timer(0.5, _check).start()
        return f"Taking off to {altitude}m"

    @skill
    def land(self) -> str:
        """Land the drone."""
        self._mode = "LAND"
        if self._controller:
            self._controller.set_velocity(0, 0, -0.5, 0)

            def _check():
                if self._data is not None:
                    bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "x2")
                    if self._data.xpos[bid][2] < 0.15:
                        self._controller.set_velocity(0, 0, 0, 0)
                        self._armed = False
                        return
                if self._running:
                    threading.Timer(0.2, _check).start()

            threading.Timer(0.5, _check).start()
        return "Landing"

    @skill
    def arm(self) -> str:
        self._armed = True
        return "Armed"

    @skill
    def disarm(self) -> str:
        self._armed = False
        if self._controller:
            self._controller.set_velocity(0, 0, 0, 0)
        return "Disarmed"

    @skill
    def set_mode(self, mode: str) -> str:
        self._mode = mode.upper()
        return f"Mode set to {self._mode}"

    @skill
    def fly_to(self, lat: float, lon: float, alt: float) -> str:
        self._on_gps_goal(LatLon(lat=lat, lon=lon))
        return f"Flying to {lat:.6f}, {lon:.6f} at {alt}m"

    @skill
    def observe(self) -> Image | None:
        if not self._renderer or not self._model or not self._data:
            return None
        self._renderer.update_scene(self._data, camera=self.camera_name)
        pixels = self._renderer.render()
        return Image(
            data=pixels.tobytes(),
            width=self.render_width,
            height=self.render_height,
            encoding="rgb8",
            step=self.render_width * 3,
        )

    @rpc
    def stop(self) -> None:
        self._running = False
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=3.0)
        if self._renderer:
            self._renderer.close()
        logger.info("MuJoCo drone simulation stopped")
        super().stop()


mujoco_drone = MuJocoDroneModule.blueprint
