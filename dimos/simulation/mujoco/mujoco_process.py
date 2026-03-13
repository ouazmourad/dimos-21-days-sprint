#!/usr/bin/env python3

# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import pickle
import signal
import sys
import time
from typing import Any

import mujoco
from mujoco import viewer
import numpy as np
from numpy.typing import NDArray
import open3d as o3d  # type: ignore[import-untyped]

from dimos.core.global_config import GlobalConfig
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.simulation.mujoco.constants import DEPTH_CAMERA_FOV
from dimos.simulation.mujoco.depth_camera import depth_image_to_point_cloud
from dimos.simulation.mujoco.model import load_model, load_scene_xml
from dimos.simulation.mujoco.person_on_track import PersonPositionController
from dimos.simulation.mujoco.shared_memory import ShmReader
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class MockController:
    """Controller that reads commands from shared memory."""

    def __init__(self, shm_interface: ShmReader) -> None:
        self.shm = shm_interface
        self._command = np.zeros(3, dtype=np.float32)

    def get_command(self) -> NDArray[Any]:
        """Get the current movement command."""
        cmd_data = self.shm.read_command()
        if cmd_data is not None:
            linear, angular = cmd_data
            # MuJoCo expects [forward, lateral, rotational]
            self._command[0] = linear[0]  # forward/backward
            self._command[1] = linear[1]  # left/right
            self._command[2] = angular[2]  # rotation
        result: NDArray[Any] = self._command.copy()
        return result

    def stop(self) -> None:
        """Stop method to satisfy InputController protocol."""
        pass


class DroneCamera:
    """Drone-style camera that tracks the robot with keyboard orbit controls.

    Controls (press in the MuJoCo viewer window):
        F           Toggle follow / free mode
        Arrow keys  Orbit camera (azimuth / elevation)
        +  /  -     Zoom in / out
        PgUp / PgDn Raise / lower altitude offset
        R           Reset camera to defaults
    """

    def __init__(
        self,
        cam: Any,
        default_distance: float,
        default_azimuth: float,
        default_elevation: float,
    ) -> None:
        self._cam = cam
        self._follow = True
        self._alt_offset = 0.0
        self._defaults = (default_distance, default_azimuth, default_elevation)

    @property
    def follow(self) -> bool:
        return self._follow

    def handle_key(self, keycode: int) -> None:
        """GLFW key callback — called from viewer thread."""
        if keycode == 70:  # F
            self._follow = not self._follow
            logger.info("[DroneCamera] %s mode", "FOLLOW" if self._follow else "FREE")
        elif keycode == 82:  # R
            self._follow = True
            self._alt_offset = 0.0
            self._cam.distance, self._cam.azimuth, self._cam.elevation = self._defaults
            logger.info("[DroneCamera] Reset")
        elif keycode == 263:  # Left arrow
            self._cam.azimuth -= 5.0
        elif keycode == 262:  # Right arrow
            self._cam.azimuth += 5.0
        elif keycode == 265:  # Up arrow
            self._cam.elevation = max(-89.0, self._cam.elevation - 3.0)
        elif keycode == 264:  # Down arrow
            self._cam.elevation = min(-1.0, self._cam.elevation + 3.0)
        elif keycode == 61:  # = (+)
            self._cam.distance = max(1.0, self._cam.distance - 0.5)
        elif keycode == 45:  # -
            self._cam.distance += 0.5
        elif keycode == 266:  # PgUp
            self._alt_offset += 0.2
        elif keycode == 267:  # PgDn
            self._alt_offset = max(0.0, self._alt_offset - 0.2)

    def update(self, robot_pos: NDArray[Any]) -> None:
        """Track robot position in follow mode. Call each frame."""
        if self._follow:
            self._cam.lookat[0] = robot_pos[0]
            self._cam.lookat[1] = robot_pos[1]
            self._cam.lookat[2] = robot_pos[2] + self._alt_offset

    def status_text(self) -> str:
        mode = "FOLLOW" if self._follow else "FREE"
        return f"Cam:{mode} | F:toggle R:reset Arrows:orbit +/-:zoom PgUp/Dn:alt"


def _run_simulation(config: GlobalConfig, shm: ShmReader) -> None:
    robot_name = config.robot_model or "unitree_go1"
    if robot_name == "unitree_go2":
        robot_name = "unitree_go1"

    controller = MockController(shm)
    model, data = load_model(
        controller, robot=robot_name, scene_xml=load_scene_xml(config), config=config
    )

    if model is None or data is None:
        raise ValueError("Failed to load MuJoCo model: model or data is None")

    match robot_name:
        case "unitree_go1":
            z = 0.3
        case "unitree_g1":
            z = 0.8
        case _:
            z = 0

    pos = config.mujoco_start_pos_float

    data.qpos[0:3] = [pos[0], pos[1], z]

    mujoco.mj_forward(model, data)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    lidar_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "lidar_front_camera")

    person_position_controller = PersonPositionController(model)

    lidar_left_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "lidar_left_camera")
    lidar_right_camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "lidar_right_camera"
    )

    width = config.mujoco_video_width
    height = config.mujoco_video_height

    # Apply shadow settings to the model before creating renderers
    if not config.mujoco_shadows:
        model.vis.quality.shadowsize = 0

    shm.set_resolution(width, height)
    shm.signal_ready()

    drone_camera: DroneCamera | None = None

    def _key_callback(keycode: int) -> None:
        if drone_camera is not None:
            drone_camera.handle_key(keycode)

    with viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False, key_callback=_key_callback
    ) as m_viewer:
        # Create renderers at configured resolution
        rgb_renderer = mujoco.Renderer(model, height=height, width=width)
        depth_renderer = mujoco.Renderer(model, height=height, width=width)
        depth_renderer.enable_depth_rendering()

        depth_left_renderer = mujoco.Renderer(model, height=height, width=width)
        depth_left_renderer.enable_depth_rendering()

        depth_right_renderer = mujoco.Renderer(model, height=height, width=width)
        depth_right_renderer.enable_depth_rendering()

        scene_option = mujoco.MjvOption()

        # Timing control
        last_video_time = 0.0
        last_lidar_time = 0.0
        video_interval = 1.0 / config.mujoco_video_fps
        lidar_interval = 1.0 / config.mujoco_lidar_fps

        # FPS tracking
        fps_frame_count = 0
        fps_last_time = time.time()
        fps_display = 0.0

        m_viewer.cam.lookat = config.mujoco_camera_position_float[0:3]
        m_viewer.cam.distance = config.mujoco_camera_position_float[3]
        m_viewer.cam.azimuth = config.mujoco_camera_position_float[4]
        m_viewer.cam.elevation = config.mujoco_camera_position_float[5]

        drone_camera = DroneCamera(
            m_viewer.cam,
            default_distance=config.mujoco_camera_position_float[3],
            default_azimuth=config.mujoco_camera_position_float[4],
            default_elevation=config.mujoco_camera_position_float[5],
        )
        logger.info(
            "[DroneCamera] Ready — F:follow/free  Arrows:orbit  +/-:zoom  PgUp/Dn:alt  R:reset"
        )

        while m_viewer.is_running() and not shm.should_stop():
            step_start = time.time()

            # Step simulation
            for _ in range(config.mujoco_steps_per_frame):
                mujoco.mj_step(model, data)

            person_position_controller.tick(data)

            # Update drone camera tracking
            drone_camera.update(data.qpos[0:3])

            # Update FPS counter
            fps_frame_count += 1
            fps_now = time.time()
            fps_elapsed = fps_now - fps_last_time
            if fps_elapsed >= 1.0:
                fps_display = fps_frame_count / fps_elapsed
                fps_frame_count = 0
                fps_last_time = fps_now
            m_viewer.set_texts([
                (mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT, f"FPS: {fps_display:.0f}", ""),
                (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, drone_camera.status_text(), ""),
            ])

            m_viewer.sync()

            # Always update odometry
            pos = data.qpos[0:3].copy()
            quat = data.qpos[3:7].copy()  # (w, x, y, z)
            shm.write_odom(pos, quat, time.time())

            current_time = time.time()

            # Video rendering
            if current_time - last_video_time >= video_interval:
                rgb_renderer.update_scene(data, camera=camera_id, scene_option=scene_option)
                pixels = rgb_renderer.render()
                shm.write_video(pixels)
                last_video_time = current_time

            # Lidar/depth rendering
            if current_time - last_lidar_time >= lidar_interval:
                # Render all depth cameras
                depth_renderer.update_scene(data, camera=lidar_camera_id, scene_option=scene_option)
                depth_front = depth_renderer.render()

                depth_left_renderer.update_scene(
                    data, camera=lidar_left_camera_id, scene_option=scene_option
                )
                depth_left = depth_left_renderer.render()

                depth_right_renderer.update_scene(
                    data, camera=lidar_right_camera_id, scene_option=scene_option
                )
                depth_right = depth_right_renderer.render()

                shm.write_depth(depth_front, depth_left, depth_right)

                # Process depth images into lidar message
                all_points = []
                cameras_data = [
                    (
                        depth_front,
                        data.cam_xpos[lidar_camera_id],
                        data.cam_xmat[lidar_camera_id].reshape(3, 3),
                    ),
                    (
                        depth_left,
                        data.cam_xpos[lidar_left_camera_id],
                        data.cam_xmat[lidar_left_camera_id].reshape(3, 3),
                    ),
                    (
                        depth_right,
                        data.cam_xpos[lidar_right_camera_id],
                        data.cam_xmat[lidar_right_camera_id].reshape(3, 3),
                    ),
                ]

                for depth_image, camera_pos, camera_mat in cameras_data:
                    points = depth_image_to_point_cloud(
                        depth_image, camera_pos, camera_mat, fov_degrees=DEPTH_CAMERA_FOV
                    )
                    if points.size > 0:
                        all_points.append(points)

                if all_points:
                    combined_points = np.vstack(all_points)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(combined_points)
                    pcd = pcd.voxel_down_sample(voxel_size=config.mujoco_lidar_resolution)

                    lidar_msg = PointCloud2(
                        pointcloud=pcd,
                        ts=time.time(),
                        frame_id="world",
                    )
                    shm.write_lidar(lidar_msg)

                last_lidar_time = current_time

            # Control simulation speed
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        person_position_controller.stop()


if __name__ == "__main__":
    global_config = pickle.loads(base64.b64decode(sys.argv[1]))
    shm_names = json.loads(sys.argv[2])

    shm = ShmReader(shm_names)

    def signal_handler(_signum: int, _frame: Any) -> None:
        # Signal the main loop to exit gracefully so the viewer context
        # manager can close the window and clean up resources.
        shm.signal_stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        _run_simulation(global_config, shm)
    finally:
        shm.cleanup()
