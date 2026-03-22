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
# import concurrent.futures
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
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.simulation.mujoco.constants import DEPTH_CAMERA_FOV
from dimos.simulation.mujoco.depth_camera import depth_image_to_point_cloud
from dimos.simulation.mujoco.model import load_model, load_scene_xml
from dimos.simulation.mujoco.person_on_track import PersonPositionController
from dimos.simulation.mujoco.shared_memory import ShmReader
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


# def _process_lidar(
#     depth_images: list[NDArray[Any]],
#     camera_positions: list[NDArray[Any]],
#     camera_matrices: list[NDArray[Any]],
#     voxel_size: float,
#     shm: ShmReader,
# ) -> None:
#     """Process depth images into a lidar point cloud (runs in background thread)."""
#     all_points = []
#     for depth_image, cam_pos, cam_mat in zip(depth_images, camera_positions, camera_matrices):
#         points = depth_image_to_point_cloud(
#             depth_image, cam_pos, cam_mat, fov_degrees=DEPTH_CAMERA_FOV
#         )
#         if points.size > 0:
#             all_points.append(points)
#
#     if all_points:
#         combined_points = np.vstack(all_points)
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(combined_points)
#         pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
#
#         lidar_msg = PointCloud2(
#             pointcloud=pcd,
#             ts=time.time(),
#             frame_id="world",
#         )
#         shm.write_lidar(lidar_msg)


class MockController:
    """Controller that reads commands from shared memory.

    When wall_avoidance is enabled, overrides forward movement with a turn
    if the front depth camera sees a wall within WALL_THRESHOLD metres.
    This keeps the RL policy stable (no physics collision forces) while
    preventing the robot from staring at a wall for minutes.
    """

    WALL_THRESHOLD = 0.8  # metres — start turning when wall is this close
    TURN_SPEED = 0.6      # rad/s yaw injected when avoiding a wall

    def __init__(self, shm_interface: ShmReader, wall_avoidance: bool = False) -> None:
        self.shm = shm_interface
        self._command = np.zeros(3, dtype=np.float32)
        self._wall_avoidance = wall_avoidance
        self._front_depth: NDArray[Any] | None = None
        self._left_depth: NDArray[Any] | None = None
        self._right_depth: NDArray[Any] | None = None
        self._wall_turn_count: int = 0

    def set_depth(
        self, front: NDArray[Any], left: NDArray[Any], right: NDArray[Any]
    ) -> None:
        """Called from the sim loop after rendering depth cameras."""
        self._front_depth = front
        self._left_depth = left
        self._right_depth = right

    @staticmethod
    def _min_depth(depth: NDArray[Any]) -> float:
        """Minimum distance in center strip of a depth image."""
        h, w = depth.shape
        centre = depth[h // 3 : 2 * h // 3, w // 4 : 3 * w // 4]
        return float(np.min(centre)) if centre.size > 0 else 999.0

    def get_command(self) -> NDArray[Any]:
        """Get the current movement command."""
        cmd_data = self.shm.read_command()
        if cmd_data is not None:
            linear, angular = cmd_data
            self._command[0] = linear[0]
            self._command[1] = linear[1]
            self._command[2] = angular[2]

        result: NDArray[Any] = self._command.copy()

        if self._wall_avoidance and self._front_depth is not None:
            front_dist = self._min_depth(self._front_depth)

            if front_dist < self.WALL_THRESHOLD:
                # Wall ahead — decide turn direction from side depths
                left_dist = self._min_depth(self._left_depth) if self._left_depth is not None else 999.0
                right_dist = self._min_depth(self._right_depth) if self._right_depth is not None else 999.0

                # Turn toward the side with MORE space
                turn_dir = 1.0 if left_dist >= right_dist else -1.0

                result[0] = 0.0  # stop forward
                result[2] = turn_dir * self.TURN_SPEED
                self._wall_turn_count += 1

                # If stuck turning for a long time (corner), increase speed
                if self._wall_turn_count > 50:
                    result[2] *= 1.5
            else:
                self._wall_turn_count = 0
                if result[0] == 0.0 and result[2] == 0.0:
                    # No agent command and no wall — gentle forward drift
                    result[0] = 0.3

        return result

    def stop(self) -> None:
        """Stop method to satisfy InputController protocol."""
        pass


def _run_simulation(config: GlobalConfig, shm: ShmReader) -> None:
    robot_name = config.robot_model or "unitree_go1"
    if robot_name == "unitree_go2":
        robot_name = "unitree_go1"

    controller = MockController(shm, wall_avoidance=config.mujoco_wall_collision)
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

    spawn_pos = config.mujoco_start_pos_float

    data.qpos[0:3] = [spawn_pos[0], spawn_pos[1], z]

    # Apply initial heading (yaw) as a quaternion rotation around Z
    yaw_deg = config.mujoco_start_yaw
    if yaw_deg != 0.0:
        import math
        yaw = math.radians(yaw_deg)
        data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    mujoco.mj_forward(model, data)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    lidar_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "lidar_front_camera")

    person_position_controller = PersonPositionController(model) if config.mujoco_person else None

    # Build wall bounding boxes for position clamping (no physics forces).
    wall_boxes: list[tuple[float, float, float, float]] = []  # (xmin, xmax, ymin, ymax)
    if config.mujoco_wall_collision:
        robot_radius = 0.3  # keep robot center this far from wall surface
        for i in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if gname.startswith("wall_"):
                p = model.geom_pos[i]
                s = model.geom_size[i]
                wall_boxes.append((
                    p[0] - s[0] - robot_radius,
                    p[0] + s[0] + robot_radius,
                    p[1] - s[1] - robot_radius,
                    p[1] + s[1] + robot_radius,
                ))
        logger.info(f"Wall collision: {len(wall_boxes)} wall bounding boxes")

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

    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as m_viewer:
        # Create renderers at configured resolution
        rgb_renderer = mujoco.Renderer(model, height=height, width=width)
        depth_renderer = mujoco.Renderer(model, height=height, width=width)
        depth_renderer.enable_depth_rendering()

        depth_left_renderer = mujoco.Renderer(model, height=height, width=width)
        depth_left_renderer.enable_depth_rendering()

        depth_right_renderer = mujoco.Renderer(model, height=height, width=width)
        depth_right_renderer.enable_depth_rendering()

        scene_option = mujoco.MjvOption()

        # # Background lidar processing (async fix)
        # lidar_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # lidar_future: concurrent.futures.Future[None] | None = None

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

        while m_viewer.is_running() and not shm.should_stop():
            step_start = time.time()

            # Step simulation
            for _ in range(config.mujoco_steps_per_frame):
                mujoco.mj_step(model, data)

            # Position-clamp: push robot out of walls without applying forces.
            if wall_boxes:
                rx, ry = data.qpos[0], data.qpos[1]
                for xmin, xmax, ymin, ymax in wall_boxes:
                    if xmin < rx < xmax and ymin < ry < ymax:
                        # Find nearest edge and push out
                        dx_left = rx - xmin
                        dx_right = xmax - rx
                        dy_bot = ry - ymin
                        dy_top = ymax - ry
                        d_min = min(dx_left, dx_right, dy_bot, dy_top)
                        if d_min == dx_left:
                            data.qpos[0] = xmin
                        elif d_min == dx_right:
                            data.qpos[0] = xmax
                        elif d_min == dy_bot:
                            data.qpos[1] = ymin
                        else:
                            data.qpos[1] = ymax
                        # Zero out the velocity component that was pushing into wall
                        data.qvel[0] *= 0.5
                        data.qvel[1] *= 0.5

            # Fall recovery: if robot z drops below threshold, reset to spawn.
            if config.mujoco_wall_collision and data.qpos[2] < 0.3:
                logger.info("Fall detected — resetting robot to spawn")
                data.qpos[0:3] = [spawn_pos[0], spawn_pos[1], z]
                data.qpos[3:7] = [1, 0, 0, 0]  # neutral orientation
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)

            if person_position_controller:
                person_position_controller.tick(data)

            # Update FPS counter
            fps_frame_count += 1
            fps_now = time.time()
            fps_elapsed = fps_now - fps_last_time
            if fps_elapsed >= 1.0:
                fps_display = fps_frame_count / fps_elapsed
                fps_frame_count = 0
                fps_last_time = fps_now
            m_viewer.set_texts(
                (mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT, f"FPS: {fps_display:.0f}", "")
            )

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
                # Render depth on main thread (requires OpenGL context)
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

                # Feed depth to controller for wall avoidance
                controller.set_depth(depth_front, depth_left, depth_right)

                # Process depth images into lidar message (synchronous -- causes stalls)
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

        # lidar_executor.shutdown(wait=False)
        if person_position_controller:
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
