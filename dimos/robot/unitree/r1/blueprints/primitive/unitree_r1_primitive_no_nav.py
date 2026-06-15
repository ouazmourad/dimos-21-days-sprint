#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
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

"""Minimal R1 stack without navigation, used as a base for larger blueprints."""

from typing import Any

from dimos_lcm.sensor_msgs import CameraInfo

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.camera.module import CameraModule
from dimos.robot.unitree.r1.sensors.pc1_camera import R1PC1Camera
from dimos.mapping.costmapper import CostMapper, costmap_to_rerun
from dimos.mapping.voxels import VoxelGridMapper
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.std_msgs.Bool import Bool
from dimos.navigation.frontier_exploration.wavefront_frontier_goal_selector import (
    WavefrontFrontierExplorer,
)
from dimos.visualization.vis_module import vis_module


def _convert_camera_info(camera_info: Any) -> Any:
    return camera_info.to_rerun(
        image_topic="/world/color_image",
        optical_frame="camera_optical",
    )


def _static_base_link(rr: Any) -> list[Any]:
    # TODO(R1): adjust camera mount Transform + base_link half-sizes to R1 geometry
    # (currently mirrors G1's half-sizes).
    return [
        rr.Boxes3D(
            half_sizes=[0.2, 0.15, 0.75],
            colors=[(0, 255, 127)],
            fill_mode="MajorWireframe",
        ),
        rr.Transform3D(parent_frame="tf#/base_link"),
    ]


def _r1_rerun_blueprint() -> Any:
    """Split layout: camera feed + 3D world view side by side."""
    import rerun as rr
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="world/color_image", name="Camera"),
            rrb.Spatial3DView(
                origin="world",
                name="3D",
                background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
                line_grid=rrb.LineGrid3D(
                    plane=rr.components.Plane3D.XY.with_distance(0.5),
                ),
            ),
            column_shares=[1, 2],
        ),
    )


rerun_config = {
    "blueprint": _r1_rerun_blueprint,
    "visual_override": {
        "world/camera_info": _convert_camera_info,
        "world/navigation_costmap": costmap_to_rerun,
    },
    "static": {
        "world/tf/base_link": _static_base_link,
    },
}

_with_vis = vis_module(viewer_backend=global_config.viewer, rerun_config=rerun_config)


def _create_r1_camera() -> R1PC1Camera:
    # The R1's own front camera, bridged from PC1 (the laptop can't read the R1's
    # DDS directly — type mismatch). Streams ~22 fps JPEG over SSH and emits DimOS
    # Image messages. Replaces the laptop webcam the G1 mirror used.
    return R1PC1Camera(fps=15)


_camera = (
    autoconnect(
        CameraModule.blueprint(
            transform=Transform(
                # TODO(R1): adjust camera mount Transform + base_link half-sizes to
                # R1 geometry (currently mirrors G1's camera mount).
                translation=Vector3(0.05, 0.0, 0.6),  # height of camera on R1 robot
                rotation=Quaternion.from_euler(Vector3(0.0, 0.2, 0.0)),
                frame_id="sensor",
                child_frame_id="camera_link",
            ),
            hardware=_create_r1_camera,
        ),
    )
    if not global_config.simulation
    else autoconnect()
)

unitree_r1_primitive_no_nav = (
    autoconnect(
        _with_vis,
        _camera,
        VoxelGridMapper.blueprint(),
        CostMapper.blueprint(),
        WavefrontFrontierExplorer.blueprint(),
    )
    .global_config(n_workers=4, robot_model="unitree_r1")
    .transports(
        {
            # R1 uses Twist for movement commands
            ("cmd_vel", Twist): LCMTransport("/cmd_vel", Twist),
            # State estimation from ROS
            ("state_estimation", Odometry): LCMTransport("/state_estimation", Odometry),
            ("odom", PoseStamped): LCMTransport("/odom", PoseStamped),
            # Navigation module topics from nav_bot
            ("goal_req", PoseStamped): LCMTransport("/goal_req", PoseStamped),
            ("goal_active", PoseStamped): LCMTransport("/goal_active", PoseStamped),
            ("path_active", Path): LCMTransport("/path_active", Path),
            ("pointcloud", PointCloud2): LCMTransport("/lidar", PointCloud2),
            ("global_pointcloud", PointCloud2): LCMTransport("/map", PointCloud2),
            # Original navigation topics for backwards compatibility
            ("goal_pose", PoseStamped): LCMTransport("/goal_pose", PoseStamped),
            ("goal_reached", Bool): LCMTransport("/goal_reached", Bool),
            ("cancel_goal", Bool): LCMTransport("/cancel_goal", Bool),
            # Camera topics
            ("color_image", Image): LCMTransport("/color_image", Image),
            ("camera_info", CameraInfo): LCMTransport("/camera_info", CameraInfo),
        }
    )
)

__all__ = ["unitree_r1_primitive_no_nav"]
