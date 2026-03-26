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

import re
from typing import Literal, TypeAlias

from pydantic_settings import BaseSettings, SettingsConfigDict

from dimos.mapping.occupancy.path_map import NavigationStrategy

ViewerBackend: TypeAlias = Literal["rerun", "rerun-web", "foxglove", "none"]
PerformanceTier: TypeAlias = Literal["low", "medium", "high"]

# Tier preset values: setting -> {tier -> value}
_TIER_PRESETS: dict[str, dict[str, object]] = {
    "mujoco_video_fps": {"low": 5, "medium": 10, "high": 20},
    "mujoco_lidar_fps": {"low": 1, "medium": 2, "high": 2},
    "mujoco_video_width": {"low": 160, "medium": 240, "high": 320},
    "mujoco_video_height": {"low": 120, "medium": 180, "high": 240},
    "mujoco_lidar_resolution": {"low": 0.1, "medium": 0.05, "high": 0.05},
    "mujoco_steps_per_frame": {"low": 3, "medium": 5, "high": 7},
    "mujoco_shadowsize": {"low": 0, "medium": 2048, "high": 8192},
    "mujoco_shadows": {"low": False, "medium": True, "high": True},
    "mujoco_reflections": {"low": False, "medium": False, "high": True},
    "n_dask_workers": {"low": 2, "medium": 4, "high": 6},
}


def _get_all_numbers(s: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", s)]


class GlobalConfig(BaseSettings):
    robot_ip: str | None = None
    simulation: bool = False
    replay: bool = False
    viewer_backend: ViewerBackend = "rerun-web"
    performance_tier: PerformanceTier = "high"
    n_dask_workers: int = 6
    memory_limit: str = "auto"
    mujoco_camera_position: str | None = None
    mujoco_room: str | None = None
    mujoco_room_from_occupancy: str | None = None
    mujoco_global_costmap_from_occupancy: str | None = None
    mujoco_global_map_from_pointcloud: str | None = None
    mujoco_start_pos: str = "-1.0, 1.0"
    mujoco_steps_per_frame: int = 7
    mujoco_video_fps: int = 20
    mujoco_lidar_fps: int = 2
    mujoco_video_width: int = 320
    mujoco_video_height: int = 240
    mujoco_lidar_resolution: float = 0.05
    mujoco_shadowsize: int = 8192
    mujoco_shadows: bool = True
    mujoco_reflections: bool = True
    mujoco_person: bool = True
    robot_model: str | None = None
    robot_ips: str | None = None
    robot_width: float = 0.3
    robot_rotation_diameter: float = 0.6
    n_workers: int = 2
    planner_strategy: NavigationStrategy = "simple"
    planner_robot_speed: float | None = None
    dtop: bool = False
    obstacle_avoidance: bool = True
    dask: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def update(self, **kwargs: object) -> None:
        """Update config fields in place."""
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"GlobalConfig has no field '{key}'")
            setattr(self, key, value)

    def resolve_performance_tier(self, explicit_overrides: set[str] | None = None) -> None:
        """Apply tier preset values for fields that were not explicitly overridden."""
        tier = self.performance_tier
        overrides = explicit_overrides or set()
        for field_name, tier_values in _TIER_PRESETS.items():
            if field_name not in overrides:
                setattr(self, field_name, tier_values[tier])

    @property
    def unitree_connection_type(self) -> str:
        if self.replay:
            return "replay"
        if self.simulation:
            return "mujoco"
        return "webrtc"

    @property
    def mujoco_start_pos_float(self) -> tuple[float, float]:
        x, y = _get_all_numbers(self.mujoco_start_pos)
        return (x, y)

    @property
    def mujoco_camera_position_float(self) -> tuple[float, ...]:
        if self.mujoco_camera_position is None:
            return (-0.906, 0.008, 1.101, 4.931, 89.749, -46.378)
        return tuple(_get_all_numbers(self.mujoco_camera_position))


global_config = GlobalConfig()
