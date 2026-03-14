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
"""DataFrameExporter — convert synchronized sensor data to pandas DataFrames."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from dimos.datasync.sync import SyncTransformer

Flattener = Callable[[Any], dict[str, Any]]


def flatten_odometry(msg: Any) -> dict[str, Any]:
    return {
        "x": msg.x, "y": msg.y, "z": msg.z,
        "roll": msg.roll, "pitch": msg.pitch, "yaw": msg.yaw,
        "vx": msg.vx, "vy": msg.vy, "vz": msg.vz,
        "wx": msg.wx, "wy": msg.wy, "wz": msg.wz,
    }


def flatten_imu(msg: Any) -> dict[str, Any]:
    return {
        "accel_x": msg.linear_acceleration.x,
        "accel_y": msg.linear_acceleration.y,
        "accel_z": msg.linear_acceleration.z,
        "gyro_x": msg.angular_velocity.x,
        "gyro_y": msg.angular_velocity.y,
        "gyro_z": msg.angular_velocity.z,
        "orient_x": msg.orientation.x,
        "orient_y": msg.orientation.y,
        "orient_z": msg.orientation.z,
        "orient_w": msg.orientation.w,
    }


def flatten_pose_stamped(msg: Any) -> dict[str, Any]:
    return {
        "x": msg.x, "y": msg.y, "z": msg.z,
        "roll": msg.roll, "pitch": msg.pitch, "yaw": msg.yaw,
    }


def flatten_image(msg: Any) -> dict[str, Any]:
    return {
        "width": msg.width, "height": msg.height, "channels": msg.channels,
        "format": msg.format.value if hasattr(msg.format, "value") else str(msg.format),
        "timestamp": msg.ts,
    }


def flatten_generic(msg: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, val in getattr(msg, "__dict__", {}).items():
        if key.startswith("_"):
            continue
        if isinstance(val, (int, float, str, bool)):
            result[key] = val
    return result


FLATTENERS: dict[str, Flattener] = {
    "nav_msgs.Odometry": flatten_odometry,
    "sensor_msgs.Imu": flatten_imu,
    "geometry_msgs.PoseStamped": flatten_pose_stamped,
    "sensor_msgs.Image": flatten_image,
}


def get_flattener(msg: Any) -> Flattener:
    msg_name = getattr(msg, "msg_name", None)
    if msg_name and msg_name in FLATTENERS:
        return FLATTENERS[msg_name]
    cls_name = type(msg).__name__
    for key, fn in FLATTENERS.items():
        if key.endswith(f".{cls_name}"):
            return fn
    return flatten_generic


def _import_pandas():  # type: ignore[no-untyped-def]
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError(
            "pandas is required for DataFrameExporter. "
            "Install it with: pip install 'dimos[datasync]'"
        ) from None


class DataFrameExporter:
    """Convert synchronized sensor data to pandas DataFrames."""

    def __init__(self, sync: SyncTransformer, flatteners: dict[str, Flattener] | None = None) -> None:
        self._sync = sync
        self._custom_flatteners = flatteners or {}

    def _flatten_row(self, grid_ts: float, row: dict[str, Any | None]) -> dict[str, Any]:
        flat: dict[str, Any] = {"timestamp": grid_ts}
        for topic_key, msg in row.items():
            if msg is None:
                continue
            flattener = self._custom_flatteners.get(topic_key) or get_flattener(msg)
            for field_name, value in flattener(msg).items():
                flat[f"{topic_key}.{field_name}"] = value
        return flat

    def to_dataframe(self, start: float | None = None, end: float | None = None) -> Any:
        pd = _import_pandas()
        rows = [self._flatten_row(ts, row) for ts, row in self._sync.iterate_synced(start, end)]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.set_index("timestamp", inplace=True)
        return df

    def to_dataframes(self, chunk_duration: float, start: float | None = None, end: float | None = None) -> Iterator[Any]:
        pd = _import_pandas()
        current_chunk: list[dict[str, Any]] = []
        chunk_start: float | None = None
        for grid_ts, row in self._sync.iterate_synced(start, end):
            if chunk_start is None:
                chunk_start = grid_ts
            if grid_ts - chunk_start >= chunk_duration:
                if current_chunk:
                    df = pd.DataFrame(current_chunk)
                    df.set_index("timestamp", inplace=True)
                    yield df
                current_chunk = []
                chunk_start = grid_ts
            current_chunk.append(self._flatten_row(grid_ts, row))
        if current_chunk:
            df = pd.DataFrame(current_chunk)
            df.set_index("timestamp", inplace=True)
            yield df

    def to_csv(self, path: str | Path) -> None:
        self.to_dataframe().to_csv(str(path))

    def to_parquet(self, path: str | Path) -> None:
        self.to_dataframe().to_parquet(str(path))
