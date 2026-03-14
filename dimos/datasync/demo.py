#!/usr/bin/env python3
"""
Demo: Multi-Sensor Time Synchronization & ML Data Export

Run with:  .venv/bin/python -m dimos.datasync.demo
"""

import math

from dimos.datasync.export import DataFrameExporter
from dimos.datasync.sync import SyncPolicy, SyncTransformer
from dimos.memory.timeseries.inmemory import InMemoryStore
from dimos.types.timestamped import Timestamped


class FakeOdometry(Timestamped):
    msg_name = "nav_msgs.Odometry"
    def __init__(self, ts: float, x: float, y: float, yaw: float) -> None:
        super().__init__(ts)
        self.x, self.y, self.z = x, y, 0.0
        self.roll = self.pitch = 0.0
        self.yaw = yaw
        self.vx, self.vy, self.vz = 0.3, 0.0, 0.0
        self.wx, self.wy, self.wz = 0.0, 0.0, 0.0


class FakeImu(Timestamped):
    msg_name = "sensor_msgs.Imu"
    def __init__(self, ts: float, accel_x: float, gyro_z: float) -> None:
        super().__init__(ts)
        class V:
            def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
        class Q:
            def __init__(self): self.x, self.y, self.z, self.w = 0.0, 0.0, 0.0, 1.0
        self.linear_acceleration = V(accel_x, 0.0, 9.81)
        self.angular_velocity = V(0.0, 0.0, gyro_z)
        self.orientation = Q()


class FakeLidar(Timestamped):
    def __init__(self, ts: float, n_points: int) -> None:
        super().__init__(ts)
        self.n_points = n_points
        self.frame_id = "lidar"


def main() -> None:
    print("=" * 60)
    print("  DimOS DataSync Demo")
    print("  Multi-Sensor Time Synchronization & ML Export")
    print("=" * 60)
    print()

    duration = 10.0
    odom_hz, imu_hz, lidar_hz = 50, 100, 2
    odom_store: InMemoryStore = InMemoryStore()
    imu_store: InMemoryStore = InMemoryStore()
    lidar_store: InMemoryStore = InMemoryStore()
    t0 = 1000.0

    print(f"Step 1: Generating {duration}s of synthetic sensor data...")
    print(f"  Odometry:  {odom_hz} Hz  ({int(duration * odom_hz)} samples)")
    print(f"  IMU:       {imu_hz} Hz  ({int(duration * imu_hz)} samples)")
    print(f"  LiDAR:     {lidar_hz} Hz  ({int(duration * lidar_hz)} samples)")

    for i in range(int(duration * odom_hz)):
        t = t0 + i / odom_hz
        angle = 0.2 * t
        odom_store.save(FakeOdometry(ts=t, x=2.0 * math.cos(angle), y=2.0 * math.sin(angle), yaw=angle))
    for i in range(int(duration * imu_hz)):
        t = t0 + i / imu_hz
        imu_store.save(FakeImu(ts=t, accel_x=0.1 * math.sin(t * 5), gyro_z=0.2))
    for i in range(int(duration * lidar_hz)):
        t = t0 + i / lidar_hz
        lidar_store.save(FakeLidar(ts=t, n_points=5000 + i * 100))

    total = len(odom_store) + len(imu_store) + len(lidar_store)
    print(f"  Total: {total} samples across 3 sensors\n")

    print(f"Step 2: Synchronizing to 10.0 Hz grid...")
    for policy in [SyncPolicy.HOLD, SyncPolicy.DROP]:
        sync = SyncTransformer(stores={"odom": odom_store, "imu": imu_store, "lidar": lidar_store}, target_hz=10.0, policy=policy)
        print(f"  Policy={policy.value:5s}: {len(list(sync.iterate_synced()))} synchronized rows")
    print()

    print("Step 3: Exporting to DataFrame...")
    sync = SyncTransformer(stores={"odom": odom_store, "imu": imu_store, "lidar": lidar_store}, target_hz=10.0, policy=SyncPolicy.HOLD)
    df = DataFrameExporter(sync).to_dataframe()
    print(f"  Shape: {df.shape}  ({df.shape[0]} rows x {df.shape[1]} columns)")
    print(f"  Columns: {', '.join(df.columns[:8])}...")
    print()

    print("Step 4: Sample data (first 5 rows):")
    print(df.head().to_string())
    print()

    print("Step 5: Chunked export (2s chunks)...")
    for i, chunk in enumerate(DataFrameExporter(sync).to_dataframes(chunk_duration=2.0)):
        print(f"  Chunk {i}: {chunk.shape[0]} rows, t=[{chunk.index[0]:.1f}, {chunk.index[-1]:.1f}]")

    print(f"\nDone! ML-ready DataFrame exported.")


if __name__ == "__main__":
    main()
