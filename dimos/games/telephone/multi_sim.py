"""MultiRobotMujocoConnection: single MuJoCo scene with 3 robots.

Uses Option B (MVP): 3 separate MujocoConnection instances sharing the
same scene XML but with different spawn positions. Each robot runs its
own MuJoCo subprocess. Robots communicate via text streams only, so
shared physics is not required.
"""

import copy
import threading
import time
from threading import Thread
from typing import Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.unitree.type.odometry import Odometry as SimOdometry
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


# Spawn positions for 3 robots in 3 rooms
ROBOT_POSITIONS = {
    "a": "0.0, 0.0",   # Room A center
    "b": "10.0, 0.0",  # Room B center
    "c": "20.0, 0.0",  # Room C center
}


class MultiSimConfig(ModuleConfig):
    scene_xml: str | None = None


class MultiRobotMujocoConnection(Module[MultiSimConfig]):
    """Single module exposing 3 robots (A, B, C) via prefixed streams.

    Internally creates 3 MujocoConnection instances, each with a
    different start position. Each runs its own MuJoCo subprocess.
    """

    default_config = MultiSimConfig

    # Robot A streams
    a_cmd_vel: In[Twist]
    a_color_image: Out[Image]
    a_odom: Out[PoseStamped]
    a_camera_info: Out[CameraInfo]

    # Robot B streams
    b_cmd_vel: In[Twist]
    b_color_image: Out[Image]
    b_odom: Out[PoseStamped]
    b_camera_info: Out[CameraInfo]

    # Robot C streams
    c_cmd_vel: In[Twist]
    c_color_image: Out[Image]
    c_odom: Out[PoseStamped]
    c_camera_info: Out[CameraInfo]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._connections: dict[str, Any] = {}
        self._stop_event = threading.Event()
        self._camera_threads: list[Thread] = []

    def _make_config(self, robot_id: str) -> GlobalConfig:
        """Create a GlobalConfig copy with the given robot's start position."""
        cfg = copy.deepcopy(self.config.g)
        cfg.mujoco_start_pos = ROBOT_POSITIONS[robot_id]
        cfg.simulation = True
        # Use low resolution to fit within the default UDP receive buffer
        # (212 KB). 160x120 RGB = 57,600 bytes — well within limits.
        cfg.mujoco_video_width = 160
        cfg.mujoco_video_height = 120
        return cfg

    @rpc
    def start(self) -> None:
        super().start()

        from concurrent.futures import ThreadPoolExecutor, as_completed

        from dimos.robot.unitree.mujoco_connection import MujocoConnection

        # Create and start all 3 connections in parallel to stay within
        # the 120 s RPC timeout (~50 s each sequentially = 150 s > 120 s).
        def _init_robot(robot_id: str) -> tuple[str, MujocoConnection]:
            cfg = self._make_config(robot_id)
            conn = MujocoConnection(cfg)
            conn.start()
            return robot_id, conn

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_init_robot, rid): rid for rid in ("a", "b", "c")}
            for future in as_completed(futures):
                robot_id, conn = future.result()
                self._connections[robot_id] = conn
                logger.info(f"[MULTI_SIM] Robot {robot_id.upper()} connection ready")

        # Wire streams after all connections are up
        for robot_id, conn in self._connections.items():
            color_out = getattr(self, f"{robot_id}_color_image")
            odom_out = getattr(self, f"{robot_id}_odom")
            cmd_in = getattr(self, f"{robot_id}_cmd_vel")
            camera_out = getattr(self, f"{robot_id}_camera_info")

            self._disposables.add(conn.video_stream().subscribe(color_out.publish))
            self._disposables.add(conn.odom_stream().subscribe(
                lambda odom, out=odom_out: out.publish(self._sim_odom_to_pose(odom))
            ))
            self._disposables.add(Disposable(cmd_in.subscribe(
                lambda twist, c=conn: c.move(twist)
            )))

            t = Thread(
                target=self._publish_camera_info_loop,
                args=(conn, camera_out),
                daemon=True,
            )
            t.start()
            self._camera_threads.append(t)

        logger.info("[MULTI_SIM] All 3 robot connections started")

    @staticmethod
    def _sim_odom_to_pose(odom: SimOdometry) -> PoseStamped:
        """Convert simulation odometry to PoseStamped (same as G1SimConnection)."""
        return PoseStamped(
            ts=odom.ts,
            frame_id=odom.frame_id,
            position=odom.position,
            orientation=odom.orientation,
        )

    def _publish_camera_info_loop(self, conn: Any, camera_out: Out[CameraInfo]) -> None:
        info = conn.camera_info_static
        while not self._stop_event.is_set():
            camera_out.publish(info)
            self._stop_event.wait(1.0)

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        for conn in self._connections.values():
            conn.stop()
        for t in self._camera_threads:
            if t.is_alive():
                t.join(timeout=1.0)
        super().stop()


multi_robot_sim = MultiRobotMujocoConnection.blueprint

__all__ = ["MultiRobotMujocoConnection", "multi_robot_sim"]
