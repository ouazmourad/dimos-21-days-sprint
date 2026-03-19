"""MultiRobotMujocoConnection: MuJoCo scenes for the telephone game.

Only Robot A and Robot C need their own simulation (A sees objects to
describe, C searches for them). Robot B is a text relay — it shares
Robot A's camera feed for visual context but doesn't need its own sim.

This keeps the memory footprint to 2 MuJoCo subprocesses instead of 3,
which is critical on 16 GB systems.
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


# Spawn positions for robots — far apart on the office floor (15x15)
# so they see different parts of the room.
ROBOT_POSITIONS = {
    "a": "-3.0, 3.0",  # Robot A — corner near curtain/window area
    "c": "3.0, -3.0",  # Robot C — opposite corner near desk/shelving
}


class MultiSimConfig(ModuleConfig):
    scene_xml: str | None = None


class MultiRobotMujocoConnection(Module[MultiSimConfig]):
    """Module exposing 3 robots' streams using only 2 MuJoCo processes.

    Robot A and C each get their own simulation. Robot B shares Robot A's
    camera feed (it only needs visual context for the text relay).
    """

    default_config = MultiSimConfig

    # Robot A streams
    a_cmd_vel: In[Twist]
    a_color_image: Out[Image]
    a_odom: Out[PoseStamped]
    a_camera_info: Out[CameraInfo]

    # Robot B shares A's camera
    b_color_image: Out[Image]

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
        cfg.performance_tier = "low"
        cfg.resolve_performance_tier()
        return cfg

    @rpc
    def start(self) -> None:
        super().start()

        from dimos.robot.unitree.mujoco_connection import MujocoConnection

        # Launch sequentially to avoid memory spikes on 16 GB systems
        for robot_id in ("a", "c"):
            logger.info(f"[MULTI_SIM] Starting Robot {robot_id.upper()} simulation...")
            cfg = self._make_config(robot_id)
            conn = MujocoConnection(cfg)
            conn.start()
            self._connections[robot_id] = conn
            logger.info(f"[MULTI_SIM] Robot {robot_id.upper()} ready")

        # Wire Robot A streams
        conn_a = self._connections["a"]
        self._disposables.add(conn_a.video_stream().subscribe(self.a_color_image.publish))
        # Robot B shares A's camera feed
        self._disposables.add(conn_a.video_stream().subscribe(self.b_color_image.publish))
        self._disposables.add(conn_a.odom_stream().subscribe(
            lambda odom: self.a_odom.publish(self._sim_odom_to_pose(odom))
        ))
        self._disposables.add(Disposable(self.a_cmd_vel.subscribe(
            lambda twist: conn_a.move(twist)
        )))

        # Wire Robot C streams
        conn_c = self._connections["c"]
        self._disposables.add(conn_c.video_stream().subscribe(self.c_color_image.publish))
        self._disposables.add(conn_c.odom_stream().subscribe(
            lambda odom: self.c_odom.publish(self._sim_odom_to_pose(odom))
        ))
        self._disposables.add(Disposable(self.c_cmd_vel.subscribe(
            lambda twist: conn_c.move(twist)
        )))

        # Camera info threads
        for rid, conn in self._connections.items():
            cam_out = getattr(self, f"{rid}_camera_info")
            t = Thread(target=self._publish_camera_info_loop, args=(conn, cam_out), daemon=True)
            t.start()
            self._camera_threads.append(t)

        logger.info("[MULTI_SIM] All robot connections started (2 sims: A + C)")

    @staticmethod
    def _sim_odom_to_pose(odom: SimOdometry) -> PoseStamped:
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
