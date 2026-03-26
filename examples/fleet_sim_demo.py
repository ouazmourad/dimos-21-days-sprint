"""Fleet simulation demo — 2 G1 humanoids in MuJoCo with performance_tier=low.

Uses MultiRobotSimConnection to run multiple robots from a single module
with prefixed streams, instead of spawning separate MuJoCo subprocesses.

Run:
    CI=1 .venv/bin/python examples/fleet_sim_demo.py
"""

import copy
import os
import threading
import time
from threading import Thread
from typing import Any

os.environ["CI"] = "1"

from dimos.core.blueprints import Blueprint, autoconnect
from dimos.core.core import rpc
from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.unitree.type.odometry import Odometry as SimOdometry
from dimos.utils.logging_config import setup_logger
from reactivex.disposable import Disposable

logger = setup_logger()


class TwoRobotSimConnection(Module):
    """Single module managing 2 G1 robots with prefixed streams.

    Each robot gets its own MujocoConnection subprocess but they're
    managed from one module. Uses performance_tier=low for each sim
    to keep total RAM under control.
    """

    # Alpha streams
    alpha_cmd_vel: In[Twist]
    alpha_color_image: Out[Image]
    alpha_odom: Out[PoseStamped]
    alpha_camera_info: Out[CameraInfo]

    # Bravo streams
    bravo_cmd_vel: In[Twist]
    bravo_color_image: Out[Image]
    bravo_odom: Out[PoseStamped]
    bravo_camera_info: Out[CameraInfo]

    SPAWN_POSITIONS = {
        "alpha": "-1.0, 1.0",
        "bravo": "1.0, -1.0",
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._connections: dict[str, Any] = {}
        self._stop_event = threading.Event()
        self._threads: list[Thread] = []

    def _make_config(self, robot_id: str) -> GlobalConfig:
        cfg = copy.deepcopy(global_config)
        cfg.simulation = True
        cfg.robot_model = "unitree_g1"
        cfg.mujoco_start_pos = self.SPAWN_POSITIONS[robot_id]
        cfg.performance_tier = "low"
        cfg.resolve_performance_tier()
        return cfg

    @rpc
    def start(self) -> None:
        super().start()

        from dimos.robot.unitree.mujoco_connection import MujocoConnection

        for robot_id in ("alpha", "bravo"):
            logger.info(f"[FLEET] Starting {robot_id} sim...")
            cfg = self._make_config(robot_id)
            conn = MujocoConnection(cfg)
            conn.start()
            self._connections[robot_id] = conn

            # Wire prefixed streams
            color_out = getattr(self, f"{robot_id}_color_image")
            odom_out = getattr(self, f"{robot_id}_odom")
            cmd_in = getattr(self, f"{robot_id}_cmd_vel")
            camera_out = getattr(self, f"{robot_id}_camera_info")

            self._disposables.add(conn.video_stream().subscribe(color_out.publish))
            self._disposables.add(conn.odom_stream().subscribe(
                lambda odom, out=odom_out: out.publish(PoseStamped(
                    ts=odom.ts, frame_id=odom.frame_id,
                    position=odom.position, orientation=odom.orientation,
                ))
            ))
            self._disposables.add(Disposable(cmd_in.subscribe(
                lambda twist, c=conn: c.move(twist)
            )))

            # Camera info thread
            t = Thread(
                target=self._pub_camera_info,
                args=(conn, camera_out),
                daemon=True,
            )
            t.start()
            self._threads.append(t)

            # Stagger launches to avoid resource spikes
            time.sleep(3)

        logger.info("[FLEET] Both robots online")

    def _pub_camera_info(self, conn: Any, out: Out[CameraInfo]) -> None:
        info = conn.camera_info_static
        while not self._stop_event.is_set():
            out.publish(info)
            self._stop_event.wait(1.0)

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        for conn in self._connections.values():
            conn.stop()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=1.0)
        super().stop()


two_robot_sim = TwoRobotSimConnection.blueprint


def main():
    global_config.simulation = True
    global_config.robot_model = "unitree_g1"
    global_config.n_workers = 1  # Single worker — one module manages both sims

    game = autoconnect(two_robot_sim())

    print("\n=== Fleet Simulation Demo (performance_tier=low) ===")
    print("  Alpha: spawns at (-1, 1)")
    print("  Bravo: spawns at (1, -1)")
    print("  Each sim uses low-tier settings (160x120, 5fps)")
    print("  Streams: alpha/odom, alpha/color_image, bravo/odom, bravo/color_image")
    print("  Single module manages both MuJoCo subprocesses")
    print()

    coordinator = game.build()
    coordinator.loop()


if __name__ == "__main__":
    main()
