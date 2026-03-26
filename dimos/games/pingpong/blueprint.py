"""Ping Pong Championship blueprint.

Minimal blueprint — just the G1 tennis sim. The match manager and
commentary run inside the MuJoCo process / dashboard respectively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dimos.core.blueprints import autoconnect
from dimos.robot.unitree.g1.sim import g1_sim_connection

if TYPE_CHECKING:
    from dimos.core.blueprints import Blueprint


def build_pingpong() -> Blueprint:
    """Build the ping pong championship blueprint."""
    from dimos.core.global_config import global_config

    global_config.n_workers = 1
    global_config.simulation = True
    global_config.robot_model = "unitree_g1_tennis"
    global_config.mujoco_room = "pingpong"
    global_config.mujoco_start_pos = "12.2, 0.0"
    global_config.mujoco_steps_per_frame = 5
    global_config.mujoco_camera_position = "10.37, 0, 1.5, 5, 0, -20"

    sim = g1_sim_connection()
    return autoconnect(sim)


def run_pingpong() -> None:
    """Run the ping pong championship."""
    game = build_pingpong()
    coordinator = game.build()
    coordinator.loop()


__all__ = ["build_pingpong", "run_pingpong"]
