"""Match Manager — tracks score, serves balls, detects points.

Runs inside the MuJoCo process alongside the controller. Monitors ball
position to detect: table bounces, net hits, out-of-bounds, and scores.
Publishes match events via shared state.
"""

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Table geometry (must match scene_pingpong.xml)
TABLE_CENTER_X = 10.37
TABLE_HALF_LEN = 1.37
TABLE_Y_HALF = 0.7625
TABLE_Z = 0.76
NET_X = TABLE_CENTER_X

NEAR_EDGE = TABLE_CENTER_X - TABLE_HALF_LEN   # 9.0
FAR_EDGE = TABLE_CENTER_X + TABLE_HALF_LEN    # 11.74

# Robot is at x=12.2 (far side). Serve from behind near edge, arc over net.
SERVE_POS = np.array([8.5, 0.0, 1.2], dtype=np.float64)   # behind table, above table height
SERVE_VEL = np.array([5.0, 0.0, 1.5], dtype=np.float64)   # fast toward robot with upward arc

FLOOR_Z = 0.05          # ball below this = dead
SERVE_COOLDOWN = 3.0     # seconds between serves
MAX_RALLY = 200          # safety limit


@dataclass
class MatchState:
    """Shared match state — read by dashboard/commentator."""
    score_robot: int = 0
    score_opponent: int = 0
    rally_count: int = 0
    best_rally: int = 0
    total_serves: int = 0
    last_event: str = ""
    serving: bool = False
    swing_count: int = 0
    match_time: float = 0.0


class MatchManager:
    """Monitors ball state and manages serve/score logic."""

    def __init__(self) -> None:
        self.state = MatchState()
        self._ball_body_id: int = -1
        self._ball_jnt_id: int = -1
        self._serve_timer: float = 0.0
        self._ball_on_table: bool = False
        self._ball_crossed_net: bool = False
        self._rally_active: bool = False
        self._initialized = False

    def init(self, model: mujoco.MjModel) -> None:
        self._ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._ball_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self._initialized = True
        logger.info("MatchManager initialized")

    def serve(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Launch ball from far end toward robot."""
        if self._ball_jnt_id < 0:
            return

        qpos_addr = model.jnt_qposadr[self._ball_jnt_id]
        qvel_addr = model.jnt_dofadr[self._ball_jnt_id]

        # Position + quaternion
        data.qpos[qpos_addr:qpos_addr + 3] = SERVE_POS
        data.qpos[qpos_addr + 3:qpos_addr + 7] = [1, 0, 0, 0]

        # Velocity with slight randomness
        vel = SERVE_VEL.copy()
        vel[1] += np.random.uniform(-0.3, 0.3)
        vel[2] += np.random.uniform(-0.2, 0.2)
        data.qvel[qvel_addr:qvel_addr + 3] = vel
        data.qvel[qvel_addr + 3:qvel_addr + 6] = [0, 0, 0]

        # Do NOT call mj_forward — it zeroes the ball velocity through constraints

        self.state.total_serves += 1
        self.state.serving = True
        self.state.rally_count = 0
        self._ball_crossed_net = False
        self._rally_active = True
        self._serve_timer = 0.0
        self.state.last_event = f"SERVE #{self.state.total_serves}"
        logger.info(f"Serve #{self.state.total_serves}")

    def tick(self, model: mujoco.MjModel, data: mujoco.MjData, dt: float) -> None:
        """Called each control step from the sim loop."""
        if not self._initialized:
            self.init(model)

        self._serve_timer += dt
        self.state.match_time += dt

        if self._ball_body_id < 0:
            return

        ball_pos = data.xpos[self._ball_body_id].copy()
        ball_vel = data.cvel[self._ball_body_id][3:6].copy()  # linear velocity

        # ── Manual bounce on table (excluded from MuJoCo contact) ──
        table_surface = TABLE_Z + 0.04  # table top + half-thickness + ball radius
        ball_on_table_xy = (
            NEAR_EDGE - 0.1 < ball_pos[0] < FAR_EDGE + 0.1
            and abs(ball_pos[1]) < TABLE_Y_HALF + 0.1
        )
        qvel_addr = model.jnt_dofadr[self._ball_jnt_id]
        qpos_addr = model.jnt_qposadr[self._ball_jnt_id]
        ball_vz = data.qvel[qvel_addr + 2]

        if ball_on_table_xy and ball_pos[2] <= table_surface and ball_vz < -0.3:
            # Bounce: reverse vertical velocity with COR=0.85
            data.qvel[qvel_addr + 2] = abs(ball_vz) * 0.85
            data.qpos[qpos_addr + 2] = table_surface + 0.01

        # ── Ball out of play ──
        if ball_pos[2] < FLOOR_Z or abs(ball_pos[1]) > 3.0 or ball_pos[0] < 6.0 or ball_pos[0] > 14.0:
            if self._rally_active:
                self._end_rally(ball_pos)
            # Auto-serve after cooldown
            if self._serve_timer > SERVE_COOLDOWN:
                self.serve(model, data)
            return

        # ── Track net crossing ──
        if self._rally_active and not self._ball_crossed_net:
            if ball_pos[0] < NET_X and ball_vel[0] < 0:
                self._ball_crossed_net = True
                self.state.rally_count += 1
                self.state.last_event = f"RALLY {self.state.rally_count}"

        # ── Detect return (ball going back over net toward far side) ──
        if self._rally_active and self._ball_crossed_net:
            if ball_pos[0] > NET_X and ball_vel[0] > 0:
                self._ball_crossed_net = False
                self.state.rally_count += 1
                self.state.last_event = f"RETURN! Rally {self.state.rally_count}"

        self.state.serving = False

    def _end_rally(self, ball_pos: NDArray[Any]) -> None:
        """Score a point based on where the ball went out."""
        self._rally_active = False

        if self.state.rally_count > self.state.best_rally:
            self.state.best_rally = self.state.rally_count

        # Simple scoring: if ball went out on robot's side, opponent scores
        if ball_pos[0] < NET_X:
            self.state.score_opponent += 1
            self.state.last_event = f"POINT OPPONENT ({self.state.score_robot}-{self.state.score_opponent})"
        else:
            self.state.score_robot += 1
            self.state.last_event = f"POINT ROBOT ({self.state.score_robot}-{self.state.score_opponent})"

        logger.info(f"Point: {self.state.last_event} | Rally: {self.state.rally_count}")
