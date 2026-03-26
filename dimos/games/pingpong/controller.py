"""Ping pong controller — uses robomotion-trained ONNX policy.

Loads the trained policy, builds observations from MuJoCo state + reference
trajectory, and applies actions via PD control with robomotion's tuned gains.
Falls back to standing pose if no policy is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# ── Joint layout (29 DOF) ──
# 0-5: left leg, 6-11: right leg, 12-14: waist, 15-21: left arm, 22-28: right arm
RIGHT_ARM_SLICE = slice(22, 29)

# Excluded from action (right wrist — holds racket rigid)
EXCLUDED_JOINTS = [26, 27, 28]  # right_wrist_roll, pitch, yaw
ACTIVE_JOINTS = [i for i in range(29) if i not in EXCLUDED_JOINTS]  # 26 active

# PD gains from robomotion tennis_config.py
KPs = np.float32([
    100, 100, 100, 200, 80, 20,   # left leg
    100, 100, 100, 200, 80, 20,   # right leg
    300, 300, 300,                  # waist
    90, 60, 20, 60, 20, 20, 20,   # left arm
    90, 60, 20, 60, 20, 20, 20,   # right arm
])
KDs = np.float32([
    2, 2, 2, 4, 2, 1,
    2, 2, 2, 4, 2, 1,
    10, 10, 10,
    2, 2, 1, 1, 1, 1, 1,
    2, 2, 1, 1, 1, 1, 1,
])
TORQUE_LIMIT = np.float32([
    88., 139., 88., 139., 50., 50.,
    88., 139., 88., 139., 50., 50.,
    88., 50., 50.,
    25., 25., 25., 25., 25., 5., 5.,
    25., 25., 25., 25., 25., 5., 5.,
])
DEFAULT_QPOS = np.float32([
    -0.1, 0, 0, 0.3, -0.2, 0,
    -0.1, 0, 0, 0.3, -0.2, 0,
    0, 0, 0,
    0.2, 0.3, 0, 1.28, 0, 0, 0,
    0.2, -0.3, 0, 1.28, 0, 0, 0,
])

# Default ONNX policy path
POLICY_PATH = Path("/home/mourad/Desktop/dimos-21days-sprint/robomotion/storage/logs/track")


def _find_latest_policy() -> Path | None:
    """Find the latest trained ONNX policy."""
    if not POLICY_PATH.exists():
        return None
    for exp_dir in sorted(POLICY_PATH.glob("*"), reverse=True):
        onnx = list(exp_dir.glob("checkpoints/*/policy.onnx"))
        if onnx:
            return sorted(onnx)[-1]
    return None


class PingPongController:
    """ONNX-policy-driven table tennis controller with PD control."""

    def __init__(
        self,
        default_angles: NDArray[Any],
        ctrl_dt: float = 0.02,
        policy_path: str | None = None,
    ) -> None:
        self._default = default_angles.copy()
        self._target = default_angles.copy()
        self._last_motor_targets = default_angles.copy()
        self._ctrl_dt = ctrl_dt

        # Load ONNX policy
        self._policy = None
        pp = Path(policy_path) if policy_path else _find_latest_policy()
        if pp and pp.exists():
            import onnxruntime as ort
            self._policy = ort.InferenceSession(str(pp), providers=ort.get_available_providers())
            logger.info(f"Loaded ONNX policy: {pp}")
        else:
            logger.warning("No ONNX policy found — using standing pose only")

        # Reference trajectory (looped from synthetic data)
        self._ref_qpos: NDArray[Any] | None = None
        self._ref_qvel: NDArray[Any] | None = None
        self._ref_frame = 0
        self._load_reference_trajectory()

        # State
        self._base_qpos: NDArray[Any] | None = None
        self._counter = 0
        self._n_substeps = 10
        self._ball_body_id = -1
        self._pelvis_imu_site_id = -1
        self._swing_count = 0

    def _load_reference_trajectory(self) -> None:
        """Load the first synthetic trajectory for reference."""
        traj_dir = Path("/home/mourad/Desktop/dimos-21days-sprint/robomotion/storage/data/mocap/Tennis/p1")
        trajs = sorted(traj_dir.glob("*.npz"))
        if trajs:
            data = np.load(trajs[0])
            self._ref_qpos = data["qpos"].astype(np.float32)
            self._ref_qvel = data["qvel"].astype(np.float32)
            logger.info(f"Loaded ref trajectory: {trajs[0].name} ({self._ref_qpos.shape[0]} frames)")

    def _get_obs(self, model: mujoco.MjModel, data: mujoco.MjData) -> NDArray[Any]:
        """Build the 151-dim observation matching training config."""
        joint_pos = data.qpos[7:7 + 29]
        joint_vel = data.qvel[6:6 + 29]

        # Gravity vector in pelvis frame
        if self._pelvis_imu_site_id < 0:
            self._pelvis_imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_pelvis")
        imu_xmat = data.site_xmat[self._pelvis_imu_site_id].reshape(3, 3)
        gvec_pelvis = imu_xmat.T @ np.array([0, 0, -1])

        # Gyroscope
        gyro_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro_pelvis")
        gyro_adr = model.sensor_adr[gyro_sensor_id]
        gyro_pelvis = data.sensordata[gyro_adr:gyro_adr + 3]

        # Reference trajectory differences
        if self._ref_qpos is not None:
            n_frames = self._ref_qpos.shape[0]
            ref_idx = self._ref_frame % n_frames
            ref_joint_pos = self._ref_qpos[ref_idx, 7:]
            ref_joint_vel = self._ref_qvel[ref_idx, 6:]
            dif_joint_pos = ref_joint_pos - joint_pos
            dif_joint_vel = (ref_joint_vel - joint_vel) * 0.1  # scaled
        else:
            dif_joint_pos = np.zeros(29, dtype=np.float32)
            dif_joint_vel = np.zeros(29, dtype=np.float32)

        # Obs: [dif_joint_pos(29), dif_joint_vel(29), gvec_pelvis(3),
        #        gyro_pelvis(3), joint_pos(29), joint_vel(29), last_motor_targets(29)]
        obs = np.concatenate([
            dif_joint_pos,
            dif_joint_vel,
            gvec_pelvis,
            gyro_pelvis * 0.1,  # scaled
            (joint_pos - DEFAULT_QPOS),
            joint_vel * 0.1,
            self._last_motor_targets,
        ]).astype(np.float32)

        return obs

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """MuJoCo control callback."""
        # Lock base
        if self._base_qpos is None:
            self._base_qpos = data.qpos[0:7].copy()
        data.qpos[0:7] = self._base_qpos
        data.qvel[0:6] = 0

        # Sanitize NaN
        if not np.all(np.isfinite(data.qvel)):
            data.qvel[~np.isfinite(data.qvel)] = 0.0

        self._counter += 1
        if self._counter % self._n_substeps != 0:
            return

        n_act = min(len(self._target), len(KPs))

        if self._policy is not None:
            # Run ONNX policy
            obs = self._get_obs(model, data)
            action = self._policy.run(
                ["continuous_actions"],
                {"obs": obs.reshape(1, -1)}
            )[0][0]

            # Map 26-dim action to 29-dim targets
            # Reference trajectory provides the base target
            if self._ref_qpos is not None:
                n_frames = self._ref_qpos.shape[0]
                ref_idx = self._ref_frame % n_frames
                ref_joint = self._ref_qpos[ref_idx, 7:]
                motor_targets = ref_joint.copy()
            else:
                motor_targets = DEFAULT_QPOS.copy()

            # Apply action deviations to active joints
            for i, j in enumerate(ACTIVE_JOINTS):
                if i < len(action):
                    motor_targets[j] = motor_targets[j] + action[i] * 1.0  # action_scale=1.0

            self._target[:n_act] = motor_targets[:n_act]
            self._last_motor_targets = motor_targets.copy()
            self._ref_frame += 1
            self._swing_count = self._ref_frame  # track progress
        else:
            # Fallback: just stand
            self._target[:] = DEFAULT_QPOS

        # PD control with robomotion gains
        joint_pos = data.qpos[7:7 + n_act]
        joint_vel = data.qvel[6:6 + n_act]
        torques = KPs[:n_act] * (self._target[:n_act] - joint_pos) + KDs[:n_act] * (-joint_vel)
        data.ctrl[:n_act] = np.clip(torques, -TORQUE_LIMIT[:n_act], TORQUE_LIMIT[:n_act])

    @property
    def swing_count(self) -> int:
        return self._swing_count

    @property
    def swing_phase(self) -> str:
        return "policy" if self._policy else "standing"
