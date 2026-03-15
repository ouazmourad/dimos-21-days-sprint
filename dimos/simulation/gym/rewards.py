"""Composable reward components for DimOS Gymnasium environments."""

from __future__ import annotations

import numpy as np


def reward_alive(height: float, min_height: float) -> float:
    """1.0 while the robot is above *min_height*, else 0.0."""
    return 1.0 if height > min_height else 0.0


def reward_velocity_tracking(
    actual: np.ndarray, target: np.ndarray, scale: float = 1.0,
) -> float:
    """Negative squared error between actual and target velocity."""
    return -scale * float(np.sum((actual - target) ** 2))


def reward_altitude_tracking(
    actual_alt: float, target_alt: float, scale: float = 1.0,
) -> float:
    """Negative squared error between actual and target altitude."""
    return -scale * (actual_alt - target_alt) ** 2


def reward_upright(up_vector: np.ndarray) -> float:
    """Reward for keeping the body z-axis aligned with world z.

    *up_vector* is the third row of the body rotation matrix (``R[2, :]``),
    which equals the world-up direction expressed in the body frame.
    Returns a value in [-1, 1], with 1.0 meaning perfectly upright.
    """
    return float(up_vector[2])


def reward_energy_penalty(ctrl: np.ndarray, scale: float = 0.01) -> float:
    """Negative sum of squared control signals."""
    return -scale * float(np.sum(ctrl ** 2))


def reward_joint_limit_penalty(
    model, data, *, margin: float = 0.1, scale: float = 0.1,
) -> float:
    """Penalty for joints whose position is within *margin* of their limit."""
    penalty = 0.0
    for i in range(model.njnt):
        if not model.jnt_limited[i]:
            continue
        lo, hi = model.jnt_range[i]
        q = data.qpos[model.jnt_qposadr[i]]
        if q < lo + margin:
            penalty += (lo + margin - q) ** 2
        elif q > hi - margin:
            penalty += (q - (hi - margin)) ** 2
    return -scale * penalty


def reward_smoothness(
    action: np.ndarray, last_action: np.ndarray, scale: float = 0.01,
) -> float:
    """Penalty for large action changes between timesteps."""
    return -scale * float(np.sum((action - last_action) ** 2))
