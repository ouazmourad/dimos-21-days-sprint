"""CLI training script using stable-baselines3 PPO.

Usage
-----
    python -m dimos.simulation.gym.train --env DroneHover --timesteps 1_000_000
    python -m dimos.simulation.gym.train --env Go2Locomotion --timesteps 5_000_000
    python -m dimos.simulation.gym.train --env DroneVelocity --timesteps 2_000_000 \\
        --learning-rate 3e-4 --batch-size 64 --n-steps 2048
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Ensure offscreen rendering (no display needed for training)
os.environ.setdefault("MUJOCO_GL", "egl")

_ENV_MAP = {
    "DroneHover": "DimOS-DroneHover-v0",
    "DroneVelocity": "DimOS-DroneVelocity-v0",
    "Go2Locomotion": "DimOS-Go2Locomotion-v0",
    "G1Standing": "DimOS-G1Standing-v0",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RL policies for DimOS robots")
    p.add_argument(
        "--env",
        required=True,
        choices=list(_ENV_MAP.keys()),
        help="Environment name",
    )
    p.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
    p.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    p.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate")
    p.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size")
    p.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout per env")
    p.add_argument("--device", default="auto", help="Torch device (auto/cuda/cpu)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--output-dir", default="models", help="Directory for saved models")
    p.add_argument("--log-dir", default="runs", help="TensorBoard log directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy imports so --help is fast and so we can set env vars first
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_util import make_vec_env

    # Register DimOS gym environments
    import dimos.simulation.gym  # noqa: F401

    gym_id = _ENV_MAP[args.env]

    print(f"Creating {args.n_envs} parallel environments: {gym_id}")
    vec_env = make_vec_env(gym_id, n_envs=args.n_envs, seed=args.seed)

    # Single eval environment for periodic evaluation
    eval_env = make_vec_env(gym_id, n_envs=1, seed=args.seed + 1000)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) / args.env
    log_dir.mkdir(parents=True, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        device=args.device,
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=args.seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / f"{args.env}_best"),
        eval_freq=max(args.timesteps // 20, 5000),
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"Training {args.env} for {args.timesteps:,} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
    )

    save_path = str(output_dir / args.env)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    print(f"TensorBoard logs in {log_dir}/")
    print(f"Run: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
