"""Evaluate a trained policy (SB3 or ONNX) in a MuJoCo viewer.

Usage
-----
    # Evaluate with interactive MuJoCo viewer (default)
    python -m dimos.simulation.gym.evaluate --env DroneHover --model models/DroneHover.zip

    # Evaluate ONNX model
    python -m dimos.simulation.gym.evaluate --env DroneHover --onnx data/mujoco_sim/drone_rl_policy.onnx

    # Headless (no viewer, just print stats)
    python -m dimos.simulation.gym.evaluate --env DroneHover --model models/DroneHover.zip --headless
"""

from __future__ import annotations

import argparse
import os
import time

_ENV_MAP = {
    "DroneHover": "DimOS-DroneHover-v0",
    "DroneVelocity": "DimOS-DroneVelocity-v0",
    "Go2Locomotion": "DimOS-Go2Locomotion-v0",
    "G1Standing": "DimOS-G1Standing-v0",
}

# Environment-specific model loading for the interactive viewer
_VIEWER_LOADERS = {
    "DroneHover": ("drone_hover", "DroneHoverEnv"),
    "DroneVelocity": ("drone_velocity", "DroneVelocityEnv"),
    "Go2Locomotion": ("go2_locomotion", "Go2LocomotionEnv"),
    "G1Standing": ("g1_standing", "G1StandingEnv"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained RL policy")
    p.add_argument("--env", required=True, choices=list(_ENV_MAP.keys()))
    p.add_argument("--model", default=None, help="Path to SB3 model .zip")
    p.add_argument("--onnx", default=None, help="Path to ONNX policy file")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    p.add_argument("--headless", action="store_true", help="No viewer, just print stats")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _make_onnx_predictor(onnx_path: str):
    """Return a callable: obs (np.ndarray) -> action (np.ndarray)."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())

    def predict(obs: np.ndarray) -> np.ndarray:
        result = session.run(
            ["continuous_actions"],
            {"obs": obs.reshape(1, -1).astype(np.float32)},
        )
        return result[0][0]

    return predict


def _run_headless(args, predict) -> None:
    """Run evaluation without a viewer, print stats."""
    os.environ.setdefault("MUJOCO_GL", "egl")

    import gymnasium
    import numpy as np

    import dimos.simulation.gym  # noqa: F401

    gym_id = _ENV_MAP[args.env]
    env = gymnasium.make(gym_id)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        steps = 0

        while True:
            action = predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

    env.close()
    print(f"\nResults over {args.episodes} episodes:")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"  Mean length: {np.mean(episode_lengths):.0f}")


def _run_viewer(args, predict) -> None:
    """Run evaluation with the interactive MuJoCo viewer."""
    import importlib

    import mujoco
    import mujoco.viewer
    import numpy as np

    # Instantiate the env to get model/data (but we drive physics ourselves)
    mod_name, cls_name = _VIEWER_LOADERS[args.env]
    mod = importlib.import_module(f"dimos.simulation.gym.envs.{mod_name}")
    env_cls = getattr(mod, cls_name)
    env = env_cls()

    model = env.model
    data = env.data

    obs, _ = env.reset(seed=args.seed)
    episode = 0
    total_reward = 0.0
    physics_steps = 0

    def controller(m, d):
        """Called by the viewer at each physics step — we only act every
        ``env._n_substeps`` steps to match control frequency."""
        nonlocal obs, episode, total_reward, physics_steps

        physics_steps += 1

        # Only run the policy at control frequency
        if physics_steps % env._n_substeps != 0:
            return

        action = predict(obs)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        env._apply_action(action)
        env._step_count += 1

        obs = env._get_obs()
        reward = env._get_reward(action)
        total_reward += reward
        terminated = env._is_terminated()
        truncated = env._step_count >= env._episode_length
        env._last_action = action.copy()

        if terminated or truncated:
            print(f"Episode {episode + 1}: reward={total_reward:.2f}, steps={env._step_count}")
            episode += 1
            if episode >= args.episodes:
                return
            obs, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            physics_steps = 0

    mujoco.set_mjcb_control(controller)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and episode < args.episodes:
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - step_start
            dt = model.opt.timestep - elapsed
            if dt > 0:
                time.sleep(dt)

    mujoco.set_mjcb_control(None)
    env.close()


def main() -> None:
    args = parse_args()

    if args.model is None and args.onnx is None:
        raise SystemExit("Provide either --model (SB3 .zip) or --onnx path")

    # Build predictor
    if args.model is not None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        from stable_baselines3 import PPO
        import numpy as np

        sb3_model = PPO.load(args.model, device="cpu")

        def predict(obs: np.ndarray) -> np.ndarray:
            action, _ = sb3_model.predict(obs, deterministic=True)
            return action
    else:
        predict = _make_onnx_predictor(args.onnx)  # type: ignore[arg-type]

    if args.headless:
        _run_headless(args, predict)
    else:
        _run_viewer(args, predict)


if __name__ == "__main__":
    main()
