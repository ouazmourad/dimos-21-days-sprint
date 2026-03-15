"""Export a trained SB3 PPO model to ONNX for deployment via OnnxController.

Usage
-----
    python -m dimos.simulation.gym.export_onnx \\
        --model models/DroneHover.zip \\
        --output data/mujoco_sim/drone_rl_policy.onnx
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn


class _OnnxablePolicy(nn.Module):
    """Wraps SB3 policy components into a single module for ONNX export.

    Forward: obs -> deterministic action (no sampling, no value head).
    """

    def __init__(self, mlp_extractor: nn.Module, action_net: nn.Module) -> None:
        super().__init__()
        self.mlp_extractor = mlp_extractor
        self.action_net = action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent_pi, _ = self.mlp_extractor(obs)
        return self.action_net(latent_pi)


def export(model_path: str, output_path: str) -> None:
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    policy = model.policy

    onnxable = _OnnxablePolicy(policy.mlp_extractor, policy.action_net)
    onnxable.eval()

    obs_size = model.observation_space.shape[0]  # type: ignore[index]
    dummy_input = torch.randn(1, obs_size)

    torch.onnx.export(
        onnxable,
        dummy_input,
        output_path,
        input_names=["obs"],
        output_names=["continuous_actions"],
        opset_version=18,
        dynamic_axes={
            "obs": {0: "batch_size"},
            "continuous_actions": {0: "batch_size"},
        },
    )
    print(f"Exported ONNX policy to {output_path}")
    print(f"  Input:  obs ({obs_size},)")
    print(f"  Output: continuous_actions ({model.action_space.shape[0]},)")  # type: ignore[index]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export SB3 model to ONNX")
    p.add_argument("--model", required=True, help="Path to SB3 model .zip")
    p.add_argument("--output", required=True, help="Output ONNX file path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export(args.model, args.output)


if __name__ == "__main__":
    main()
