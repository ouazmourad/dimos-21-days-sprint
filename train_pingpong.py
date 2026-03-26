"""Custom training — forces debug-sized batch to fit 4GB GPU, but runs longer."""
import os, sys
sys.path.insert(0, "/home/mourad/Desktop/dimos-21days-sprint/robomotion")
os.environ["GLI_PATH"] = "/home/mourad/Desktop/dimos-21days-sprint/robomotion"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_PROJECT"] = "robomotion"
os.environ["WANDB_ENTITY"] = "local"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.chdir("/home/mourad/Desktop/dimos-21days-sprint/robomotion")

# Use debug name to trigger small batch, but override timesteps
from robomotion.algorithms.runners.tennis_ppo import Args, train
args = Args(
    task="G1TrackingTennis",
    exp_name="debug_long",   # "debug" triggers small batch (16 envs)
    num_timesteps=10_000_000,  # but run 10x longer
)
train(args)
