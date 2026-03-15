"""Gymnasium environments for DimOS MuJoCo simulations.

Importing this module registers the environments with Gymnasium so they can
be created via ``gymnasium.make("DimOS-DroneHover-v0")`` etc.
"""

from gymnasium.envs.registration import register

register(
    id="DimOS-DroneHover-v0",
    entry_point="dimos.simulation.gym.envs.drone_hover:DroneHoverEnv",
)
register(
    id="DimOS-DroneVelocity-v0",
    entry_point="dimos.simulation.gym.envs.drone_velocity:DroneVelocityEnv",
)
register(
    id="DimOS-Go2Locomotion-v0",
    entry_point="dimos.simulation.gym.envs.go2_locomotion:Go2LocomotionEnv",
)
register(
    id="DimOS-G1Standing-v0",
    entry_point="dimos.simulation.gym.envs.g1_standing:G1StandingEnv",
)
