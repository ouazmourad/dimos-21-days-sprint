"""Fleet simulation demo — 2 G1 humanoids in the same MuJoCo office.

Run:
    CI=1 .venv/bin/python examples/fleet_sim_demo.py

What it does:
    - Spawns 2 G1 robots (alpha at (-1,1), bravo at (1,-1))
    - Each has its own camera, odom, lidar streams (namespaced)
    - Prints odom from both robots to show they're independent
    - Opens MuJoCo viewer so you can see both robots
"""

import os
os.environ["CI"] = "1"

from dimos.core.fleet import fleet, RobotConfig
from dimos.core.global_config import global_config
from dimos.robot.unitree.g1.sim import G1SimConnection


def main():
    # Configure simulation
    global_config.simulation = True
    global_config.robot_model = "unitree_g1"
    global_config.n_workers = 2
    global_config.mujoco_start_pos = "-1.0, 1.0"

    # Build fleet — 2 G1 humanoids, each with independent streams
    game = fleet(
        robots=[
            RobotConfig("alpha", G1SimConnection),
            RobotConfig("bravo", G1SimConnection),
        ],
    )

    print("\n=== Fleet Simulation Demo ===")
    print(f"Modules: {[a.module.__name__ for a in game._active_blueprints]}")
    print(f"Remappings ({len(game.remapping_map)}):")
    for (cls, stream), target in sorted(game.remapping_map.items(), key=lambda x: x[1]):
        print(f"  {cls.__name__}.{stream} → {target}")

    print("\nTopics that will be created:")
    print("  alpha/odom, alpha/color_image, alpha/cmd_vel, alpha/lidar, ...")
    print("  bravo/odom, bravo/color_image, bravo/cmd_vel, bravo/lidar, ...")
    print("\nEach robot has fully isolated sensor streams.")
    print("\nNote: Running 2 MuJoCo subprocesses requires ~4GB RAM each.\n")

    coordinator = game.build()
    coordinator.loop()


if __name__ == "__main__":
    main()
