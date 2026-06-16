#!/usr/bin/env python3
"""Demonstrate DimOS controlling the real Unitree R1.

Drives the physical robot through the DimOS R1 connection's "pc1" backend
(``R1LocoProxy`` — the exact code path ``R1Connection(backend="pc1")`` uses),
sending DimOS ``Twist`` messages that get proxied to PC1's native
``r1_loco_client``. Run from the laptop with the robot link up.

    python tools/r1_dimos_demo.py                # connect + read live FSM state (NO motion)
    python tools/r1_dimos_demo.py turn 0.3       # gentle in-place yaw (~1.5s)  -- MOTION
    python tools/r1_dimos_demo.py move 0.15 0    # small forward nudge (~1.5s)  -- MOTION
    python tools/r1_dimos_demo.py damp           # relax to compliant hold

Safety: motion commands move a real humanoid. Keep ~2 m clear, e-stop in hand.
"""

import sys
import time

from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.robot.unitree.r1.effectors.high_level.loco_proxy import R1LocoProxy

MOVE_DURATION = 2.5


def main() -> int:
    args = sys.argv[1:]

    # This is the DimOS R1 connection's pc1 backend.
    proxy = R1LocoProxy()
    proxy.start()  # SSH to PC1 + log current FSM state (DimOS -> robot, read-only)
    print(f"[demo] DimOS read live robot state through R1 backend: FSM = {proxy.get_state()}")

    try:
        if not args:
            print("[demo] read-only demo done (no motion). Pass move/turn/damp to command it.")
            return 0

        cmd = args[0]
        if cmd in ("turn", "move") and proxy.get_state() != "811":
            # Velocity is ignored unless the robot is in Start/locomotion mode.
            print("[demo] entering Start mode (FSM 811) so the robot accepts velocity ...")
            proxy.start_locomotion()
            time.sleep(3.0)
            print(f"[demo] FSM now {proxy.get_state()}")

        if cmd == "turn":
            yaw = float(args[1]) if len(args) > 1 else 0.3
            twist = Twist(linear=Vector3(0, 0, 0), angular=Vector3(0, 0, yaw))
            print(f"[demo] DimOS Twist -> robot: yaw={yaw} rad/s for {MOVE_DURATION}s")
            proxy.move(twist, duration=MOVE_DURATION)
            time.sleep(MOVE_DURATION + 0.3)
        elif cmd == "move":
            vx = float(args[1]) if len(args) > 1 else 0.15
            vy = float(args[2]) if len(args) > 2 else 0.0
            twist = Twist(linear=Vector3(vx, vy, 0), angular=Vector3(0, 0, 0))
            print(f"[demo] DimOS Twist -> robot: vx={vx} vy={vy} m/s for {MOVE_DURATION}s")
            proxy.move(twist, duration=MOVE_DURATION)
            time.sleep(MOVE_DURATION + 0.3)
        elif cmd == "damp":
            print("[demo] DimOS -> robot: damp (compliant hold)")
            proxy.damp()
        else:
            print(f"[demo] unknown command {cmd!r}")
            return 2

        print(f"[demo] FSM after command: {proxy.get_state()}")
        return 0
    finally:
        proxy.stop()  # sends stop_move (safety) and closes the PC1 session


if __name__ == "__main__":
    sys.exit(main())
