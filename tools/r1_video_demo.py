#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DimOS <-> Unitree R1 — live control demo, paced for a screen recording.

Connects through the DimOS R1 connection ("pc1" backend), reads the robot's live
state, then commands a gentle in-place turn — the real R1 moves on a DimOS
command. The pauses are sized for an on-camera segment (~25 s).

SAFETY: this drives a real humanoid. Robot on the floor, ~2 m clear all around,
e-stop / remote in hand. Confirm the ethernet link first
(``cat /sys/class/net/enp2s0/carrier`` should print 1).

    python tools/r1_video_demo.py            # gentle in-place turn (default)
    python tools/r1_video_demo.py forward    # small forward step instead
"""

import sys
import time

from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.robot.unitree.r1.effectors.high_level.loco_proxy import R1LocoProxy


def _banner(msg: str) -> None:
    line = "=" * 62
    print(f"\n{line}\n  {msg}\n{line}", flush=True)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "turn"
    _banner("DimOS  <->  Unitree R1   .   live control on real hardware")
    time.sleep(1.5)

    # R1LocoProxy is exactly what R1Connection(backend="pc1") drives the robot with.
    proxy = R1LocoProxy()
    print("\n[1/3]  Connecting to the R1 through DimOS ...", flush=True)
    try:
        proxy.start()  # logs 'R1 loco proxy connected' + the current FSM state
    except Exception as e:
        print(f"\n  x  Could not reach the robot: {e}", flush=True)
        print("     Check the link: cat /sys/class/net/enp2s0/carrier  (want 1)", flush=True)
        return 1
    time.sleep(1.5)

    print("\n[2/3]  DimOS reading the robot's live state ...", flush=True)
    print(f"        FSM state = {proxy.get_state()}   (standing, ready)", flush=True)
    time.sleep(2.5)

    try:
        if mode == "forward":
            print("\n[3/3]  DimOS command -> move forward.  Watch the robot ...", flush=True)
            twist = Twist(linear=Vector3(0.15, 0.0, 0.0), angular=Vector3(0, 0, 0))
        else:
            print("\n[3/3]  DimOS command -> turn in place.  Watch the robot ...", flush=True)
            twist = Twist(linear=Vector3(0, 0, 0), angular=Vector3(0, 0, 0.5))
        time.sleep(1.0)
        proxy.move(twist, duration=2.5)  # enters Start/locomotion mode, then moves
        time.sleep(3.0)
        print("\n  +  The Unitree R1 moved on a DimOS command.", flush=True)
    finally:
        proxy.stop()

    _banner("Demo complete  .  DimOS is driving the real R1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
