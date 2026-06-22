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

"""Go2 real-robot keyboard teleop WITH the live camera on rerun.

Combines the proven WebRTC keyboard-teleop (drive the real dog with the
keyboard, via the control coordinator) with the rerun camera/3D view — on a
single WebRTC connection (the Go2 allows only one client). For the REAL robot:

    export ROBOT_IP=<go2-ip>
    export GO2_AES_KEY=<key>          # data2=3 firmware (Go2 >= 1.1.15)
    # close the Unitree app first (one WebRTC client at a time)
    dimos --rerun-open web run unitree-go2-keyboard-view

The camera the dog actually sees shows in the rerun web viewer; WASD/arrows
drive it. Stop = release keys / Ctrl-C (obstacle avoidance also stays on).
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import _with_vis
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_webrtc_keyboard_teleop import (
    unitree_go2_webrtc_keyboard_teleop,
)

# _with_vis contributes only the rerun/websocket viz modules (no GO2Connection of
# its own), so this stays a single connection: the coordinator's GO2Connection
# publishes color_image, and the vis subscribes + renders it.
unitree_go2_keyboard_view = autoconnect(
    unitree_go2_webrtc_keyboard_teleop,
    _with_vis,
)

__all__ = ["unitree_go2_keyboard_view"]
