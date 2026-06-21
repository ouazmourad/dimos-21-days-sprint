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

"""Go2 "narrate what it sees" demo.

The robot continuously describes its camera view out loud using a vision model
plus third-party TTS — a lightweight, autonomous fallback demo that needs only
the camera (no navigation, no human input). Run with:

    dimos run unitree-go2-narrate --robot-ip <go2-ip>
"""

from dimos.agents.scene_narrator import SceneNarrator
from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import _with_vis
from dimos.robot.unitree.go2.connection import GO2Connection

# Passive camera-only connection (camera_only=True): the narrator only watches
# and speaks; DimOS never stands or drives the robot, so an external controller
# (e.g. the Unitree phone app) stays the sole driver. Mirrors unitree_go2_basic
# (vis + GO2Connection) but with the connection in camera-only mode.
unitree_go2_narrate = autoconnect(
    _with_vis,
    GO2Connection.blueprint(camera_only=True),
    SceneNarrator.blueprint(),
).global_config(n_workers=4, robot_model="unitree_go2")

__all__ = ["unitree_go2_narrate"]
