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

"""Lean R1 agentic stack: real-robot control + LLM chat, no perception/viz.

For the R1 driven from the laptop, the camera/lidar/odom sensor streams don't
reach DimOS (DDS XTypes type mismatch; no Go2-style WebRTC), so the perception,
mapping and Rerun modules in ``unitree_r1_agentic`` either idle or error (e.g.
the laptop-webcam open failure, the Rerun viewer version mismatch). This
blueprint drops all of them and keeps only what works on the R1 today: the
connection ("pc1" backend), the MCP agent, chat/voice input, and the loco/arm
skills. Chat UI at http://localhost:5555.
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.robot.unitree.r1.blueprints.agentic._agentic_skills import _agentic_skills
from dimos.robot.unitree.r1.connection import R1Connection

unitree_r1_control = (
    autoconnect(
        R1Connection.blueprint(),
        _agentic_skills,
    )
    .global_config(n_workers=4, robot_model="unitree_r1")
    .transports(
        {
            ("cmd_vel", Twist): LCMTransport("/cmd_vel", Twist),
        }
    )
)

__all__ = ["unitree_r1_control"]
