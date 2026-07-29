#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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

"""Lightweight agentic Go2 blueprint with hand-gesture control.

Same low-RAM base as `unitree-go2-agentic-lite` (unitree_go2_basic: camera/observe,
odom, sport commands — no CUDA voxel/nav/SLAM), plus the gesture container that
turns MediaPipe hand gestures into motion ("go where I point", emotes).

The person controls the robot directly with their hands via `follow_gestures`; the
agent can also just walk/dance/speak as usual. MediaPipe runs on CPU, so this stays
within the ~3 GB budget of the lite base.

Requirements:
  - mediapipe + opencv in the dimos venv
  - the go2_gesture_recognition checkout for the .task models (auto-discovered at
    the repo root, or point $GO2_GESTURE_MODEL_DIR at it).

Run:
    dimos --rerun-open none run unitree-go2-gesture-lite -o rerunbridgemodule.memory_limit=1GB
"""

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.skills.speak_skill import SpeakSkill
from dimos.agents.web_human_input import WebInput
from dimos.core.coordination.blueprints import autoconnect
from dimos.perception.perceive_loop_skill import PerceiveLoopSkill
from dimos.robot.unitree.gesture.gesture_skill_container import GestureSkillContainer
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import unitree_go2_basic
from dimos.robot.unitree.unitree_skill_container import UnitreeSkillContainer

unitree_go2_gesture_lite = autoconnect(
    unitree_go2_basic,
    McpServer.blueprint(),
    McpClient.blueprint(),
    UnitreeSkillContainer.blueprint(),
    SpeakSkill.blueprint(),
    WebInput.blueprint(),
    PerceiveLoopSkill.blueprint(),
    GestureSkillContainer.blueprint(),
).global_config(n_workers=4, robot_model="unitree_go2")

__all__ = ["unitree_go2_gesture_lite"]
