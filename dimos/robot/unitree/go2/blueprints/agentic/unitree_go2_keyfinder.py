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

"""Go2 "find my lost keys" blueprint — autonomous floor patrol + key detection.

Deliberately minimal AND headless: no CUDA voxel mapper, no SLAM/spatial memory, no
nav planners (those OOM a 15 GB laptop and the Air can't use the costmap anyway) and
no visualisation stack (that was ~5 GB of RAM by itself). Just camera + control + the
agent loop + speech + the key-finder behaviour.

Run:
    dimos --rerun-open none run unitree-go2-keyfinder

Then drive it with `dimos humancli` ("find my keys"), or call the skills directly:
    dimos mcp call look_for_keys          # scan current view (never moves)
    dimos mcp call inspect_below          # StandDown close-check (no patrol)
    dimos mcp call find_my_keys -a minutes=5
    dimos mcp call search_status
    dimos mcp call stop_searching

Needs an open-vocabulary detector for "keys" (COCO YOLO has no keys class):
set `detection_model=openai` in .env so gpt-4o's box detector is used.
"""

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.skills.speak_skill import SpeakSkill
from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.keyfinder_skill_container import KeyFinderSkillContainer
from dimos.robot.unitree.unitree_skill_container import UnitreeSkillContainer

# HEADLESS on purpose — measured on a 15 GB laptop, the visualisation stack was the
# dominant memory cost and made the machine thrash:
#   RerunBridgeModule alone held ~5.3 GB in its own worker, and the Rerun web viewer
#   added ~4-5 GB inside Chrome. Together with the rest, 14/15 GB were used, swap was
#   100% full, and only ~77 MB stayed available.
# A key-finding patrol needs no 3D/camera visualisation, so this blueprint drops
# unitree_go2_basic's vis stack (RerunBridgeModule, RerunWebSocketServer,
# WebsocketVisModule) and WebInput, and halves the worker pool. Drive it with
# `dimos humancli` or `dimos mcp call` (both use LCM, no web UI needed).
# If you *do* want a live camera view, run `unitree-go2-agentic-lite` instead and
# expect the extra memory cost.
unitree_go2_keyfinder = autoconnect(
    GO2Connection.blueprint(),
    McpServer.blueprint(),
    McpClient.blueprint(),
    UnitreeSkillContainer.blueprint(),
    SpeakSkill.blueprint(),
    KeyFinderSkillContainer.blueprint(),
).global_config(n_workers=2, robot_model="unitree_go2")

__all__ = ["unitree_go2_keyfinder"]
