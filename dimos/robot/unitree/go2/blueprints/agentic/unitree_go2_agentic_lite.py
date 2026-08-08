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

"""Lightweight agentic Go2 blueprint for low-RAM machines (e.g. a 15 GB laptop).

`unitree-go2-agentic` builds on `unitree_go2_spatial` -> the full `unitree_go2`
(CUDA VoxelGridMapper, CostMapper, A* planner, frontier explorer, patrol) PLUS
SpatialMemory (CLIP + chromadb) PLUS n_workers=8. That stack OOMs a 15 GB laptop.

This trims to exactly what a scripted talk/move/dance/guard demo needs:
  - GO2Connection (camera/observe, odom for closed-loop move, sport commands) via
    the light `unitree_go2_basic` base (throttled vis, no CUDA voxel/nav/SLAM)
  - the agent loop (McpServer + McpClient, driven by /human_input -> humancli)
  - UnitreeSkillContainer (move, execute_sport_command, wait)
  - SpeakSkill (speak), WebInput (/human_input)
  - PerceiveLoopSkill (look_out_for -> intruder detection; uses the OpenAI cloud
    VLM, so no local GPU/RAM model)

Dropped vs. full agentic: VoxelGridMapper, CostMapper, ReplanningAStarPlanner,
WavefrontFrontierExplorer, PatrollingModule, NavigationSkillContainer,
PersonFollowSkillContainer, SpatialMemory. The agent loses nav/patrol/follow
tools (which don't work on the Air anyway) but keeps move/turn/dance/speak/
observe/look_out_for.

Run:
    dimos --rerun-open none run unitree-go2-agentic-lite -o rerunbridgemodule.memory_limit=1GB
"""

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.skills.speak_skill import SpeakSkill
from dimos.agents.web_human_input import WebInput
from dimos.core.coordination.blueprints import autoconnect
from dimos.perception.perceive_loop_skill import PerceiveLoopSkill
from dimos.robot.unitree.football_skill_container import FootballSkillContainer
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import unitree_go2_basic
from dimos.robot.unitree.unitree_skill_container import UnitreeSkillContainer

unitree_go2_agentic_lite = autoconnect(
    unitree_go2_basic,
    McpServer.blueprint(),
    McpClient.blueprint(),
    UnitreeSkillContainer.blueprint(),
    SpeakSkill.blueprint(),
    WebInput.blueprint(),
    PerceiveLoopSkill.blueprint(),
    FootballSkillContainer.blueprint(),
).global_config(n_workers=4, robot_model="unitree_go2")

__all__ = ["unitree_go2_agentic_lite"]
