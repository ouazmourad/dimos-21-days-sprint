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

"""Key-finder driven by REAL autonomous exploration (frontier + A* replanning).

`unitree-go2-keyfinder` sweeps with a hand-rolled step/turn pattern, which covers a
house poorly. This blueprint instead builds on `unitree_go2` (the "smart" stack:
VoxelGridMapper -> CostMapper -> WavefrontFrontierExplorer -> ReplanningAStarPlanner
-> MovementManager) so the robot explores systematically — the same thing the Command
Center's "Start Exploration" button triggers. KeyFinderSkillContainer publishes
`explore_cmd` to start it, pauses it (`stop_explore_cmd`) whenever it spots a key
candidate so it can approach + StandDown-verify, then resumes coverage.

COST: this is the heavy configuration — it maps with CUDA (VoxelGridMapper) and runs
~10 workers. On a 15 GB / 4 GB-VRAM laptop run it HEADLESS and trim the voxel grid:

    dimos --rerun-open none run unitree-go2-keyfinder-explore \
        -o voxelgridmapper.block_count=400000

Add `-o rerunbridgemodule.memory_limit=512MB` if you also want the viewer, and expect
several GB more RAM. Use `unitree-go2-keyfinder` (headless, ~3 GB, no mapping) when you
just want the cheapest possible key search.
"""

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.skills.speak_skill import SpeakSkill
from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2 import unitree_go2
from dimos.robot.unitree.keyfinder_skill_container import KeyFinderSkillContainer
from dimos.robot.unitree.unitree_skill_container import UnitreeSkillContainer

unitree_go2_keyfinder_explore = autoconnect(
    unitree_go2,
    McpServer.blueprint(),
    McpClient.blueprint(),
    UnitreeSkillContainer.blueprint(),
    SpeakSkill.blueprint(),
    KeyFinderSkillContainer.blueprint(),
).global_config(n_workers=10, robot_model="unitree_go2")

__all__ = ["unitree_go2_keyfinder_explore"]
