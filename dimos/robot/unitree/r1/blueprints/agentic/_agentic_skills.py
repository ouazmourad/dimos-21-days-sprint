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

"""Agentic skills used by higher-level R1 blueprints."""

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.skills.speak_skill import SpeakSkill
from dimos.agents.web_human_input import WebInput
from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.unitree.r1.skill_container import UnitreeR1SkillContainer
from dimos.robot.unitree.r1.system_prompt import R1_SYSTEM_PROMPT

# NavigationSkillContainer is intentionally omitted: it hard-requires a
# SpatialMemory module (built from camera/lidar), and the R1's sensor streams do
# not reach DimOS on the laptop (DDS XTypes type mismatch; no Go2-style WebRTC).
# WebInput supplies the chat/voice human input (port 5555) that drives the agent;
# UnitreeR1SkillContainer's move/loco skills reach the robot via the connection's
# "pc1" backend. Re-add NavigationSkillContainer + a spatial tier once R1 sensor
# streams reach DimOS (matching IDL types or a PC1 sensor bridge).
_agentic_skills = autoconnect(
    McpServer.blueprint(),
    # Use Claude Haiku 4.5 via the ANTHROPIC_API_KEY (fast + low-cost, good for
    # the R1 command loop). The McpClient default model is "gpt-4o" (OpenAI),
    # which hangs at "thinking..." without an OPENAI_API_KEY.
    McpClient.blueprint(model="anthropic:claude-haiku-4-5", system_prompt=R1_SYSTEM_PROMPT),
    WebInput.blueprint(),
    SpeakSkill.blueprint(),
    UnitreeR1SkillContainer.blueprint(),
)

__all__ = ["_agentic_skills"]
