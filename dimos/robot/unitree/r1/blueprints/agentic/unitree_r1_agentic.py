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

"""Full R1 stack with agentic skills.

Composes agentic skills directly on top of ``unitree_r1_basic`` (rather than a
perceptive layer like G1's ``unitree_g1``) — the G1 perceptive/memory layer is
heavy and the task explicitly blesses building R1 agentic on top of basic.
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.unitree.r1.blueprints.agentic._agentic_skills import _agentic_skills
from dimos.robot.unitree.r1.blueprints.basic.unitree_r1_basic import unitree_r1_basic

unitree_r1_agentic = autoconnect(
    unitree_r1_basic,
    _agentic_skills,
)

__all__ = ["unitree_r1_agentic"]
