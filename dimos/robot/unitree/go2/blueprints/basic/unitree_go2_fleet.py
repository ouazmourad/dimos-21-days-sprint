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

"""Blueprint for Go2 fleet — multiple Go2 robots.

**Independent mode** (recommended)::

    # Each robot has its own sensor streams, nav, and agent
    ROBOT_IPS=10.0.0.102,10.0.0.209 dimos run unitree-go2-fleet-independent

**Broadcast mode** (legacy)::

    # All robots receive the same commands; only primary publishes sensors
    ROBOT_IPS=10.0.0.102,10.0.0.209 dimos run unitree-go2-fleet
"""

from dimos.core.blueprints import autoconnect
from dimos.core.fleet import RobotConfig, fleet
from dimos.core.global_config import global_config
from dimos.protocol.service.system_configurator import ClockSyncConfigurator
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import with_vis
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.go2.fleet_connection import go2_fleet_connection
from dimos.web.websocket_vis.websocket_vis_module import websocket_vis


# ── Legacy broadcast mode ──
# All robots get the same commands; only primary publishes sensors.
unitree_go2_fleet = (
    autoconnect(
        with_vis,
        go2_fleet_connection(),
        websocket_vis(),
    )
    .global_config(n_workers=4, robot_model="unitree_go2")
    .configurators(ClockSyncConfigurator())
)


# ── Independent mode (fleet API) ──
# Each robot has its own namespaced streams, can be controlled individually.
def build_go2_fleet_independent() -> "Blueprint":
    """Build a fleet where each Go2 has independent sensor streams.

    Robot IPs are read from the ``ROBOT_IPS`` environment variable
    (comma-separated).  Each robot gets auto-namespaced streams:
    ``alpha/odom``, ``bravo/odom``, etc.

    Returns:
        A :class:`Blueprint` ready for ``.build()``.
    """
    raw = global_config.robot_ips
    if not raw:
        raise ValueError(
            "Set ROBOT_IPS (e.g. ROBOT_IPS=10.0.0.102,10.0.0.209)"
        )
    ips = [ip.strip() for ip in raw.split(",") if ip.strip()]

    # Generate robot names: alpha, bravo, charlie, delta, ...
    names = ["alpha", "bravo", "charlie", "delta", "echo",
             "foxtrot", "golf", "hotel", "india", "juliet"]

    robots = [
        RobotConfig(
            name=names[i] if i < len(names) else f"robot_{i}",
            connection=GO2Connection,
            kwargs={"ip": ip},
        )
        for i, ip in enumerate(ips)
    ]

    return (
        fleet(robots=robots)
        .global_config(n_workers=len(ips) + 2, robot_model="unitree_go2")
        .configurators(ClockSyncConfigurator())
    )


__all__ = ["unitree_go2_fleet", "build_go2_fleet_independent"]
