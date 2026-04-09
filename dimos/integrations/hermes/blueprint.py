"""Hermes-DIMOS bridge blueprints.

Single Go2 quadruped in MuJoCo simulation, exposing DIMOS skills via
the built-in MCP server. The local LangGraph Agent is intentionally
OMITTED — Hermes is the brain.

Two variants:

  - hermes_dimos_go2_lite : sim + Go2 movement skills + speak + MCP server.
                            Roughly 4-5 GB RAM, n_workers=1. Best for laptops.
                            navigate_with_text NOT available (no spatial memory).

  - hermes_dimos_go2_full : full perception (spatial memory + object tracker)
                            + navigate_with_text + everything in lite.
                            Roughly 10+ GB RAM, n_workers=7. Use on bigger boxes.

Both expose the MCP server at:
    http://127.0.0.1:9990/mcp

Hermes config to consume it (in ~/.hermes/config.yaml):
    mcp_servers:
      dimos:
        url: http://127.0.0.1:9990/mcp
        timeout: 60
        connect_timeout: 30
"""

from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.skills.navigation import navigation_skill
from dimos.agents.skills.speak_skill import speak_skill
from dimos.core.blueprints import autoconnect
from dimos.integrations.hermes.move_skill import direct_move_skill
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import unitree_go2_basic
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2_spatial import unitree_go2_spatial
from dimos.robot.unitree.unitree_skill_container import unitree_skills
from dimos.visualization.rerun.bridge import RerunBridgeModule
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule

# ── Lite: basic Go2 sim + DIRECT cmd_vel movement + speak + MCP. ─────
#
# IMPORTANT: we DO NOT include unitree_skills() here. Its skills are
# either broken in simulation (`execute_sport_command` calls
# `MujocoConnection.publish_request` which is a no-op) or require a
# navigation planner (`relative_move` needs NavigationInterface, which
# would pull in the costmap + perception stack — defeating "lite").
#
# Instead we expose `direct_move_skill` which writes Twists straight
# to `cmd_vel`. MujocoConnection consumes that and the robot walks.
#
# We also strip Rerun and the websocket viz server — Rerun fails on
# hosts without the `rerun` CLI on PATH, and WebsocketVisModule starts
# a second uvicorn on port 7779 that duplicates every log line.
hermes_dimos_go2_lite = (
    autoconnect(
        unitree_go2_basic,
        speak_skill(),
        direct_move_skill(),
        McpServer.blueprint(),
    )
    .disabled_modules(RerunBridgeModule, WebsocketVisModule)
    .global_config(n_workers=1, robot_model="unitree_go2")
)


# ── Full: adds perception, spatial memory, navigate_with_text. ───────
hermes_dimos_go2_full = autoconnect(
    unitree_go2_spatial,
    navigation_skill(),
    speak_skill(),
    unitree_skills(),
    McpServer.blueprint(),
)


# Default alias used by run_dimos_mcp.py
hermes_dimos_g1 = hermes_dimos_go2_lite  # legacy alias kept for back-compat
hermes_dimos_go2 = hermes_dimos_go2_lite

__all__ = [
    "hermes_dimos_go2",
    "hermes_dimos_go2_lite",
    "hermes_dimos_go2_full",
    "hermes_dimos_g1",  # back-compat alias
]
