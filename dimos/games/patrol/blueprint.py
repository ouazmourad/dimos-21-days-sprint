"""Collaborative patrol blueprint — wires two autonomous patrol robots.

Each robot gets its own Agent (LangGraph), VLMAgent, SimpleNavSkill,
RadioSkill, and PatrolObserver. A RadioBridge cross-wires their
communications. A PatrolCoordinator manages mission lifecycle.

The multi-instance problem (2 Agents, 2 VLMs, etc.) is solved the
same way as in the telephone game: subclass for unique type identity,
then use remappings to wire each to the correct sim streams.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dimos.agents.agent import Agent, AgentConfig
from dimos.agents.vlm_agent import VLMAgent
from dimos.core.blueprints import autoconnect
from dimos.core.core import rpc
from dimos.core.rpc_client import RPCClient
from dimos.games.patrol.coordinator import PatrolCoordinator, patrol_coordinator
from dimos.games.patrol.nav_simple import SimpleNavSkill
from dimos.games.patrol.observer import PatrolObserver
from dimos.games.patrol.radio import RadioBridge, RadioSkill, radio_bridge
from dimos.games.telephone.multi_sim import MultiRobotMujocoConnection, multi_robot_sim
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.core.blueprints import Blueprint

logger = setup_logger()

# ═══════════════════════════════════════════════════════════════════
# System prompts
# ═══════════════════════════════════════════════════════════════════

ALPHA_PROMPT = """\
You are Alpha, a patrol robot in a MuJoCo office simulation.

MISSION: Patrol your zone, observe surroundings, and share findings via radio with your partner Charlie.

BEHAVIOR:
- Start by calling describe_surroundings to see what's around you.
- Move around using move_forward, turn_left, turn_right to explore.
- After observing something noteworthy, use broadcast to tell Charlie.
- If you see a person in distress or something urgent, use request_help.
- When you receive a [RADIO from Charlie] message, react appropriately — \
if Charlie reports an emergency, move toward that area to assist.
- Keep patrolling: observe, move, broadcast, repeat.

TOOLS: describe_surroundings, move_forward, move_backward, turn_left, \
turn_right, stop_moving, broadcast, request_help, start_mission, end_mission

Be concise in your radio messages. Focus on what you see, where you are, \
and what action you're taking."""

CHARLIE_PROMPT = """\
You are Charlie, a patrol robot in a MuJoCo office simulation.

MISSION: Patrol your zone, observe surroundings, and share findings via radio with your partner Alpha.

BEHAVIOR:
- Start by calling describe_surroundings to see what's around you.
- Move around using move_forward, turn_left, turn_right to explore.
- After observing something noteworthy, use broadcast to tell Alpha.
- If you see a person in distress or something urgent, use request_help.
- When you receive a [RADIO from Alpha] message, react appropriately — \
if Alpha reports an emergency, move toward that area to assist.
- Keep patrolling: observe, move, broadcast, repeat.

TOOLS: describe_surroundings, move_forward, move_backward, turn_left, \
turn_right, stop_moving, broadcast, request_help, start_mission, end_mission

Be concise in your radio messages. Focus on what you see, where you are, \
and what action you're taking."""

# ═══════════════════════════════════════════════════════════════════
# VLM subclasses with visual_query RPC
# ═══════════════════════════════════════════════════════════════════

PATROL_VLM_PROMPT = (
    "You are the vision system of a patrol robot in a simulated office. "
    "When shown an image, describe what you see in detail. Focus on people, "
    "objects, furniture, anything unusual, and spatial layout. "
    "Be specific about locations and states (e.g., 'person lying on floor' "
    "vs 'person sitting in chair'). Never refuse to describe what you see."
)


class _PatrolVLM(VLMAgent):
    """VLMAgent with visual_query RPC for patrol robots."""

    @rpc
    def visual_query(self, query: str) -> str:
        """Query the VLM with the latest camera image."""
        if self._latest_image is None:
            return "No image available yet — camera may not be streaming."
        response = self._invoke_image(self._latest_image, query)
        content = response.content
        return content if isinstance(content, str) else str(content)


class VLMAgentAlpha(_PatrolVLM):
    """VLMAgent for Robot Alpha."""


class VLMAgentCharlie(_PatrolVLM):
    """VLMAgent for Robot Charlie."""


# ═══════════════════════════════════════════════════════════════════
# Agent subclasses with skill filtering
# ═══════════════════════════════════════════════════════════════════


class _PatrolAgent(Agent):
    """Agent subclass that filters skills to only those owned by this robot."""

    OWNED_SKILL_CLASSES: set[str] = set()

    @rpc
    def on_system_modules(self, modules: list[RPCClient]) -> None:
        """Override to filter skills — each Agent only sees its own robot's
        skills plus shared ones (PatrolCoordinator)."""
        if not self.OWNED_SKILL_CLASSES:
            # No filter configured — accept all (fallback)
            super().on_system_modules(modules)
            return

        filtered = []
        for m in modules:
            try:
                skills = m.get_skills() or []
            except Exception:
                filtered.append(m)
                continue

            if not skills:
                # Non-skill modules (sim, bridge, etc.) — include
                filtered.append(m)
            elif any(s.class_name in self.OWNED_SKILL_CLASSES for s in skills):
                filtered.append(m)

        logger.info(
            f"[{self.__class__.__name__}] Skill filter: "
            f"{len(filtered)}/{len(modules)} modules accepted"
        )
        super().on_system_modules(filtered)


class PatrolAgentAlpha(_PatrolAgent):
    """Agent for Robot Alpha."""
    OWNED_SKILL_CLASSES: set[str] = {
        "RadioSkillAlpha", "PatrolObserverAlpha", "NavSkillAlpha", "PatrolCoordinator",
    }


class PatrolAgentCharlie(_PatrolAgent):
    """Agent for Robot Charlie."""
    OWNED_SKILL_CLASSES: set[str] = {
        "RadioSkillCharlie", "PatrolObserverCharlie", "NavSkillCharlie", "PatrolCoordinator",
    }


# ═══════════════════════════════════════════════════════════════════
# Per-robot module subclasses
# ═══════════════════════════════════════════════════════════════════

class NavSkillAlpha(SimpleNavSkill):
    """Navigation for Alpha."""


class NavSkillCharlie(SimpleNavSkill):
    """Navigation for Charlie."""


class RadioSkillAlpha(RadioSkill):
    """Radio for Alpha."""


class RadioSkillCharlie(RadioSkill):
    """Radio for Charlie."""


class PatrolObserverAlpha(PatrolObserver):
    """Observer for Alpha."""
    vlm_rpc: str = "VLMAgentAlpha.visual_query"
    rpc_calls: list[str] = ["VLMAgentAlpha.visual_query"]


class PatrolObserverCharlie(PatrolObserver):
    """Observer for Charlie."""
    vlm_rpc: str = "VLMAgentCharlie.visual_query"
    rpc_calls: list[str] = ["VLMAgentCharlie.visual_query"]


# ═══════════════════════════════════════════════════════════════════
# Blueprint composition
# ═══════════════════════════════════════════════════════════════════

def build_patrol(model: str = "gpt-4o") -> Blueprint:
    """Build the collaborative patrol blueprint.

    Args:
        model: LLM model for Agents and VLMs (e.g., "gpt-4o", "anthropic/claude-sonnet-4-20250514").

    Returns:
        Blueprint ready to .build()
    """
    sim = multi_robot_sim()
    coordinator = patrol_coordinator()
    bridge = radio_bridge()

    # Force minimal workers to save RAM on 16GB systems
    from dimos.core.global_config import global_config
    global_config.n_workers = 1

    # Robot Alpha stack
    vlm_a = VLMAgentAlpha.blueprint(model=model, system_prompt=PATROL_VLM_PROMPT)
    agent_a = PatrolAgentAlpha.blueprint(model=model, system_prompt=ALPHA_PROMPT)
    nav_a = NavSkillAlpha.blueprint()
    radio_a = RadioSkillAlpha.blueprint()
    observer_a = PatrolObserverAlpha.blueprint()

    # Robot Charlie stack
    vlm_c = VLMAgentCharlie.blueprint(model=model, system_prompt=PATROL_VLM_PROMPT)
    agent_c = PatrolAgentCharlie.blueprint(model=model, system_prompt=CHARLIE_PROMPT)
    nav_c = NavSkillCharlie.blueprint()
    radio_c = RadioSkillCharlie.blueprint()
    observer_c = PatrolObserverCharlie.blueprint()

    game = autoconnect(
        sim, coordinator, bridge,
        # Alpha
        vlm_a, agent_a, nav_a, radio_a, observer_a,
        # Charlie
        vlm_c, agent_c, nav_c, radio_c, observer_c,
    )

    game = game.remappings([
        # ── Alpha camera / odom / cmd_vel ──
        (VLMAgentAlpha, "color_image", "a_color_image"),
        (PatrolObserverAlpha, "color_image", "a_color_image"),
        (NavSkillAlpha, "odom", "a_odom"),
        (NavSkillAlpha, "cmd_vel", "a_cmd_vel"),

        # ── Charlie camera / odom / cmd_vel ──
        (VLMAgentCharlie, "color_image", "c_color_image"),
        (PatrolObserverCharlie, "color_image", "c_color_image"),
        (NavSkillCharlie, "odom", "c_odom"),
        (NavSkillCharlie, "cmd_vel", "c_cmd_vel"),

        # ── Agent human_input disambiguation ──
        (PatrolAgentAlpha, "human_input", "a_human_input"),
        (PatrolAgentCharlie, "human_input", "c_human_input"),

        # ── Agent output disambiguation ──
        (PatrolAgentAlpha, "agent", "a_agent"),
        (PatrolAgentAlpha, "agent_idle", "a_agent_idle"),
        (PatrolAgentCharlie, "agent", "c_agent"),
        (PatrolAgentCharlie, "agent_idle", "c_agent_idle"),

        # ── VLM stream disambiguation ──
        (VLMAgentAlpha, "query_stream", "a_query_stream"),
        (VLMAgentAlpha, "answer_stream", "a_answer_stream"),
        (VLMAgentCharlie, "query_stream", "c_query_stream"),
        (VLMAgentCharlie, "answer_stream", "c_answer_stream"),

        # ── Radio cross-wiring ──
        (RadioSkillAlpha, "radio_out", "radio_a_out"),
        (RadioSkillCharlie, "radio_out", "radio_c_out"),
        (RadioBridge, "radio_a_in", "radio_a_out"),
        (RadioBridge, "radio_c_in", "radio_c_out"),
        (RadioBridge, "inject_to_a", "a_human_input"),
        (RadioBridge, "inject_to_c", "c_human_input"),

    ])

    return game


def run_patrol(model: str = "gpt-4o") -> None:
    """Run the collaborative patrol system."""
    game = build_patrol(model=model)
    coordinator = game.build()
    coordinator.loop()


__all__ = ["build_patrol", "run_patrol"]
