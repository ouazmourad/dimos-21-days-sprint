"""Robot Escape Room blueprint.

Trapped robot (with MuJoCo sim) + Guide agent (no sim, text-only).
GameMaster tracks puzzle progress and validates discoveries.

Only 1 MuJoCo process needed — the Guide is pure LLM, saving RAM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dimos.agents.agent import Agent
from dimos.agents.vlm_agent import VLMAgent
from dimos.core.blueprints import autoconnect
from dimos.core.core import rpc
from dimos.core.rpc_client import RPCClient
from dimos.games.escape_room.game_master import GameMaster, game_master
from dimos.games.patrol.nav_simple import SimpleNavSkill
from dimos.games.patrol.observer import PatrolObserver
from dimos.games.patrol.radio import RadioBridge, RadioSkill, radio_bridge
from dimos.robot.unitree.g1.sim import g1_sim_connection
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.core.blueprints import Blueprint

logger = setup_logger()

# ═══════════════════════════════════════════════════════════════════
# System prompts
# ═══════════════════════════════════════════════════════════════════

TRAPPED_PROMPT = """\
You are the Trapped robot in an Escape Room game. You are stuck in \
a simulated office and must find 3 hidden clues to escape.

HOW TO PLAY:
- Your partner (the Guide) will give you hints about what to look for.
- Use describe_surroundings to see what's around you.
- Use move_forward, turn_left, turn_right to navigate the room.
- When you think you've found a clue, use broadcast to tell the Guide \
exactly what you see in detail.
- The Guide will tell the GameMaster to check if you found the right thing.
- Find all 3 clues to escape!

STRATEGY:
- Move around the room systematically. Don't stay in one spot.
- After each move, call describe_surroundings to check your new view.
- Describe objects in detail when broadcasting — color, shape, position.
- Listen to the Guide's hints carefully.

TOOLS: describe_surroundings, move_forward, move_backward, turn_left, \
turn_right, stop_moving, broadcast

Keep moving and searching. You want to ESCAPE!"""

GUIDE_PROMPT = """\
You are the Guide in an Escape Room game. Your partner robot is \
trapped in a room and needs to find 3 clues to escape. You can NOT \
see the room — you only hear what the Trapped robot tells you via radio.

HOW TO PLAY:
- Call start_game to begin. You'll receive the first hint.
- Relay the hint to the Trapped robot using broadcast.
- When the Trapped robot describes finding something, call \
submit_discovery with their description to check if it's correct.
- If correct, you'll get the next hint. Relay it.
- If wrong, encourage them and repeat the hint.
- Call get_current_hint if you forget the current clue.

STRATEGY:
- Give the hint clearly via broadcast.
- When the Trapped robot broadcasts what they see, call submit_discovery.
- Be encouraging! Guide them with radio messages.
- You cannot see the room. Only the Trapped robot can see.

TOOLS: broadcast, start_game, submit_discovery, get_current_hint

Start by calling start_game, then broadcast the first hint!"""

ESCAPE_VLM_PROMPT = (
    "You are the vision system of a robot trapped in an escape room. "
    "Describe everything you see in vivid detail — furniture, objects, "
    "colors, textures, spatial layout. Be thorough — your description "
    "determines if a clue is found. Never refuse to describe."
)


# ═══════════════════════════════════════════════════════════════════
# Module subclasses
# ═══════════════════════════════════════════════════════════════════

class _EscapeVLM(VLMAgent):
    @rpc
    def visual_query(self, query: str) -> str:
        if self._latest_image is None:
            return "No image available yet."
        response = self._invoke_image(self._latest_image, query)
        content = response.content
        return content if isinstance(content, str) else str(content)


class TrappedVLM(_EscapeVLM):
    """VLM for the Trapped robot."""


class _EscapeAgent(Agent):
    """Agent subclass with skill filtering and thread-start guard."""
    OWNED_SKILL_CLASSES: set[str] = set()
    _modules_initialized: bool = False

    @rpc
    def on_system_modules(self, modules: list[RPCClient]) -> None:
        if self._modules_initialized:
            return  # Guard against double-init
        self._modules_initialized = True

        if not self.OWNED_SKILL_CLASSES:
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
                filtered.append(m)
            elif any(s.class_name in self.OWNED_SKILL_CLASSES for s in skills):
                filtered.append(m)
        super().on_system_modules(filtered)


class TrappedAgent(_EscapeAgent):
    """Agent for the Trapped robot — only uses its own skills."""
    OWNED_SKILL_CLASSES: set[str] = {
        "TrappedNav", "TrappedObserver", "TrappedRadio",
    }


class GuideAgent(_EscapeAgent):
    """Agent for the Guide — uses radio + game master skills."""
    OWNED_SKILL_CLASSES: set[str] = {
        "GuideRadio", "GameMaster",
    }


class TrappedNav(SimpleNavSkill):
    """Nav for the Trapped robot."""


class TrappedObserver(PatrolObserver):
    """Observer for the Trapped robot."""
    vlm_rpc: str = "TrappedVLM.visual_query"
    rpc_calls: list[str] = ["TrappedVLM.visual_query"]


class TrappedRadio(RadioSkill):
    """Radio for the Trapped robot."""


class GuideRadio(RadioSkill):
    """Radio for the Guide."""


# ═══════════════════════════════════════════════════════════════════
# Blueprint
# ═══════════════════════════════════════════════════════════════════

def build_escape_room(
    agent_model: str = "claude-3-haiku-20240307",
    vlm_model: str = "gpt-4o",
) -> Blueprint:
    """Build the escape room blueprint.

    Only 1 MuJoCo sim (Trapped robot). Guide is text-only agent.

    Args:
        model: LLM model for both agents.

    Returns:
        Blueprint ready to .build()
    """
    from dimos.core.global_config import global_config
    global_config.n_workers = 1
    global_config.simulation = True
    global_config.performance_tier = "low"
    global_config.mujoco_room = "escape_maze"
    global_config.mujoco_start_pos = "-4.0, -4.0"
    global_config.mujoco_start_yaw = 0.0  # default upright orientation
    global_config.mujoco_person = False  # no walking person in the maze
    global_config.resolve_performance_tier()

    # Trapped robot — has sim (in the maze)
    sim = g1_sim_connection()
    vlm = TrappedVLM.blueprint(model=vlm_model, system_prompt=ESCAPE_VLM_PROMPT)
    trapped_agent = TrappedAgent.blueprint(model=agent_model, system_prompt=TRAPPED_PROMPT)
    nav = TrappedNav.blueprint()
    observer = TrappedObserver.blueprint()
    trapped_radio = TrappedRadio.blueprint()

    # Guide — no sim, just agent + radio + game master
    guide_agent = GuideAgent.blueprint(model=agent_model, system_prompt=GUIDE_PROMPT)
    guide_radio = GuideRadio.blueprint()
    gm = game_master()

    # Bridge routes radio between them
    bridge = radio_bridge()

    game = autoconnect(
        sim, vlm, trapped_agent, nav, observer, trapped_radio,
        guide_agent, guide_radio, gm, bridge,
    )

    game = game.remappings([
        # ── Trapped robot streams ──
        (TrappedAgent, "human_input", "trapped_input"),
        (TrappedAgent, "agent", "trapped_agent_out"),
        (TrappedAgent, "agent_idle", "trapped_idle"),
        (TrappedVLM, "query_stream", "trapped_query"),
        (TrappedVLM, "answer_stream", "trapped_answer"),
        (TrappedRadio, "radio_out", "radio_trapped_out"),

        # ── Guide agent streams ──
        (GuideAgent, "human_input", "guide_input"),
        (GuideAgent, "agent", "guide_agent_out"),
        (GuideAgent, "agent_idle", "guide_idle"),

        # Guide has no VLM/camera — but VLMAgent creates these streams
        # so we must disambiguate them
        (GuideRadio, "radio_out", "radio_guide_out"),

        # ── Radio bridge wiring ──
        (RadioBridge, "radio_a_in", "radio_trapped_out"),
        (RadioBridge, "radio_c_in", "radio_guide_out"),
        (RadioBridge, "inject_to_a", "trapped_input"),
        (RadioBridge, "inject_to_c", "guide_input"),
    ])

    return game


def run_escape_room(agent_model: str = "claude-3-haiku-20240307", vlm_model: str = "gpt-4o") -> None:
    """Run the escape room game."""
    game = build_escape_room(agent_model=agent_model, vlm_model=vlm_model)
    coordinator = game.build()
    coordinator.loop()


__all__ = ["build_escape_room", "run_escape_room"]
