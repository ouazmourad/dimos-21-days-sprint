"""Robot Escape Room blueprint.

Trapped robot (with MuJoCo sim) + Guide agent (no sim, text-only).
GameMaster tracks puzzle progress and validates discoveries.

Only 1 MuJoCo process needed — the Guide is pure LLM, saving RAM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dimos.agents.agent import Agent
from dimos.agents.annotation import skill
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
You are the Trapped robot in an Escape Room game. You are inside \
a maze in a simulated office and must find 3 hidden clues to escape.

HOW TO PLAY:
- The Guide will radio you hints about what to look for.
- Use describe_surroundings to see what is around you.
- Use move_forward, turn_left, turn_right to navigate corridors.
- When you spot a colored object (red, blue, green) on a pedestal, \
use broadcast to tell the Guide the exact color and shape you see.
- The Guide will check if it matches. Find all 3 clues to escape!

CRITICAL RULES:
1. NEVER call describe_surroundings twice in a row. Always MOVE between looks.
2. If you see only walls or a corner: call turn_right IMMEDIATELY. Do NOT look first.
3. If stuck in a corner or dead end: call turn_around to do a 180, then move_forward.
4. The pattern is always: look ONCE → move/turn → look ONCE → move/turn.
5. When you see a colored object on a pedestal, IMMEDIATELY broadcast \
its color and shape (e.g. "I found a bright red sphere on a grey pedestal!").

TOOLS: describe_surroundings, move_forward, move_backward, turn_left, \
turn_right, turn_around, stop_moving, broadcast

Start NOW: describe_surroundings once, then turn or move!"""

GUIDE_PROMPT = """\
You are the Guide in an Escape Room game. Your partner robot is \
trapped in a maze and needs to find 3 clues to escape. You CANNOT \
see the room — you only hear what the Trapped robot tells you via radio.

HOW TO PLAY:
1. Call start_game IMMEDIATELY to begin. You will receive the first hint.
2. Broadcast the hint to the Trapped robot right away.
3. When the Trapped robot mentions seeing a COLORED OBJECT (red, blue, \
green) with a SHAPE (sphere, cylinder, cube), IMMEDIATELY call \
submit_discovery with their full description.
4. If correct, you get the next hint — broadcast it immediately.
5. If wrong, encourage them and repeat the current hint.

IMPORTANT RULES:
- Act FAST. Do not deliberate — broadcast hints and submit discoveries \
the moment you have them.
- When the robot says anything about a colored object, call submit_discovery. \
Do not wait for a perfect description.
- If the robot seems lost, re-broadcast the current hint.

TOOLS: broadcast, start_game, submit_discovery, get_current_hint

Call start_game NOW!"""

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
    """Nav for the Trapped robot — short moves and pure rotation."""

    @skill
    def move_forward(self, duration: float = 1.5) -> str:
        """Walk forward for a short distance (~1.2 metres per 1.5 seconds).
        Use shorter durations (0.5-1.0) in tight corridors.

        Args:
            duration: Seconds to walk (default 1.5, max 3.0).

        Returns:
            Status message.
        """
        duration = min(duration, 3.0)
        self._send_velocity(0.8, 0.0, duration)
        return f"Moving forward {duration:.1f}s. Call describe_surroundings to check."

    @skill
    def move_backward(self, duration: float = 1.0) -> str:
        """Back up to escape a wall or dead end.

        Args:
            duration: Seconds to back up (default 1.0).

        Returns:
            Status message.
        """
        duration = min(duration, 2.0)
        self._send_velocity(-0.3, 0.0, duration)
        return f"Backing up {duration:.1f}s."

    @skill
    def turn_left(self, duration: float = 2.0) -> str:
        """Turn left ~90 degrees.

        Args:
            duration: Seconds to turn (default 2.0 = ~90 degrees).

        Returns:
            Status message.
        """
        self._send_velocity(0.0, 0.8, duration)
        return f"Turned left ~{duration * 45:.0f} degrees."

    @skill
    def turn_right(self, duration: float = 2.0) -> str:
        """Turn right ~90 degrees.

        Args:
            duration: Seconds to turn (default 2.0 = ~90 degrees).

        Returns:
            Status message.
        """
        self._send_velocity(0.0, -0.8, duration)
        return f"Turned right ~{duration * 45:.0f} degrees."

    @skill
    def turn_around(self) -> str:
        """Turn 180 degrees. Use this when stuck in a corner or dead end.

        Returns:
            Status message.
        """
        self._send_velocity(0.0, 0.8, 4.0)
        return "Turning around 180 degrees."


class TrappedObserver(PatrolObserver):
    """Observer for the Trapped robot — escape-room-tuned vision."""
    vlm_rpc: str = "TrappedVLM.visual_query"
    rpc_calls: list[str] = ["TrappedVLM.visual_query"]

    @skill
    def describe_surroundings(self) -> str:
        """Look around and describe what you see. Focus on colored objects,
        pedestals, walls, and anything that could be a clue.

        Returns:
            A detailed description of the current scene.
        """
        try:
            vlm_query = self.get_rpc_calls(self.vlm_rpc)
        except Exception:
            return "Vision system not available — cannot observe."
        response = vlm_query(
            "You are a robot trapped in a maze escape room. Describe EVERYTHING "
            "you see in vivid detail. Pay special attention to: colored objects "
            "(red, blue, green), objects on pedestals or stands, spheres, "
            "cylinders, cubes/boxes, walls, corridors, and openings. "
            "Mention colors, shapes, sizes, and positions. Be thorough!"
        )
        logger.info(f"[ESCAPE-OBSERVER] {response}")
        return response


class TrappedRadio(RadioSkill):
    """Radio for the Trapped robot."""


class GuideRadio(RadioSkill):
    """Radio for the Guide."""


# ═══════════════════════════════════════════════════════════════════
# Blueprint
# ═══════════════════════════════════════════════════════════════════

def build_escape_room(
    agent_model: str = "claude-haiku-4-5-20251001",
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
    global_config.performance_tier = "medium"
    global_config.robot_model = "unitree_g1"
    global_config.mujoco_room = "escape_maze"
    global_config.mujoco_start_pos = "3.0, -2.0"
    global_config.mujoco_start_yaw = 180.0
    global_config.mujoco_person = False
    global_config.mujoco_wall_collision = True
    global_config.mujoco_steps_per_frame = 7
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


def run_escape_room(agent_model: str = "claude-haiku-4-5-20251001", vlm_model: str = "gpt-4o") -> None:
    """Run the escape room game."""
    game = build_escape_room(agent_model=agent_model, vlm_model=vlm_model)
    coordinator = game.build()
    coordinator.loop()


__all__ = ["build_escape_room", "run_escape_room"]
