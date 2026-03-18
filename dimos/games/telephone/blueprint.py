"""Robot Telephone game blueprint - wires all modules together.

Strategy: Subclass each shared module type (VLMAgent, etc.) to give
unique class identity for autoconnect. Then use .remappings() to rename
their `color_image` streams to match the multi-sim's prefixed outputs
(a_color_image, b_color_image, c_color_image).

The telephone chain is wired via remappings:
  Describer.telephone_out  ->  renamed to "telephone_ab"
  Relay.telephone_in       ->  renamed to "telephone_ab"
  Relay.telephone_out      ->  renamed to "telephone_bc"
  Seeker.telephone_in      ->  renamed to "telephone_bc"

Transcript streams are similarly disambiguated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dimos.agents.vlm_agent import VLMAgent
from dimos.core.blueprints import autoconnect
from dimos.games.telephone.controller import GameController, game_controller
from dimos.games.telephone.describer import TelephoneDescriber
from dimos.games.telephone.multi_sim import MultiRobotMujocoConnection, multi_robot_sim
from dimos.games.telephone.relay import TelephoneRelay
from dimos.games.telephone.seeker import TelephoneSeeker
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.core.blueprints import Blueprint

logger = setup_logger()


# =====================================================================
# Per-robot subclasses give each module a unique type identity so
# autoconnect can distinguish them and remappings can target them.
# No new streams needed — we remap the inherited ones.
# =====================================================================


class VLMAgentA(VLMAgent):
    """VLMAgent for Robot A (The Describer)."""


class VLMAgentB(VLMAgent):
    """VLMAgent for Robot B (The Relay)."""


class VLMAgentC(VLMAgent):
    """VLMAgent for Robot C (The Seeker)."""


class DescriberA(TelephoneDescriber):
    """Describer module scoped to Robot A."""


class RelayB(TelephoneRelay):
    """Relay module scoped to Robot B."""


class SeekerC(TelephoneSeeker):
    """Seeker module scoped to Robot C."""


def build_telephone_game(
    vlm_model: str = "gpt-4o",
    judge_model: str = "gpt-4o",
) -> Blueprint:
    """Build the complete Robot Telephone blueprint.

    Args:
        vlm_model: Model to use for VLMAgents (all 3 robots).
        judge_model: Model to use for the scoring judge.

    Returns:
        Blueprint ready to .build().loop()
    """
    sim = multi_robot_sim()
    controller = game_controller(judge_model=judge_model)

    vlm_a = VLMAgentA.blueprint(model=vlm_model)
    describer = DescriberA.blueprint()

    vlm_b = VLMAgentB.blueprint(model=vlm_model)
    relay = RelayB.blueprint()

    vlm_c = VLMAgentC.blueprint(model=vlm_model)
    seeker = SeekerC.blueprint()

    game = autoconnect(
        sim,
        controller,
        vlm_a, describer,
        vlm_b, relay,
        vlm_c, seeker,
    )

    # Wire each robot's VLM + module to the correct sim camera via remappings.
    # Remapping format: (ModuleClass, "original_stream_name", "new_stream_name")
    #
    # This renames `color_image` on each subclass so autoconnect wires it
    # to the corresponding prefixed output from MultiRobotMujocoConnection.
    game = game.remappings([
        # --- Robot A camera wiring ---
        (VLMAgentA, "color_image", "a_color_image"),
        (DescriberA, "color_image", "a_color_image"),

        # --- Robot B camera wiring ---
        (VLMAgentB, "color_image", "b_color_image"),
        (RelayB, "color_image", "b_color_image"),

        # --- Robot C camera wiring ---
        (VLMAgentC, "color_image", "c_color_image"),
        (SeekerC, "color_image", "c_color_image"),
        (SeekerC, "odom", "c_odom"),

        # --- Telephone chain ---
        # A's output -> B's input (shared topic "telephone_ab")
        (DescriberA, "telephone_out", "telephone_ab"),
        (RelayB, "telephone_in", "telephone_ab"),

        # B's output -> C's input (shared topic "telephone_bc")
        (RelayB, "telephone_out", "telephone_bc"),
        (SeekerC, "telephone_in", "telephone_bc"),

        # --- Transcript wiring to controller ---
        (DescriberA, "transcript", "transcript_a"),
        (RelayB, "transcript", "transcript_b"),
    ])

    return game


def run_telephone(vlm_model: str = "gpt-4o") -> None:
    """Run the Robot Telephone game."""
    game = build_telephone_game(vlm_model=vlm_model)
    coordinator = game.build()
    coordinator.loop()


__all__ = ["build_telephone_game", "run_telephone"]
