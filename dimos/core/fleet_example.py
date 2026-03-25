"""Example: Patrol game — before and after fleet().

BEFORE (current approach): ~80 lines of manual subclasses + remappings
AFTER (with fleet):        ~15 lines
"""

# ═══════════════════════════════════════════════════════════════════
# BEFORE — the current approach (simplified from patrol/blueprint.py)
# ═══════════════════════════════════════════════════════════════════
#
#   class PatrolAgentAlpha(Agent): ...
#   class PatrolAgentCharlie(Agent): ...
#   class VLMAgentAlpha(VLMAgent): ...
#   class VLMAgentCharlie(VLMAgent): ...
#   class NavSkillAlpha(SimpleNavSkill): ...
#   class NavSkillCharlie(SimpleNavSkill): ...
#   class RadioSkillAlpha(RadioSkill): ...
#   class RadioSkillCharlie(RadioSkill): ...
#   class ObserverAlpha(PatrolObserver): ...
#   class ObserverCharlie(PatrolObserver): ...
#
#   # 10 subclasses just to have 2 robots!
#
#   game = autoconnect(
#       sim, vlm_a, agent_a, nav_a, obs_a, radio_a,
#       vlm_c, agent_c, nav_c, obs_c, radio_c,
#       bridge, coordinator,
#   )
#
#   game = game.remappings([
#       (VLMAgentAlpha, "color_image", "a_color_image"),
#       (NavSkillAlpha, "cmd_vel", "a_cmd_vel"),
#       (NavSkillAlpha, "odom", "a_odom"),
#       (PatrolAgentAlpha, "human_input", "a_human_input"),
#       (PatrolAgentAlpha, "agent", "a_agent_out"),
#       (PatrolAgentAlpha, "agent_idle", "a_idle"),
#       (RadioSkillAlpha, "radio_out", "radio_a_out"),
#       (ObserverAlpha, "color_image", "a_color_image"),
#       # ... same 8 lines again for Charlie ...
#       (VLMAgentCharlie, "color_image", "c_color_image"),
#       (NavSkillCharlie, "cmd_vel", "c_cmd_vel"),
#       # ... etc etc ...
#       (RadioBridge, "radio_a_in", "radio_a_out"),
#       (RadioBridge, "radio_c_in", "radio_c_out"),
#       (RadioBridge, "inject_to_a", "a_human_input"),
#       (RadioBridge, "inject_to_c", "c_human_input"),
#   ])
#   # Total: ~10 subclasses + ~20 remapping lines = ~80 lines of boilerplate


# ═══════════════════════════════════════════════════════════════════
# AFTER — with fleet()
# ═══════════════════════════════════════════════════════════════════

from dimos.core.fleet import fleet, RobotConfig, SharedModule

# These are the REAL module classes — no subclassing needed
from dimos.agents.vlm_agent import VLMAgent
from dimos.robot.unitree.g1.sim import G1SimConnection


def build_patrol_fleet() -> "Blueprint":
    """Two-robot fleet — 10 lines instead of 80."""
    from dimos.core.blueprints import Blueprint

    return fleet(
        robots=[
            RobotConfig(
                name="alpha",
                connection=G1SimConnection,
                modules=[VLMAgent],
            ),
            RobotConfig(
                name="charlie",
                connection=G1SimConnection,
                modules=[VLMAgent],
            ),
        ],
    )


# This produces:
# - Alpha_G1SimConnection, Alpha_VLMAgent, Alpha_Agent, Alpha_SimpleNavSkill, ...
# - Charlie_G1SimConnection, Charlie_VLMAgent, Charlie_Agent, ...
# - RadioBridge (shared, unnamespaced)
#
# With auto-generated remappings:
# - (Alpha_G1SimConnection, "odom")        → "alpha/odom"
# - (Alpha_G1SimConnection, "color_image") → "alpha/color_image"
# - (Alpha_Agent, "human_input")           → "alpha/human_input"
# - (Charlie_Agent, "human_input")         → "charlie/human_input"
# - ... etc for all streams on all modules
#
# The RadioBridge still needs manual remapping to connect
# alpha/radio_out → charlie/human_input and vice versa.
# This can be done with a single .remappings() call on the result.


if __name__ == "__main__":
    import os
    os.environ["CI"] = "1"
    from dimos.core.global_config import global_config

    global_config.simulation = True
    global_config.robot_model = "unitree_g1"

    bp = build_patrol_fleet()

    print("Fleet modules:")
    for atom in bp._active_blueprints:
        print(f"  {atom.module.__name__}")

    print(f"\nRemappings ({len(bp.remapping_map)}):")
    for (cls, stream), target in sorted(bp.remapping_map.items(), key=lambda x: x[1]):
        print(f"  {cls.__name__}.{stream} → {target}")
