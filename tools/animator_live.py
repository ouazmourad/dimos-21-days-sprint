"""Live interactive demo of the Robot Technical Animator.

Opens a MuJoCo viewer with the wooden expressive-head Go2 and runs the
layered animation engine in real time. The background idle layer is
always on (gaze drift, breathing, blinks); you trigger artist intents
and switch personalities live from the keyboard.

Run:
    python tools/animator_live.py
    # (sets MUJOCO_GL=glfw itself; needs a display)

Keyboard (focus the MuJoCo window, not the terminal):
    1  curious      2  shy        3  proud
    4  nervous      5  calm
    N  notice guest      C  curious head-tilt
    P  proud chest-lift  R  search room ("look around")
    SPACE  stop the current intent (return to idle)

The current personality + state is printed to the terminal on each
change. Mouse-orbit the viewer as usual.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Interactive viewer needs a real GL backend, not the headless EGL one.
os.environ["MUJOCO_GL"] = "glfw"
os.environ.setdefault("CI", "1")  # skip the LCM/sudo system-config prompt

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dimos.animator.engine import AnimationEngine
from dimos.animator.intents import (
    curious_head_tilt,
    notice_guest,
    proud_chest_lift,
    search_room,
)
from dimos.animator.orchestrator import PerformanceOrchestrator
from dimos.animator.personality import Personality
from dimos.animator.retargeter import GO1_JOINT_ORDER
from dimos.animator.rig import CharacterRig
from dimos.animator.sim_head import HEAD_JOINTS, compose_animator_go1_xml
from dimos.simulation.mujoco.model import get_assets
from dimos.utils.data import get_data

PERSONALITIES = ("curious", "shy", "proud", "nervous", "calm")

# GLFW key codes.
_K_1, _K_5 = 49, 53
_K_N, _K_C, _K_P, _K_R, _K_SPACE = 78, 67, 80, 82, 32


def main() -> None:
    rig = CharacterRig.from_yaml(ROOT / "data/animator/rigs/unitree_go2.yaml")
    scene = (Path(str(get_data("mujoco_sim"))) / "scene_empty.xml").read_text()
    xml, assets, _ = compose_animator_go1_xml(scene, get_assets())
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    jq = {}
    for j in list(GO1_JOINT_ORDER) + list(HEAD_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jid >= 0:
            jq[j] = model.jnt_qposadr[jid]

    # Mutable demo state, rebuilt on personality switch.
    state = {"name": "curious"}

    def build_engine(name: str):
        p = Personality.from_yaml(ROOT / f"data/animator/personalities/{name}.yaml")
        orch = PerformanceOrchestrator(rig, p, tick_hz=50.0)
        eng = AnimationEngine(orch, p)
        return p, eng

    personality, engine = build_engine(state["name"])
    engine_dt = 1.0 / 50.0  # engine / orchestrator tick period

    def banner() -> None:
        p = personality
        print(
            f"\n[ {state['name'].upper():8s} ]  "
            f"cur={p.curiosity:+.1f} en={p.energy:+.1f} conf={p.confidence:+.1f} "
            f"soft={p.softness:+.1f} play={p.playfulness:+.1f}"
            f"   {'PLAYING' if engine.is_playing else 'idle'}",
            flush=True,
        )

    def key_callback(keycode: int) -> None:
        nonlocal personality, engine
        if _K_1 <= keycode <= _K_5:
            idx = keycode - _K_1
            if idx < len(PERSONALITIES):
                state["name"] = PERSONALITIES[idx]
                personality, engine = build_engine(state["name"])
                banner()
        elif keycode == _K_N:
            engine.trigger(notice_guest(0.6, 0.25, personality, tick_dt=engine_dt))
            print("  -> notice_guest", flush=True)
        elif keycode == _K_C:
            engine.trigger(curious_head_tilt(direction=1, personality=personality, tick_dt=engine_dt))
            print("  -> curious_head_tilt", flush=True)
        elif keycode == _K_P:
            engine.trigger(proud_chest_lift(personality=personality, tick_dt=engine_dt))
            print("  -> proud_chest_lift", flush=True)
        elif keycode == _K_R:
            engine.trigger(search_room(yaw_range_rad=0.6, n_stops=5,
                                       personality=personality, tick_dt=engine_dt))
            print("  -> search_room", flush=True)
        elif keycode == _K_SPACE:
            # Re-trigger an empty/finished clip = drop back to idle.
            personality, engine = build_engine(state["name"])
            print("  -> stop (idle)", flush=True)

    print(__doc__)
    banner()

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        # Frame the head nicely.
        viewer.cam.distance = 1.1
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -12
        head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
        viewer.cam.lookat[:] = data.xpos[head_id]

        while viewer.is_running():
            t0 = time.perf_counter()
            tick = engine.tick()
            for nm, v in tick.command.angles.items():
                if nm in jq:
                    data.qpos[jq[nm]] = v
            mujoco.mj_forward(model, data)
            viewer.sync()
            # Real-time pace at 50 Hz.
            dt = engine_dt - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
