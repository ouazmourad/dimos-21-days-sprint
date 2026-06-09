"""Live interactive demo of the Robot Technical Animator.

Opens a MuJoCo viewer with the wooden expressive-head Go2 and runs the
layered animation engine in real time. The background idle layer is
always on (gaze drift, breathing, blinks); you trigger artist intents
and switch personalities live — by typing single keys in THIS TERMINAL
(not the viewer window, which reserves its own keys for render toggles).

Run:
    python tools/animator_live.py

Controls (type in the terminal, no Enter needed):
    1  curious      2  shy        3  proud
    4  nervous      5  calm
    n  notice guest         c  curious head-tilt
    p  proud chest-lift     r  search room ("look around")
    w  walk forward (toggle)   a/d  steer left/right while walking
    space  return to idle (and stop walking)
    q  quit

While walking, the trot gait drives the legs + body across the floor
and the expressive head rides on top. Mouse-orbit / zoom as usual.
"""

from __future__ import annotations

import atexit
import math
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

os.environ["MUJOCO_GL"] = "glfw"      # interactive viewer needs real GL
os.environ.setdefault("CI", "1")       # skip the LCM/sudo system-config prompt

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dimos.animator.engine import AnimationEngine
from dimos.animator.gait import GaitGenerator
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

# Leg joints the gait owns while walking.
_LEG_JOINTS = set(GO1_JOINT_ORDER)

PERSONALITIES = {"1": "curious", "2": "shy", "3": "proud", "4": "nervous", "5": "calm"}
TICK_DT = 1.0 / 50.0


def _stdin_reader(cmd_q: "queue.Queue[str]", stop: threading.Event) -> None:
    """Read single keypresses from the terminal (raw mode) into a queue."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    atexit.register(termios.tcsetattr, fd, termios.TCSADRAIN, old)
    tty.setcbreak(fd)
    while not stop.is_set():
        r, _, _ = select.select([sys.stdin], [], [], 0.1)
        if r:
            ch = sys.stdin.read(1)
            cmd_q.put(ch)


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
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    base_adr = model.jnt_qposadr[model.body_jntadr[trunk_id]]  # free-joint xyz+quat
    base_z0 = float(data.qpos[base_adr + 2])

    gait = GaitGenerator(rig.default_pose)
    walk = {"on": False, "turn": 0.0, "yaw": 0.0}

    def build(name: str):
        p = Personality.from_yaml(ROOT / f"data/animator/personalities/{name}.yaml")
        return p, AnimationEngine(PerformanceOrchestrator(rig, p, tick_hz=50.0), p)

    name = "curious"
    personality, engine = build(name)

    def banner() -> None:
        p = personality
        print(
            f"[ {name.upper():8s} ] cur={p.curiosity:+.1f} en={p.energy:+.1f} "
            f"conf={p.confidence:+.1f} soft={p.softness:+.1f} play={p.playfulness:+.1f}",
            flush=True,
        )

    cmd_q: "queue.Queue[str]" = queue.Queue()
    stop = threading.Event()
    reader = threading.Thread(target=_stdin_reader, args=(cmd_q, stop), daemon=True)
    reader.start()

    print(__doc__)
    banner()

    def handle(ch: str) -> bool:
        """Return False to quit."""
        nonlocal personality, engine, name
        if ch in PERSONALITIES:
            name = PERSONALITIES[ch]
            personality, engine = build(name)
            banner()
        elif ch == "n":
            engine.trigger(notice_guest(0.6, 0.25, personality, tick_dt=TICK_DT))
            print("  -> notice_guest", flush=True)
        elif ch == "c":
            engine.trigger(curious_head_tilt(direction=1, personality=personality, tick_dt=TICK_DT))
            print("  -> curious_head_tilt", flush=True)
        elif ch == "p":
            engine.trigger(proud_chest_lift(personality=personality, tick_dt=TICK_DT))
            print("  -> proud_chest_lift", flush=True)
        elif ch == "r":
            engine.trigger(search_room(yaw_range_rad=0.8, n_stops=5,
                                       personality=personality, tick_dt=TICK_DT))
            print("  -> search_room", flush=True)
        elif ch == "w":
            walk["on"] = not walk["on"]
            gait.set_active(walk["on"])
            engine._orch.set_walking(walk["on"])  # noqa: SLF001 — tail wags while trotting
            if not walk["on"]:
                walk["turn"] = 0.0
            print(f"  -> walk {'ON' if walk['on'] else 'off'}", flush=True)
        elif ch == "a":
            walk["turn"] = 1.0
            print("  -> steer left", flush=True)
        elif ch == "d":
            walk["turn"] = -1.0
            print("  -> steer right", flush=True)
        elif ch == " ":
            personality, engine = build(name)
            walk["on"] = False
            gait.set_active(False)
            walk["turn"] = 0.0
            print("  -> idle", flush=True)
        elif ch in ("q", "\x03"):  # q or Ctrl-C
            return False
        return True

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False,
    ) as viewer:
        viewer.cam.distance = 1.1
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -12
        viewer.cam.lookat[:] = data.xpos[head_id]

        while viewer.is_running():
            t0 = time.perf_counter()

            # Drain any pending keypresses.
            try:
                while True:
                    if not handle(cmd_q.get_nowait()):
                        stop.set()
                        return
            except queue.Empty:
                pass

            tick = engine.tick()
            # Engine drives everything (head + expressive legs) when standing.
            for nm, v in tick.command.angles.items():
                if nm in jq:
                    data.qpos[jq[nm]] = v

            # The gait is ALWAYS stepped so its start/stop ramps complete;
            # while it has weight it owns the 12 leg joints and moves the
            # base, and the expressive head keeps riding on top.
            out = gait.step(TICK_DT, personality, turn=walk["turn"])
            if out.weight > 0.0:
                # Crossfade legs between the engine's expressive pose and
                # the gait by the gait's own ramp weight — buttery starts
                # and stops even mid-emote.
                for nm, v in out.leg_angles.items():
                    prev = data.qpos[jq[nm]]
                    data.qpos[jq[nm]] = prev + (v - prev) * out.weight
                walk["yaw"] += out.base_dyaw
                cy, sy = math.cos(walk["yaw"] / 2), math.sin(walk["yaw"] / 2)
                data.qpos[base_adr + 0] += out.base_dx * math.cos(walk["yaw"])
                data.qpos[base_adr + 1] += out.base_dx * math.sin(walk["yaw"])
                data.qpos[base_adr + 2] = base_z0 + out.base_dz
                data.qpos[base_adr + 3] = cy   # quat w
                data.qpos[base_adr + 6] = sy   # quat z (yaw)
                # Steering auto-centres after each press.
                walk["turn"] *= 0.92

            mujoco.mj_forward(model, data)
            # Keep the camera on the (possibly moving) robot.
            viewer.cam.lookat[:] = data.xpos[trunk_id]
            viewer.sync()

            dt = TICK_DT - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)

    stop.set()


if __name__ == "__main__":
    main()
