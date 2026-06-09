"""Continuous-performance demo of the layered animation engine.

Shows the Disney-style architecture: an always-on background (idle gaze
drift + breathing + blinks), with a triggered intent that ramps in over
the background, plays, and ramps back out — the robot returns to idle on
its own. Renders a filmstrip across the whole timeline so the blend
in/out is visible.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CI", "1")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dimos.animator.engine import AnimationEngine
from dimos.animator.intents import notice_guest
from dimos.animator.orchestrator import PerformanceOrchestrator
from dimos.animator.personality import Personality
from dimos.animator.retargeter import GO1_JOINT_ORDER
from dimos.animator.rig import CharacterRig
from dimos.animator.sim_head import HEAD_JOINTS, compose_animator_go1_xml
from dimos.simulation.mujoco.model import get_assets
from dimos.utils.data import get_data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personality", default="curious")
    ap.add_argument("--idle-before", type=float, default=2.0,
                    help="Seconds of pure background idle before the trigger.")
    ap.add_argument("--idle-after", type=float, default=2.5,
                    help="Seconds of background idle to show after the clip.")
    ap.add_argument("--trigger-at", type=float, default=2.0)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--frame-w", type=int, default=360)
    ap.add_argument("--frame-h", type=int, default=320)
    ap.add_argument("--out", default="/tmp/animator_engine_demo.png")
    args = ap.parse_args()

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

    p = Personality.from_yaml(ROOT / f"data/animator/personalities/{args.personality}.yaml")
    orch = PerformanceOrchestrator(rig, p, tick_hz=50.0)
    eng = AnimationEngine(orch, p)

    renderer = mujoco.Renderer(model, args.frame_h, args.frame_w)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 0.7
    cam.azimuth = 176
    cam.elevation = -6

    dt = orch.tick_dt
    total_s = args.idle_before + 4.0 + args.idle_after
    total_ticks = int(total_s / dt)
    trigger_tick = int(args.trigger_at / dt)
    snap_ticks = np.linspace(0, total_ticks - 1, args.n_frames).astype(int)

    frames = []
    si = 0
    triggered = False
    for n in range(total_ticks):
        if not triggered and n >= trigger_tick:
            eng.trigger(notice_guest(0.7, 0.25, p, tick_dt=dt))
            triggered = True
        tick = eng.tick()
        for nm, v in tick.command.angles.items():
            if nm in jq:
                data.qpos[jq[nm]] = v
        mujoco.mj_forward(model, data)
        if si < len(snap_ticks) and n == snap_ticks[si]:
            cam.lookat[:] = data.xpos[head_id]
            renderer.update_scene(data, camera=cam)
            frames.append((n * dt, eng.is_playing, renderer.render()))
            si += 1
    renderer.close()

    fh, fw, _ = frames[0][2].shape
    pad = 5
    cols = len(frames)
    W = cols * fw + (cols + 1) * pad
    H = fh + 2 * pad + 26
    canvas = Image.new("RGB", (W, H), (245, 245, 248))
    d = ImageDraw.Draw(canvas)
    try:
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except OSError:
        fs = fb = ImageFont.load_default()

    for i, (t_s, playing, im) in enumerate(frames):
        x = pad + i * (fw + pad)
        canvas.paste(Image.fromarray(im), (x, 22))
        label = f"t={t_s:.1f}s"
        state = "NOTICE" if playing else "idle"
        d.text((x + 2, 4), label, fill=(30, 30, 30), font=fs)
        color = (200, 40, 40) if playing else (120, 120, 120)
        d.text((x + 2, fh + 24), state, fill=color, font=fb)

    canvas.save(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
