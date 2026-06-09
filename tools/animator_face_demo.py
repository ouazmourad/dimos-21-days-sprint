"""Demo grid for the articulated-head animator (option B).

Renders the same intent across N personalities with the expressive
head (neck + eyelids + brows), using a face-focused camera so the
expression is visible. Writes a labelled comparison PNG.
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

from dimos.animator.intents import INTENTS, notice_guest
from dimos.animator.orchestrator import PerformanceOrchestrator
from dimos.animator.personality import Personality
from dimos.animator.retargeter import GO1_JOINT_ORDER
from dimos.animator.rig import CharacterRig
from dimos.animator.sim_head import HEAD_JOINTS, compose_animator_go1_xml
from dimos.simulation.mujoco.model import get_assets
from dimos.utils.data import get_data


def _make_intent(name: str, p: Personality, tick_dt: float):
    if name == "notice_guest":
        return notice_guest(0.6, 0.2, p, tick_dt)
    if name == "curious_head_tilt":
        return INTENTS[name](direction=1, personality=p, tick_dt=tick_dt)
    if name == "proud_chest_lift":
        return INTENTS[name](personality=p, tick_dt=tick_dt)
    if name == "search_room":
        return INTENTS[name](yaw_range_rad=0.6, n_stops=4, personality=p, tick_dt=tick_dt)
    raise ValueError(name)


def render_row(persona_name, rig, model, intent_name, times, frame_w, frame_h):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    all_joints = list(GO1_JOINT_ORDER) + list(HEAD_JOINTS)
    jq = {}
    for j in all_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jid >= 0:
            jq[j] = model.jnt_qposadr[jid]

    p = Personality.from_yaml(ROOT / f"data/animator/personalities/{persona_name}.yaml")
    orch = PerformanceOrchestrator(rig, p, tick_hz=50.0)
    intent = _make_intent(intent_name, p, orch.tick_dt)

    renderer = mujoco.Renderer(model, frame_h, frame_w)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 0.62
    cam.azimuth = 176
    cam.elevation = -4
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")

    snap = sorted({int(t / orch.tick_dt) for t in times})
    si = 0
    n = 0
    done = False
    frames = []
    it = orch.run_intent(intent)
    while si < len(snap):
        if not done:
            try:
                tk = next(it)
                cmd = tk.command
                if tk.finished:
                    done = True
            except StopIteration:
                done = True
                cmd = orch.idle_tick().command
        else:
            cmd = orch.idle_tick().command
        for nm, v in cmd.angles.items():
            if nm in jq:
                data.qpos[jq[nm]] = v
        mujoco.mj_forward(model, data)
        n += 1
        if n == snap[si]:
            cam.lookat[:] = data.xpos[head_id]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
            si += 1
    renderer.close()
    return p, frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personalities", nargs="+", default=["curious", "shy", "proud"])
    ap.add_argument("--intent", default="notice_guest", choices=list(INTENTS.keys()))
    ap.add_argument("--times", nargs="+", type=float, default=[0.5, 1.5, 2.5, 3.5, 4.5])
    ap.add_argument("--frame-w", type=int, default=400)
    ap.add_argument("--frame-h", type=int, default=340)
    ap.add_argument("--out", default="/tmp/animator_face_demo.png")
    args = ap.parse_args()

    rig = CharacterRig.from_yaml(ROOT / "data/animator/rigs/unitree_go2.yaml")
    data_dir = Path(str(get_data("mujoco_sim")))
    scene = (data_dir / "scene_empty.xml").read_text()
    xml, assets, _ = compose_animator_go1_xml(scene, get_assets())
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)

    rows = []
    for name in args.personalities:
        print(f"Rendering {name}...", flush=True)
        p, frames = render_row(name, rig, model, args.intent, args.times,
                               args.frame_w, args.frame_h)
        rows.append((name, p, frames))

    fh, fw, _ = rows[0][2][0].shape
    label_w = 210
    col_h = 38
    pad = 6
    W = label_w + len(args.times) * fw + (len(args.times) + 1) * pad
    H = col_h + len(rows) * fh + (len(rows) + 1) * pad
    canvas = Image.new("RGB", (W, H), (235, 235, 240))
    d = ImageDraw.Draw(canvas)
    try:
        fl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        fl = fs = ImageFont.load_default()

    for c, t in enumerate(args.times):
        x = label_w + pad + c * (fw + pad)
        d.text((x + fw // 2 - 28, 9), f"t = {t:.1f} s", fill=(30, 30, 30), font=fl)

    for r, (name, p, frames) in enumerate(rows):
        y = col_h + pad + r * (fh + pad)
        d.text((10, y + 8), name.upper(), fill=(20, 20, 80), font=fl)
        for i, line in enumerate([
            f"curiosity   {p.curiosity:+.1f}",
            f"energy      {p.energy:+.1f}",
            f"confidence  {p.confidence:+.1f}",
            f"softness    {p.softness:+.1f}",
            f"playfulness {p.playfulness:+.1f}",
        ]):
            d.text((10, y + 36 + i * 16), line, fill=(60, 60, 60), font=fs)
        for c, f in enumerate(frames):
            canvas.paste(Image.fromarray(f), (label_w + pad + c * (fw + pad), y))

    canvas.save(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
