"""Render the animator running an intent on the wooden-toy Go1 in MuJoCo.

Usage:
    .venv/bin/python tools/animator_preview.py \\
        --personality data/animator/personalities/curious.yaml \\
        --intent notice_guest \\
        --out /tmp/animator_curious.mp4

Records a video of the simulated robot performing the chosen intent.
Used to make the v1 demo videos (same intent × three personalities).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

# Make sure we can find the local dimos package when run from a worktree.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dimos.animator.intents import INTENTS, notice_guest
from dimos.animator.orchestrator import PerformanceOrchestrator
from dimos.animator.personality import Personality
from dimos.animator.retargeter import GO1_JOINT_ORDER
from dimos.animator.rig import CharacterRig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Animator preview")
    p.add_argument("--personality", required=True,
                   help="Path to a personality YAML file.")
    p.add_argument("--intent", default="notice_guest",
                   choices=list(INTENTS.keys()),
                   help="Which intent to perform.")
    p.add_argument("--rig", default=str(ROOT / "data/animator/rigs/unitree_go2.yaml"),
                   help="Path to the rig YAML.")
    p.add_argument("--target-yaw", type=float, default=0.4,
                   help="For intents that take a target bearing (radians).")
    p.add_argument("--target-pitch", type=float, default=0.1,
                   help="For intents that take a target pitch (radians).")
    p.add_argument("--duration", type=float, default=8.0,
                   help="Total preview duration in seconds (incl. settle).")
    p.add_argument("--tick-hz", type=float, default=50.0,
                   help="Animator control rate.")
    p.add_argument("--out", default="/tmp/animator_preview.png",
                   help="Where to save the final-frame screenshot.")
    p.add_argument("--snap-at", type=float, default=None,
                   help="Optional: also save the frame at this many seconds in.")
    p.add_argument("--mp4", default=None,
                   help="Optional: write an MP4 of the whole performance here.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    return p.parse_args()


def _build_go1_xml(wooden: bool = True) -> str:
    """Compose the Go1 in scene_empty, with optional wooden-toy skin.

    Reuses ``get_model_xml`` from dimos.simulation.mujoco.model so the
    wooden-toy textures and material injection happen exactly the same
    way they do in the production sim — no parallel path to maintain.
    """
    import xml.etree.ElementTree as ET

    from dimos.simulation.mujoco.model import get_model_xml
    from dimos.utils.data import get_data

    data_dir = Path(str(get_data("mujoco_sim")))
    scene_xml = (data_dir / "scene_empty.xml").read_text()
    # get_model_xml expects a config object for the wooden-skin toggle in some
    # branches; on this branch it accepts a bare ``wooden_skin`` kwarg.
    try:
        composed = get_model_xml("unitree_go1", scene_xml, wooden_skin=wooden)
    except TypeError:
        # Older signature — falls back to default behaviour (wooden on).
        composed = get_model_xml("unitree_go1", scene_xml)

    root = ET.fromstring(composed)
    # Bigger offscreen framebuffer for nicer demo frames.
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    g = visual.find("global")
    if g is None:
        g = ET.SubElement(visual, "global")
    g.set("offwidth", "1280")
    g.set("offheight", "960")
    return ET.tostring(root, encoding="unicode")


def _get_assets() -> dict[str, bytes]:
    """Forward to dimos' asset loader (it already injects wood + toy XML)."""
    from dimos.simulation.mujoco.model import get_assets
    return get_assets()


def _make_intent_iter(name: str, args: argparse.Namespace, p: Personality, tick_dt: float):
    if name == "notice_guest":
        return notice_guest(args.target_yaw, args.target_pitch, p, tick_dt)
    if name == "curious_head_tilt":
        return INTENTS[name](direction=1, personality=p, tick_dt=tick_dt)
    if name == "proud_chest_lift":
        return INTENTS[name](personality=p, tick_dt=tick_dt)
    if name == "search_room":
        return INTENTS[name](yaw_range_rad=0.5, n_stops=4, personality=p, tick_dt=tick_dt)
    raise ValueError(f"Unknown intent {name}")


def main() -> int:
    args = parse_args()

    rig = CharacterRig.from_yaml(args.rig)
    personality = Personality.from_yaml(args.personality)
    print(f"Rig         : {rig.robot} ({len(rig.roles)} expressive channels)", flush=True)
    print(f"Personality : {args.personality}", flush=True)
    print(f"             curiosity={personality.curiosity:+.2f} "
          f"energy={personality.energy:+.2f} confidence={personality.confidence:+.2f} "
          f"softness={personality.softness:+.2f} playfulness={personality.playfulness:+.2f}",
          flush=True)

    # Load MuJoCo model.
    xml_string = _build_go1_xml()
    model = mujoco.MjModel.from_xml_string(xml_string, assets=_get_assets())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Map joint names → qpos indices once.
    joint_qpos: dict[str, int] = {}
    for jname in GO1_JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise RuntimeError(f"joint {jname!r} not found in MuJoCo model")
        joint_qpos[jname] = model.jnt_qposadr[jid]

    # Renderer + camera framing the trunk.
    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 1.4
    cam.azimuth = 135
    cam.elevation = -15

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

    # Orchestrator
    orch = PerformanceOrchestrator(rig, personality, tick_hz=args.tick_hz)
    intent = _make_intent_iter(args.intent, args, personality, orch.tick_dt)

    # MP4 writer (optional).
    mp4_writer = None
    if args.mp4:
        try:
            import imageio.v3 as iio  # noqa: F401
            frames: list[np.ndarray] = []
            mp4_writer = frames  # use a list, write at the end
        except ImportError:
            print("imageio not installed; --mp4 ignored", flush=True)
            mp4_writer = None

    # Tick loop. Each tick: orchestrator → joint commands → write into qpos.
    # We are NOT running physics (would fight the joint targets); we just
    # animate the kinematic pose. This is exactly what an animator preview
    # should do — kinematic preview, not dynamic simulation.
    n_total = int(args.duration / orch.tick_dt)
    n_done = 0
    finished_intent = False
    snap_tick = int(args.snap_at / orch.tick_dt) if args.snap_at is not None else None
    snap_saved = False
    t0 = time.monotonic()

    def _maybe_snapshot() -> None:
        nonlocal snap_saved
        if snap_tick is not None and not snap_saved and n_done >= snap_tick:
            from PIL import Image as _Img
            cam.lookat[:] = data.xpos[trunk_id]
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            snap_path = args.out.replace(".png", f"_t{args.snap_at:.1f}s.png")
            _Img.fromarray(frame).save(snap_path)
            print(f"Wrote mid-performance snapshot to {snap_path}", flush=True)
            snap_saved = True

    for tick in orch.run_intent(intent):
        for name, value in tick.command.angles.items():
            data.qpos[joint_qpos[name]] = value
        mujoco.mj_forward(model, data)

        if mp4_writer is not None:
            cam.lookat[:] = data.xpos[trunk_id]
            renderer.update_scene(data, camera=cam)
            mp4_writer.append(renderer.render())
        n_done += 1
        _maybe_snapshot()
        if tick.finished:
            finished_intent = True
            break

    # After the intent finishes, keep ticking (idle / breathing) until we
    # fill the requested duration.
    while n_done < n_total:
        tick = orch.idle_tick()
        for name, value in tick.command.angles.items():
            data.qpos[joint_qpos[name]] = value
        mujoco.mj_forward(model, data)
        if mp4_writer is not None:
            cam.lookat[:] = data.xpos[trunk_id]
            renderer.update_scene(data, camera=cam)
            mp4_writer.append(renderer.render())
        n_done += 1
        _maybe_snapshot()

    # Save last frame as PNG.
    from PIL import Image
    cam.lookat[:] = data.xpos[trunk_id]
    renderer.update_scene(data, camera=cam)
    final = renderer.render()
    Image.fromarray(final).save(args.out)
    print(f"Wrote final frame to {args.out}", flush=True)

    # Save MP4 if requested.
    if mp4_writer is not None and args.mp4:
        import imageio.v3 as iio
        iio.imwrite(args.mp4, np.stack(mp4_writer, axis=0), fps=int(args.tick_hz))
        print(f"Wrote {len(mp4_writer)} frames to {args.mp4}", flush=True)

    print(f"Ran {n_done} ticks ({n_done * orch.tick_dt:.2f} s sim time)", flush=True)
    print(f"Intent finished? {finished_intent}", flush=True)
    print(f"Wall time: {time.monotonic() - t0:.2f} s", flush=True)
    renderer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
