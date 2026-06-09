"""THE SHOW — a fully autonomous, choreographed character performance.

One command, zero interaction: the wooden character sleeps, wakes,
notices you, does a double-take, trots across the stage tail-wagging,
strikes a proud pose, gets startled into shyness when the camera rushes
in, recovers, searches the room, takes a bow, and ends on a hero shot.

Every beat runs through the layered animation engine (background idle +
ramped intents), the saccadic eyes, the spring-loaded ears and tail, and
the foot-synced trot — staged with three-point lighting on a glossy
floor and a programmed cinematic camera.

Run live (opens a window, real-time):
    python tools/animator_show.py

Render headless to a contact sheet (no display needed):
    python tools/animator_show.py --headless --out /tmp/show.png

Record every Nth frame as PNGs (for making a video):
    python tools/animator_show.py --headless --record /tmp/show_frames
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CI", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TICK_DT = 1.0 / 50.0

# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

STAGE_XML = """<?xml version="1.0"?>
<mujoco model="animator_stage">
  <compiler angle="radian"/>
  <asset>
    <texture type="skybox" builtin="gradient"
             rgb1="0.09 0.11 0.16" rgb2="0.015 0.02 0.04"
             width="800" height="800"/>
    <texture type="2d" name="stagefloor" builtin="checker" mark="edge"
             rgb1="0.10 0.11 0.13" rgb2="0.08 0.09 0.11"
             markrgb="0.16 0.17 0.20" width="400" height="400"/>
    <material name="stagefloor" texture="stagefloor" texuniform="true"
              texrepeat="12 12" reflectance="0.38" specular="0.6"
              shininess="0.6"/>
    <material name="backdrop" rgba="0.05 0.07 0.12 1" specular="0.1"/>
  </asset>
  <worldbody>
    <geom name="floor" size="0 0 0.01" type="plane" material="stagefloor"/>
    <geom name="cyc" type="box" size="14 0.1 5" pos="0 7.5 2.5"
          material="backdrop" contype="0" conaffinity="0"/>
    <light name="key"  pos="2.5 -2.5 3.2" dir="-0.55 0.55 -0.62"
           diffuse="0.95 0.88 0.78" specular="0.35 0.32 0.28"
           castshadow="true" cutoff="60"/>
    <light name="fill" pos="-3 -1.5 2.2" dir="0.7 0.35 -0.55"
           diffuse="0.38 0.42 0.52" specular="0 0 0" castshadow="false"/>
    <light name="rim"  pos="0 3.5 2.8" dir="0 -0.75 -0.66"
           diffuse="0.5 0.48 0.55" specular="0.4 0.4 0.45"
           castshadow="false"/>
  </worldbody>
  <visual>
    <headlight diffuse="0.14 0.14 0.16" ambient="0.25 0.25 0.30"
               specular="0 0 0"/>
    <rgba haze="0.04 0.05 0.09 1"/>
    <map znear="0.01" zfar="60"/>
    <quality shadowsize="4096"/>
  </visual>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Camera direction
# ---------------------------------------------------------------------------

@dataclass
class Shot:
    """One camera shot: interpolates az/el/dist over [t0, t1)."""

    t0: float
    t1: float
    az: tuple[float, float]
    el: tuple[float, float]
    dist: tuple[float, float]
    look: str = "trunk"          # "trunk" | "head"
    look_dz: float = 0.0         # extra lookat height

    def params(self, t: float) -> tuple[float, float, float]:
        u = 0.0 if self.t1 <= self.t0 else (t - self.t0) / (self.t1 - self.t0)
        u = max(0.0, min(1.0, u))
        s = u * u * (3 - 2 * u)  # smoothstep
        lerp = lambda a, b: a + (b - a) * s  # noqa: E731
        return lerp(*self.az), lerp(*self.el), lerp(*self.dist)


SHOTS = [
    # Sleeping — slow push-in on the curled character.
    Shot(0.0, 5.0, az=(160, 168), el=(-9, -7), dist=(1.8, 1.25), look="head"),
    # Wake + notice — face close-up.
    Shot(5.0, 11.5, az=(172, 178), el=(-4, -4), dist=(0.95, 0.85), look="head"),
    # The walk — wide tracking shot from the side.
    Shot(11.5, 19.5, az=(95, 120), el=(-13, -11), dist=(2.4, 2.0), look="trunk"),
    # Proud pose — low heroic angle, slow push (stays outside the body).
    Shot(19.5, 24.0, az=(150, 160), el=(-4, -7), dist=(1.7, 1.3), look="head"),
    # The scare — camera RUSHES in (this is what startles it).
    Shot(24.0, 25.2, az=(176, 176), el=(-3, -2), dist=(1.6, 0.85), look="head"),
    # Shy — hold the uncomfortable close-up.
    Shot(25.2, 29.0, az=(176, 183), el=(-2, -4), dist=(0.9, 1.0), look="head"),
    # Recovery + search — pull back to a 3/4.
    Shot(29.0, 35.5, az=(183, 145), el=(-6, -10), dist=(1.25, 1.7), look="head"),
    # Bow — frontal, respectful distance.
    Shot(35.5, 39.0, az=(178, 178), el=(-7, -8), dist=(1.7, 1.55), look="trunk"),
    # Final hero — slow orbit, pull out wide.
    Shot(39.0, 45.0, az=(150, 95), el=(-9, -13), dist=(1.6, 2.5), look="trunk"),
]

SHOW_LEN_S = 45.0


def camera_at(t: float):
    for s in SHOTS:
        if s.t0 <= t < s.t1:
            return s
    return SHOTS[-1]


# ---------------------------------------------------------------------------
# The performance
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true",
                    help="Render a contact sheet instead of a window.")
    ap.add_argument("--out", default="/tmp/animator_show.png")
    ap.add_argument("--record", default=None,
                    help="Directory: dump a PNG every 4 ticks (12.5 fps).")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    args = ap.parse_args()

    if args.headless or args.record:
        os.environ.setdefault("MUJOCO_GL", "egl")
    else:
        os.environ["MUJOCO_GL"] = "glfw"

    import mujoco
    import numpy as np

    from dimos.animator.channels.gaze import GazeTarget
    from dimos.animator.channels.posture import PostureTarget
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

    rig = CharacterRig.from_yaml(ROOT / "data/animator/rigs/unitree_go2.yaml")
    xml, assets, _ = compose_animator_go1_xml(STAGE_XML, get_assets())
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
    base = model.jnt_qposadr[model.body_jntadr[trunk_id]]
    base_z0 = float(data.qpos[base + 2])

    # Personalities used through the arc.
    P = {
        "sleepy": Personality(curiosity=-0.3, energy=-0.8, confidence=-0.2,
                              softness=0.9, playfulness=-0.3),
        "curious": Personality.from_yaml(ROOT / "data/animator/personalities/curious.yaml"),
        "proud": Personality.from_yaml(ROOT / "data/animator/personalities/proud.yaml"),
        "shy": Personality.from_yaml(ROOT / "data/animator/personalities/shy.yaml"),
    }

    personality = P["sleepy"]
    orch = PerformanceOrchestrator(rig, personality, tick_hz=50.0)
    engine = AnimationEngine(orch, personality)
    gait = GaitGenerator(rig.default_pose)
    walk_yaw = 0.0
    walk_turn = 0.0

    def switch(name: str) -> None:
        nonlocal personality
        personality = P[name]
        orch.set_personality(personality)
        engine._personality = personality  # noqa: SLF001 — director interface

    # ---- Beat list: (time_s, callable) -----------------------------------
    def beat_sleep() -> None:
        orch._expression.override_openness = 0.03   # noqa: SLF001
        orch._expression.override_brow = -0.3       # noqa: SLF001
        orch._gaze.set_target(GazeTarget(pitch_rad=-0.35), personality)  # head down
        orch._posture.set_target(PostureTarget(z_offset_rad=-0.12))     # crouched

    def beat_wake() -> None:
        switch("curious")
        orch._expression.override_openness = None   # noqa: SLF001
        orch._expression.override_brow = None       # noqa: SLF001
        orch._posture.set_target(PostureTarget())   # rise
        engine.trigger(curious_head_tilt(direction=1, personality=personality,
                                         tick_dt=TICK_DT))

    def beat_notice() -> None:
        engine.trigger(notice_guest(0.25, 0.22, personality, tick_dt=TICK_DT))

    def beat_walk_on() -> None:
        nonlocal walk_turn
        gait.set_active(True)
        orch.set_walking(True)
        walk_turn = 0.28  # gentle arc keeps it on stage

    def beat_walk_off() -> None:
        nonlocal walk_turn
        gait.set_active(False)
        orch.set_walking(False)
        walk_turn = 0.0

    def beat_proud() -> None:
        switch("proud")
        engine.trigger(proud_chest_lift(personality=personality, tick_dt=TICK_DT))

    def beat_scare() -> None:
        switch("shy")
        # Recoil: shrink back and down, eyes will droop from personality.
        orch._posture.set_target(PostureTarget(x_offset_rad=-0.14,   # noqa: SLF001
                                               z_offset_rad=-0.10))
        orch._gaze.set_target(GazeTarget(yaw_rad=0.5, pitch_rad=-0.15),  # noqa: SLF001
                              personality)  # look away

    def beat_recover() -> None:
        switch("curious")
        orch._posture.set_target(PostureTarget())   # noqa: SLF001
        engine.trigger(search_room(yaw_range_rad=0.85, n_stops=5,
                                   personality=personality, tick_dt=TICK_DT))

    def beat_bow() -> None:
        orch._gaze.set_target(GazeTarget(pitch_rad=-0.32), personality)  # noqa: SLF001
        orch._posture.set_target(PostureTarget(z_offset_rad=-0.15,      # noqa: SLF001
                                               x_offset_rad=0.06))

    def beat_finale() -> None:
        switch("proud")
        orch._posture.set_target(PostureTarget())   # noqa: SLF001
        orch._gaze.set_target(GazeTarget(pitch_rad=0.12), personality)  # noqa: SLF001
        engine.trigger(proud_chest_lift(personality=personality, tick_dt=TICK_DT))
        orch._secondary.excite(0.8)                 # noqa: SLF001 — happy wag

    BEATS = [
        (0.0, beat_sleep,   "sleeping"),
        (4.2, beat_wake,    "waking up"),
        (6.5, beat_notice,  "noticing you (double-take)"),
        (11.5, beat_walk_on, "trotting across the stage"),
        (18.0, beat_walk_off, "settling"),
        (19.8, beat_proud,  "proud pose"),
        (24.0, beat_scare,  "startled -> shy"),
        (29.0, beat_recover, "recovering, searching"),
        (35.5, beat_bow,    "taking a bow"),
        (39.0, beat_finale, "finale"),
    ]

    # ---- Run loop ---------------------------------------------------------
    total_ticks = int(SHOW_LEN_S / TICK_DT)
    beat_idx = 0

    record_dir = Path(args.record) if args.record else None
    if record_dir:
        record_dir.mkdir(parents=True, exist_ok=True)

    contact_times = [0.5, 3.0, 5.5, 8.0, 10.0, 14.0, 18.0, 21.5,
                     25.0, 27.0, 31.0, 36.5, 40.5, 44.0]
    contact_frames: list[tuple[float, str, np.ndarray]] = []
    contact_idx = 0
    beat_label = "sleeping"

    headless = args.headless or bool(record_dir)
    renderer = None
    viewer_ctx = None
    if headless:
        renderer = mujoco.Renderer(model, args.height, args.width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
    else:
        import mujoco.viewer
        viewer_ctx = mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False)
        viewer = viewer_ctx.__enter__()
        cam = viewer.cam

    def apply_camera(t: float) -> None:
        shot = camera_at(t)
        az, el, dist = shot.params(t)
        # The walk rotates the robot's heading; keep shots framed relative
        # to the character by offsetting azimuth with the current yaw.
        cam.azimuth = az + math.degrees(walk_yaw)
        cam.elevation = el
        cam.distance = dist
        body = head_id if shot.look == "head" else trunk_id
        look = data.xpos[body].copy()
        look[2] += shot.look_dz
        cam.lookat[:] = look

    print("THE SHOW — 45 s autonomous performance. Beats:")
    for t, _, label in BEATS:
        print(f"  {t:5.1f}s  {label}")

    # Pre-roll: let the sleep pose settle before the curtain rises, so the
    # opening frame is a properly curled-up character, not a mid-transition.
    beat_sleep()
    for _ in range(int(1.5 / TICK_DT)):
        warm = engine.tick()
        for nm, v in warm.command.angles.items():
            if nm in jq:
                data.qpos[jq[nm]] = v
    mujoco.mj_forward(model, data)

    try:
        for n in range(total_ticks):
            t = n * TICK_DT
            t_wall0 = time.perf_counter()

            while beat_idx < len(BEATS) and t >= BEATS[beat_idx][0]:
                BEATS[beat_idx][1]()
                beat_label = BEATS[beat_idx][2]
                print(f"  [{t:5.1f}s] {beat_label}", flush=True)
                beat_idx += 1

            tick = engine.tick()
            for nm, v in tick.command.angles.items():
                if nm in jq:
                    data.qpos[jq[nm]] = v

            out = gait.step(TICK_DT, personality, turn=walk_turn)
            if out.weight > 0.0:
                for nm, v in out.leg_angles.items():
                    prev = data.qpos[jq[nm]]
                    data.qpos[jq[nm]] = prev + (v - prev) * out.weight
                walk_yaw += out.base_dyaw
                data.qpos[base + 0] += out.base_dx * math.cos(walk_yaw)
                data.qpos[base + 1] += out.base_dx * math.sin(walk_yaw)
                data.qpos[base + 2] = base_z0 + out.base_dz
                data.qpos[base + 3] = math.cos(walk_yaw / 2)
                data.qpos[base + 6] = math.sin(walk_yaw / 2)

            mujoco.mj_forward(model, data)
            apply_camera(t)

            if headless:
                if record_dir and n % 4 == 0:
                    renderer.update_scene(data, camera=cam)
                    from PIL import Image
                    Image.fromarray(renderer.render()).save(
                        record_dir / f"frame_{n // 4:05d}.png")
                if (contact_idx < len(contact_times)
                        and t >= contact_times[contact_idx]):
                    renderer.update_scene(data, camera=cam)
                    contact_frames.append((t, beat_label, renderer.render()))
                    contact_idx += 1
            else:
                viewer.sync()
                if not viewer.is_running():
                    break
                dt_sleep = TICK_DT - (time.perf_counter() - t_wall0)
                if dt_sleep > 0:
                    time.sleep(dt_sleep)
    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)

    if headless and contact_frames:
        from PIL import Image, ImageDraw, ImageFont
        fh, fw, _ = contact_frames[0][2].shape
        cols = 7
        rows = math.ceil(len(contact_frames) / cols)
        pad = 4
        W = cols * fw + (cols + 1) * pad
        H = rows * (fh + 22) + pad
        canvas = Image.new("RGB", (W, H), (8, 10, 16))
        d = ImageDraw.Draw(canvas)
        try:
            fnt = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        except OSError:
            fnt = ImageFont.load_default()
        for i, (t, label, im) in enumerate(contact_frames):
            r, c = divmod(i, cols)
            x = pad + c * (fw + pad)
            y = pad + r * (fh + 22)
            canvas.paste(Image.fromarray(im), (x, y))
            d.text((x + 4, y + fh + 3), f"{t:.1f}s  {label}",
                   fill=(235, 235, 240), font=fnt)
        canvas.save(args.out)
        print(f"Contact sheet -> {args.out}")
    if record_dir:
        print(f"Frames -> {record_dir}/ (12.5 fps)")


if __name__ == "__main__":
    main()
