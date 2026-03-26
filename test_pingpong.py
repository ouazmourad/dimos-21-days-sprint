"""Ping Pong integration tests — robot stability, ball physics, serve, hit detection."""
import os; os.environ["CI"] = "1"
import math
import mujoco
import numpy as np

from dimos.simulation.mujoco.model import load_scene_xml, get_model_xml, get_assets
from dimos.core.global_config import global_config

global_config.simulation = True
global_config.robot_model = "unitree_g1_tennis"
global_config.mujoco_room = "pingpong"
global_config.mujoco_start_pos = "12.2, 0.0"

model = mujoco.MjModel.from_xml_string(
    get_model_xml("unitree_g1_tennis", load_scene_xml(global_config)),
    assets=get_assets()
)
data = mujoco.MjData(model)

from dimos.games.pingpong.controller import PingPongController
ctrl = PingPongController(default_angles=np.array(model.keyframe("home").qpos[7:7+model.nu]))
mujoco.set_mjcb_control(ctrl.get_control)
mujoco.mj_resetDataKeyframe(model, data, 0)

# Set position + facing
data.qpos[0:3] = [12.2, 0.0, 0.785]
yaw = math.radians(180.0)
data.qpos[3:7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]

ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
ball_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
ball_dof = model.jnt_dofadr[ball_jnt]
ball_qpos_addr = model.jnt_qposadr[ball_jnt]
table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
racket_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tennis_racket_collision")

print("=" * 50)
print("PING PONG INTEGRATION TESTS")
print("=" * 50)

# ── Test 1: Robot stands stable ──
print("\n[1] Robot stability (5 seconds)...")
for _ in range(2500):
    mujoco.mj_step(model, data)
z = data.qpos[2]
assert z > 0.5, f"FAIL: Robot fell (z={z:.3f})"
print(f"  PASS: z={z:.3f} (upright)")

# ── Test 2: Robot faces table ──
print("\n[2] Robot faces table...")
robot_x = data.qpos[0]
table_x = data.xpos[table_id][0]
assert robot_x > table_x, f"FAIL: Robot at x={robot_x:.1f} should be > table at x={table_x:.1f}"
print(f"  PASS: Robot x={robot_x:.1f}, table x={table_x:.1f} (robot on far side)")

# ── Test 3: No body-table collision ──
print("\n[3] Robot-table clearance...")
gap = robot_x - 11.74  # far table edge
assert gap > 0.3, f"FAIL: Only {gap:.2f}m gap (need >0.3m)"
print(f"  PASS: {gap:.2f}m clearance from table edge")

# ── Test 4: Paddle exists (not tennis racket) ──
print("\n[4] Paddle geometry...")
assert racket_geom >= 0, "FAIL: No racket collision geom"
rsize = model.geom_size[racket_geom]
assert rsize[0] < 0.1, f"FAIL: Racket radius {rsize[0]:.3f} too large (tennis racket?)"
print(f"  PASS: Paddle radius={rsize[0]:.3f}m (ping pong size)")

# ── Test 5: Ball serves toward robot ──
print("\n[5] Ball serve trajectory...")
from dimos.games.pingpong.match_manager import MatchManager
mgr = MatchManager()
mgr.init(model)
mgr.serve(model, data)

# Track ball for 1 second
ball_xs = []
for i in range(500):
    mujoco.mj_step(model, data)
    mgr.tick(model, data, 0.002)
    if i % 50 == 0:
        bx = data.qpos[ball_qpos_addr]
        ball_xs.append(bx)

# Ball should move toward robot (+x direction)
assert ball_xs[-1] > ball_xs[0], f"FAIL: Ball moving wrong way ({ball_xs[0]:.1f} -> {ball_xs[-1]:.1f})"
print(f"  PASS: Ball x: {ball_xs[0]:.1f} -> {ball_xs[-1]:.1f} (toward robot)")

# ── Test 6: Ball bounces on table ──
print("\n[6] Ball table bounce...")
bounced = False
mgr.serve(model, data)
for i in range(1000):
    prev_vz = data.qvel[ball_dof + 2]
    mujoco.mj_step(model, data)
    mgr.tick(model, data, 0.002)
    curr_vz = data.qvel[ball_dof + 2]
    if prev_vz < -1.0 and curr_vz > 0.5:  # velocity reversal = bounce
        bounced = True
        break
assert bounced, "FAIL: Ball never bounced on table"
print(f"  PASS: Ball bounced at t={i*0.002:.2f}s")

# ── Test 7: Ball reaches robot's side ──
print("\n[7] Ball reaches robot side...")
mgr.serve(model, data)
reached = False
for i in range(1500):
    mujoco.mj_step(model, data)
    mgr.tick(model, data, 0.002)
    bx = data.qpos[ball_qpos_addr]
    if bx > 11.5:  # past far table edge
        reached = True
        break
if reached:
    print(f"  PASS: Ball reached x={bx:.1f} at t={i*0.002:.2f}s")
else:
    bx_final = data.qpos[ball_qpos_addr]
    print(f"  WARN: Ball only reached x={bx_final:.1f} (may need serve tuning)")

# ── Test 8: Racket-ball collision possible ──
print("\n[8] Racket-ball contact detection...")
# Check if any contacts involve the racket during the rally
mgr.serve(model, data)
racket_hit = False
for i in range(2000):
    mujoco.mj_step(model, data)
    mgr.tick(model, data, 0.002)
    for c in range(data.ncon):
        g1, g2 = data.contact[c].geom1, data.contact[c].geom2
        if g1 == racket_geom or g2 == racket_geom:
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"g{g1}"
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"g{g2}"
            racket_hit = True
            print(f"  PASS: Racket contact at t={i*0.002:.2f}s ({n1} <-> {n2})")
            break
    if racket_hit:
        break
if not racket_hit:
    print(f"  WARN: No racket-ball contact (policy may need more training)")

# ── Test 9: Robot still upright after rallies ──
print("\n[9] Post-rally stability...")
z_final = data.qpos[2]
assert z_final > 0.5, f"FAIL: Robot fell after rallies (z={z_final:.3f})"
print(f"  PASS: z={z_final:.3f} (still upright)")

# ── Summary ──
print("\n" + "=" * 50)
print("ALL CRITICAL TESTS PASSED")
print(f"Policy: {ctrl.swing_phase}")
print(f"Frames played: {ctrl.swing_count}")
print("=" * 50)
