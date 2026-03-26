"""Generate synthetic mocap with ALL required TrajectoryModel fields."""
import numpy as np
import mujoco
from pathlib import Path

xml_path = "/home/mourad/Desktop/dimos-21days-sprint/robomotion/storage/assets/unitree_g1/scene_mjx_racket_wo_ball_flat_terrain.xml"
spec = mujoco.MjSpec.from_file(xml_path)
model = spec.compile()
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
home_qpos = data.qpos.copy()
nq, nv = model.nq, model.nv
RA = 7 + 22
dt = 1.0 / 50.0

def gen_swing(n_frames=100, var=0.0):
    qpos_all, qvel_all, xpos_all, xquat_all = [], [], [], []
    cvel_all, subtree_com_all, site_xpos_all, site_xmat_all = [], [], [], []

    for i in range(n_frames):
        t = i / n_frames
        q = home_qpos.copy()
        if t < 0.3:
            s = t / 0.3
            q[RA+0] = 0.2 + s * (0.6 + var * 0.1)
            q[RA+1] = -0.3 - s * 0.2
            q[RA+2] = -s * 0.3
            q[RA+3] = 1.28 + s * 0.2
        elif t < 0.6:
            s = (t - 0.3) / 0.3
            q[RA+0] = 0.8 - s * 1.3
            q[RA+1] = -0.5 + s * 0.3
            q[RA+2] = -0.3 + s * 0.6
            q[RA+3] = 1.48 - s * 0.8
        else:
            s = (t - 0.6) / 0.4
            q[RA+0] = -0.5 + s * 0.7
            q[RA+1] = -0.2 - s * 0.1
            q[RA+2] = 0.3 - s * 0.3
            q[RA+3] = 0.68 + s * 0.6
        q[7+12] = -0.1 * np.sin(max(0, (t-0.2)/0.5) * np.pi) if 0.2 < t < 0.7 else 0
        q[7+3] = 0.3 + 0.1 * np.sin(t * np.pi)
        q[7+9] = 0.3 + 0.1 * np.sin(t * np.pi)

        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        qpos_all.append(q.copy())
        qvel_all.append(np.zeros(nv, dtype=np.float32))
        xpos_all.append(data.xpos.copy())
        xquat_all.append(data.xquat.copy())
        cvel_all.append(data.cvel.copy())
        subtree_com_all.append(data.subtree_com.copy())
        site_xpos_all.append(data.site_xpos.copy())
        site_xmat_all.append(data.site_xmat.copy())

    qpos_arr = np.array(qpos_all, dtype=np.float32)
    qvel_arr = np.array(qvel_all, dtype=np.float32)
    for i in range(1, len(qpos_all)):
        qvel_arr[i, 6:] = (qpos_arr[i, 7:] - qpos_arr[i-1, 7:]) / dt

    return (qpos_arr, qvel_arr, np.array(xpos_all), np.array(xquat_all),
            np.array(cvel_all), np.array(subtree_com_all),
            np.array(site_xpos_all), np.array(site_xmat_all))

out_dir = Path("/home/mourad/Desktop/dimos-21days-sprint/robomotion/storage/data/mocap/Tennis/p1")
joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
site_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]

for idx in range(1, 5):
    n = 80 + idx * 10
    qpos, qvel, xpos, xquat, cvel, scom, sxpos, sxmat = gen_swing(n, idx*0.5)
    name = f"Random_00{idx}_Tennis 001"
    path = out_dir / f"{name}.npz"

    np.savez(path,
        # TrajectoryInfo
        joint_names=joint_names, frequency=50.0,
        body_names=body_names, site_names=site_names, metadata=None,
        # TrajectoryModel — ALL fields
        njnt=model.njnt, jnt_type=model.jnt_type.copy(),
        nbody=model.nbody,
        body_rootid=model.body_rootid.copy(),
        body_weldid=model.body_weldid.copy(),
        body_mocapid=model.body_mocapid.copy(),
        body_pos=model.body_pos.copy(),
        body_quat=model.body_quat.copy(),
        body_ipos=model.body_ipos.copy(),
        body_iquat=model.body_iquat.copy(),
        site_bodyid=model.site_bodyid.copy(),
        site_pos=model.site_pos.copy(),
        site_quat=model.site_quat.copy(),
        # TrajectoryData
        qpos=qpos, qvel=qvel, xpos=xpos, xquat=xquat,
        cvel=cvel, subtree_com=scom, site_xpos=sxpos, site_xmat=sxmat,
        split_points=np.array([0, n]),
    )
    print(f"Generated {name}: {n} frames")

print("Done!")
