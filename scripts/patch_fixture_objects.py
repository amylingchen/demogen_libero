"""Add the stove and the wooden cabinet to already-generated paired demos.

The spatial instructions refer to these two fixtures ("the bowl ON THE STOVE",
"in the top drawer of THE WOODEN CABINET"), but `object_order` lists only the
five free objects, so `build_seg_lut` mapped the fixtures to background: the
recorded data has no mask, no 3D pose and no box for either of them. This adds
them as seg ids 110 (stove) and 120 (cabinet), appended so the existing 60..100
ids are untouched.

Trajectories are NOT re-synthesised. Every frame's full sim state is stored, so
each frame is restored and only re-rendered -- the demos' content is identical,
they just gain two more labelled objects.

Fixture poses are model data (fixtures have no joint), and the per-demo value was
not logged. It is reconstructed instead: jitter moves a relation group rigidly
about its own bowl, so the bowl's frame-0 xy gives the translation and its
quaternion gives the yaw, which together place the fixture exactly. Every demo is
verified before being written -- the re-rendered masks of the five ORIGINAL
objects must match the stored ones (IoU >= --min-iou); a misplaced fixture
changes occlusion and fails that check.

Usage:
    .venv\\Scripts\\python.exe scripts\\patch_fixture_objects.py --task on_the_stove
    .venv\\Scripts\\python.exe scripts\\patch_fixture_objects.py            # all tasks
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import h5py
import numpy as np

import screen_spatial_pairs as P
from run_pair_oc_demo import group_members
from demogen_libero.convert import load_demo
from demogen_libero import libero_replay as R, oc_obs, spatial_scene as S

# These now live in spatial_scene so the generator can record the fixtures as it
# goes instead of this script reconstructing them afterwards; re-exported here
# because build_pair_eval_suite.py imports them from this module.
FIXTURES = S.FIXTURE_INSTANCES                         # segmentation instance names
FIXTURE_BODY = S.FIXTURE_INSTANCE_BODY
DRAWER_BODY = S.DRAWER_BODY
DRAWER_JOINT = S.DRAWER_JOINT


def _yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def fixture_poses(scene, rel, st0, addrs, role):
    """Where the two fixtures stood for THIS demo, recovered from its frame-0
    state (see module docstring)."""
    def read(j):
        a = addrs[j]
        return st0[a:a + 2].copy(), st0[a + 3:a + 7].copy()

    pa, qa = read(P.BOWL_A)
    pb, qb = read(P.BOWL_B)
    if role == "b":                       # undo the target/distractor swap
        pa, qa, pb, qb = pb, qb, pa, qa

    out = {}
    for which, (pos, quat) in (("a", (pa, qa)), ("b", (pb, qb))):
        bowl, _anchors, fixture = group_members(scene, rel, which)
        if not fixture:
            continue
        pivot = np.array(scene["free"][bowl]["xy"])
        th = _yaw(quat) - _yaw(np.asarray(scene["free"][bowl]["quat"]))
        d = pos - pivot
        v = np.array(scene["fixtures"][fixture]["xy"]) - pivot
        c, s = np.cos(th), np.sin(th)
        out[fixture] = {
            "pos": np.array([pivot[0] + c * v[0] - s * v[1] + d[0],
                             pivot[1] + s * v[0] + c * v[1] + d[1],
                             scene["fixtures"][fixture]["z"]]),
            "quat": S._quat_mul(S._yaw_to_quat(th),
                                np.asarray(scene["fixtures"][fixture]["quat"])),
        }
    for fb, v in scene["fixtures"].items():           # unowned fixtures never moved
        if fb not in out:
            out[fb] = {"pos": np.array([v["xy"][0], v["xy"][1], v["z"]]),
                       "quat": np.asarray(v["quat"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/libero_spatial_pairs/train")
    ap.add_argument("--plan", default="output/pair_split.json")
    ap.add_argument("--task", default=None, help="short task name; default = all")
    ap.add_argument("--min-iou", type=float, default=0.95,
                    help="re-rendered vs stored mask agreement required on the five "
                         "original objects before a demo is written")
    ap.add_argument("--min-area", type=int, default=50,
                    help="ignore an object in the IoU check when the stored mask has "
                         "fewer than this many pixels in that frame")
    ap.add_argument("--limit", type=int, default=0, help="0 = every demo")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    plan = json.load(open(args.plan))
    cfg_all = S.load_spatial_config()
    rel = P.relations_from_config(cfg_all)
    import mujoco

    dirs = sorted(d for d in os.listdir(args.root)
                  if os.path.isdir(os.path.join(args.root, d))
                  and (args.task is None or d == args.task))
    grand = [0, 0]
    for short in dirs:
        d = os.path.join(args.root, short)
        h5p = glob.glob(os.path.join(d, "*_demo.hdf5"))
        log_p = os.path.join(d, "scene_log.json")
        if not h5p or not os.path.exists(log_p):
            continue
        h5p = h5p[0]
        log = {r["demo_name"]: r for r in json.load(open(log_p))}
        task = next(iter(log.values()))["task"]
        old_order = cfg_all[task]["object_order"]
        new_order = old_order + FIXTURES

        demo0 = load_demo(os.path.join(P.DATA_DIR, f"{task}_demo.hdf5"), "demo_0")
        env = oc_obs.make_oc_env(demo0.bddl_file)
        env.reset()
        R.reset_to_init_state(env, demo0.init_state)
        lut_old = oc_obs.build_seg_lut(env, old_order)
        lut_new = oc_obs.build_seg_lut(env, new_order)
        lay = S.read_layout(env, demo0.init_state, P.FREE_JOINTS, S.ALL_FIXTURES)
        addrs = {j: lay["free"][j]["addr"] for j in P.FREE_JOINTS}
        bids = {f: env.sim.model.body_name2id(FIXTURE_BODY[f]) for f in FIXTURES}
        mdl = env.sim.model._model
        d_bid = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_BODY, DRAWER_BODY)
        d_jid = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_JOINT, DRAWER_JOINT)
        d_adr = int(mdl.jnt_qposadr[d_jid]) if d_jid >= 0 else None

        with h5py.File(h5p, "r") as f:
            names = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        if args.limit:
            names = names[:args.limit]

        n_ok = n_bad = n_fast = 0
        for dn in names:
            with h5py.File(h5p, "r") as f:
                g = f["data"][dn]
                seg_done = json.loads(g.attrs["object_instances"]) == new_order
                drawer_done = "drawer_pos" in g["obs"]
                if seg_done and drawer_done:
                    n_ok += 1
                    continue
                states = np.array(g["states"])
                stored_exo = np.array(g["obs"]["agentview_seg"])
                op = np.array(g["obs"]["obj_pos"])
                oq = np.array(g["obs"]["obj_quat"])
            row = log.get(dn)
            if row is None:
                n_bad += 1
                continue
            scene = plan["scenes"][row["scene_index"]]
            fx = fixture_poses(scene, rel, states[0], addrs, row["target_role"])

            if seg_done and not drawer_done:
                # segmentation is already correct; the drawer fields need only
                # the sim state, no rendering, so this path is ~50x cheaper
                T = states.shape[0]
                dpos = np.zeros((T, 3)); dq = np.zeros((T, 1))
                R.reset_to_init_state(env, states[0])
                S.apply_fixture_edits(env, fx)
                for t in range(T):
                    env.sim.set_state_from_flattened(states[t])
                    env.sim.forward()
                    dpos[t] = env.sim.data.body_xpos[d_bid]
                    dq[t, 0] = env.sim.data.qpos[d_adr] if d_adr is not None else 0.0
                with h5py.File(h5p, "a") as f:
                    o = f["data"][dn]["obs"]
                    o.create_dataset("drawer_pos", data=dpos)
                    o.create_dataset("drawer_qpos", data=dq)
                n_ok += 1
                n_fast += 1
                continue

            T = states.shape[0]
            exo = np.zeros_like(stored_exo)
            ego = np.zeros((T, *stored_exo.shape[1:]), dtype=np.uint8)
            fpos = np.zeros((T, len(FIXTURES), 3), dtype=np.float64)
            fquat = np.zeros((T, len(FIXTURES), 4), dtype=np.float64)
            dpos = np.zeros((T, 3), dtype=np.float64)
            dqpos = np.zeros((T, 1), dtype=np.float64)
            iou_min = 1.0
            # reset (and therefore the model rebuild that wipes body_pos) happens
            # ONCE per demo; per-frame we only write qpos/qvel, which is what makes
            # re-rendering 70k frames feasible at all
            R.reset_to_init_state(env, states[0])
            S.apply_fixture_edits(env, fx)
            for t in range(T):
                env.sim.set_state_from_flattened(states[t])
                env.sim.forward()
                obs = env.env._get_observations(force_update=True)
                a = lut_new[np.clip(obs["agentview_segmentation_instance"][..., 0],
                                    0, len(lut_new) - 1)][::-1]
                e = lut_new[np.clip(
                    obs["robot0_eye_in_hand_segmentation_instance"][..., 0],
                    0, len(lut_new) - 1)][::-1]
                exo[t], ego[t] = a, e
                for i, fb in enumerate(FIXTURES):
                    fpos[t, i] = env.sim.data.body_xpos[bids[fb]]
                    q = env.sim.data.body_xquat[bids[fb]]        # wxyz
                    fquat[t, i] = [q[1], q[2], q[3], q[0]]       # -> xyzw, as obj_quat
                dpos[t] = env.sim.data.body_xpos[d_bid]
                dqpos[t, 0] = env.sim.data.qpos[d_adr] if d_adr is not None else 0.0
                if t % max(T // 8, 1) == 0:                      # spot-check frames
                    for sid in (60, 70, 80, 90, 100):
                        x, y = stored_exo[t] == sid, a == sid
                        # an object the arm has almost fully covered occupies a
                        # handful of pixels, where IoU swings wildly on a 1-pixel
                        # difference and says nothing about the fixture pose
                        if x.sum() < args.min_area:
                            continue
                        u = (x | y).sum()
                        if u:
                            iou_min = min(iou_min, (x & y).sum() / u)
            if iou_min < args.min_iou:
                print(f"  {short}/{dn}: mask mismatch (IoU {iou_min:.3f}) -- skipped",
                      flush=True)
                n_bad += 1
                continue

            with h5py.File(h5p, "a") as f:
                g = f["data"][dn]
                g["obs"]["agentview_seg"][...] = exo
                g["obs"]["eye_in_hand_seg"][...] = ego
                for key, old, new in (("obj_pos", op, fpos), ("obj_quat", oq, fquat)):
                    del g["obs"][key]
                    g["obs"].create_dataset(key, data=np.concatenate([old, new], axis=1))
                for key, arr in (("drawer_pos", dpos), ("drawer_qpos", dqpos)):
                    if key in g["obs"]:
                        del g["obs"][key]
                    g["obs"].create_dataset(key, data=arr)
                g.attrs["object_instances"] = json.dumps(new_order)
                g.attrs["object_names"] = json.dumps(
                    [oc_obs.display_name(x) for x in new_order])
            n_ok += 1
            if n_ok % 10 == 0:
                print(f"  {short}: {n_ok}/{len(names)}", flush=True)
        env.close()

        meta_p = os.path.join(d, "metainfo.json")
        if os.path.exists(meta_p):
            meta = json.load(open(meta_p))
            names_disp = [oc_obs.display_name(x) for x in new_order]
            with h5py.File(h5p, "r") as f:
                for dn, entry in meta.get(task, {}).items():
                    if dn not in f["data"] or not isinstance(entry, dict):
                        continue
                    g = f["data"][dn]
                    entry["object_names"] = names_disp
                    entry["exo_boxes"] = oc_obs.boxes_for_seg(
                        np.array(g["obs"]["agentview_seg"]), names_disp)
                    entry["ego_boxes"] = oc_obs.boxes_for_seg(
                        np.array(g["obs"]["eye_in_hand_seg"]), names_disp)
            json.dump(meta, open(meta_p, "w"), indent=2)

        grand[0] += n_ok
        grand[1] += n_bad
        print(f"[{short}] patched {n_ok} ({n_fast} drawer-only), skipped {n_bad}",
              flush=True)
    print(f"\nTOTAL patched {grand[0]}, skipped {grand[1]}")


if __name__ == "__main__":
    main()
