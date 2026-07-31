"""object_geometry.json for the libero_spatial scenes -- all SEVEN objects.

dump_object_geometry.py resolves each instance through its free joint
(`<inst>_joint0`) and skips anything without one, so the two fixtures the
spatial instructions name -- flat_stove_1, wooden_cabinet_1 -- are silently
absent from the file it writes. precompute_stageb_pack.py needs an entry per
obj_pos row, and the paired dataset records the fixtures as rows 5 and 6, so
that file cannot be used as-is.

Reference frame per row, matching how each row's pose is recorded:
  free objects (0-4)  obj_pos = free-joint origin   -> offset from that origin
  fixtures    (5,6)   obj_pos = body_xpos of the fixture body (oc_obs records
                      `env.sim.data.body_xpos[bid]`) -> offset from the BODY
                      origin, in the body frame
so `world_center = obj_pos + R(obj_quat) @ offset_body` holds for all seven.

The cabinet's subtree contains its sliding drawer, so its bbox depends on the
drawer's state. Every task is measured and the spread reported: a fixture whose
extents move across tasks is telling you its bbox is not a rigid property, and
the reported value is the one at that task's recorded initial state.

Run with the demogen venv:
    .venv\\Scripts\\python.exe scripts\\dump_spatial_object_geometry.py
    .venv\\Scripts\\python.exe scripts\\dump_spatial_object_geometry.py \\
        --out output/libero_spatial_pairs/object_geometry.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

import screen_spatial_pairs as P
from dump_object_geometry import geom_world_corners
from demogen_libero.convert import load_demo
from demogen_libero import libero_replay as R, oc_obs, spatial_scene as S


def subtree_ids(model):
    children = {}
    for b in range(model.nbody):
        children.setdefault(int(model.body_parentid[b]), []).append(b)

    def walk(root):
        out, stack = [], [root]
        while stack:
            b = stack.pop()
            out.append(b)
            stack.extend(c for c in children.get(b, []) if c != b)
        return out
    return walk


def measure_root(model, data, root, walk, exclude=()):
    """bbox of a body's whole subtree, taken IN THAT BODY'S FRAME.

    dump_object_geometry.py takes the world axis-aligned bbox and then rotates
    its centre back, which makes `offset_body` depend on the object's current
    pose: a bowl settled at a tilt on the ramekin grows a taller world AABB, and
    its "body offset" moves with it. Measured across the 10 spatial tasks that
    way, akita_black_bowl_1's z extent swings 2.68 cm while the identical
    akita_black_bowl_2, which stays flat on the table, swings 0.06 cm.

    Transforming the corners into the body frame FIRST gives the pose-invariant
    box, so offset_body is a genuine constant of the asset -- which is what
    `world_center = obj_pos + R(quat) @ offset_body` assumes downstream.
    """
    bodies = set(walk(root))
    for e in exclude:
        bodies -= set(walk(e))
    pts = [geom_world_corners(model, data, gid) for gid in range(model.ngeom)
           if int(model.geom_bodyid[gid]) in bodies]
    if not pts:
        return None
    pts = np.concatenate(pts)
    Rt = data.xmat[root].reshape(3, 3).T
    pts_b = (Rt @ (pts - data.xpos[root]).T).T
    lo, hi = pts_b.min(0), pts_b.max(0)
    return {"offset_body": ((lo + hi) / 2).round(5).tolist(),
            "extents": (hi - lo).round(5).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/spatial_object_geometry.json")
    ap.add_argument("--task", default=None,
                    help="short task name; default = measure all and report spread")
    args = ap.parse_args()
    import mujoco

    cfg = S.load_spatial_config()
    tasks = ([t for t in cfg if P.short(t) == args.task] if args.task else list(cfg))
    assert tasks, args.task

    per_task, env = {}, None
    try:
        for task in tasks:
            d = load_demo(os.path.join(P.DATA_DIR, f"{task}_demo.hdf5"), "demo_0")
            if env is not None:
                env.close()
            env = oc_obs.make_oc_env(d.bddl_file)
            env.reset()
            R.reset_to_init_state(env, d.init_state)
            env.sim.forward()
            model, data = env.sim.model._model, env.sim.data._data
            walk = subtree_ids(model)
            out = {}
            # rows 0-4: free objects, keyed off their free joint
            for inst in cfg[task]["object_order"]:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{inst}_joint0")
                assert jid >= 0, f"{inst} has no free joint in {task}"
                m = measure_root(model, data, int(model.jnt_bodyid[jid]), walk)
                out[oc_obs.display_name(inst)] = m
            # rows 5-6: fixtures, keyed off their BODY (no joint exists).
            # The cabinet's subtree contains the sliding drawer, whose pose is
            # NOT a property of the cabinet -- measured with it in, the cabinet's
            # y extent swings 15.5 cm between the drawer-open task and the other
            # nine. The drawer already has its own recorded fields
            # (drawer_pos / drawer_qpos) and deliberately is not an 8th obj_pos
            # row, so it is excluded here too and the cabinet becomes rigid.
            d_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                      S.DRAWER_BODY)
            for inst in S.FIXTURE_INSTANCES:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                        S.FIXTURE_INSTANCE_BODY[inst])
                assert bid >= 0, f"{inst} body missing in {task}"
                m = measure_root(model, data, int(bid), walk,
                                 exclude=[d_bid] if d_bid >= 0 else ())
                out[oc_obs.display_name(inst)] = m
            per_task[P.short(task)] = out
            print(f"  measured {P.short(task)}", flush=True)
    finally:
        if env is not None:
            env.close()

    names = sorted({n for v in per_task.values() for n in v})
    print(f"\n{'object':26s} {'extents (m)':>26s}   spread across tasks (cm)")
    result = {}
    for n in names:
        ext = np.array([per_task[t][n]["extents"] for t in per_task])
        off = np.array([per_task[t][n]["offset_body"] for t in per_task])
        spread = (ext.max(0) - ext.min(0)) * 100
        # the value written is the median across tasks; anything whose bbox is
        # not a rigid property (the cabinet, whose drawer slides) shows up in
        # the spread column instead of being silently averaged away
        result[n] = {"offset_body": np.median(off, 0).round(5).tolist(),
                     "extents": np.median(ext, 0).round(5).tolist(),
                     "extents_spread_cm": spread.round(2).tolist()}
        print(f"{n:26s} {str(np.round(np.median(ext,0),4).tolist()):>26s}   "
              f"{np.round(spread,2).tolist()}")

    result["_convention"] = (
        "offset_body = bbox center relative to the row's recorded origin, in that "
        "frame; rows 0-4 (free objects) use the free-joint origin, rows 5-6 "
        "(flat stove 1, wooden cabinet 1) use the fixture BODY origin, which is "
        "what oc_obs records into obj_pos for them. "
        "world center = obj_pos + R(obj_quat)@offset_body for all seven. "
        "Values are the median across the 10 spatial tasks; extents_spread_cm "
        "reports the max-min, which is non-zero for the cabinet because its "
        "subtree includes the sliding drawer.")
    result["_per_task"] = per_task
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}  ({len(names)} objects, {len(per_task)} tasks)")


if __name__ == "__main__":
    main()
