"""Dump each task object's physical bounding box relative to its free-joint
origin (body frame): bbox center offset + full extents, computed from mesh
vertices / primitive geom sizes at env.reset().

The OC dataset's GT object pose is the free-joint origin, which for assets
like the basket is NOT the geometric center — evaluation of "3D center error"
needs this offset. Run with the demogen venv:
    .venv\\Scripts\\python.exe scripts\\dump_object_geometry.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from demogen_libero.convert import load_demo
from demogen_libero import oc_obs

from demogen_libero.config import DATA_DIR as SOURCE_BASE_DIR


def geom_world_corners(model, data, gid):
    """8 world-frame AABB corner candidates for one geom (mesh: exact vertex
    bounds; primitives: size-based box)."""
    import mujoco
    gtype = model.geom_type[gid]
    xpos = data.geom_xpos[gid]
    xmat = data.geom_xmat[gid].reshape(3, 3)
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gid]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[adr:adr + num]          # geom frame
        return (xmat @ verts.T).T + xpos
    size = model.geom_size[gid]
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        half = np.full(3, size[0])
    elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        half = np.array([size[0], size[0], size[1] + (size[0] if
                         gtype == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)])
    else:                                               # box & fallback
        half = size[:3].copy()
    corners = np.array([[sx, sy, sz] for sx in (-half[0], half[0])
                        for sy in (-half[1], half[1])
                        for sz in (-half[2], half[2])])
    return (xmat @ corners.T).T + xpos


def measure_task(env, object_instances):
    import mujoco

    model, data = env.sim.model._model, env.sim.data._data
    children = {}
    for b in range(model.nbody):
        children.setdefault(int(model.body_parentid[b]), []).append(b)

    def subtree(root):
        out, stack = [], [root]
        while stack:
            b = stack.pop()
            out.append(b)
            stack.extend(c for c in children.get(b, []) if c != b)
        return out

    result = {}
    for inst in object_instances:
        joint = f"{inst}_joint0"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0:
            continue
        root = int(model.jnt_bodyid[jid])
        bodies = set(subtree(root))
        pts = [geom_world_corners(model, data, gid) for gid in range(model.ngeom)
               if int(model.geom_bodyid[gid]) in bodies]
        pts = np.concatenate(pts)
        lo, hi = pts.min(0), pts.max(0)
        center_w = (lo + hi) / 2
        origin_w = data.xpos[root].copy()
        offset_body = data.xmat[root].reshape(3, 3).T @ (center_w - origin_w)
        # key by display name ("salad dressing 1") for consistency with metainfo
        result[oc_obs.display_name(inst)] = {
            "offset_body": offset_body.round(5).tolist(),
            "extents": (hi - lo).round(5).tolist(),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None,
                        help="single task (short name); default = all tasks (full pool)")
    parser.add_argument("--source-base-dir", type=str, default=SOURCE_BASE_DIR)
    parser.add_argument("--out", type=str, default="output/object_geometry.json")
    args = parser.parse_args()

    cfg = oc_obs.load_task_config()
    if args.task:
        tk = args.task if args.task.startswith("pick_up") else f"pick_up_the_{args.task}_and_place_it_in_the_basket"
        tasks = [tk]
    else:
        tasks = list(cfg.keys())

    result = {}
    env = None
    try:
        for task in tasks:
            d = load_demo(os.path.join(args.source_base_dir, f"{task}_demo.hdf5"), "demo_0")
            if env is not None:
                env.close()
            env = oc_obs.make_oc_env(d.bddl_file)
            env.reset()
            result.update(measure_task(env, cfg[task]["object_order"]))
    finally:
        if env is not None:
            env.close()

    result["_convention"] = ("offset_body = bbox center relative to free-joint origin in body frame; "
                             "extents = full bbox size; world center = obj_pos + R(quat)@offset_body")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    n = len([k for k in result if not k.startswith("_")])
    print(f"wrote {args.out}  ({n} objects)")
    for k, v in result.items():
        if not k.startswith("_"):
            print(f"  {k:18s} offset_body={v['offset_body']} extents={v['extents']}")


if __name__ == "__main__":
    main()
