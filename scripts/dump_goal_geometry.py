"""Dump the goal suite's object and fixture geometry (plan §7.4).

For the four movable objects this mirrors the object suite's file: the bbox
center offset relative to the free-joint origin plus full extents, because the
recorded GT pose is the joint origin and for assets like the bottle that is
NOT the geometric center -- a "3D center error" metric needs the offset.

For the three fixtures it additionally records the task-relevant reference
points in the fixture's own frame: cabinet top surface height and the drawer
handle travel, the rack slot, and the stove burner / knob. Those offsets are
the ones goal_scene.layout_reach_points uses, and they were derived from
source-demo EE positions (scripts/measure_goal_event_ee.py); dumping them from
the compiled model here is an INDEPENDENT second source, and the file records
both so a disagreement is visible instead of averaged away.

Usage:
    .venv\\Scripts\\python.exe scripts\\dump_goal_geometry.py --out output/goal_gen_v3
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from demogen_libero import goal_scene as G
from demogen_libero import oc_obs

from libero.libero import get_libero_path

# body-frame reference points measured from the source demos' EE at the goal
# event (measure_goal_event_ee.py); compared here against the model geometry
MEASURED_OFFSETS = {
    "wooden_cabinet_1_main": {"cabinet_top": (-0.03, 0.05, 1.16),
                              "drawer_handle_closed": (0.0, 0.10, 1.03),
                              "drawer_handle_open": (0.0, 0.28, 1.02)},
    "flat_stove_1_main": {"burner": (0.16, 0.05, 0.96), "knob": (0.0, 0.0, 0.93)},
    "wine_rack_1_main": {"rack_slot": (0.083, 0.0, 1.20)},
}


def geom_world_corners(model, data, gid):
    """8 world-frame AABB corner candidates for one geom (mesh: exact vertex
    bounds; primitives: size-based box). Same implementation the object suite's
    dumper uses -- a cruder "largest half-size as a sphere radius" version
    inflated the wine bottle to 0.21 m across and the rack to 0.57 m."""
    import mujoco
    gtype = model.geom_type[gid]
    xpos = data.geom_xpos[gid]
    xmat = data.geom_xmat[gid].reshape(3, 3)
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gid]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[adr:adr + num]
        return (xmat @ verts.T).T + xpos
    size = model.geom_size[gid]
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        half = np.full(3, size[0])
    elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        half = np.array([size[0], size[0], size[1] + (size[0] if
                         gtype == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)])
    else:
        half = size[:3].copy()
    corners = np.array([[sx, sy, sz] for sx in (-half[0], half[0])
                        for sy in (-half[1], half[1])
                        for sz in (-half[2], half[2])])
    return (xmat @ corners.T).T + xpos


def body_geom_bounds(model, data, body_name):
    """World-frame AABB over every geom attached to a body."""
    bid = model.body_name2id(body_name)
    pts = []
    n = 0
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != bid:
            continue
        pts.append(geom_world_corners(model, data, g))
        n += 1
    if not n:
        return None, None, 0
    allp = np.concatenate(pts, axis=0)
    return allp.min(0), allp.max(0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("output", "goal_gen_v3"))
    args = ap.parse_args()

    env = oc_obs.make_oc_env(os.path.join(
        get_libero_path("bddl_files"), "libero_goal",
        "open_the_middle_drawer_of_the_cabinet.bddl"))
    env.reset()
    m, d = env.sim.model, env.sim.data
    out = {"_convention": {
        "objects": ("offset_body / extents are metres in the body frame; the "
                    "recorded GT pose is the FREE-JOINT origin, which is not "
                    "the geometric centre for every asset"),
        "fixtures": ("offsets are metres from the fixture's base body position "
                     "(the value stored in each demo's fixture_edits); z is "
                     "absolute world height, since the fixtures rest on the "
                     "table at a fixed height"),
    }}

    # --- movable objects: bbox relative to the free-joint origin ---
    for jn in G.GOAL_JOINTS:
        inst = jn.replace("_joint0", "")
        origin = np.asarray(d.get_joint_qpos(jn)[:3])
        lo, hi, n = body_geom_bounds(m, d, f"{inst}_main")
        if lo is None:
            out[inst] = {"error": "no geoms found"}
            continue
        center = (lo + hi) / 2.0
        out[inst] = {"n_geoms": n,
                     "offset_body": np.round(center - origin, 5).tolist(),
                     "extents": np.round(hi - lo, 5).tolist(),
                     "joint_origin_world": np.round(origin, 5).tolist()}
        print(f"  {inst:22s} extents={np.round(hi - lo, 3)} "
              f"offset={np.round(center - origin, 4)}", flush=True)

    # --- fixtures: footprint, height, and the task reference points ---
    print("\nfixtures:", flush=True)
    for fb in G.GOAL_FIXTURES:
        base = np.asarray(d.get_body_xpos(fb))
        lo, hi, n = body_geom_bounds(m, d, fb)
        rec = {"base_world": np.round(base, 5).tolist(), "n_geoms_own_body": n}
        if lo is not None:
            rec["own_body_extents"] = np.round(hi - lo, 5).tolist()
            rec["top_surface_z"] = round(float(hi[2]), 4)
        # child bodies (the cabinet's drawers) with their own AABBs
        children = {}
        for b in range(m.nbody):
            nm = m.body_id2name(b)
            if nm and nm != fb and nm.startswith(fb.replace("_main", "") + "_"):
                clo, chi, cn = body_geom_bounds(m, d, nm)
                if clo is not None:
                    children[nm] = {"n_geoms": cn,
                                    "extents": np.round(chi - clo, 5).tolist(),
                                    "top_z": round(float(chi[2]), 4),
                                    "offset_from_base": np.round(
                                        (clo + chi) / 2.0 - base, 4).tolist()}
        if children:
            rec["child_bodies"] = children
        all_top = ([float(hi[2])] if lo is not None else []) +                   [c["top_z"] for c in children.values()]
        if all_top:
            rec["top_surface_z_incl_children"] = round(max(all_top), 4)
        # measured vs model for the task reference points
        pts = {}
        for name, off in MEASURED_OFFSETS.get(fb, {}).items():
            pts[name] = {"measured_offset_xy": [off[0], off[1]],
                         "measured_z": off[2]}
        # the cabinet's top surface from the model, to compare with the measured
        # place height
        # the cabinet's own body carries no geoms -- its geometry lives in the
        # child bodies -- so the top surface must come from those
        top_z_candidates = [float(hi[2])] if lo is not None else []
        top_z_candidates += [c["top_z"] for c in children.values()]
        if fb == "wooden_cabinet_1_main" and top_z_candidates:
            model_top = max(top_z_candidates)
            meas = MEASURED_OFFSETS[fb]["cabinet_top"][2]
            pts["cabinet_top"]["model_top_surface_z"] = round(model_top, 4)
            pts["cabinet_top"]["measured_minus_model_z"] = round(meas - model_top, 4)
        rec["task_reference_points"] = pts
        out[fb] = rec
        print(f"  {fb.replace('_1_main',''):16s} base={np.round(base, 3)} "
              f"own_extents={np.round(hi - lo, 3) if lo is not None else None} "
              f"children={len(children)}", flush=True)
        for cn_, cv in children.items():
            print(f"      {cn_.split('cabinet_')[-1]:10s} extents="
                  f"{np.round(cv['extents'], 3)} top_z={cv['top_z']}", flush=True)
    env.close()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "object_geometry.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    cab = out.get("wooden_cabinet_1_main", {}).get("task_reference_points", {})
    if "cabinet_top" in cab and "measured_minus_model_z" in cab["cabinet_top"]:
        print(f"\ncabinet top: model surface z={cab['cabinet_top']['model_top_surface_z']}, "
              f"measured place height minus that = "
              f"{cab['cabinet_top']['measured_minus_model_z']:+.3f} m "
              f"(the EE releases above the surface, so a small positive value "
              f"is expected)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
