"""Feasibility screen for TWO-RELATION (paired-task) spatial scenes.

A paired scene puts BOTH black bowls into meaningful, mutually-exclusive spatial
relations at once -- e.g. bowl A on the stove, bowl B next to the plate -- so the
instruction is the only thing that says which bowl to pick. The two bowls are
visually identical, so one geometry yields TWO episodes by swapping which bowl
is `akita_black_bowl_1` (the joint every BDDL goal and the synthesis pipeline
are keyed on).

This script does NOT generate demos and does NOT modify the production pipeline.

WHAT DECIDES WHETHER A SCENE IS VALID

Geometry is only a cheap prefilter here. Bounding volumes cannot represent these
assets: the stove's burner is a raised ring whose bounding box swallows the bowl
that rests inside it, so any box/disc/slab model either rejects the real LIBERO
"bowl on the stove" scene or is loosened until it stops detecting real overlaps.
So the authorities are measurements, each calibrated on real source scenes:

  collision     MuJoCo contact penetration depth after writing the state
  stability     object displacement after settling (a scene that explodes is not
                a usable init state, however clean it looks geometrically)
  visibility    rendered segmentation area per object

Every tolerance below is read off the distribution of the same quantity over
LIBERO's own 500 source init states, not chosen. Phase 0 additionally runs a
FALSIFICATION test -- deliberately overlapped object pairs, which the collision
check must reject -- because a control made only of valid scenes cannot tell a
working checker from one that always says yes.

Usage:
    .venv\\Scripts\\python.exe scripts\\screen_spatial_pairs.py --n 40
    .venv\\Scripts\\python.exe scripts\\screen_spatial_pairs.py --n 40 --mosaics
    .venv\\Scripts\\python.exe scripts\\screen_spatial_pairs.py --exclusivity global
    # add the on-vs-beside shared-anchor pairs to an existing screen
    .venv\\Scripts\\python.exe scripts\\screen_spatial_pairs.py --n 40 \\
        --shared-anchor only --merge
"""
import argparse
import itertools
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np

from demogen_libero.convert import resolve_bddl_file
from demogen_libero import libero_replay as R
from demogen_libero import oc_obs
from demogen_libero import spatial_scene as S

DATA_DIR = "D:/Data/LingLing/libero/hf/libero_spatial"

FREE_INST = ["akita_black_bowl_1", "akita_black_bowl_2", "plate_1", "cookies_1",
             "glazed_rim_porcelain_ramekin_1"]
FREE_JOINTS = [f"{i}_joint0" for i in FREE_INST]
BOWL_A, BOWL_B = "akita_black_bowl_1_joint0", "akita_black_bowl_2_joint0"
PLATE = "plate_1_joint0"
DRAWER_TASK = ("pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet"
               "_and_place_it_on_the_plate")
DRAWER_JOINT = "wooden_cabinet_1_top_level"
HOST_TASK = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"


def short(task):
    return (task.replace("pick_up_the_black_bowl_", "")
                .replace("_and_place_it_on_the_plate", ""))


def _yaw_of(quat):
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _rot(v, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


# ------------------------------------------------------------- sim measurement

def _subtree(model, root):
    children = {}
    for b in range(model.nbody):
        children.setdefault(int(model.body_parentid[b]), []).append(b)
    out, stack = [], [root]
    while stack:
        b = stack.pop()
        out.append(b)
        stack.extend(c for c in children.get(b, []) if c != b)
    return set(out)


def _geom_world_pts(model, data, gid):
    import mujoco
    gtype = model.geom_type[gid]
    xpos, xmat = data.geom_xpos[gid], data.geom_xmat[gid].reshape(3, 3)
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gid]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        return (xmat @ model.mesh_vert[adr:adr + num].T).T + xpos
    size = model.geom_size[gid]
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        half = np.full(3, size[0])
    elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        half = np.array([size[0], size[0],
                         size[1] + (size[0] if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)])
    else:
        half = size[:3].copy()
    corners = np.array([[sx, sy, sz] for sx in (-half[0], half[0])
                        for sy in (-half[1], half[1]) for sz in (-half[2], half[2])])
    return (xmat @ corners.T).T + xpos


def measure_scene_geometry(env):
    """Per scene item: the radius of the smallest disc about its BODY ORIGIN that
    contains its footprint, plus its z extent, plus the set of geom ids that
    belong to it. The disc is used only as a lenient prefilter -- MuJoCo contacts
    decide collisions -- so it is deliberately a simple, conservative shape.
    Also returns the table's measured xy extent."""
    import mujoco
    model, data = env.sim.model._model, env.sim.data._data
    items, geom_owner = {}, {}

    def record(key, root):
        bodies = _subtree(model, root)
        gids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies]
        pts = np.concatenate([_geom_world_pts(model, data, g) for g in gids])
        origin = data.xpos[root].copy()
        rot = data.xmat[root].reshape(3, 3)
        local = (rot.T @ (pts - origin).T).T          # body frame
        lo, hi = local[:, :2].min(0), local[:, :2].max(0)
        # the disc about the footprint's own CENTRE, not about the body origin:
        # the stove's origin sits 0.097 m outside its footprint centre, so an
        # origin-centred disc reads 0.263 m where the true half-diagonal is 0.176
        ctr = (lo + hi) / 2.0
        items[key] = {
            "off": ctr.tolist(),
            # tightest circumscribed radius about the footprint centre. The AABB
            # half-diagonal would inflate a round object by sqrt(2) -- for the
            # bowl, 0.056 -> 0.079 -- which is enough to veto every scene where
            # two bowls share one anchor and therefore sit a fixed 0.117 m apart.
            "r": float(np.max(np.linalg.norm(local[:, :2] - ctr, axis=1))),
            "r_box": float(np.linalg.norm((hi - lo) / 2.0)),
            "r_origin": float(np.max(np.linalg.norm(local[:, :2], axis=1))),
            "dz_lo": float(local[:, 2].min()), "dz_hi": float(local[:, 2].max()),
        }
        for g in gids:
            geom_owner[g] = key

    for inst in FREE_INST:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{inst}_joint0")
        record(f"{inst}_joint0", int(model.jnt_bodyid[jid]))
    for fb in S.ALL_FIXTURES:
        record(fb, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fb))

    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    tpts = np.concatenate([_geom_world_pts(model, data, g) for g in range(model.ngeom)
                           if int(model.geom_bodyid[g]) == tb])
    table = {"lo": tpts[:, :2].min(0).tolist(), "hi": tpts[:, :2].max(0).tolist(),
             "top_z": float(tpts[:, 2].max())}
    return items, geom_owner, table


def scene_contacts(env, geom_owner):
    """Deepest penetration between any two SCENE items in the current sim state.
    MuJoCo's own contact solver is the collision authority: it uses the actual
    collision meshes, so it is not fooled by the burner ring or the plate rim
    that defeat every bounding-volume model of these assets."""
    data = env.sim.data._data
    worst, pair = 0.0, None
    for i in range(data.ncon):
        c = data.contact[i]
        a, b = geom_owner.get(int(c.geom1)), geom_owner.get(int(c.geom2))
        if a is None or b is None or a == b:
            continue
        depth = -float(c.dist)
        if depth > worst:
            worst, pair = depth, (a, b)
    return worst, pair


# ------------------------------------------------------ relations and layouts

def relations_from_config(cfg_all):
    """Anchors per task, taken from tasks_config_spatial.json so this screen can
    never drift from the generator's own notion of a task's anchors."""
    return {t: {"anchors": list(c["group_joints"]), "fixture": c["group_fixture"],
                "kind": c["anchor_kind"]}
            for t, c in cfg_all.items()}


def build_layout_cache(cache_path, tasks, n_demos):
    """Per task: each source demo's free-object and fixture poses, plus the
    drawer joint value. Cached, but the cache records how many demos it holds and
    is rebuilt when fewer than requested -- a silently short cache would make the
    control claim a sample size it does not have."""
    if os.path.exists(cache_path):
        blob = json.load(open(cache_path))
        if blob.get("n_demos", 0) >= n_demos and set(blob.get("tasks", [])) >= set(tasks):
            return {t: v[:n_demos] for t, v in blob["cache"].items()}
        print(f"  layout cache holds {blob.get('n_demos', 0)} demos/task "
              f"(need {n_demos}) -- rebuilding", flush=True)
    import mujoco
    cache = {}
    for task in tasks:
        with h5py.File(os.path.join(DATA_DIR, f"{task}_demo.hdf5"), "r") as f:
            bddl = resolve_bddl_file(f["data"].attrs["bddl_file_name"])
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:n_demos]
            states = [np.array(f["data"][k]["states"][0], dtype=np.float64) for k in keys]
        env = R.make_env(bddl)
        env.reset()
        m = env.sim.model._model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, DRAWER_JOINT)
        dadr = int(m.jnt_qposadr[jid]) if jid >= 0 else None
        demos = []
        for st in states:
            lay = S.read_layout(env, st, FREE_JOINTS, S.ALL_FIXTURES)
            demos.append({
                "free": {j: {"pos": lay["free"][j]["pos"].tolist(),
                             "quat": lay["free"][j]["quat"].tolist()} for j in FREE_JOINTS},
                "fixtures": {f: {"pos": lay["fixtures"][f]["pos"].tolist(),
                                 "quat": lay["fixtures"][f]["quat"].tolist()}
                             for f in S.ALL_FIXTURES},
                "drawer": float(st[1 + dadr]) if dadr is not None else 0.0,
            })
        env.close()
        cache[task] = demos
        print(f"  layout cache: {short(task)} ({len(demos)} demos)", flush=True)
    json.dump({"n_demos": n_demos, "tasks": list(tasks), "cache": cache},
              open(cache_path, "w"))
    return cache


def measure_relation_points(cache, rel):
    """Where each task's relation actually holds, in its ANCHOR's body frame:
    the mean position the target bowl occupies when that task's instruction is
    true. Keyed by TASK, not by anchor -- the wooden cabinet anchors two tasks
    whose naming points ("on the cabinet top" and "in the open drawer") are
    0.14 m apart, so one entry per fixture would silently keep only one."""
    pts = {}
    for task, demos in cache.items():
        r = rel[task]
        keys = list(r["anchors"]) + ([r["fixture"]] if r["fixture"] else [])
        per = {}
        for k in keys:
            offs = []
            for d in demos:
                b = np.array(d["free"][BOWL_A]["pos"][:2])
                src = d["fixtures"][k] if k in d["fixtures"] else d["free"][k]
                offs.append(_rot(b - np.array(src["pos"][:2]), -_yaw_of(src["quat"])))
            per[k] = np.mean(offs, axis=0)
        pts[task] = per
    return pts


def measure_relation_ranges(cache, rel):
    """Planar distance from the target bowl to each of its own anchors over every
    source demo: what "the relation holds" looks like in the data. Exclusion
    thresholds are compared against this rather than guessed."""
    out = {}
    for task, demos in cache.items():
        r = rel[task]
        keys = list(r["anchors"]) + ([r["fixture"]] if r["fixture"] else [])
        per = {}
        for k in keys:
            ds = [float(np.linalg.norm(
                np.array(d["free"][BOWL_A]["pos"][:2]) -
                np.array((d["fixtures"][k] if k in d["fixtures"] else d["free"][k])["pos"][:2])))
                for d in demos]
            per[k] = {"min": min(ds), "med": float(np.median(ds)), "max": max(ds)}
        if r["kind"] == "absolute":
            p = np.array([d["free"][BOWL_A]["pos"][:2] for d in demos])
            per["__center__"] = {"min": 0.0, "med": 0.0,
                                 "max": float(np.max(np.linalg.norm(p - p.mean(0), axis=1))),
                                 "xy": p.mean(0).tolist()}
        out[task] = per
    return out


# --------------------------------------------------------------------- spec

class Spec:
    """Sampling bounds for a PAIRED scene. Workspace, place zone and yaw limits
    are inherited from the single-task SpatialSpec so the two screens stay
    comparable; the tolerances that decide validity are measured in phase 0."""

    def __init__(self, exclusion=0.24, margin=0.01, exclusivity="pair",
                 fixture_band=None, max_yaw_object=None, reverse_plate=True):
        s = S.SpatialSpec()
        self.x_range, self.y_range = s.x_range, s.y_range
        self.dest_x, self.dest_y = s.dest_x, s.dest_y
        self.indep_x_range, self.indep_y_range = s.indep_x_range, s.indep_y_range
        self.center_x, self.center_y = s.center_x, s.center_y
        # LIBERO's source scenes lay every object-anchored relation out along the
        # x axis (the between-plate-and-ramekin line spans only 26 deg of
        # direction across all 50 demos), and the single-task +-30 deg group yaw
        # keeps that bias. These relations are rotation-invariant -- a bowl
        # between two objects is still between them when the line runs along y --
        # so the limit is exposed as a knob.
        self.max_yaw_object = (np.deg2rad(max_yaw_object) if max_yaw_object is not None
                               else s.max_yaw_object)
        self.max_yaw_fixture = s.max_yaw_fixture
        self.center_keepout = s.center_keepout
        self.group_tries, self.tries = s.group_tries, s.indep_tries
        # the single-task bands keep the low stove forward so it stays visible; in
        # a paired scene that band runs through the middle of the table, so it is
        # exposed as a knob instead of assumed
        self.fixture_y_range = ({fb: tuple(fixture_band) for fb in S.ALL_FIXTURES}
                                if fixture_band else dict(S.FIXTURE_Y_RANGE))
        self.margin = margin          # prefilter slack only; the sim decides collisions
        self.exclusion = exclusion
        self.exclusivity = exclusivity
        # When a relation names the plate ("next to the plate", "between the
        # plate and the ramekin") the plate rides with the group, so positioning
        # the group by its BOWL leaves the plate to land where it may -- and it
        # must land in the 0.18 x 0.16 m reachable zone. Drawing the bowl from
        # the 0.40 x 0.50 m workspace makes that a 1-6% event, which is where
        # 89-92% of those pairs' rejected draws go. Sampling the PLATE inside the
        # place zone and deriving the bowl from the same rigid offset explores
        # the same feasible set from the small side instead.
        self.reverse_plate = reverse_plate
        self.table_lo = np.array([-0.5, -0.6])   # replaced by the measured table
        self.table_hi = np.array([0.5, 0.6])

    def set_table(self, table):
        self.table_lo = np.array(table["lo"])
        self.table_hi = np.array(table["hi"])


# ----------------------------------------------------------- scene assembly

def place_group(demo_layout, rel, bowl_joint, bowl_xy, yaw):
    """Rigid-transplant one task's relation onto a chosen bowl: the group's
    internal pose is copied verbatim from a source demo (so the relation is one a
    human actually demonstrated), then the whole group is translated so the bowl
    lands on `bowl_xy` and rotated by `yaw` about it. Each member keeps its own
    z, so a stacked bowl stays stacked."""
    yq = S._yaw_to_quat(yaw)
    src_bowl = np.array(demo_layout["free"][BOWL_A]["pos"])
    src_bq = np.array(demo_layout["free"][BOWL_A]["quat"])
    out = {bowl_joint: {"xy": np.asarray(bowl_xy, float), "z": float(src_bowl[2]),
                        "yaw": _yaw_of(src_bq) + yaw, "quat": S._quat_mul(yq, src_bq),
                        "is_fixture": False, "shape": BOWL_A}}
    members = [(a, False) for a in rel["anchors"]]
    if rel["fixture"]:
        members.append((rel["fixture"], True))
    for key, is_fix in members:
        src = demo_layout["fixtures"][key] if is_fix else demo_layout["free"][key]
        p, q = np.array(src["pos"]), np.array(src["quat"])
        out[key] = {"xy": np.asarray(bowl_xy, float) + _rot(p[:2] - src_bowl[:2], yaw),
                    "z": float(p[2]), "yaw": _yaw_of(q) + yaw,
                    "quat": S._quat_mul(yq, q), "is_fixture": is_fix, "shape": key}
    return out


def _footprint(it, geom):
    g = geom[it["shape"]]
    return it["xy"] + _rot(np.asarray(g["off"]), it["yaw"]), g["r"]


def place_shared_group(layA, layB, relA, relB, shared, anchor_xy, yaw):
    """Both relations name the SAME anchor ("on the cookie box" + "next to the
    cookie box"), so that anchor must be placed ONCE and both bowls hung off it.
    Placing the two groups independently would put two copies of the anchor in
    different spots, and whichever was written last would win.

    Each bowl keeps its own source offset from the anchor and its own z, so both
    relations stay exactly as demonstrated: one bowl concentric and raised, the
    other beside it on the table."""
    yq = S._yaw_to_quat(yaw)
    out = {}
    key = shared[0]
    src = layA["free"][key]
    out[key] = {"xy": np.asarray(anchor_xy, float), "z": float(src["pos"][2]),
                "yaw": _yaw_of(np.asarray(src["quat"])) + yaw,
                "quat": S._quat_mul(yq, np.asarray(src["quat"])),
                "is_fixture": False, "shape": key}
    for lay, bowl in ((layA, BOWL_A), (layB, BOWL_B)):
        b = np.array(lay["free"][BOWL_A]["pos"])
        a = np.array(lay["free"][key]["pos"])
        q = np.asarray(lay["free"][BOWL_A]["quat"])
        out[bowl] = {"xy": np.asarray(anchor_xy, float) + _rot(b[:2] - a[:2], yaw),
                     "z": float(b[2]), "yaw": _yaw_of(q) + yaw,
                     "quat": S._quat_mul(yq, q), "is_fixture": False,
                     "shape": BOWL_A}
    return out


def prefilter_ok(placed, geom, spec):
    """Lenient disc test used only to skip hopeless draws cheaply. It must NOT be
    the arbiter: two objects whose discs overlap can still be a perfectly legal
    stack (a bowl on a ramekin is concentric), so only pairs that overlap in
    height AND are interpenetrating well past their footprint discs are dropped
    here. MuJoCo decides the rest."""
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            a, b = placed[i], placed[j]
            if a["gid"] == b["gid"] and a["gid"] >= 0:
                continue
            ga, gb = geom[a["shape"]], geom[b["shape"]]
            if min(a["z"] + ga["dz_hi"], b["z"] + gb["dz_hi"]) <= \
               max(a["z"] + ga["dz_lo"], b["z"] + gb["dz_lo"]):
                continue
            ca, ra = _footprint(a, geom)
            cb, rb = _footprint(b, geom)
            if float(np.linalg.norm(ca - cb)) < (ra + rb) * 0.75 + spec.margin:
                return False, (f"prefilter:{a['key'].split('_joint')[0]}"
                               f"|{b['key'].split('_joint')[0]}")
    return True, None


def check_on_table(placed, geom, spec):
    for it in placed:
        c, r = _footprint(it, geom)
        if np.any(c - r < spec.table_lo) or np.any(c + r > spec.table_hi):
            return False, f"off_table:{it['key'].split('_joint')[0]}"
    return True, None


def check_exclusivity(bowls, naming, own, spec, center_xy, center_bowls):
    """Neither bowl may match the instruction meant for the other.

      "pair"   (default) only the two paired relations must be mutually
               exclusive; those are the only instructions ever issued against
               the scene.
      "global" no bowl may sit near ANY anchor its own relation does not name,
               so the scene is unambiguous under all ten instructions.

    Both are planar-distance rules against each relation's NAMING POINT (the
    burner, not the stove's origin). They cannot express relations that share an
    xy position but differ in height or containment -- see the note that phase 1
    prints for the cabinet-top / in-drawer pair.
    """
    for bj, bxy in bowls.items():
        others = [o for o in bowls if o != bj]
        if spec.exclusivity == "global":
            foreign = [k for k in naming if k not in own[bj]]
        else:
            foreign = [k for o in others for k in own[o]
                       if k not in own[bj] and k in naming]
        for key in foreign:
            for pt in naming[key]:
                if np.linalg.norm(np.asarray(bxy) - pt) < spec.exclusion:
                    return False, f"ambiguous:{bj.split('_joint')[0]}~{key.split('_joint')[0]}"
        needs_center = (spec.exclusivity == "global"
                        or any(o in center_bowls for o in others))
        if bj not in center_bowls and needs_center and \
                np.linalg.norm(np.asarray(bxy) - center_xy) < spec.center_keepout:
            return False, f"ambiguous:{bj.split('_joint')[0]}~table_center"
    return True, None


def bowl_ranges(rel_task, spec):
    if rel_task["kind"] == "absolute":
        return spec.center_x, spec.center_y, spec.max_yaw_object
    if rel_task["fixture"]:
        return (spec.x_range,
                spec.fixture_y_range.get(rel_task["fixture"], (-0.12, 0.06)),
                spec.max_yaw_fixture)
    return spec.x_range, spec.y_range, spec.max_yaw_object


def sample_pair_geometry(rng, taskA, taskB, cache, rel, geom, spec, rel_points,
                         demo_a, demo_b, stats):
    """Draw one candidate paired geometry. Both groups are drawn together rather
    than one being frozen first: freezing group A and then discovering that group
    B's stove has nowhere to go makes a pair look infeasible when it is only
    mis-ordered. Returns (scene|None, reason)."""
    layA, layB = cache[taskA][demo_a], cache[taskB][demo_b]
    relA, relB = rel[taskA], rel[taskB]

    owned = set(relA["anchors"]) | set(relB["anchors"])
    for r in (relA, relB):
        if r["fixture"]:
            owned.add(r["fixture"])
    stationary = [{"key": fb, "xy": np.array(layA["fixtures"][fb]["pos"][:2]),
                   "z": float(layA["fixtures"][fb]["pos"][2]),
                   "yaw": _yaw_of(layA["fixtures"][fb]["quat"]),
                   "quat": np.array(layA["fixtures"][fb]["quat"]),
                   "is_fixture": True, "shape": fb, "gid": -2}
                  for fb in S.ALL_FIXTURES if fb not in owned]
    for st in stationary:
        st["key"] = st["key"]

    center_xy = np.array([np.mean(spec.center_x), np.mean(spec.center_y)])
    center_bowls = {b for b, r in ((BOWL_A, relA), (BOWL_B, relB))
                    if r["kind"] == "absolute"}

    def naming_points(groups, tasks, indep=None):
        """Every point an instruction could name. A stationary fixture carries the
        naming points of EVERY task anchored to it, since any of those
        instructions could be read against the scene."""
        pts = {}
        for g, tk in zip(groups, tasks):
            for k, m in g.items():
                if k in (BOWL_A, BOWL_B):
                    continue
                off = rel_points.get(tk, {}).get(k)
                pts.setdefault(k, []).append(
                    m["xy"] + (_rot(off, m["yaw"]) if off is not None else 0.0))
        for st in stationary:
            offs = [rel_points[t][st["key"]] for t in rel_points
                    if st["key"] in rel_points[t]] or [np.zeros(2)]
            pts.setdefault(st["key"], []).extend(
                st["xy"] + _rot(o, st["yaw"]) for o in offs)
        for k, v in (indep or {}).items():
            pts.setdefault(k, []).append(np.asarray(v, float))
        return pts

    def as_items(groups, indep=None):
        out = []
        for gid, g in enumerate(groups):
            for k, m in g.items():
                out.append({**m, "key": k, "gid": gid})
        out.extend(stationary)
        for k, m in (indep or {}).items():
            out.append({**m, "key": k, "gid": -1})
        return out

    xa, ya, wa = bowl_ranges(relA, spec)
    xb, yb, wb = bowl_ranges(relB, spec)

    def draw(lay, r, bowl_joint, xr, yr, wmax):
        """One group pose. If the group owns the destination plate, the plate is
        drawn inside the place zone and the bowl derived from it; the bowl is
        then required to land in the reachable workspace, so the feasible set is
        identical to forward sampling, only reached far more often."""
        yaw = rng.uniform(-wmax, wmax)
        if spec.reverse_plate and PLATE in r["anchors"]:
            want = np.array([rng.uniform(*spec.dest_x), rng.uniform(*spec.dest_y)])
            off = _rot(np.array(lay["free"][PLATE]["pos"][:2])
                       - np.array(lay["free"][BOWL_A]["pos"][:2]), yaw)
            bxy = want - off
            if not (spec.x_range[0] <= bxy[0] <= spec.x_range[1]
                    and spec.y_range[0] <= bxy[1] <= spec.y_range[1]):
                return None
        else:
            bxy = np.array([rng.uniform(*xr), rng.uniform(*yr)])
        return place_group(lay, r, bowl_joint, bxy, yaw)

    shared = [k for k in relA["anchors"] if k in relB["anchors"]]

    gA = gB = None
    why = "groups"
    for _ in range(spec.group_tries):
        stats["draws"] += 1
        if shared:
            # one anchor, two relations: place it once, hang both bowls off it
            g = place_shared_group(layA, layB, relA, relB, shared,
                                   (rng.uniform(*spec.x_range),
                                    rng.uniform(*spec.y_range)),
                                   rng.uniform(-wa, wa))
            # the draw positions the ANCHOR, but the "beside" bowl hangs 0.117-
            # 0.126 m off it, so it can land outside the target workspace. Left
            # unchecked its checkerboard cell would be silently CLAMPED to an edge
            # cell (Board.cell clips), contaminating the seen/unseen label -- and
            # it would sit outside the reach envelope the rest of the pipeline
            # assumes. Require both bowls inside instead.
            if any(not (spec.x_range[0] <= g[b]["xy"][0] <= spec.x_range[1]
                        and spec.y_range[0] <= g[b]["xy"][1] <= spec.y_range[1])
                   for b in (BOWL_A, BOWL_B)):
                why = "bowl_unreachable"
                stats[f"bind/{why}"] += 1
                continue
            cA = {k: v for k, v in g.items() if k != BOWL_B}
            cB = {BOWL_B: g[BOWL_B]}
        else:
            cA = draw(layA, relA, BOWL_A, xa, ya, wa)
            cB = draw(layB, relB, BOWL_B, xb, yb, wb)
        if cA is None or cB is None:
            why = "bowl_unreachable"
            stats[f"bind/{why}"] += 1
            continue
        items = as_items([cA, cB])
        ok, w = check_on_table(items, geom, spec)
        # when the plate rides with a relation group its reachability is decided
        # by the group's pose, so it has to be tested here where the sampler can
        # still redraw -- testing it after the loop only reports the failure
        if ok and (PLATE in cA or PLATE in cB):
            pxy = (cA.get(PLATE) or cB.get(PLATE))["xy"]
            if not (spec.dest_x[0] <= pxy[0] <= spec.dest_x[1] and
                    spec.dest_y[0] <= pxy[1] <= spec.dest_y[1]):
                ok, w = False, "plate_unreachable"
        if ok:
            ok, w = prefilter_ok(items, geom, spec)
        if ok:
            ok, w = check_exclusivity(
                {BOWL_A: cA[BOWL_A]["xy"], BOWL_B: cB[BOWL_B]["xy"]},
                naming_points([cA, cB], [taskA, taskB]),
                {BOWL_A: set(cA) | set(shared), BOWL_B: set(cB) | set(shared)},
                spec, center_xy, center_bowls)
        if ok:
            gA, gB = cA, cB
            stats["draw_accept"] += 1
            break
        why = w
        stats[f"bind/{w}"] += 1
    if gA is None:
        return None, f"groups/{why}"

    # a shared anchor belongs to BOTH relations, so neither bowl may be judged
    # "ambiguously close" to it -- that is the whole point of the pair
    own = {BOWL_A: set(gA) | set(shared), BOWL_B: set(gB) | set(shared)}
    bowls = {BOWL_A: gA[BOWL_A]["xy"], BOWL_B: gB[BOWL_B]["xy"]}
    indep_keys = [j for j in FREE_JOINTS if j not in set(gA) | set(gB)]
    indep, placed = {}, as_items([gA, gB])
    for jn in ([PLATE] if PLATE in indep_keys else []) + \
              [j for j in indep_keys if j != PLATE]:
        xr, yr = ((spec.dest_x, spec.dest_y) if jn == PLATE
                  else (spec.indep_x_range, spec.indep_y_range))
        got = None
        for _ in range(spec.tries):
            xy = np.array([rng.uniform(*xr), rng.uniform(*yr)])
            yaw = rng.uniform(-np.pi, np.pi)
            cand = {"key": jn, "gid": -1, "xy": xy, "z": S._rest_z(jn) + 0.01,
                    "yaw": yaw, "is_fixture": False, "shape": jn,
                    "quat": S._quat_mul(S._yaw_to_quat(yaw),
                                        np.array(layA["free"][jn]["quat"]))}
            ok, _ = prefilter_ok(placed + [cand], geom, spec)
            if not ok:
                continue
            # a loose plate / cookie box / ramekin is itself an anchor some task
            # names, so under "global" it must clear both bowls; under "pair"
            # only the two paired instructions matter and a loose object is in
            # neither of them
            if spec.exclusivity == "global" and any(
                    np.linalg.norm(xy - b) < spec.exclusion for b in bowls.values()):
                continue
            got = cand
            break
        if got is None:
            return None, f"indep:{jn.split('_joint')[0]}"
        indep[jn] = got
        placed.append(got)

    # the destination plate must be reachable wherever it ended up -- it is a
    # group member for two of the tasks, and skipping the test in that case is
    # what lets a scene be scored feasible while the plate sits out of reach
    plate_xy = (indep[PLATE]["xy"] if PLATE in indep
                else (gA.get(PLATE) or gB.get(PLATE))["xy"])
    if not (spec.dest_x[0] <= plate_xy[0] <= spec.dest_x[1] and
            spec.dest_y[0] <= plate_xy[1] <= spec.dest_y[1]):
        return None, "plate_unreachable"

    ok, w = check_exclusivity(
        bowls, naming_points([gA, gB], [taskA, taskB],
                             {k: v["xy"] for k, v in indep.items()}),
        own, spec, center_xy, center_bowls)
    if not ok:
        return None, w

    return {"gA": gA, "gB": gB, "indep": indep, "stationary": stationary,
            "taskA": taskA, "taskB": taskB, "demo_a": demo_a, "demo_b": demo_b,
            "drawer": max(cache[taskA][demo_a]["drawer"],
                          cache[taskB][demo_b]["drawer"], key=abs)}, None


# ---------------------------------------------------------- sim-side checking

def write_scene(scene, host_init, addrs, nq, drawer_adr):
    state = np.asarray(host_init, dtype=np.float64).copy()
    fx = {}

    def put(jn, m):
        a = addrs[jn]
        state[a], state[a + 1], state[a + 2] = m["xy"][0], m["xy"][1], m["z"]
        q = np.asarray(m["quat"], float)
        state[a + 3:a + 7] = q / max(np.linalg.norm(q), 1e-9)

    for g in (scene["gA"], scene["gB"]):
        for key, m in g.items():
            if m["is_fixture"]:
                fx[key] = {"pos": np.array([m["xy"][0], m["xy"][1], m["z"]]),
                           "quat": np.asarray(m["quat"], float)}
            else:
                put(key, m)
    for jn, m in scene["indep"].items():
        put(jn, m)
    for st in scene["stationary"]:
        fx[st["key"]] = {"pos": np.array([st["xy"][0], st["xy"][1], st["z"]]),
                         "quat": np.asarray(st["quat"], float)}
    # the in-drawer relation is only meaningful with the drawer actually open;
    # the host scene has it shut, and nothing else in the pipeline opens it
    if drawer_adr is not None:
        state[1 + drawer_adr] = scene.get("drawer", 0.0)
    state[1 + nq:] = 0.0
    return state, fx


def write_source(demo, host_init, addrs, nq, drawer_adr):
    state = np.asarray(host_init, dtype=np.float64).copy()
    for jn in FREE_JOINTS:
        a, src = addrs[jn], demo["free"][jn]
        state[a:a + 3] = src["pos"]
        q = np.asarray(src["quat"], float)
        state[a + 3:a + 7] = q / max(np.linalg.norm(q), 1e-9)
    if drawer_adr is not None:
        state[1 + drawer_adr] = demo.get("drawer", 0.0)
    state[1 + nq:] = 0.0
    fx = {fb: {"pos": np.asarray(demo["fixtures"][fb]["pos"], float),
               "quat": np.asarray(demo["fixtures"][fb]["quat"], float)}
          for fb in S.ALL_FIXTURES}
    return state, fx


def _anchor_xy(env, key):
    if key in S.ALL_FIXTURES:
        bid = env.sim.model.body_name2id(key)
        return env.sim.model.body_pos[bid][:2].copy()
    return env.sim.data.get_joint_qpos(key)[:2].copy()


def evaluate_state(env, state, fx, geom_owner, lut, seg_ids, settle_steps,
                   claims=(), want_image=False):
    """Measure everything that decides whether an init state is usable.

    `claims` are the relations the scene is supposed to express, as
    (bowl_joint, anchor_key, d_lo, d_hi, z_gap). They are re-measured AFTER
    settling, because that is the question that actually matters: a raw
    displacement threshold cannot distinguish a bowl re-seating harmlessly on
    its ramekin (19 mm in LIBERO's own scenes) from a bowl sliding off it, and
    only the second one breaks the instruction.
    """
    R.reset_to_init_state(env, state)
    S.apply_fixture_edits(env, fx)
    env.sim.forward()
    depth, pair = scene_contacts(env, geom_owner)
    before = {j: env.sim.data.get_joint_qpos(j)[:3].copy() for j in FREE_JOINTS}
    for _ in range(settle_steps):
        env.sim.step()
    moved = max(float(np.linalg.norm(env.sim.data.get_joint_qpos(j)[:3] - before[j]))
                for j in FREE_JOINTS)

    broken, worst_slack = None, 0.0
    for bowl, anchor, d_lo, d_hi, z_gap in claims:
        b = env.sim.data.get_joint_qpos(bowl)[:3]
        a_xy = _anchor_xy(env, anchor)
        d = float(np.linalg.norm(b[:2] - a_xy))
        slack = max(d_lo - d, d - d_hi)
        if z_gap is not None:
            a_z = (env.sim.model.body_pos[env.sim.model.body_name2id(anchor)][2]
                   if anchor in S.ALL_FIXTURES
                   else env.sim.data.get_joint_qpos(anchor)[2])
            slack = max(slack, abs((b[2] - a_z) - z_gap))
        if slack > worst_slack:
            worst_slack, broken = slack, (bowl.split("_joint")[0],
                                          anchor.split("_joint")[0])

    obs = env.env._get_observations()
    raw = obs["agentview_segmentation_instance"][..., 0]
    segids = lut[np.clip(raw, 0, len(lut) - 1)]
    px = min(int((segids == sid).sum()) for sid in seg_ids.values())
    img = (env.sim.render(camera_name="agentview", height=256, width=256)[::-1]
           if want_image else None)
    return {"depth": depth, "pair": pair, "moved": moved, "min_px": px,
            "relation_slack": worst_slack, "broken": broken}, img


def claims_for(task_pairs, rel, ranges, cache):
    """The relations a scene asserts, with the distance band each one holds in
    across the source demos and the bowl-to-anchor height gap."""
    out = []
    for bowl, task, demo_i in task_pairs:
        demo = cache[task][demo_i]
        r = rel[task]
        for k in list(r["anchors"]) + ([r["fixture"]] if r["fixture"] else []):
            src = demo["fixtures"][k] if k in demo["fixtures"] else demo["free"][k]
            z_gap = float(demo["free"][BOWL_A]["pos"][2] - src["pos"][2])
            rg = ranges[task][k]
            out.append((bowl, k, rg["min"], rg["max"], z_gap))
    return out


def mosaic(imgs, cols=6):
    rows = (len(imgs) + cols - 1) // cols
    h, w, _ = imgs[0].shape
    canvas = np.full((rows * h, cols * w, 3), 40, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return canvas


# ------------------------------------------------------ phase 1: compatibility

CONCENTRIC = 0.05      # |naming-point offset| below this = the bowl sits ON the anchor


def stacked_vs_beside(taskA, taskB, rel, rel_points, shared):
    """True when the two relations name the SAME anchor but in different ways:
    one puts the bowl ON it (concentric, measured 0.000-0.011 m offset) and the
    other BESIDE it (0.117-0.123 m). Planar naming-point distance calls these
    ambiguous, but "the bowl on the cookie box" and "the bowl next to the cookie
    box" are plainly different objects -- the discriminator is support, not
    distance, and the two bowls still end up ~0.11 m apart.

    It also has to be a pair `place_shared_group` can actually EXPRESS, which is
    narrower than "the relations differ". That routine places ONE anchor and
    hangs both bowls off it, so two conditions are structural, not stylistic:

      no fixture   a fixture has no free joint, so there is nothing to write the
                   shared pose into; two independently drawn groups would each
                   carry their own copy of the stove/cabinet and whichever was
                   written last would silently win. This is what keeps
                   on_the_wooden_cabinet + in_the_top_drawer out, even though its
                   naming points (0.009 m vs 0.144 m) pass the support test.
      nothing else in either anchor set
                   an anchor named by only ONE of the two relations is not part
                   of the shared group, so it would be scattered as an
                   independent object and that relation would simply be false --
                   which is what would happen to the plate in
                   between_the_plate_and_the_ramekin + on_the_ramekin.
    """
    a, b = rel[taskA], rel[taskB]
    if a["fixture"] or b["fixture"]:
        return False
    if set(a["anchors"]) != set(shared) or set(b["anchors"]) != set(shared):
        return False
    for k in shared:
        a_pt, b_pt = rel_points[taskA].get(k), rel_points[taskB].get(k)
        if a_pt is None or b_pt is None:
            return False
        na, nb = float(np.linalg.norm(a_pt)), float(np.linalg.norm(b_pt))
        if not ((na < CONCENTRIC) ^ (nb < CONCENTRIC)):
            return False
    return True


def compat(taskA, taskB, rel, rel_points, exclusion, allow_shared_anchor=False):
    """Can these two relations coexist? Two relations conflict when their NAMING
    POINTS -- the spot each instruction actually refers to -- are closer than the
    exclusion radius, which is a stronger and more accurate test than asking
    whether they share an anchor object: the cabinet anchors both 'on the wooden
    cabinet' and 'in the top drawer', two clearly different places."""
    a, b = rel[taskA], rel[taskB]
    sa = set(a["anchors"]) | ({a["fixture"]} if a["fixture"] else set())
    sb = set(b["anchors"]) | ({b["fixture"]} if b["fixture"] else set())
    shared = sa & sb
    if not shared:
        return "candidate", []
    gaps = {}
    for k in shared:
        pa, pb = rel_points[taskA].get(k), rel_points[taskB].get(k)
        if pa is None or pb is None:
            return "shared_anchor", sorted(shared)
        gaps[k] = float(np.linalg.norm(pa - pb))
    if min(gaps.values()) < exclusion:
        if allow_shared_anchor and stacked_vs_beside(taskA, taskB, rel, rel_points, shared):
            return "stacked_vs_beside", [f"{k.split('_joint')[0]}:{v:.3f}m"
                                         for k, v in gaps.items()]
        return "shared_anchor", [f"{k.split('_joint')[0]}:{v:.3f}m" for k, v in gaps.items()]
    # distinct places on one anchor: feasible in principle, but both relations
    # would have to share a single placement of that anchor, which the two
    # independent groups sampled here cannot express
    return "shared_anchor_distinct", [f"{k.split('_joint')[0]}:{v:.3f}m"
                                      for k, v in gaps.items()]


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="scene attempts per pair")
    ap.add_argument("--source-demos", type=int, default=50)
    ap.add_argument("--exclusion", type=float, default=0.24)
    ap.add_argument("--exclusivity", choices=["pair", "global"], default="pair")
    ap.add_argument("--fixture-band", type=float, nargs=2, default=None,
                    metavar=("Y_MIN", "Y_MAX"))
    ap.add_argument("--settle-steps", type=int, default=200,
                    help="physics steps used to test that a scene is stable")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="only screen pairs whose short names contain all of these "
                         "substrings (e.g. --pairs from_table_center on_the_stove)")
    ap.add_argument("--shared-anchor", choices=["off", "include", "only"], default="off",
                    help="'off' (default) keeps every pair whose two relations name "
                         "the same anchor out, as the planar exclusion rule decides. "
                         "'include' also admits ON-vs-BESIDE pairs, where one relation "
                         "puts the bowl on the shared anchor and the other beside it "
                         "(next_to_the_ramekin + on_the_ramekin, on_the_cookie_box + "
                         "next_to_the_cookie_box) -- the discriminator there is support, "
                         "not planar distance. 'only' screens just those pairs, for "
                         "adding them to an existing screen with --merge")
    ap.add_argument("--merge", action="store_true",
                    help="update the matrix and results already in the output json "
                         "instead of replacing them, so a run restricted to a few "
                         "pairs does not discard the rest of the screen")
    ap.add_argument("--mosaics", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="output/spatial_pairs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    cfg_all = S.load_spatial_config()
    tasks = list(cfg_all.keys())
    rel = relations_from_config(cfg_all)

    print("=" * 96)
    print("PHASE 0  calibration, control, and falsification")
    print("=" * 96)
    cache = build_layout_cache(os.path.join(args.out_dir, "layout_cache.json"),
                               tasks, args.source_demos)
    n_src = min(len(v) for v in cache.values())

    import mujoco
    with h5py.File(os.path.join(DATA_DIR, f"{HOST_TASK}_demo.hdf5"), "r") as f:
        host_bddl = resolve_bddl_file(f["data"].attrs["bddl_file_name"])
        host_init = np.array(f["data"]["demo_0"]["states"][0], dtype=np.float64)
    env = oc_obs.make_oc_env(host_bddl)
    env.reset()
    R.reset_to_init_state(env, host_init)
    geom, geom_owner, table = measure_scene_geometry(env)
    m = env.sim.model._model
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, DRAWER_JOINT)
    drawer_adr = int(m.jnt_qposadr[jid]) if jid >= 0 else None
    host_layout = S.read_layout(env, host_init, FREE_JOINTS, S.ALL_FIXTURES)
    addrs = {j: host_layout["free"][j]["addr"] for j in FREE_JOINTS}
    nq = env.sim.model.nq
    obj_order = cfg_all[HOST_TASK]["object_order"]
    lut = oc_obs.build_seg_lut(env, obj_order)
    seg_ids = {inst: 60 + 10 * obj_order.index(inst) for inst in FREE_INST}

    print(f"\nmeasured prefilter discs (about each body origin) vs the hand-tuned "
          f"constants in spatial_scene.py:")
    print(f"  {'item':<34}{'r':>8}{'origin offset':>16}{'z extent':>18}{'code r':>9}")
    for k, g in geom.items():
        base = k.replace("_joint0", "").rsplit("_", 1)[0]
        code = S.OBJECT_RADIUS.get(base, S.FIXTURE_RADIUS.get(k))
        zext = f"{g['dz_lo']:+.4f}..{g['dz_hi']:+.4f}"
        code_txt = f"{code:.3f}" if code else "-"
        print(f"  {k:<34}{g['r']:>8.4f}{str(np.round(g['off'], 4)):>16}"
              f"{zext:>18}{code_txt:>9}")
    print(f"  table xy extent (measured): lo={np.round(table['lo'],3)} "
          f"hi={np.round(table['hi'],3)} top_z={table['top_z']:.3f}")

    rel_points = measure_relation_points(cache, rel)
    print("\nrelation naming points (offset from the anchor's origin, body frame):")
    for task in tasks:
        bits = ", ".join(f"{k.split('_joint')[0][:22]}=[{v[0]:+.3f},{v[1]:+.3f}]"
                         f" |{np.linalg.norm(v):.3f}|" for k, v in rel_points[task].items())
        print(f"  {short(task):<40}{bits or '(table centre)'}")

    ranges = measure_relation_ranges(cache, rel)
    max_sat = max(v["max"] for per in ranges.values() for k, v in per.items()
                  if k != "__center__")
    print(f"\nwidest distance at which any relation still holds: {max_sat:.3f} m")
    print(f"exclusion threshold under test: {args.exclusion:.2f} m "
          f"(margin over the widest: {args.exclusion - max_sat:+.3f} m)")

    spec = Spec(exclusion=args.exclusion, exclusivity=args.exclusivity,
                fixture_band=args.fixture_band)
    spec.set_table(table)

    # ---------- control: what do REAL LIBERO init states measure? ----------
    print(f"\nCONTROL: the same measurements on {n_src * len(tasks)} real LIBERO "
          f"source init states")
    ctrl = {"depth": [], "moved": [], "min_px": [], "slack": []}
    for task, demos in cache.items():
        for i, demo in enumerate(demos):
            st, fx = write_source(demo, host_init, addrs, nq, drawer_adr)
            r, _ = evaluate_state(env, st, fx, geom_owner, lut, seg_ids,
                                  args.settle_steps,
                                  claims=claims_for([(BOWL_A, task, i)], rel, ranges, cache))
            for k, v in (("depth", r["depth"]), ("moved", r["moved"]),
                         ("min_px", r["min_px"]), ("slack", r["relation_slack"])):
                ctrl[k].append(v)
    dep, mov, pxs, slk = (np.array(ctrl[k]) for k in ("depth", "moved", "min_px", "slack"))
    print(f"  contact penetration      med={np.median(dep):.5f}  max={dep.max():.5f} m")
    print(f"  settle displacement      med={np.median(mov):.5f}  max={mov.max():.5f} m "
          f"({args.settle_steps} steps)")
    print(f"  relation drift on settle med={np.median(slk):.5f}  max={slk.max():.5f} m "
          f"(how far the task's own relation moves out of its source band)")
    print(f"  min rendered object      med={int(np.median(pxs))}  min={int(pxs.min())} px")
    tol_depth = float(max(dep.max(), 5e-4) * 1.5)
    tol_moved = float(max(mov.max(), 5e-3) * 3.0)     # explosion guard only
    tol_slack = float(max(slk.max(), 5e-3) * 1.5)
    tol_px = float(max(np.percentile(pxs, 5) * 0.5, 50))
    print(f"  -> penetration <= {tol_depth:.5f} m, relation drift <= {tol_slack:.5f} m, "
          f"displacement <= {tol_moved:.5f} m, area >= {tol_px:.0f} px")
    print(f"     (displacement is only an explosion guard -- LIBERO's own "
          f"'bowl on the ramekin' scenes re-seat by {mov.max():.3f} m and are fine,")
    print(f"      so relation drift, not raw motion, is what decides usability)")

    # ---------- falsification: the checks must be able to say no ----------
    print("\nFALSIFICATION: configurations that MUST be rejected")
    print("  (a control made only of valid scenes cannot tell a working check")
    print("   from one that always says yes)")
    base_demo = cache[HOST_TASK][0]
    base_claims = claims_for([(BOWL_A, HOST_TASK, 0)], rel, ranges, cache)

    def probe(label, mutate):
        st, fx = write_source(base_demo, host_init, addrs, nq, drawer_adr)
        mutate(st, fx)
        r, _ = evaluate_state(env, st, fx, geom_owner, lut, seg_ids,
                              args.settle_steps, claims=base_claims)
        hit = (r["depth"] > tol_depth or r["moved"] > tol_moved
               or r["relation_slack"] > tol_slack)
        print(f"    {'REJECTED' if hit else '** MISSED **':<14}{label:<44}"
              f"pen={r['depth']:.5f} drift={r['relation_slack']:.5f} "
              f"moved={r['moved']:.5f}")
        return hit

    def at_fixture(jn, fb):
        def f(st, fx):
            a = addrs[jn]
            st[a], st[a + 1] = fx[fb]["pos"][0], fx[fb]["pos"][1]
        return f

    def onto(dst, src):
        def f(st, fx):
            a, b = addrs[dst], addrs[src]
            st[a], st[a + 1] = st[b], st[b + 1]      # same xy, each keeps its own z
        return f

    probes = [
        ("bowl_1 inside the stove body", at_fixture(BOWL_A, "flat_stove_1_main")),
        ("bowl_2 inside the cabinet body", at_fixture(BOWL_B, "wooden_cabinet_1_main")),
        ("cookie box on top of the ramekin",
         onto("cookies_1_joint0", "glazed_rim_porcelain_ramekin_1_joint0")),
        ("ramekin on top of the cookie box",
         onto("glazed_rim_porcelain_ramekin_1_joint0", "cookies_1_joint0")),
        ("bowl_2 on top of bowl_1", onto(BOWL_B, BOWL_A)),
        ("plate under the ramekin", onto(PLATE, "glazed_rim_porcelain_ramekin_1_joint0")),
    ]
    caught = sum(probe(lbl, fn) for lbl, fn in probes)
    print(f"  invalid configurations rejected: {caught}/{len(probes)}")
    if caught < len(probes):
        print("  !! the collision/stability check is NOT sound; yields below are void")

    # a bowl that starts on the destination plate makes the episode already
    # solved -- a distinct failure from any collision, so it is tested separately
    def on_plate(st, fx):
        a, p = addrs[BOWL_B], addrs[PLATE]
        st[a], st[a + 1] = st[p], st[p + 1]
        st[a + 2] = st[p + 2] + 0.02
    st, fx = write_source(base_demo, host_init, addrs, nq, drawer_adr)
    on_plate(st, fx)
    r, _ = evaluate_state(env, st, fx, geom_owner, lut, seg_ids, args.settle_steps,
                          claims=base_claims)
    print(f"    {'(separate)':<14}{'bowl already resting on the plate':<44}"
          f"pen={r['depth']:.5f} drift={r['relation_slack']:.5f} "
          f"moved={r['moved']:.5f}")
    print(f"      -> not a collision; caught instead by requiring every bowl to stay "
          f">= {args.exclusion:.2f} m")
    print(f"         from the plate unless its own relation names the plate")

    # ------------------------------------------------- phase 1
    print("\n" + "=" * 96)
    print("PHASE 1  compatibility matrix")
    print("=" * 96)
    allow_shared = args.shared_anchor != "off"
    keep = {"off": ("candidate",),
            "include": ("candidate", "stacked_vs_beside"),
            "only": ("stacked_vs_beside",)}[args.shared_anchor]
    names = [short(t) for t in tasks]
    w = max(len(n) for n in names) + 1
    print(" " * w + "".join(f"{n[:7]:>9}" for n in names))
    matrix, notes, shared_notes = {}, [], []
    for i, ta in enumerate(tasks):
        row = f"{names[i]:<{w}}"
        for j, tb in enumerate(tasks):
            if i == j:
                row += f"{'.':>9}"
                continue
            verdict, detail = compat(ta, tb, rel, rel_points, args.exclusion, allow_shared)
            matrix[f"{names[i]}|{names[j]}"] = {"verdict": verdict, "detail": detail}
            mark = {"candidate": "ok", "shared_anchor": "X",
                    "shared_anchor_distinct": "X*", "stacked_vs_beside": "S"}[verdict]
            row += f"{mark:>9}"
            if verdict == "shared_anchor_distinct" and i < j:
                notes.append((names[i], names[j], detail))
            if verdict == "stacked_vs_beside" and i < j:
                shared_notes.append((names[i], names[j], detail))
        print(row)
    print("\n  ok = the two relations name places further apart than the exclusion radius")
    print("  X  = the two instructions would point at effectively the same place")
    print("  X* = same anchor, DIFFERENT places on it -- feasible in principle, but")
    print("       both relations must share one placement of that anchor, which the")
    print("       two independently-placed groups sampled here cannot express:")
    for a, b, d in notes:
        print(f"         {a} + {b}   naming points {d}")
    if allow_shared:
        print("  S  = ON vs BESIDE the same anchor: the anchor is placed ONCE and both")
        print("       bowls hang off it, so what separates the two instructions is")
        print("       support, not planar distance. Admitted only when the shared anchor")
        print("       is a FREE object and is the ONLY anchor either relation names:")
        for a, b, d in shared_notes:
            print(f"         {a} + {b}   naming points {d}")
        if not shared_notes:
            print("         (none)")

    pairs = [(a, b) for a, b in itertools.combinations(tasks, 2)
             if compat(a, b, rel, rel_points, args.exclusion,
                       allow_shared)[0] in keep]
    if args.pairs:
        pairs = [(a, b) for a, b in pairs
                 if all(s in f"{short(a)}|{short(b)}" for s in args.pairs)]
    print(f"\n  {len(pairs)} candidate pairs out of "
          f"{len(tasks) * (len(tasks) - 1) // 2}")

    # ------------------------------------------------- phase 2
    print("\n" + "=" * 96)
    print(f"PHASE 2  end-to-end yield ({args.n} attempts/pair, "
          f"exclusivity={args.exclusivity}, exclusion={args.exclusion:.2f}, "
          f"fixture y-band={args.fixture_band or 'single-task default'})")
    print("=" * 96)
    print(f"{'pair':<44}{'usable':>7}{'draw':>7}   "
          f"attrition geom>pen>rel>px    dominant loss")
    results = {}
    for ta, tb in pairs:
        rng = np.random.default_rng(args.seed)
        stats = Counter()
        n_geom = n_pen = n_rel = n_px = 0
        imgs, reasons = [], Counter()
        for _ in range(args.n):
            da, db = int(rng.integers(len(cache[ta]))), int(rng.integers(len(cache[tb])))
            scene, why = sample_pair_geometry(
                rng, ta, tb, cache, rel, geom, spec, rel_points, da, db, stats)
            if scene is None:
                reasons[why] += 1
                continue
            n_geom += 1
            st, fx = write_scene(scene, host_init, addrs, nq, drawer_adr)
            claims = claims_for([(BOWL_A, ta, da), (BOWL_B, tb, db)], rel, ranges, cache)
            r, img = evaluate_state(env, st, fx, geom_owner, lut, seg_ids,
                                    args.settle_steps, claims=claims,
                                    want_image=args.mosaics and len(imgs) < 12)
            if r["depth"] > tol_depth:
                who = r["pair"][0].split("_joint")[0] if r["pair"] else "?"
                reasons[f"penetration:{who}"] += 1
                continue
            n_pen += 1
            if r["relation_slack"] > tol_slack or r["moved"] > tol_moved:
                who = "|".join(r["broken"]) if r["broken"] else "explode"
                reasons[f"relation_broken:{who}"] += 1
                continue
            n_rel += 1
            if r["min_px"] < tol_px:
                reasons["occluded_rendered"] += 1
                continue
            n_px += 1
            if img is not None:
                imgs.append(img)
        dom = reasons.most_common(1)[0] if reasons else ("-", 0)
        p_draw = stats["draw_accept"] / max(stats["draws"], 1)
        key = f"{short(ta)} + {short(tb)}"
        print(f"{key[:43]:<44}{100 * n_px / args.n:6.0f}%{100 * p_draw:6.2f}%   "
              f"{args.n:>3}>{n_geom:>3}>{n_pen:>3}>{n_rel:>3}>{n_px:>3}   "
              f"{dom[0][:34]} ({dom[1]})")
        results[key] = {"usable_rate": n_px / args.n, "draw_accept": p_draw,
                        "attrition": {"attempts": args.n, "geometry": n_geom,
                                      "penetration": n_pen, "relation": n_rel,
                                      "rendered": n_px},
                        "n_attempts": args.n, "failures": dict(reasons),
                        "binding": {k[5:]: v for k, v in stats.items()
                                    if k.startswith("bind/")}}
        if args.mosaics and imgs:
            from PIL import Image
            Image.fromarray(mosaic(imgs)).save(
                os.path.join(args.out_dir, f"{short(ta)}__{short(tb)}.png"))

    env.close()
    out = {"geometry": geom, "table": table,
           "relation_points": {t: {k: v.tolist() for k, v in p.items()}
                               for t, p in rel_points.items()},
           "relation_ranges": ranges,
           "control": {k: list(map(float, v)) for k, v in ctrl.items()},
           "tolerances": {"depth": tol_depth, "moved": tol_moved,
                          "slack": tol_slack, "min_px": tol_px},
           "falsification": {"caught": caught, "probes": len(probes),
                             "labels": [lbl for lbl, _ in probes]},
           "matrix": matrix, "results": results,
           "params": vars(args), "n_source_demos": n_src}
    path = os.path.join(args.out_dir, "pair_screen.json")
    if args.merge and os.path.exists(path):
        prev = json.load(open(path))
        kept = {k: v for k, v in prev.get("results", {}).items() if k not in results}
        out["results"] = {**kept, **results}
        out["matrix"] = {**prev.get("matrix", {}), **matrix}
        out["merged_over"] = {"n_results_kept": len(kept),
                              "previous_params": prev.get("params")}
        print(f"\nmerged: kept {len(kept)} previously screened pairs, "
              f"{'updated' if set(results) & set(prev.get('results', {})) else 'added'} "
              f"{len(results)}")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
