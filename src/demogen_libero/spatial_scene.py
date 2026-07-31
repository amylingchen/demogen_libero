"""Scene placement for LIBERO_Spatial tasks.

Unlike libero_object (free target + distractors + basket on a flat table), a
spatial task's LANGUAGE *is* a spatial relation ("the black bowl on the stove",
"between the plate and the ramekin", "next to the ramekin"). Randomly relocating
the target would make the instruction false. So we augment with a RELATIONAL
RIGID GROUP:

    group = {target bowl} u {anchor objects} u {anchor fixture}

The whole group is moved by one rigid transform (planar translation + yaw about
the group centroid), so the internal relation -- hence the instruction -- is
preserved exactly while the group travels across the table. Each member keeps
its own z (a bowl stacked on the ramekin stays stacked; a bowl on the moved
stove stays on the stove). Free objects move via their free-joint qpos; a
fixture anchor (stove/cabinet, which has no free joint) moves via model.body_pos
+ body_quat. Objects outside the group are placed independently on the table
(occlusion- and spacing-constrained); the identical twin bowl additionally must
stay clear of the anchor so it can't satisfy the same relation (anti-ambiguity).

The synthesis downstream only needs the target's net displacement (obj_t) and
the destination plate's net displacement (tar_t); everything else here is just
setting up a consistent, fully visible init scene.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .gridscene import _occludes, object_state_slices

_CONFIG_PATH = Path(__file__).with_name("tasks_config_spatial.json")


def load_spatial_config() -> dict:
    cfg = json.loads(_CONFIG_PATH.read_text())
    cfg.pop("_schema", None)
    return cfg


# canonical table-rest heights (m) for placing independent objects flat on the
# main table (measured from the from_table_center source init state)
TABLE_REST_Z = {
    "akita_black_bowl": 0.898,
    "plate": 0.902,
    "cookies": 0.909,
    "glazed_rim_porcelain_ramekin": 0.899,
}


def _rest_z(joint_name: str) -> float:
    base = joint_name.replace("_joint0", "").rsplit("_", 1)[0]
    return TABLE_REST_Z.get(base, 0.90)


# approximate horizontal footprint radius (m) per object type, used for
# radius-aware non-overlap so the large plate never clips a bowl/stove (mesh geom
# sizes under-report, so these are visual estimates)
OBJECT_RADIUS = {
    "akita_black_bowl": 0.075,
    "plate": 0.10,
    "glazed_rim_porcelain_ramekin": 0.06,
    "cookies": 0.05,
}
# effective keep-out radius per fixture body (platform half-width; the offset
# burner / cabinet top the target sits on is handled as a group member)
FIXTURE_RADIUS = {"flat_stove_1_main": 0.12, "wooden_cabinet_1_main": 0.13}
SPACING_MARGIN = 0.03

# per-fixture target y-band: the low, flat stove is easily cut off / hidden at
# the back so it is kept in a forward band; the tall cabinet (bowl rests on its
# elevated top, always visible) can stay in its native back region
FIXTURE_Y_RANGE = {"flat_stove_1_main": (-0.12, 0.06),
                   "wooden_cabinet_1_main": (-0.26, 0.00)}

# how wide a fixture blocks the agentview camera. The cabinet is a TALL, wide box
# (plus its pulled-out drawer) so it hides much more than its ground radius
# suggests; the flat stove barely occludes anything. Used for the "is X hidden
# behind a fixture" test -- too small a value lets a moved cabinet clip the plate.
FIXTURE_OCC_LATERAL = {"wooden_cabinet_1_main": 0.20, "flat_stove_1_main": 0.14}


def _radius(joint_name: str) -> float:
    base = joint_name.replace("_joint0", "").rsplit("_", 1)[0]
    return OBJECT_RADIUS.get(base, 0.07)


# every fixture in the libero_spatial scene (present in all 10 tasks); the group
# fixture moves, the rest stay put and must be avoided by the moved fixture and
# by independent objects
ALL_FIXTURES = ["flat_stove_1_main", "wooden_cabinet_1_main"]

# The same two fixtures under their SEGMENTATION INSTANCE names (ALL_FIXTURES
# holds their body names). The spatial instructions name them ("the bowl ON THE
# STOVE", "in the top drawer of THE WOODEN CABINET"), so they are recorded as
# objects 5 and 6 -- seg ids 110 and 120 -- appended AFTER the five free objects
# so ids 60..100 keep their meaning.
FIXTURE_INSTANCES = ["flat_stove_1", "wooden_cabinet_1"]
FIXTURE_INSTANCE_BODY = {"flat_stove_1": "flat_stove_1_main",
                         "wooden_cabinet_1": "wooden_cabinet_1_main"}

# The top drawer is an ARTICULATED part, not a rigid child: its body_pos is
# [0,0,0] and its true pose comes entirely from the slide joint, so reading
# body_pos (which works for the stove's burner, a static [0.15,0,0] offset) would
# place it on the cabinet body -- 0.155 m out. It is recorded as its own fields
# rather than an 8th obj_pos entry, because it is only pulled open in the
# in_the_top_drawer task; in the other nine it sits at qpos 0 hidden inside the
# cabinet, which would make an obj_pos slot degenerate for 90% of the data.
DRAWER_BODY = "wooden_cabinet_1_cabinet_top"
DRAWER_JOINT = "wooden_cabinet_1_top_level"


CABINET_TOP_Z = 1.126   # rest height of a bowl on the wooden-cabinet top (measured)


@dataclass
class SpatialSpec:
    x_range: tuple = (-0.26, 0.14)       # target/group workspace (reachable table)
    y_range: tuple = (-0.26, 0.24)
    fixture_y_range: tuple = (-0.12, 0.06)  # fixture-anchor target kept in a mid band:
                                            # forward enough that the big stove/cabinet
                                            # stays fully in frame + clear of the back
                                            # cabinet, back enough to leave the plate zone
    center_x: tuple = (-0.13, -0.02)     # table_center target region (small jitter only)
    center_y: tuple = (-0.06, 0.06)
    center_keepout: float = 0.26         # twin must stay this far from table centre so the
                                         # central bowl is unambiguous (req 2; target sits
                                         # <=0.08 from centre -> >=0.18 clear margin)
    dest_x: tuple = (-0.05, 0.13)        # reachable place zone for the plate: kept
    dest_y: tuple = (0.10, 0.26)         # front-central so the goal is never off to a
                                         # far corner / out of gripper reach (req 1)
    indep_x_range: tuple = (-0.28, 0.20)  # independent-object table box
    indep_y_range: tuple = (-0.26, 0.30)
    twin_clearance: float = 0.24         # twin bowl kept this far from every anchor so
                                         # a "next to X" relation names only the target
    occlusion_lateral: float = 0.05      # anti-occlusion ray tolerance (small objects)
    fixture_occ_lateral: float = 0.13    # wider ray tol treating a fixture as an occluder (req 6)
    cabinet_plate_clearance: float = 0.22  # plate kept this far from the cabinet body (req 6);
                                           # cabinet stays behind the front plate so depth
                                           # separation does most of the work
    max_yaw_object: float = np.deg2rad(30)   # group yaw for object-anchor tasks
    max_yaw_fixture: float = np.deg2rad(12)  # smaller for fixture groups (keep drawer/burner facing robot)
    fixture_clearance: float = 0.34      # min center distance moved-fixture <-> stationary fixture
    fixture_footprint: float = 0.20      # keep-out radius around any fixture for independent objects
    target_cell: float = 0.09            # OOD checkerboard cell size on target xy (req 4)
    avoid_radius: float = 0.07           # within-scene: steer a retry away from a target
                                         # whose scene just failed the visibility gate
    twin_on_support_p: float = 0.35      # prob. of resting the twin bowl on the cabinet top (req 5)
    group_tries: int = 600
    indep_tries: int = 400
    settle_physics_steps: int = 200       # PHYSICS steps (env.sim.step()), NOT
                                         # control steps.  One env.step() is 25
                                         # of these (control 50 ms / model 2 ms),
                                         # so the old value of 6 settled for
                                         # 12 ms where seating a dropped object
                                         # takes ~120 ms -- measured, a bowl
                                         # placed on the ramekin had fallen
                                         # 0.06 of its 1.83 cm at step 6 and
                                         # converged at step 60.  200 matches
                                         # screen_spatial_pairs' --settle-steps
                                         # default, and it has to: the screener
                                         # certifies the state it screened, so
                                         # emitting a less-settled one emits a
                                         # scene nothing ever validated.


def _yaw_to_quat(theta: float) -> np.ndarray:
    """z-axis rotation as wxyz quaternion (MuJoCo convention)."""
    return np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _rot2d(xy: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])


def read_layout(env, init_state: np.ndarray, joints: list, fixtures: list) -> dict:
    """Absolute reference layout from a source init state: each free joint's
    (xyz, wxyz quat) and each fixture body's (pos, quat)."""
    slices = object_state_slices(env, joints)
    s = np.asarray(init_state, dtype=np.float64)
    free = {}
    for jn in joints:
        a = slices[jn]
        free[jn] = {"pos": s[a:a + 3].copy(), "quat": s[a + 3:a + 7].copy(), "addr": a}
    fix = {}
    m = env.sim.model
    for fb in fixtures:
        bid = m.body_name2id(fb)
        fix[fb] = {"pos": m.body_pos[bid].copy(), "quat": m.body_quat[bid].copy(), "bid": bid}
    return {"free": free, "fixtures": fix}


def _group_centroid(layout: dict, target_joint: str, group_joints: list,
                    group_fixture: str) -> np.ndarray:
    pts = [layout["free"][target_joint]["pos"][:2]]
    for jn in group_joints:
        pts.append(layout["free"][jn]["pos"][:2])
    if group_fixture:
        pts.append(layout["fixtures"][group_fixture]["pos"][:2])
    return np.mean(pts, axis=0)


def transform_group(layout: dict, cfg: dict, dx: float, dy: float, theta: float) -> dict:
    """New planar xy of the target + every rigid-group member (and the group
    fixture under key "__fixture__") under a translate-then-rotate-about-centroid
    transform, computed from THIS layout so each source demo keeps its own exact
    internal relation."""
    target_joint = cfg["target_joint"]
    group_joints = cfg["group_joints"]
    group_fixture = cfg["group_fixture"]
    centroid = _group_centroid(layout, target_joint, group_joints, group_fixture)
    d = np.array([dx, dy])
    out = {}
    for jn in [target_joint, *group_joints]:
        old = layout["free"][jn]["pos"][:2]
        out[jn] = centroid + _rot2d(old - centroid, theta) + d
    if group_fixture:
        old = layout["fixtures"][group_fixture]["pos"][:2]
        out["__fixture__"] = centroid + _rot2d(old - centroid, theta) + d
    return out


def _occluded_by(pt, occ, cam_xy, lateral) -> bool:
    """True if `pt` is hidden from the agentview camera by `occ`: pt is FARTHER
    from the camera than occ and lies within `lateral` of the camera ray through
    occ. Directional (unlike _occludes) so an object in FRONT of a fixture is not
    wrongly rejected."""
    if cam_xy is None:
        return False
    p = np.asarray(pt) - cam_xy
    o = np.asarray(occ) - cam_xy
    if np.linalg.norm(p) <= np.linalg.norm(o):
        return False
    u = o / max(np.linalg.norm(o), 1e-9)
    perp = p - np.dot(p, u) * u
    return float(np.linalg.norm(perp)) < lateral and np.dot(p, u) > 0


def _cabinet_xy(layout: dict, group_xy: dict, group_fixture: str):
    """Planar xy of the wooden cabinet (its moved position if it is the group
    fixture, else its stationary layout position); None if absent."""
    for fb in layout["fixtures"]:
        if "cabinet" in fb:
            if group_fixture == fb:
                return np.asarray(group_xy["__fixture__"])
            return layout["fixtures"][fb]["pos"][:2]
    return None


def _in_box(xy, x_range, y_range):
    return x_range[0] <= xy[0] <= x_range[1] and y_range[0] <= xy[1] <= y_range[1]


def sample_scene(rng, spec: SpatialSpec, cfg: dict, layout: dict,
                 cam_xy: np.ndarray, used_target_xy: list = None) -> dict:
    """Sample one scene perturbation (relative to a source layout): a rigid group
    transform (dx, dy, theta) plus absolute positions for the independent
    objects. Returned as offsets/positions so the same scene can be applied to
    several source demos.

    used_target_xy: prior target positions; the group transform is chosen to
    spread target placements (farthest-point) so a post-hoc checkerboard split
    into seen/unseen cells has coverage on both sides (req 4).

    Raises RuntimeError if no valid configuration is found.
    """
    target_joint = cfg["target_joint"]
    dest_joint = cfg["dest_joint"]
    group_joints = cfg["group_joints"]
    group_fixture = cfg["group_fixture"]
    indep_joints = cfg["indep_joints"]
    twin_joint = cfg.get("twin_joint")
    anchor_kind = cfg["anchor_kind"]
    used_target_xy = used_target_xy or []
    dest_in_group = dest_joint in ([target_joint] + group_joints)

    target_old = layout["free"][target_joint]["pos"][:2]
    centroid = _group_centroid(layout, target_joint, group_joints, group_fixture)
    max_yaw = spec.max_yaw_fixture if group_fixture else spec.max_yaw_object
    # target region: small centre box for table_center (req 2); back-biased for
    # fixture anchors so the plate zone stays clear (req 1); full workspace else
    if anchor_kind == "absolute":
        tx_range, ty_range = spec.center_x, spec.center_y
    elif group_fixture:
        tx_range = spec.x_range
        ty_range = FIXTURE_Y_RANGE.get(group_fixture, spec.fixture_y_range)
    else:
        tx_range, ty_range = spec.x_range, spec.y_range

    # --- sample the group transform; take the first valid candidate that also
    #     stays clear of `avoid` (targets whose scene just failed the rendered
    #     visibility gate), else any valid candidate. NOT a global farthest-point
    #     spread: that saturates and can't fill to the target count once the
    #     workspace is covered -- uniform random already covers the OOD cells. ---
    avoid = used_target_xy
    best = None
    fallback = None
    for _ in range(spec.group_tries):
        tx = rng.uniform(*tx_range)
        ty = rng.uniform(*ty_range)
        theta = rng.uniform(-max_yaw, max_yaw)
        rot_target = centroid + _rot2d(target_old - centroid, theta)
        dx, dy = tx - rot_target[0], ty - rot_target[1]
        cand = transform_group(layout, cfg, dx, dy, theta)
        xs = [v[0] for v in cand.values()]
        ys = [v[1] for v in cand.values()]
        if min(xs) < -0.45 or max(xs) > 0.30 or min(ys) < -0.42 or max(ys) > 0.40:
            continue
        # a moved fixture must not collide with any stationary fixture
        stat_fix = [(fb, layout["fixtures"][fb]["pos"][:2]) for fb in layout["fixtures"]
                    if fb != group_fixture]
        if group_fixture:
            moved = cand["__fixture__"]
            if any(np.linalg.norm(moved - sp) < spec.fixture_clearance for fb, sp in stat_fix):
                continue
        # group FREE members (bowl/anchors resting on the table) must clear a
        # stationary fixture's footprint (radius-aware)
        grp_free = [(j, cand[j]) for j in [target_joint, *group_joints]]
        if any(np.linalg.norm(g - sp) < _radius(j) + FIXTURE_RADIUS.get(fb, 0.16) + SPACING_MARGIN
               for j, g in grp_free for fb, sp in stat_fix):
            continue
        # the target (and a moved fixture) must not be hidden behind a stationary
        # fixture from the agentview camera (req: no occlusion)
        occ_pts = [cand[target_joint]] + ([cand["__fixture__"]] if group_fixture else [])
        if any(_occluded_by(p, sp, cam_xy, FIXTURE_OCC_LATERAL.get(fb, 0.16))
               for p in occ_pts for fb, sp in stat_fix):
            continue
        # if the plate is a group member it must land in the reachable place zone
        # and clear the cabinet (reqs 1 & 6)
        if dest_in_group:
            plate_xy = cand[dest_joint]
            if not _in_box(plate_xy, spec.dest_x, spec.dest_y):
                continue
            cab = _cabinet_xy(layout, cand, group_fixture)
            if cab is not None and (np.linalg.norm(plate_xy - cab) < spec.cabinet_plate_clearance
                                    or _occluded_by(plate_xy, cab, cam_xy, spec.fixture_occ_lateral)):
                continue
        cand_tuple = (dx, dy, theta, cand)
        if fallback is None:
            fallback = cand_tuple
        tgt = np.array([tx, ty])
        if all(np.linalg.norm(tgt - q) > spec.avoid_radius for q in avoid):
            best = cand_tuple
            break
    best = best or fallback
    if best is None:
        raise RuntimeError("could not place the relational group in bounds")
    dx, dy, theta, group_xy = best

    target_new = group_xy[target_joint]
    anchor_pts = [group_xy[j] for j in group_joints] + [target_new]
    if group_fixture:
        anchor_pts.append(group_xy["__fixture__"])
    # placed objects as (xy, radius) for radius-aware non-overlap: the target and
    # any free group members
    placed = [(group_xy[j], _radius(j)) for j in [target_joint, *group_joints]]

    # fixtures as (name, xy) -- the group fixture at its moved xy, the rest fixed
    fixture_pts = []
    for fb in layout["fixtures"]:
        xy = group_xy["__fixture__"] if (group_fixture and fb == group_fixture) \
            else layout["fixtures"][fb]["pos"][:2]
        fixture_pts.append((fb, np.asarray(xy)))
    cab_xy = _cabinet_xy(layout, group_xy, group_fixture)
    cab_stationary = group_fixture is None or "cabinet" not in (group_fixture or "")
    center_pt = np.array([np.mean(spec.center_x), np.mean(spec.center_y)])

    # --- place independent objects: the plate (destination) in the reachable
    #     zone, others on the table; all non-conflicting, unoccluded (req 1, 5) ---
    indep_xy, indep_z = {}, {}

    def try_place(jn, x_range, y_range, is_plate, on_support):
        r = _radius(jn)
        for _ in range(spec.indep_tries):
            if on_support:  # rest the twin on the cabinet top (req 5)
                xy = cab_xy + rng.uniform(-0.03, 0.03, size=2)
            else:
                xy = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
            # radius-aware overlap with already-placed objects
            if any(np.linalg.norm(xy - q) < r + rq + SPACING_MARGIN for q, rq in placed):
                continue
            # clear each fixture footprint (skip the twin's own support fixture)
            if any(np.linalg.norm(xy - fp) < r + FIXTURE_RADIUS.get(fb, 0.16) + SPACING_MARGIN
                   for fb, fp in fixture_pts
                   if not (on_support and "cabinet" in fb)):
                continue
            if any(_occludes(xy, q, cam_xy, spec.occlusion_lateral) for q, _ in placed):
                continue
            # not hidden behind any fixture (directional, fixture-specific width)
            if any(_occluded_by(xy, fp, cam_xy, FIXTURE_OCC_LATERAL.get(fb, 0.16))
                   for fb, fp in fixture_pts):
                continue
            if jn == twin_joint:
                if min(np.linalg.norm(xy - q) for q in anchor_pts) < spec.twin_clearance:
                    continue
                if anchor_kind == "absolute" and np.linalg.norm(xy - center_pt) < spec.center_keepout:
                    continue
            return xy
        return None

    # order: plate first (tight zone), then the rest
    order = ([dest_joint] if (dest_joint in indep_joints) else []) + \
            [j for j in indep_joints if j != dest_joint]
    for jn in order:
        is_plate = (jn == dest_joint)
        on_support = (jn == twin_joint and cab_xy is not None and cab_stationary
                      and rng.random() < spec.twin_on_support_p)
        xr, yr = (spec.dest_x, spec.dest_y) if is_plate else (spec.indep_x_range, spec.indep_y_range)
        got = try_place(jn, xr, yr, is_plate, on_support)
        if got is None and on_support:            # support crowded -> fall back to table
            got = try_place(jn, spec.indep_x_range, spec.indep_y_range, is_plate, False)
            on_support = False
        if got is None:
            raise RuntimeError(f"could not place independent object {jn}")
        indep_xy[jn] = got
        indep_z[jn] = CABINET_TOP_Z if on_support else (_rest_z(jn) + 0.01)
        placed.append((got, _radius(jn)))

    indep_yaw = {jn: float(rng.uniform(-np.pi, np.pi)) for jn in indep_joints}
    cell = (int(np.floor(target_new[0] / spec.target_cell)),
            int(np.floor(target_new[1] / spec.target_cell)))

    return {
        "dx": float(dx), "dy": float(dy), "theta": float(theta),
        "target_new_xy": target_new.tolist(),
        "target_cell": list(cell),
        "group_xy": {k: v.tolist() for k, v in group_xy.items()},
        "indep_xy": {k: v.tolist() for k, v in indep_xy.items()},
        "indep_z": {k: float(v) for k, v in indep_z.items()},
        "indep_yaw": indep_yaw,
    }


def jitter_scene(scene: dict, rng, pos_amp: float = 0.02,
                 yaw_amp: float = np.deg2rad(3)) -> dict:
    """A slightly perturbed copy of a base scene: small translation/rotation of
    the rigid group (dx, dy, theta) and a small planar nudge of each independent
    object, so the several demos rolled out on one scene don't share identical
    object start positions (still recognizably the same scene). indep_z / support
    membership is preserved."""
    s = dict(scene)
    s["dx"] = scene["dx"] + float(rng.uniform(-pos_amp, pos_amp))
    s["dy"] = scene["dy"] + float(rng.uniform(-pos_amp, pos_amp))
    s["theta"] = scene["theta"] + float(rng.uniform(-yaw_amp, yaw_amp))
    s["indep_xy"] = {jn: [xy[0] + float(rng.uniform(-pos_amp, pos_amp)),
                          xy[1] + float(rng.uniform(-pos_amp, pos_amp))]
                     for jn, xy in scene["indep_xy"].items()}
    s["indep_yaw"] = {jn: y + float(rng.uniform(-yaw_amp, yaw_amp))
                      for jn, y in scene["indep_yaw"].items()}
    return s


def apply_scene(scene: dict, init_state: np.ndarray, env, cfg: dict, layout: dict):
    """Write a sampled scene into a source init state.

    Returns (new_init_state, fixture_edits, obj_t, tar_t, info):
        new_init_state  flattened sim state with all FREE objects relocated
        fixture_edits   {body_name: {"pos":(3,), "quat":(4,)}} to apply via
                        apply_fixture_edits() AFTER every env reset (body_pos is
                        model data, wiped by env.reset())
        obj_t, tar_t    target and destination-plate net planar displacement
                        (3-vectors, z=0) for trajectory.synthesize_uniform
    """
    target_joint = cfg["target_joint"]
    dest_joint = cfg["dest_joint"]
    group_joints = cfg["group_joints"]
    group_fixture = cfg["group_fixture"]
    indep_joints = cfg["indep_joints"]
    theta = scene["theta"]
    yaw_q = _yaw_to_quat(theta)

    state = np.asarray(init_state, dtype=np.float64).copy()
    free = layout["free"]

    def set_free(jn, new_xy, new_quat, z=None):
        a = free[jn]["addr"]
        state[a] = new_xy[0]
        state[a + 1] = new_xy[1]
        if z is not None:
            state[a + 2] = z
        state[a + 3:a + 7] = new_quat / max(np.linalg.norm(new_quat), 1e-9)

    target_old_xy = free[target_joint]["pos"][:2].copy()
    dest_old_xy = free[dest_joint]["pos"][:2].copy()

    # recompute the rigid group from THIS demo's layout so its exact internal
    # relation is preserved (the sampled scene only fixes the transform params)
    group_xy = transform_group(layout, cfg, scene["dx"], scene["dy"], theta)

    # group free members: rigid transform, keep each member's own z + apply yaw
    for jn in [target_joint, *group_joints]:
        new_xy = group_xy[jn]
        new_quat = _quat_mul(yaw_q, free[jn]["quat"])
        set_free(jn, new_xy, new_quat)  # z unchanged (rest on same support)

    # independent free members: drop onto their support (table rest height, or
    # the cabinet top for a twin placed on-support) + random yaw
    indep_z = scene.get("indep_z", {})
    for jn in indep_joints:
        new_xy = np.array(scene["indep_xy"][jn])
        rand_q = _quat_mul(_yaw_to_quat(scene["indep_yaw"][jn]), free[jn]["quat"])
        set_free(jn, new_xy, rand_q, z=indep_z.get(jn, _rest_z(jn) + 0.01))

    # zero velocities
    nq = env.sim.model.nq
    state[1 + nq:] = 0.0

    # fixture edits (absolute body pose)
    fixture_edits = {}
    if group_fixture:
        old = layout["fixtures"][group_fixture]
        new_xy = group_xy["__fixture__"]
        new_pos = old["pos"].copy()
        new_pos[0], new_pos[1] = new_xy[0], new_xy[1]
        new_quat = _quat_mul(yaw_q, old["quat"])
        fixture_edits[group_fixture] = {"pos": new_pos, "quat": new_quat}

    target_new_xy = group_xy[target_joint]
    dest_new_xy = group_xy[dest_joint] if dest_joint in group_xy \
        else np.array(scene["indep_xy"][dest_joint])
    obj_t = np.array([*(target_new_xy - target_old_xy), 0.0])
    tar_t = np.array([*(dest_new_xy - dest_old_xy), 0.0])

    info = {
        "target_old_xy": target_old_xy.tolist(),
        "target_new_xy": target_new_xy.tolist(),
        "dest_old_xy": dest_old_xy.tolist(),
        "dest_new_xy": dest_new_xy.tolist(),
        "theta": theta,
        "fixture": group_fixture,
    }
    return state, fixture_edits, obj_t, tar_t, info


def apply_fixture_edits(env, fixture_edits: dict):
    """Set fixture body_pos/body_quat (model data). Call AFTER env.reset()/
    reset_to_init_state, since those rebuild the model and reset body_pos."""
    if not fixture_edits:
        return
    m = env.sim.model
    for fb, e in fixture_edits.items():
        bid = m.body_name2id(fb)
        m.body_pos[bid][:] = e["pos"]
        m.body_quat[bid][:] = e["quat"]
    env.sim.forward()


def settle(env, steps: int, tol: float = 1e-3, require_converged: bool = False):
    """Let dropped independent objects seat on the table.

    `steps` is a CAP in PHYSICS steps (`env.sim.step()`), not control steps --
    one `env.step()` is ~25 of these.  The distinction is not pedantic: this
    function ran for 6 of them for months, which is 12 ms, over which a bowl
    dropped onto the ramekin falls 0.6 of the 18.3 mm it eventually falls.  The
    scenes were fine; the states written out were premature, and every
    downstream artifact rendered from them -- rgb, seg, depth, obj_pos --
    showed a layout that exists for two physics steps and never again.

    Nothing caught it because the old version returned nothing.  So this one
    stops when the free bodies actually stop, and says so:

      returns {"steps": used, "max_speed": m/s, "converged": bool,
               "max_disp_cm": how far anything moved while settling}

    `require_converged` raises instead of returning. It defaults OFF because
    the callers that matter are candidate loops: a scene still moving at the
    cap should be REJECTED and the next jitter tried, exactly like the
    visibility gate, not turned into a crash that abandons the whole task.
    Callers that emit data must gate on `converged` themselves -- and a
    non-converging settle prints a warning either way, because returning
    nothing at all is how a 10x-short settle survived unnoticed.
    """
    free = [env.sim.model.joint_id2name(j)
            for j in range(env.sim.model.njnt)
            if env.sim.model.jnt_type[j] == 0]
    before = {j: env.sim.data.get_joint_qpos(j)[:3].copy() for j in free}
    used, speed = 0, float("inf")
    for used in range(1, steps + 1):
        env.sim.step()
        speed = max((float(np.linalg.norm(env.sim.data.get_joint_qvel(j)[:3]))
                     for j in free), default=0.0)
        if speed < tol:
            break
    disp = max((float(np.linalg.norm(
        env.sim.data.get_joint_qpos(j)[:3] - before[j])) for j in free),
        default=0.0) * 100.0
    rep = {"steps": used, "max_speed": speed, "converged": speed < tol,
           "max_disp_cm": disp}
    if not rep["converged"]:
        msg = (f"settle did not converge: {speed:.2e} m/s still moving after "
               f"{used} physics steps (tol {tol:.0e}); objects moved "
               f"{disp:.2f} cm. This state is still being resolved by the "
               f"simulator and should not be emitted.")
        if require_converged:
            raise RuntimeError(msg)
        print(f"[settle] {msg}", flush=True)
    return rep
