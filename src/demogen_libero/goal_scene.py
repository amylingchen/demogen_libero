"""LIBERO-Goal layout sampling + per-task anchor transforms (plan doc:
docs/LIBERO_goal_12布局_轨迹生成计划.md).

Mechanism facts (probed 2026-08-22 on this repo's compiled goal scene, which
CONTRADICT the plan doc's §1.3): the three fixtures (wooden_cabinet_1,
flat_stove_1, wine_rack_1) have NO free joint -- the bddl :fixtures assembly
path strips the class default. Free joints exist only for the 4 movable
objects. Fixture relocation therefore uses the spatial-suite mechanism
(model.body_pos edits via spatial_scene.apply_fixture_edits, re-applied after
every env reset), NOT qpos. Upside: fixtures are welded, so they cannot be
knocked away mid-episode and layout labels cannot drift (plan §7.7 is moot).

Flattened sim state = [time, qpos(41), qvel(37)] = 79 dims; qpos addr a is
state column 1+a.
"""
from dataclasses import dataclass, field

import h5py
import numpy as np

from . import spatial_scene as S

# free-joint objects (qpos addr from probe_goal_env.py: bowl 9, cheese 16,
# bottle 23, plate 30)
GOAL_JOINTS = [
    "akita_black_bowl_1_joint0",
    "cream_cheese_1_joint0",
    "wine_bottle_1_joint0",
    "plate_1_joint0",
]
GOAL_FIXTURES = ["wooden_cabinet_1_main", "flat_stove_1_main", "wine_rack_1_main"]

# entity -> MuJoCo body, in GOAL_SEG_IDS order minus the gripper, i.e. the order
# obs/obj_pos and obs/obj_quat columns follow. Five of the nine have no free
# joint; their pose still comes out of data.body_xpos, which mj_forward
# recomputes from the joints -- measured, not assumed: driving
# wooden_cabinet_1_middle_level by -0.16 m moves that body origin 16.00 cm and
# leaves the other eight at 0.00 (scripts/probe_goal_entity_pose.py). The
# comment in oc_obs.EXTRA_OBS_KEYS, that an articulated part's pose "lives in a
# slide joint rather than in body_pos", is true of model.body_pos and false of
# data.body_xpos.
GOAL_ENTITY_BODIES = [
    ("akita_black_bowl_1", "akita_black_bowl_1_main"),
    ("cream_cheese_1", "cream_cheese_1_main"),
    ("wine_bottle_1", "wine_bottle_1_main"),
    ("plate_1", "plate_1_main"),
    ("wooden_cabinet_1", "wooden_cabinet_1_main"),
    ("wooden_cabinet_1_middle_drawer", "wooden_cabinet_1_cabinet_middle"),
    ("wooden_cabinet_1_top_drawer", "wooden_cabinet_1_cabinet_top"),
    ("flat_stove_1", "flat_stove_1_main"),
    ("wine_rack_1", "wine_rack_1_main"),
]


# display names for the same nine entities, in the same order. Single source:
# build_goal_metainfo writes these into every metainfo entry's object_names, and
# the pack builder looks geometry up by them, so a second hand-written copy is
# how the two drift apart.
GOAL_ENTITY_DISPLAY = ["bowl", "cream cheese", "wine bottle", "plate", "cabinet",
                       "middle drawer", "top drawer", "stove", "wine rack"]


def entity_poses(env):
    """Per-frame world pose of all nine goal entities, in GOAL_ENTITY_BODIES
    order: positions (9,3) and quaternions (9,4) as **xyzw**.

    xyzw, not MuJoCo's wxyz, because that is what the object and spatial suites
    store (oc_obs writes robosuite's `<name>_quat` observable) and what the
    downstream pack builder decodes with quat_xyzw_to_mat. For the four
    free-joint objects this returns exactly what get_joint_qpos did before
    (verified identical to 0.000 mm and 0.0 on the quaternion).
    """
    m, d = env.sim.model, env.sim.data
    pos, quat = [], []
    for _name, body in GOAL_ENTITY_BODIES:
        i = m.body_name2id(body)
        pos.append(np.array(d.body_xpos[i], dtype=np.float64))
        w, x, y, z = np.array(d.body_xquat[i], dtype=np.float64)
        quat.append(np.array([x, y, z, w]))
    return np.stack(pos), np.stack(quat)

# state columns (flattened state = time + qpos + qvel)
STATE_COL_MIDDLE_DRAWER = 1 + 38   # wooden_cabinet_1_middle_level (slide)
STATE_COL_STOVE_KNOB = 1 + 40      # flat_stove_1_button (hinge)
STATE_COL_PLATE_X = 1 + 30         # plate_1_joint0 qpos x

# CALIBRATED 2026-08-24 against the layout LIBERO itself ships. The previous
# values (bowl .075, cheese .05, bottle .05, plate .10 with a 3cm margin) were
# inherited from the spatial suite and never checked here -- they REJECT the
# official goal layout, which replays all 9 tasks: its bowl-plate gap is
# 0.142 m vs the 0.205 m the old model demanded, and its plate sits 3.5 cm
# inside the old cabinet keep-out. That over-conservatism removed 50-62% of
# every placement zone and is what made the plate zone and the fixture
# corridors look saturated. These radii admit the official layout with ~1cm
# slack; the geometry filter is deliberately permissive now and the REPLAY
# gate (end-state + joint margin + final pose) is what actually validates a
# candidate.
OBJECT_RADIUS = {
    "akita_black_bowl_1_joint0": 0.060,
    "cream_cheese_1_joint0": 0.040,
    "wine_bottle_1_joint0": 0.040,
    "plate_1_joint0": 0.070,
}
# fixture footprints as circles in the fixture's own frame (the stove burner
# sticks out +0.15 x from its base and is where the bowl lands)
FIXTURE_CIRCLES = {
    "wooden_cabinet_1_main": [((0.0, 0.0), 0.13)],
    "flat_stove_1_main": [((0.0, 0.0), 0.10), ((0.15, 0.0), 0.11)],
    "wine_rack_1_main": [((0.0, 0.0), 0.10)],
}
# how wide a fixture blocks the agentview ray (cabinet is tall+wide)
FIXTURE_OCC_LATERAL = {"wooden_cabinet_1_main": 0.20,
                       "flat_stove_1_main": 0.14,
                       "wine_rack_1_main": 0.14}

# task table (plan §3; open_the_top_drawer_and_put_the_bowl_inside excluded by D5)
#   pick_place: obj_t = target delta, tar_t = dest delta (dest joint or fixture)
#   fixture_op: whole trajectory translates with the fixture (obj_t = tar_t)
#   push:       whole trajectory anchored to the plate (plan flags this task as
#               un-splittable; kept honestly, low yield expected)
GOAL_TASKS = {
    "open_the_middle_drawer_of_the_cabinet": {
        "kind": "fixture_op", "anchor_fixture": "wooden_cabinet_1_main",
        "contact": ("state_col", STATE_COL_MIDDLE_DRAWER, 0.003)},
    "turn_on_the_stove": {
        "kind": "fixture_op", "anchor_fixture": "flat_stove_1_main",
        "contact": ("state_col", STATE_COL_STOVE_KNOB, 0.05)},
    "put_the_bowl_on_the_plate": {
        "kind": "pick_place", "target_joint": "akita_black_bowl_1_joint0",
        "dest": ("joint", "plate_1_joint0")},
    "put_the_bowl_on_the_stove": {
        "kind": "pick_place", "target_joint": "akita_black_bowl_1_joint0",
        "dest": ("fixture", "flat_stove_1_main")},
    "put_the_bowl_on_top_of_the_cabinet": {
        "kind": "pick_place", "target_joint": "akita_black_bowl_1_joint0",
        "dest": ("fixture", "wooden_cabinet_1_main")},
    "put_the_cream_cheese_in_the_bowl": {
        "kind": "pick_place", "target_joint": "cream_cheese_1_joint0",
        "dest": ("joint", "akita_black_bowl_1_joint0")},
    "put_the_wine_bottle_on_the_rack": {
        "kind": "pick_place", "target_joint": "wine_bottle_1_joint0",
        "dest": ("fixture", "wine_rack_1_main")},
    "put_the_wine_bottle_on_top_of_the_cabinet": {
        "kind": "pick_place", "target_joint": "wine_bottle_1_joint0",
        "dest": ("fixture", "wooden_cabinet_1_main")},
    "push_the_plate_to_the_front_of_the_stove": {
        "kind": "push", "target_joint": "plate_1_joint0",
        "contact": ("plate_xy", STATE_COL_PLATE_X, 0.005)},
}


@dataclass
class GoalSpec:
    """Smoke-stage sampling spec (plan §4; every number marked 待试点校准 there
    starts at the doc's suggested value)."""
    # Fixture corridors, MEASURED by corner-replay probe 2026-08-22
    # (scripts/probe_goal_fixture_extremes.py, output/goal_fixture_probe*):
    # fixtures sample UNIFORMLY over their corridor (not nominal±delta -- the
    # ±12cm box was too small for unseen layouts to differ meaningfully in the
    # fixture dimension, which anchors 6 of 9 tasks). Corner evidence:
    #   cabinet: (-0.17,-0.08) drawer+top OK; (0.13,-0.38) drawer 1/2 + top OK;
    #            (-0.17,-0.38) 0/2 both tasks -> back-left diagonal cut;
    #            (0.13,-0.08) chokes object space (auto-gated by sample_objects)
    #   stove:   (-0.15,0.33) and (-0.50,0.33) knob+bowl OK; (-0.50,0.08) 0/2;
    #            burner at y<=0.14 chokes object space -> y floor 0.16
    #   rack:    3 of 4 corners OK incl. both far ones; (-0.10,-0.14) 0/2
    fixture_corridor: dict = field(default_factory=lambda: {
        "wooden_cabinet_1_main": ((-0.17, 0.13), (-0.38, -0.10)),
        "flat_stove_1_main": ((-0.50, -0.15), (0.16, 0.33)),
        "wine_rack_1_main": ((-0.42, -0.10), (-0.38, -0.14)),
    })
    fixture_clearance: float = 0.04      # extra margin between fixture circles
    # Leakage floors (2026-08-23 review round 1 + user decision): ONE
    # multi-task policy trains on frames of ALL 8 seen layouts, so the guard is
    # unseen vs EVERY seen layout, per entity (all 7), sampled unseen-FIRST so
    # the floor holds by construction. 0.08 chosen over the plan's 0.10 for
    # geometric feasibility (corridors ~30cm; 4 r=0.10 keep-outs outsize the
    # cabinet corridor).
    unseen_seen_min_dist: float = 0.08   # per entity, unseen vs every seen
    unseen_internal_min_dist: float = 0.06  # per entity, within the unseen set
    seen_fixture_min_dist: float = 0.03  # seen-seen fixture diversity floor
    # push feasibility disk: push's goal region is table-fixed, whole-traj
    # translation tolerates only ~3-4cm of plate displacement from the source
    # placement, so push train cells require the plate within this radius of
    # the nominal plate position (user decision: keep push on feasible layouts)
    push_disk_center: tuple = (0.047, -0.008)
    push_disk_radius: float = 0.03
    n_push_seen: int = 3                 # sampled seen layouts forced into the disk
    # object placement zone (table frame); edges pulled in 3cm from the
    # spatial-suite values 2026-08-23: with the unseen-first 8cm-from-nominal
    # exclusion, free objects (cheese/bottle) were being pushed to the zone
    # rim where the replay gate failed repeatedly while the same sources are
    # 86-93% healthy at nominal (output/goal_source_screening.json) -- i.e.
    # rim placements, not sources, were the failure mode
    # Zones widened toward the ROBOT on 2026-08-24 while the reach circle
    # (REACH_FLAT/REACH_MIN) took over as the binding far-side constraint.
    # Widening -x costs nothing in arm stretch -- joint saturation came from the
    # +x far side -- and it buys the area the unseen exclusion circles need:
    # with 2 unseen already placed, geometry success went 2% -> 12%.
    obj_x: tuple = (-0.32, 0.14)
    obj_y: tuple = (-0.28, 0.28)
    # placement-RECEIVING objects (plate receives the bowl + the push, bowl
    # receives the cheese) must stay in a central reachable band: the first
    # smoke layout put the plate at (0.137, 0.259) -- the far workspace corner
    # -- and bowl->plate yield collapsed to 1/8 with rim-graze misses (spatial
    # suite hit the same wall and tuned its dest zone to x<=0.13)
    receiver_zone: dict = field(default_factory=lambda: {
        "plate_1_joint0": ((-0.20, 0.13), (-0.24, 0.24)),
        "akita_black_bowl_1_joint0": ((-0.30, 0.13), (-0.26, 0.26)),
        # cheese is not a receiver, but its grasp fails at the far-left rim:
        # in the 2026-08-23 seed-43 sampling run all 3 cheese-task rejections
        # had cheese at x <= -0.23 while cheese sources are 12/14 healthy at
        # nominal -- rim placement, not sources
        "cream_cheese_1_joint0": ((-0.30, 0.14), (-0.26, 0.26)),
    })
    spacing_margin: float = 0.01   # calibrated with OBJECT_RADIUS above
    occlusion_lateral: float = 0.06      # object-object camera-ray block width
    settle_steps: int = 300              # physics-step cap for the settle gate
    min_px: int = 60                     # per-entity visibility gate (rendered)
    max_tries: int = 400


def _circles(fb: str, center_xy: np.ndarray):
    return [(np.asarray(center_xy) + np.asarray(off), r)
            for off, r in FIXTURE_CIRCLES[fb]]


# ---------------------------------------------------------------------------
# Reach limits (measured 2026-08-24 from the v1 500-demo dataset + the source
# demos). The human demos' EE never goes beyond 0.75 m from the robot base;
# our retargeted replays reached 0.82-0.86 m and drove joints 2/4 into their
# hard limits on 60/420 demos, while the SAME sources saturate 0/30 times at
# their own placements. Every saturated demo had a reach point >= 0.715 m, so
# the limits below sit just under that, and are validated to admit the
# official nominal layout (its worst points: flat 0.723, elev 0.683,
# handle 0.687). Points are checked at sampling AND at per-demo jitter time.
ROBOT_BASE_XY = np.array([-0.66, 0.0])
REACH_FLAT = 0.73      # table-level grasp/place points
REACH_ELEV = 0.70      # elevated targets (cabinet top, rack slot): the EE must
                       # reach up and over, costing ~5cm more than a flat point
REACH_HANDLE = 0.73    # drawer handle at its fully-open travel
REACH_MIN = 0.35       # too CLOSE is also unreachable: the corner probe's stove
                       # at (-0.50,0.08) puts the knob 0.18 m from the base and
                       # replayed 0/2, while (-0.50,0.33) at 0.37 m passed; the
                       # source demos' nearest object sits at 0.46 m


def _reach(xy) -> float:
    return float(np.linalg.norm(np.asarray(xy) - ROBOT_BASE_XY))


def object_reach_ok(xy) -> bool:
    """Table-level object placement within the arm's demonstrated envelope."""
    return REACH_MIN <= _reach(xy) <= REACH_FLAT


def fixture_reach_violations(fb: str, xy) -> list:
    """Goal points of a fixture that fall outside the demonstrated envelope."""
    xy = np.asarray(xy)
    checks = {
        "wooden_cabinet_1_main": [("cabinet_top", [-0.03, 0.05], REACH_ELEV),
                                  ("drawer_handle_open", [0.0, 0.28], REACH_HANDLE)],
        "flat_stove_1_main": [("burner", [0.16, 0.05], REACH_FLAT),
                              ("knob", [0.0, 0.0], REACH_FLAT)],
        "wine_rack_1_main": [("rack_slot", [0.083, 0.0], REACH_ELEV)],
    }.get(fb, [])
    return [(nm, round(_reach(xy + np.asarray(off)), 3), lim)
            for nm, off, lim in checks if _reach(xy + np.asarray(off)) > lim]


def layout_reach_violations(layout: dict) -> list:
    """Every reach point of a full layout that exceeds its limit."""
    out = []
    for jn, xy in layout["objects"].items():
        if not object_reach_ok(xy):
            out.append((jn, round(_reach(xy), 3), REACH_FLAT))
    for fb, xy in layout["fixtures"].items():
        out += fixture_reach_violations(fb, xy)
    return out


def _corridor_cut(fb: str, xy) -> bool:
    """Probed dead corners inside the rectangular corridors (True = reject),
    implemented as rectangular cuts: cabinet back-left corner 0/2, stove
    back-left extreme (-0.50,0.08) 0/2, rack center-front (-0.10,-0.14) 0/2
    in the 2026-08-22 corner probes (output/goal_fixture_probe*)."""
    x, y = float(xy[0]), float(xy[1])
    if fb == "flat_stove_1_main":
        # supported by the 2026-08-24 grid: no feasible point in this region
        return x < -0.45 and y < 0.26
    # The cabinet back-left and rack center-front cuts were REMOVED on
    # 2026-08-24: both came from the 2026-08-22 corner probe, which pinned the
    # other two fixtures at their nominal poses and therefore forced the probed
    # fixture INTO them -- cabinet (-0.17,-0.38) sat 0.147 m from the nominal
    # rack (needs 0.23), rack (-0.10,-0.14) sat 0.162 m from the nominal
    # cabinet. Those 0/2 readings measured interpenetrating scenes, not
    # reachability. The corrected grid finds feasible points throughout both
    # regions, so the cheap filter no longer excludes them and the replay gate
    # decides.
    return False


def _excluded(cand, key, exclusions) -> bool:
    """True if candidate position violates a per-entity exclusion circle.
    exclusions: {entity_key: [(center_xy, min_dist), ...]} -- used by the
    unseen-first suite sampler to enforce the leakage floors DURING placement
    (checking them only on the finished layout multiplies 7 per-entity pass
    probabilities into a ~1% joint acceptance and starves the sampler)."""
    if not exclusions or key not in exclusions:
        return False
    return any(np.linalg.norm(np.asarray(cand) - np.asarray(c)) < d
               for c, d in exclusions[key])


def sample_goal_layout(rng: np.random.Generator, spec: GoalSpec, ref_layout: dict,
                       cam_xy: np.ndarray, exclusions: dict = None) -> dict:
    """One full layout: new planar xy for the 3 fixtures + 4 objects.
    Geometry gates only (spacing / zones / camera-ray occlusion / per-entity
    exclusion circles); the caller must still run the settle +
    rendered-visibility gates on the applied scene.
    Raises RuntimeError if no layout found within spec.max_tries."""
    for _ in range(spec.max_tries):
        fixtures = {}
        ok = True
        for fb in GOAL_FIXTURES:
            (x0, x1), (y0, y1) = spec.fixture_corridor[fb]
            for _f in range(60):
                cand = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
                if _corridor_cut(fb, cand):
                    continue
                if fixture_reach_violations(fb, cand):
                    continue
                if _excluded(cand, fb, exclusions):
                    continue
                clash = False
                for other, oxy in fixtures.items():
                    for c1, r1 in _circles(fb, cand):
                        for c2, r2 in _circles(other, oxy):
                            if np.linalg.norm(c1 - c2) < r1 + r2 + spec.fixture_clearance:
                                clash = True
                if not clash:
                    fixtures[fb] = cand
                    break
            else:
                ok = False
                break
        if not ok:
            continue

        objects = sample_objects(rng, spec, fixtures, cam_xy, exclusions)
        if objects is None:
            continue
        return {"fixtures": {fb: xy.tolist() for fb, xy in fixtures.items()},
                "objects": {jn: xy.tolist() for jn, xy in objects.items()}}
    raise RuntimeError("no valid goal layout found")


def sample_objects(rng: np.random.Generator, spec: GoalSpec, fixtures: dict,
                   cam_xy: np.ndarray, exclusions: dict = None):
    """Place the 4 movable objects given FIXED fixture positions ({body: xy}).
    Returns {joint: xy ndarray} or None if no non-conflicting placement exists.
    Also used by the fixture-extreme probe, where fixture positions are forced
    rather than sampled."""
    fixtures = {fb: np.asarray(xy) for fb, xy in fixtures.items()}
    fixture_circles = [c for fb, xy in fixtures.items() for c in _circles(fb, xy)]
    objects = {}
    # zone-constrained receivers first (tightest region, fewest options)
    placement_order = sorted(GOAL_JOINTS, key=lambda j: j not in spec.receiver_zone)
    for jn in placement_order:
        r = OBJECT_RADIUS[jn]
        (zx, zy) = spec.receiver_zone.get(jn, (spec.obj_x, spec.obj_y))
        for _o in range(80):
            cand = np.array([rng.uniform(*zx), rng.uniform(*zy)])
            if not object_reach_ok(cand):
                continue
            if _excluded(cand, jn, exclusions):
                continue
            if any(np.linalg.norm(cand - q) < r + OBJECT_RADIUS[j2] + spec.spacing_margin
                   for j2, q in objects.items()):
                continue
            if any(np.linalg.norm(cand - c) < r + cr + spec.spacing_margin
                   for c, cr in fixture_circles):
                continue
            if any(S._occludes(cand, q, cam_xy, spec.occlusion_lateral)
                   for q in objects.values()):
                continue
            if any(S._occluded_by(cand, fixtures[fb], cam_xy, FIXTURE_OCC_LATERAL[fb])
                   for fb in GOAL_FIXTURES):
                continue
            if (jn in DRAWER_SWEEP_KEEPOUT_JOINTS
                    and in_drawer_sweep(cand, fixtures["wooden_cabinet_1_main"])):
                continue
            objects[jn] = cand
            break
        else:
            return None
    return objects


def apply_goal_layout(layout: dict, init_state: np.ndarray, demo_layout: dict):
    """Write a sampled layout into a source demo's init state.

    Returns (new_init_state, fixture_edits): objects get new planar xy in qpos
    (each keeps its own z and quat -- D1: no yaw change between layouts);
    fixtures become body_pos edits for spatial_scene.apply_fixture_edits
    (applied AFTER every reset; quat kept at the scene default)."""
    state = np.asarray(init_state, dtype=np.float64).copy()
    for jn in GOAL_JOINTS:
        a = demo_layout["free"][jn]["addr"]
        state[a:a + 2] = layout["objects"][jn]
    fixture_edits = {}
    for fb in GOAL_FIXTURES:
        ref = demo_layout["fixtures"][fb]
        fixture_edits[fb] = {
            "pos": np.array([layout["fixtures"][fb][0], layout["fixtures"][fb][1],
                             ref["pos"][2]]),
            "quat": ref["quat"].copy(),
        }
    return state, fixture_edits


def anchor_deltas(task_cfg: dict, layout: dict, demo_layout: dict):
    """(obj_t, tar_t) 3-vectors (z=0) for trajectory.synthesize_uniform,
    measured from THIS source demo's own placements (objects vary ±1cm between
    source demos; fixtures are the same scene default in all of them)."""
    def joint_delta(jn):
        d = np.zeros(3)
        d[:2] = np.asarray(layout["objects"][jn]) - demo_layout["free"][jn]["pos"][:2]
        return d

    def fixture_delta(fb):
        d = np.zeros(3)
        d[:2] = np.asarray(layout["fixtures"][fb]) - demo_layout["fixtures"][fb]["pos"][:2]
        return d

    kind = task_cfg["kind"]
    if kind == "fixture_op":
        d = fixture_delta(task_cfg["anchor_fixture"])
        return d, d.copy()
    if kind == "push":
        d = joint_delta(task_cfg["target_joint"])
        return d, d.copy()
    obj_t = joint_delta(task_cfg["target_joint"])
    dkind, dname = task_cfg["dest"]
    tar_t = joint_delta(dname) if dkind == "joint" else fixture_delta(dname)
    return obj_t, tar_t


class ReachMap:
    """Empirical EE reach map from scripts/probe_reach_map.py: servo residual
    on a grid at 3 z-planes. reachable(xy, z) uses the nearest plane and the
    WORST residual of the 4 surrounding grid cells (conservative)."""

    def __init__(self, path: str):
        import json as _json
        d = _json.load(open(path))
        self.xs = np.asarray(d["xs"])
        self.ys = np.asarray(d["ys"])
        self.z_planes = d["z_planes"]
        self.tol = d["tol"]
        self.res = {float(z): np.asarray(d["residual"][str(z)]) for z in d["z_planes"]}

    def residual(self, xy, z: float) -> float:
        zp = min(self.z_planes, key=lambda p: abs(p - z))
        r = self.res[float(zp)]
        i = np.searchsorted(self.xs, xy[0]) - 1
        j = np.searchsorted(self.ys, xy[1]) - 1
        vals = []
        for di in (0, 1):
            for dj in (0, 1):
                ii, jj = i + di, j + dj
                if 0 <= ii < len(self.xs) and 0 <= jj < len(self.ys):
                    vals.append(r[ii, jj])
        return float(max(vals)) if vals else np.inf

    def reachable(self, xy, z: float, tol: float = None) -> bool:
        return self.residual(xy, z) < (tol if tol is not None else self.tol)


# Task goal points, MEASURED from source-demo EE positions at the goal event
# (evidence: scripts/measure_goal_event_ee.py ->
# output/goal_geometry/goal_event_ee.json; 8 demos each, std <= 3cm): the
# drawer slides open along +y (handle closed = cab+(0,0.10) z1.03, fully open
# = cab+(0,0.28) z1.02 -- NOT +x as the cabinet's visual front suggests,
# yaw=pi puts the slide on y); knob EE = stove base+(−0.01,0.01) z0.93;
# burner release = stove+(0.16,0.05) z0.96; cabinet-top release =
# cab+(−0.03,0.05) z1.16 (bowl) and cab+(−0.06,0.02) z1.24 (bottle); rack
# slot release = rack+(0.083,0) z1.20. push's goal region
# (main_table_stove_front_region) is FIXED ON THE TABLE at (-0.05,0.21) --
# the one goal that does NOT follow its fixture (plan doc §1.4 is wrong for
# that task). NOTE: layout_reach_points/ReachMap below are currently UNUSED
# (the servo reach map failed cross-validation and was replaced by the
# replay gate); kept for reference only.
Z_TABLE, Z_MID, Z_TOP = 1.00, 1.15, 1.26


def layout_reach_points(layout: dict) -> list:
    """Every point the arm must reach for SOME task on this layout, as
    (label, xy, z). A layout serves all 9 tasks, so all points must pass."""
    obj = {jn: np.asarray(layout["objects"][jn]) for jn in GOAL_JOINTS}
    fix = {fb: np.asarray(layout["fixtures"][fb]) for fb in GOAL_FIXTURES}
    cab, stove, rack = (fix["wooden_cabinet_1_main"], fix["flat_stove_1_main"],
                        fix["wine_rack_1_main"])
    pts = [(f"grasp:{jn.split('_joint')[0]}", obj[jn], Z_TABLE) for jn in GOAL_JOINTS]
    pts += [
        ("place:plate", obj["plate_1_joint0"], Z_TABLE),
        ("place:bowl", obj["akita_black_bowl_1_joint0"], Z_TABLE),
        ("place:burner", stove + [0.16, 0.05], Z_TABLE),
        ("op:knob", stove, Z_TABLE),
        ("place:cabinet_top_bowl", cab + [-0.03, 0.05], Z_MID),
        ("place:cabinet_top_bottle", cab + [-0.06, 0.02], Z_TOP),
        ("place:rack_slot", rack + [0.083, 0.0], Z_MID),
        ("op:drawer_handle_closed", cab + [0.0, 0.10], Z_TABLE),
        ("op:drawer_handle_open", cab + [0.0, 0.28], Z_TABLE),
    ]
    return pts


# The middle drawer slides +y by 0.16 m when opened (joint axis (0,1,0) local,
# range [-0.16,0.01], body yaw=pi). Measured drawer geometry (evidence:
# scripts/measure_goal_event_ee.py -> output/goal_geometry/goal_event_ee.json):
# body spans x_rel [-0.111,+0.100], closed front face y_rel +0.064, open
# handle y_rel +0.25; drawer BOTTOM z=0.986. The bowl (rim z~0.97), plate and
# cheese all pass UNDER it -- only the 25cm bottle collides (verified the
# other way on L00: nominal bowl sits in-sweep and the drawer replay
# succeeds).
DRAWER_SWEEP_KEEPOUT_JOINTS = ("wine_bottle_1_joint0",)


def in_drawer_sweep(xy, cab_xy) -> bool:
    return (-0.12 < xy[0] - cab_xy[0] < 0.11
            and cab_xy[1] + 0.05 < xy[1] < cab_xy[1] + 0.27)


def unreachable_points(layout: dict, reach: ReachMap, tol: float = None) -> list:
    return [(lbl, list(np.round(xy, 3)), z)
            for lbl, xy, z in layout_reach_points(layout)
            if not reach.reachable(xy, z, tol)]


def read_demo_layout(env, init_state: np.ndarray, fixture_ref: dict) -> dict:
    """Per-source-demo layout with the fixture block taken from `fixture_ref`
    (captured ONCE right after a fresh env.reset()). Never read fixtures from
    the live model between attempts: apply_fixture_edits mutates model.body_pos
    and it stays mutated until the next reset, so a live read on attempt 2+
    returns the ALREADY-MOVED pose and every fixture delta collapses to zero
    (bug caught in the first smoke run: put_the_wine_bottle_on_the_rack
    tar_t=[0,0] from the second source onward)."""
    layout = S.read_layout(env, init_state, GOAL_JOINTS, GOAL_FIXTURES)
    layout["fixtures"] = {fb: {"pos": fixture_ref[fb]["pos"].copy(),
                               "quat": fixture_ref[fb]["quat"].copy(),
                               "bid": fixture_ref[fb]["bid"]}
                          for fb in GOAL_FIXTURES}
    return layout


def capture_fixture_ref(env) -> dict:
    """Snapshot the pristine fixture body poses. Call immediately after a
    fresh env.reset(), before any apply_fixture_edits."""
    m = env.sim.model
    out = {}
    for fb in GOAL_FIXTURES:
        bid = m.body_name2id(fb)
        out[fb] = {"pos": m.body_pos[bid].copy(), "quat": m.body_quat[bid].copy(),
                   "bid": bid}
    return out


def contact_frame(hdf5_path: str, demo_key: str, contact_spec, margin: int = 8) -> int:
    """First frame the anchored entity starts moving (drawer slides, knob
    turns, plate slides), minus a small margin so the final approach is part of
    the verbatim segment. Read from the stored full state series."""
    mode, col, thresh = contact_spec
    with h5py.File(hdf5_path, "r") as f:
        st = np.array(f["data"][demo_key]["states"])
    if mode == "state_col":
        dev = np.abs(st[:, col] - st[0, col])
    else:  # plate_xy: planar displacement of the plate
        dev = np.linalg.norm(st[:, col:col + 2] - st[0, col:col + 2], axis=1)
    moving = np.where(dev > thresh)[0]
    assert moving.size > 0, f"{demo_key}: anchored entity never moves (thresh={thresh})"
    return max(int(moving[0]) - margin, 1)



# ---------------------------------------------------------------------------
# Segmentation entity table for the goal suite (plan §7.3).
#
# The INSTANCE segmentation this repo uses for libero_object returns one mask
# per object instance, which for the cabinet means a single `wooden_cabinet_1`
# covering the whole unit -- "open the middle drawer" then has no mask to
# ground its noun on. The ELEMENT segmentation returns per-geom ids, and
# regrouping those by `geom_bodyid` yields per-drawer masks; probed feasible
# 2026-08-25 (scripts/probe_drawer_seg.py): with the cabinet CLOSED the middle
# drawer is 412 px and the top drawer 469 px, both groundable from the initial
# frame.
#
# CAVEAT recorded with the probe: a STATIONARY drawer's mask grows when a
# neighbour opens, because the gap exposes the neighbour's box (opening the
# middle drawer takes the top drawer from 469 to 4249 px). Never pick the
# referred drawer by "largest cabinet part mask".
#
# Ids follow the repo convention: gripper 50, then entities from 60 in steps
# of 10, in the order below.
GOAL_SEG_IDS = {
    "robot_gripper": 50,
    "akita_black_bowl_1": 60,
    "cream_cheese_1": 70,
    "wine_bottle_1": 80,
    "plate_1": 90,
    "wooden_cabinet_1": 100,          # frame only; the drawers get their own ids
    "wooden_cabinet_1_middle_drawer": 110,
    "wooden_cabinet_1_top_drawer": 120,
    "flat_stove_1": 130,
    "wine_rack_1": 140,
}
# body name -> entity, for the bodies that need to be split out or renamed
_BODY_TO_ENTITY = {
    "wooden_cabinet_1_cabinet_middle": "wooden_cabinet_1_middle_drawer",
    "wooden_cabinet_1_cabinet_top": "wooden_cabinet_1_top_drawer",
    # the bottom drawer is not referred to by any task instruction, so it stays
    # part of the cabinet frame rather than getting its own id
    "wooden_cabinet_1_cabinet_bottom": "wooden_cabinet_1",
}
_GRIPPER_KEYWORDS = ("gripper", "finger", "hand")


def build_goal_seg_lut(env) -> np.ndarray:
    """LUT indexed by ELEMENT segmentation value + 1, giving the goal entity
    seg id. Index 0 (background) and anything unmapped stay 0.

    The +1 is NOT already in the render output: robosuite adds it only for the
    instance/class levels, where the raw ids are remapped
    (robot_env._create_camera_sensors: `mapping.get(x, -1) + 1`, and `mapping
    is None` for "element"). The ELEMENT level therefore returns the raw geom
    index, 0-based, with -1 for background. Indexing this LUT with the value
    as-is shifts every geom onto its predecessor's label.

    Always go through map_element_seg rather than indexing this array directly:
        ids = map_element_seg(lut, obs["agentview_segmentation_element"][..., 0])
    """
    m = env.sim.model
    lut = np.zeros(m.ngeom + 2, dtype=np.uint8)
    for g in range(m.ngeom):
        bid = int(m.geom_bodyid[g])
        body = m.body_id2name(bid)
        if body is None:
            continue
        entity = _BODY_TO_ENTITY.get(body)
        if entity is None:
            low = body.lower()
            if any(k in low for k in _GRIPPER_KEYWORDS):
                entity = "robot_gripper"
            else:
                # body names are "<instance>_main" / "<instance>_g12" etc.
                for inst in GOAL_SEG_IDS:
                    if body.startswith(inst + "_") or body == inst:
                        entity = inst
                        break
        if entity in GOAL_SEG_IDS:
            lut[g + 1] = GOAL_SEG_IDS[entity]
    return lut


def map_element_seg(lut: np.ndarray, seg_element: np.ndarray) -> np.ndarray:
    """Element segmentation (raw geom index, -1 = background) -> goal entity
    seg ids, using a LUT from build_goal_seg_lut.

    The single place the +1 lives. Background (-1) maps to index 0, which the
    LUT never writes, so it stays 0 instead of aliasing onto geom 0's label.
    """
    return lut[np.clip(np.asarray(seg_element).astype(np.int32) + 1,
                       0, len(lut) - 1)]


def orientation_of(seg_ids: np.ndarray) -> str:
    """Tell whether a segmentation/image array is stored UPRIGHT (row 0 = top,
    the viewer convention this repo's OC format uses) or in the raw GL
    orientation (row 0 = bottom), from the content itself.

    The discriminator is physical rather than conventional: the gripper hangs
    ABOVE the table, so in an upright array its rows are SMALLER than the
    plate's; in a GL array the order inverts. Reporting a row number alone
    cannot distinguish the two, which is the whole failure mode this guards
    against -- a caller that forgets the flip produces arrays whose per-entity
    pixel counts are identical and whose top/bottom membership is reversed.

    Returns "upright", "gl", or "undecided" (an entity was missing).
    """
    g = np.nonzero(seg_ids == GOAL_SEG_IDS["robot_gripper"])[0]
    p = np.nonzero(seg_ids == GOAL_SEG_IDS["plate_1"])[0]
    if g.size == 0 or p.size == 0:
        return "undecided"
    return "upright" if g.mean() < p.mean() else "gl"


def flip_fingerprint(env, lut: np.ndarray) -> dict:
    """Per-entity pixel counts plus the orientation verdict for both the raw
    render and its flip, so the check demonstrates it can tell them apart."""
    obs = env.env._get_observations(force_update=True)
    seg = obs["agentview_segmentation_element"][..., 0]
    ids = map_element_seg(lut, seg)
    entities = {}
    for name, sid in GOAL_SEG_IDS.items():
        mask = ids == sid
        if mask.any():
            entities[name] = {"px": int(mask.sum()),
                              "mean_row_raw": round(float(np.nonzero(mask)[0].mean()), 1)}
    return {"entities": entities,
            "verdict_raw_render": orientation_of(ids),
            "verdict_flipped": orientation_of(ids[::-1]),
            "discriminative": orientation_of(ids) != orientation_of(ids[::-1])}


def jitter_ok(entity_key: str, xy, unseen_layouts: list,
              floor: float = 0.06) -> bool:
    """GENERATION-TIME leakage guard (user decision 2026-08-23, round-2 review
    B5): the manifest's 0.08 floor holds between layout CENTERS; per-demo
    jitter (plan §4, objects ±2.5cm) would erode the realized train-vs-unseen
    separation to ~1.4cm worst-case. Every jittered entity position in a
    TRAINING demo (and every jittered unseen eval init) must keep >= `floor`
    from the same entity in every unseen (resp. seen) layout -- resample the
    jitter until this returns True."""
    group = "objects" if entity_key.endswith("_joint0") else "fixtures"
    return all(np.linalg.norm(np.asarray(xy) -
                              np.asarray(l["layout"][group][entity_key])) >= floor
               for l in unseen_layouts)


def whole_traj_frames(f1: int, T: int):
    """Frames for a rigidly-translated trajectory (obj_t == tar_t): closed-loop
    tracked reach up to f1, everything from f1 on verbatim. The 1-frame
    motion-2 carries no offset change, so skill-1/skill-2 split is nominal."""
    from .trajectory import Frames
    assert f1 + 2 < T, f"contact frame {f1} too late for T={T}"
    return Frames(f1, f1 + 1, f1 + 2)
