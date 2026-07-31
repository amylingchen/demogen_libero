"""Generate OC-format demos for PAIRED (two-relation) spatial scenes from a
split plan produced by build_pair_split.py.

Each planned episode names a geometry and which of the two bowls the instruction
refers to. That bowl is written as `akita_black_bowl_1` and the other takes the
remaining pose, so every task's BDDL goal `(On akita_black_bowl_1 plate_1)` and
the OC seg-id convention (60 = target) hold unchanged -- the two bowls are the
same asset (identical geoms, sizes, rgba, friction and mass), so the swap is
invisible in the image and changes only which bowl the episode is about.

Recording matches run_spatial_oc_demo.py exactly (same env, seg LUT, extractor,
subtask annotation and write path), so paired demos are format-identical to the
existing single-task dataset. The stove and the cabinet are recorded as objects
5 and 6 (seg 110 / 120) as the demo is generated, which is what
patch_fixture_objects.py had to reconstruct after the fact for the first batch.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_pair_oc_demo.py --split train
    .venv\\Scripts\\python.exe scripts\\run_pair_oc_demo.py --split train --task on_the_stove
    # top an existing task up using only one batch of newly added episodes
    .venv\\Scripts\\python.exe scripts\\run_pair_oc_demo.py --split train \\
        --task on_the_ramekin --batch shared_anchor --target-count 52
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import h5py
import numpy as np

import screen_spatial_pairs as P
from run_spatial_oc_demo import segment_for
from demogen_libero.convert import load_demo
from demogen_libero.trajectory import synthesize_uniform
from demogen_libero import libero_replay as R, oc_obs, spatial_scene as S


def _rot(v, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def group_members(scene, rel, which):
    """Which free joints and fixture move rigidly with one relation's bowl."""
    task = scene["task_a"] if which == "a" else scene["task_b"]
    bowl = P.BOWL_A if which == "a" else P.BOWL_B
    r = rel[task]
    return bowl, list(r["anchors"]), r["fixture"]


def shared_anchors(scene, rel):
    """Anchors named by BOTH of the scene's relations. One object carrying two
    instructions ("the bowl ON the ramekin" / "the bowl NEXT TO the ramekin")
    means the two relation groups are not disjoint, so they cannot be transformed
    independently."""
    _ba, anchors_a, fix_a = group_members(scene, rel, "a")
    _bb, anchors_b, fix_b = group_members(scene, rel, "b")
    keys = [k for k in anchors_a if k in anchors_b]
    if fix_a and fix_a == fix_b:
        keys.append(fix_a)
    return keys


def jitter_scene(scene, rel, rng, pos_amp, yaw_amp):
    """A slightly perturbed copy. Each relation group moves RIGIDLY about its own
    bowl so the relation it encodes stays exactly true; objects in no group get an
    independent nudge. Mirrors spatial_scene.jitter_scene, which does the same for
    the single-relation case.

    A scene whose two relations SHARE an anchor is jittered as ONE rigid body.
    Jittering such a scene per group is not a smaller error, it is a broken
    relation: group A's transform moves the shared anchor, then group B's
    transform moves it again -- recomputed from the ORIGINAL pose, so A's
    transform is silently discarded and the bowl that sat ON the anchor is left
    hanging beside it. Nothing downstream would notice: the visibility gate
    passes, and the episode can still end with the bowl on the plate, so the
    demo would be kept while its instruction had become false.
    """
    out = {"free": {k: dict(v) for k, v in scene["free"].items()},
           "fixtures": {k: dict(v) for k, v in scene["fixtures"].items()}}
    groups = [("a", "b")] if shared_anchors(scene, rel) else [("a",), ("b",)]
    moved = set()
    for grp in groups:
        d = rng.uniform(-pos_amp, pos_amp, size=2)
        th = rng.uniform(-yaw_amp, yaw_amp)
        yq = S._yaw_to_quat(th)
        members, fixtures, pivot = [], [], None
        for which in grp:
            bowl, anchors, fixture = group_members(scene, rel, which)
            if pivot is None:
                pivot = np.array(scene["free"][bowl]["xy"])
            members += [bowl, *anchors]
            if fixture:
                fixtures.append(fixture)
        for k in dict.fromkeys(members):
            src = scene["free"][k]
            out["free"][k]["xy"] = (pivot + _rot(np.array(src["xy"]) - pivot, th) + d).tolist()
            out["free"][k]["quat"] = S._quat_mul(yq, np.asarray(src["quat"])).tolist()
            moved.add(k)
        for fixture in dict.fromkeys(fixtures):
            src = scene["fixtures"][fixture]
            out["fixtures"][fixture]["xy"] = (
                pivot + _rot(np.array(src["xy"]) - pivot, th) + d).tolist()
            out["fixtures"][fixture]["quat"] = S._quat_mul(
                yq, np.asarray(src["quat"])).tolist()
    for k, v in scene["free"].items():
        if k in moved:
            continue
        out["free"][k]["xy"] = (np.array(v["xy"]) + rng.uniform(-pos_amp, pos_amp, 2)).tolist()
        out["free"][k]["quat"] = S._quat_mul(
            S._yaw_to_quat(rng.uniform(-yaw_amp, yaw_amp)), np.asarray(v["quat"])).tolist()
    for k in ("task_a", "task_b", "drawer_qpos"):
        out[k] = scene[k]
    return out


def build_state(scene, role, base_init, addrs, nq, drawer_adr):
    state = np.asarray(base_init, dtype=np.float64).copy()
    free = dict(scene["free"])
    if role == "b":
        free[P.BOWL_A], free[P.BOWL_B] = scene["free"][P.BOWL_B], scene["free"][P.BOWL_A]
    for jn, m in free.items():
        a = addrs[jn]
        state[a], state[a + 1], state[a + 2] = m["xy"][0], m["xy"][1], m["z"]
        q = np.asarray(m["quat"], float)
        state[a + 3:a + 7] = q / max(np.linalg.norm(q), 1e-9)
    if drawer_adr is not None:
        state[1 + drawer_adr] = scene.get("drawer_qpos", 0.0)
    state[1 + nq:] = 0.0
    fx = {k: {"pos": np.array([v["xy"][0], v["xy"][1], v["z"]]),
              "quat": np.asarray(v["quat"], float)}
          for k, v in scene["fixtures"].items()}
    return state, fx


def relation_holds(env, bands, tol, z_tol):
    """Is the target bowl still where its instruction says, AFTER settling?

    This is the check that separates "a demo of this task" from "a demo that
    merely ends with a bowl on the plate". It matters most for the stacked
    relations: a bowl balanced ON the ramekin or the cookie box can slide off
    during the settle, and nothing else in the pipeline would object -- the scene
    still renders, every object is still visible, and the robot can still pick
    the bowl up off the table and complete the episode. The demo would be kept
    with an instruction that had quietly become false."""
    # the target always occupies the akita_black_bowl_1 slot -- build_state writes
    # whichever bowl the instruction names into it -- so this reads the target
    b = env.sim.data.get_joint_qpos(P.BOWL_A)[:3]
    for key, band in bands.items():
        if key in S.ALL_FIXTURES:
            # a fixture has no joint; its pose is model data, which is also where
            # the bands were measured from and where apply_fixture_edits writes
            bid = env.sim.model.body_name2id(key)
            a_xy, a_z = env.sim.model.body_pos[bid][:2], env.sim.model.body_pos[bid][2]
        else:
            q = env.sim.data.get_joint_qpos(key)
            a_xy, a_z = q[:2], q[2]
        name = key.split("_joint")[0]
        d = float(np.linalg.norm(b[:2] - a_xy))
        if d < band["lo"] - tol or d > band["hi"] + tol:
            return False, f"relation:{name}:xy"
        dz = float(b[2] - a_z)
        if dz < band["z_lo"] - z_tol or dz > band["z_hi"] + z_tol:
            return False, f"relation:{name}:z"
    return True, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="output/pair_split.json")
    ap.add_argument("--split", default="train", choices=["train", "counterfact", "unseen"])
    ap.add_argument("--task", default=None, help="short task name; default = all")
    ap.add_argument("--demos-per-scene", type=int, default=None,
                    help="default: the plan's own value")
    ap.add_argument("--source-retries", type=int, default=6,
                    help="alternative source demos to try when a rollout misses")
    ap.add_argument("--jitter-pos", type=float, default=0.02)
    ap.add_argument("--jitter-yaw", type=float, default=3.0)
    ap.add_argument("--jitter-tries", type=int, default=6)
    ap.add_argument("--min-px", type=int, default=100)
    ap.add_argument("--min-px-bowl", type=int, default=250,
                    help="stricter floor for BOTH bowls: the target must be visible "
                         "and so must the distractor it is being chosen over")
    ap.add_argument("--batch", default=None,
                    help="only use episodes carrying this `batch` tag, and count "
                         "progress against that batch alone -- for ADDING episodes "
                         "to a task whose demos are already generated")
    ap.add_argument("--target-count", type=int, default=None,
                    help="stop once the task's hdf5 holds this many demos in total "
                         "(instead of demos-per-scene x episodes)")
    ap.add_argument("--relation-tol", type=float, default=0.02,
                    help="planar slack allowed on the target's measured relation "
                         "band after settling; 0 disables the check entirely")
    ap.add_argument("--relation-z-tol", type=float, default=0.008,
                    help="height slack, kept well under the 0.020 m gap change that "
                         "a bowl sliding off the 2 cm cookie box produces")
    ap.add_argument("--relation-calib", type=int, default=20,
                    help="source init states settled to measure the bands (0 = all)")
    ap.add_argument("--no-fixtures", action="store_true",
                    help="record only the five free objects, as the first batch was "
                         "generated before patch_fixture_objects.py existed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="output/libero_spatial_pairs")
    ap.add_argument("--no-depth-mm", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    plan = json.load(open(args.plan))
    scenes = plan["scenes"]
    n_per = args.demos_per_scene or plan["conventions"]["demos_per_scene"]
    board = __import__("build_pair_split").Board(plan["board"]["n"])
    cfg_all = S.load_spatial_config()
    rel = P.relations_from_config(cfg_all)
    jitter_yaw = np.deg2rad(args.jitter_yaw)

    eps = defaultdict(list)
    for e in plan["episodes"]:
        if e["split"] == args.split and (args.batch is None
                                        or e.get("batch") == args.batch):
            eps[e["task"]].append(e)
    tasks = ([t for t in eps if P.short(t) == args.task] if args.task else list(eps))
    if not tasks:
        raise SystemExit(f"no episodes for split={args.split} task={args.task}"
                         + (f" batch={args.batch}" if args.batch else ""))

    grand = [0, 0]
    for task in tasks:
        short = P.short(task)
        out_dir = os.path.join(args.out_root, args.split, short)
        os.makedirs(out_dir, exist_ok=True)
        hdf5_out = os.path.join(out_dir, f"{task}_demo.hdf5")
        meta_out = os.path.join(out_dir, "metainfo.json")
        json.dump(oc_obs.PHASE_MAP, open(os.path.join(out_dir, "phase_map.json"), "w"),
                  indent=2)
        meta = json.load(open(meta_out)) if os.path.exists(meta_out) else {task: {}}
        meta.setdefault(task, {})
        log_path = os.path.join(out_dir, "scene_log.json")
        scene_log = json.load(open(log_path)) if os.path.exists(log_path) else []
        n_done = 0
        if os.path.exists(hdf5_out):
            with h5py.File(hdf5_out, "r") as f:
                n_done = len(f["data"].keys()) if "data" in f else 0
            print(f"[resume] {short}: {n_done} existing demos", flush=True)
        # `n_done` names the next demo; progress is tracked PER EPISODE from the
        # scene_log instead of by dividing a running total by demos-per-scene. That
        # division assumes every episode yielded its full quota, so one episode that
        # yields nothing (an occluded anchor, say) shifts the whole tail by one and
        # a resume re-runs an episode that was already complete.
        ep_done = Counter((r["scene_index"], r["target_role"]) for r in scene_log
                          if r.get("batch") == args.batch and r["split"] == args.split)
        if args.batch:
            print(f"[resume] {short}: {sum(ep_done.values())} demos over "
                  f"{len(ep_done)} episodes of batch {args.batch!r}", flush=True)
        if args.target_count is not None and n_done >= args.target_count:
            print(f"[{short}] already at {n_done} >= target {args.target_count}, skipping",
                  flush=True)
            continue

        cfg = cfg_all[task]
        obj_order = list(cfg["object_order"])
        if not args.no_fixtures:
            obj_order += S.FIXTURE_INSTANCES
        src_path = os.path.join(P.DATA_DIR, f"{task}_demo.hdf5")
        with h5py.File(src_path, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        sources = {}
        for k in keys:
            d = load_demo(src_path, k)
            try:
                sources[k] = (d, segment_for(d)[0])
            except AssertionError:
                pass
        print(f"[{short}] {len(sources)}/{len(keys)} usable source demos", flush=True)
        any_demo = sources[next(iter(sources))][0]

        env = oc_obs.make_oc_env(any_demo.bddl_file)
        env.reset()
        lut = oc_obs.build_seg_lut(env, obj_order)
        seg_ids = {inst: 60 + 10 * obj_order.index(inst) for inst in P.FREE_INST}
        import mujoco
        mdl = env.sim.model._model
        jid = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_JOINT, P.DRAWER_JOINT)
        drawer_adr = int(mdl.jnt_qposadr[jid]) if jid >= 0 else None
        f_bids = {f: env.sim.model.body_name2id(S.FIXTURE_INSTANCE_BODY[f])
                  for f in S.FIXTURE_INSTANCES}
        d_bid = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_BODY, S.DRAWER_BODY)

        def extract(e, obs):
            # obj_pos/obj_quat in extract_oc_frame come from robosuite's per-object
            # observations, which exist only for FREE-JOINTED objects -- a fixture
            # has no joint and no `<name>_pos` entry. So ask the extractor for the
            # five free objects and append the two fixtures from the sim, where
            # their pose actually lives. The seg LUT already covers all seven.
            #
            # The five must be in the TASK's object_order, not P.FREE_INST: those
            # two differ by a transposition (FREE_INST is bowl,bowl,plate...,
            # object_order is bowl,plate,bowl...), and obj_pos rows are only
            # meaningful against the object_instances attr the file also stores.
            row = oc_obs.extract_oc_frame(e, obs, lut, object_order=obj_order[:5],
                                          depth_mm=not args.no_depth_mm)
            if not args.no_fixtures:
                fp = np.zeros((len(S.FIXTURE_INSTANCES), 3))
                fq = np.zeros((len(S.FIXTURE_INSTANCES), 4))
                for i, fb in enumerate(S.FIXTURE_INSTANCES):
                    fp[i] = e.sim.data.body_xpos[f_bids[fb]]
                    q = e.sim.data.body_xquat[f_bids[fb]]        # wxyz
                    fq[i] = [q[1], q[2], q[3], q[0]]             # -> xyzw, as obj_quat
                row["obj_pos"] = np.concatenate([row["obj_pos"], fp], axis=0)
                row["obj_quat"] = np.concatenate([row["obj_quat"], fq], axis=0)
            row["drawer_pos"] = e.sim.data.body_xpos[d_bid].copy()
            row["drawer_qpos"] = np.array(
                [e.sim.data.qpos[drawer_adr] if drawer_adr is not None else 0.0])
            return row

        R.reset_to_init_state(env, any_demo.init_state)
        lay0 = S.read_layout(env, any_demo.init_state, P.FREE_JOINTS, S.ALL_FIXTURES)
        addrs = {j: lay0["free"][j]["addr"] for j in P.FREE_JOINTS}
        nq = env.sim.model.nq
        src_xy = {}
        for k, (d, _) in sources.items():
            l = S.read_layout(env, d.init_state, P.FREE_JOINTS, S.ALL_FIXTURES)
            src_xy[k] = (l["free"][P.BOWL_A]["pos"][:2].copy(),
                         l["free"][P.PLATE]["pos"][:2].copy())

        # The band each of the task's own relations holds in, measured on LIBERO's
        # own init states AFTER THE SAME SETTLE the gate below applies -- so the
        # comparison is like-for-like and the margin does not have to absorb the
        # settle. Sizing this by eye would not work: a bowl sliding off the COOKIE
        # BOX changes the height gap by exactly 0.020 m (the box is 2 cm tall with
        # its origin at the centre), so a hand-picked 0.02 m tolerance would sit
        # right on top of the very failure the check exists to catch. The ramekin,
        # 4.3 cm tall, would have been caught either way.
        r_task = rel[task]
        band_keys = (list(r_task["anchors"])
                     + ([r_task["fixture"]] if r_task["fixture"] else []))
        bands = {}
        if args.relation_tol > 0 and band_keys:
            obs_d = {k: ([], []) for k in band_keys}
            calib = list(sources)[:args.relation_calib or len(sources)]
            for k in calib:
                R.reset_to_init_state(env, sources[k][0].init_state)
                S.settle(env, S.SpatialSpec().settle_physics_steps)
                b = env.sim.data.get_joint_qpos(P.BOWL_A)[:3]
                for key in band_keys:
                    if key in S.ALL_FIXTURES:
                        bid = env.sim.model.body_name2id(key)
                        a = env.sim.model.body_pos[bid]
                    else:
                        a = env.sim.data.get_joint_qpos(key)[:3]
                    obs_d[key][0].append(float(np.linalg.norm(b[:2] - a[:2])))
                    obs_d[key][1].append(float(b[2] - a[2]))
            bands = {k: {"lo": min(v[0]), "hi": max(v[0]),
                         "z_lo": min(v[1]), "z_hi": max(v[1])} for k, v in obs_d.items()}
            print(f"[{short}] relation bands from {len(calib)} settled source states: "
                  + ", ".join(f"{k.split('_joint')[0]} d=[{v['lo']:.3f},{v['hi']:.3f}] "
                              f"dz=[{v['z_lo']:.3f},{v['z_hi']:.3f}]"
                              for k, v in bands.items())
                  + f"  (+/- {args.relation_tol:.3f} xy, {args.relation_z_tol:.3f} z)",
                  flush=True)

        def usable(state, fx):
            R.reset_to_init_state(env, state)
            S.apply_fixture_edits(env, fx)
            S.settle(env, S.SpatialSpec().settle_physics_steps)
            # force_update is required, not optional: settle() drives env.sim.step()
            # directly, which never touches robosuite's observable cache, and
            # reset_to_init_state() left that cache holding the env's DEFAULT reset
            # scene. Without it this gate renders a layout that is not the one being
            # tested -- and the default layout has all five objects in plain view,
            # so it would pass every candidate.
            obs = env.env._get_observations(force_update=True)
            raw = obs["agentview_segmentation_instance"][..., 0]
            seg = lut[np.clip(raw, 0, len(lut) - 1)]
            for inst, sid in seg_ids.items():
                # BOTH bowls carry a higher bar than the rest of the scene. In a
                # shared-anchor pair they sit 0.117-0.126 m apart with one of them
                # raised on the anchor, so the elevated bowl can clip the other in
                # the agentview -- and a policy that cannot see the distractor
                # cannot be said to have chosen between them. 250 px is the screen's
                # own calibrated floor (the 500 real LIBERO init states render every
                # object at 491 px or more).
                floor = args.min_px_bowl if inst.startswith("akita_black_bowl") \
                    else args.min_px
                if int((seg == sid).sum()) < floor:
                    return False, f"occluded:{inst}"
            return relation_holds(env, bands, args.relation_tol, args.relation_z_tol)

        rng = np.random.default_rng(args.seed + hash(task) % 10000)
        n_ok, n_try = 0, 0
        rejected = defaultdict(int)
        try:
            for ep_i, e in enumerate(eps[task]):
                if args.target_count is not None and n_done + n_ok >= args.target_count:
                    break
                base = scenes[e["scene_index"]]
                role = e["target_role"]
                if ep_done[(e["scene_index"], role)] >= n_per:   # done on a prior run
                    continue
                want_cell = tuple(e["target_cell"])
                order = list(sources)
                rng.shuffle(order)
                if f"demo_{base['source_demo_a' if role == 'a' else 'source_demo_b']}" in order:
                    k0 = f"demo_{base['source_demo_a' if role == 'a' else 'source_demo_b']}"
                    order.remove(k0)
                    order.insert(0, k0)
                got = ep_done[(e["scene_index"], role)]   # top this episode up
                for src_key in order:
                    if got >= n_per:
                        break
                    if args.target_count is not None and n_done + n_ok >= args.target_count:
                        break
                    demo, frames = sources[src_key]
                    sb, sp = src_xy[src_key]
                    # jitter, then RE-DERIVE the cell from the jittered target: the
                    # board cell is 0.100 x 0.125 m and the jitter is 0.02 m, so a
                    # scene near a boundary can otherwise carry the wrong split label
                    sc, state, fx = None, None, None
                    for _ in range(args.jitter_tries):
                        cand = (base if got == 0 else
                                jitter_scene(base, rel, rng, args.jitter_pos, jitter_yaw))
                        bowl = P.BOWL_A if role == "a" else P.BOWL_B
                        if board.cell(cand["free"][bowl]["xy"]) != want_cell:
                            continue
                        st, f2 = build_state(cand, role, demo.init_state, addrs, nq,
                                             drawer_adr)
                        ok, why = usable(st, f2)
                        if not ok:
                            rejected[why] += 1
                            continue
                        sc, state, fx = cand, st, f2
                        break
                    if sc is None:
                        continue
                    bowl = P.BOWL_A if role == "a" else P.BOWL_B
                    tgt = np.array(sc["free"][bowl]["xy"])
                    plate = np.array(sc["free"][P.PLATE]["xy"])
                    obj_t = np.array([*(tgt - sb), 0.0])
                    tar_t = np.array([*(plate - sp), 0.0])
                    n_try += 1
                    try:
                        ref, base_actions, new_frames = synthesize_uniform(
                            demo.state, demo.action, frames, obj_t, tar_t)
                        R.reset_to_init_state(env, state)
                        S.apply_fixture_edits(env, fx)
                        success, _obs, rollout = R.replay_uniform(
                            env, base_actions, ref, new_frames, collect=True, extract=extract)
                    except Exception as exc:
                        print(f"  [{short} ep{ep_i}] {src_key} ERROR {exc!r}"[:120], flush=True)
                        continue
                    if not success:
                        continue
                    name = f"demo_{n_done + n_ok}"
                    rollout = {k: np.asarray(v) for k, v in rollout.items()}
                    rollout["subtask_id"], subtasks = oc_obs.annotate_subtasks(
                        rollout["actions"], oc_obs.display_name(obj_order[0]),
                        oc_obs.display_name(obj_order[1]))
                    oc_obs.write_oc_demo(hdf5_out, name, rollout, success=True,
                                         object_order=obj_order, subtasks=subtasks)
                    meta[task][name] = oc_obs.metainfo_entry(
                        task, rollout, obj_order, state, subtasks=subtasks)
                    scene_log.append({
                        "demo_name": name, "split": args.split, "task": task,
                        "pair": e["pair"], "target_role": role,
                        "scene_index": e["scene_index"], "source_demo": src_key,
                        # same identity build_pair_eval_suite.py writes for the
                        # eval splits: one geometry+role, however many demos or
                        # arm variants were rendered from it. Written here too so
                        # all three splits can be grouped by one key.
                        "scene_uid": f"scene{e['scene_index']}_{role}",
                        "target_xy": tgt.tolist(), "target_cell": list(want_cell),
                        "target_in_unseen_cell": bool(e["target_in_unseen_cell"]),
                        "counterfact_partner_cell": base[f"cell_{'b' if role == 'a' else 'a'}"],
                        "batch": e.get("batch"),
                        "with_fixtures": not args.no_fixtures,
                    })
                    n_ok += 1
                    got += 1
                    if n_ok % 5 == 0:
                        json.dump(meta, open(meta_out, "w"), indent=2)
                        json.dump(scene_log, open(log_path, "w"), indent=2)
                if got < n_per:
                    print(f"  [{short} ep{ep_i}] only {got}/{n_per} demos", flush=True)
                if (ep_i + 1) % 5 == 0:
                    print(f"[{short}] episode {ep_i+1}/{len(eps[task])}  "
                          f"demos={n_ok}  attempts={n_try}  "
                          f"rate={100*n_ok/max(n_try,1):.0f}%", flush=True)
        finally:
            env.close()
            json.dump(meta, open(meta_out, "w"), indent=2)
            json.dump(scene_log, open(log_path, "w"), indent=2)
        grand[0] += n_ok
        grand[1] += n_try
        print(f"[{short}] done: {n_ok} demos from {n_try} attempts "
              f"({100*n_ok/max(n_try,1):.0f}%), hdf5 now holds {n_done + n_ok}"
              f" -> {hdf5_out}", flush=True)
        if rejected:
            print(f"[{short}] init states rejected before rollout: "
                  + ", ".join(f"{k}={v}" for k, v in
                              sorted(rejected.items(), key=lambda kv: -kv[1])[:6]),
                  flush=True)

    print(f"\nTOTAL {grand[0]} demos from {grand[1]} attempts "
          f"({100*grand[0]/max(grand[1],1):.0f}% success)")


if __name__ == "__main__":
    main()
