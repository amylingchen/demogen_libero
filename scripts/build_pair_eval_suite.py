"""Render the counterfactual and unseen splits of a paired plan as EVALUATION
scenes: initial states plus frame-0 observations, no trajectories.

A policy is rolled out from these states and scored, so a reference trajectory
would be unused -- the same convention `build_eval_suite.py` follows for
libero_object. Solvability was checked separately by sampling
(`probe_pair_trajectories.py --split counterfact|unseen`), so a scene appearing
here is known to admit a working trajectory at roughly the training rate.

Each episode is T=1: the geometry from the plan, with the bowl the instruction
names written as `akita_black_bowl_1` and the other taking the remaining pose.
Fixtures (stove, cabinet) and the articulated drawer are recorded from the start,
so no post-hoc patch is needed the way the train split needed one.

Usage:
    .venv\\Scripts\\python.exe scripts\\build_pair_eval_suite.py --split counterfact
    .venv\\Scripts\\python.exe scripts\\build_pair_eval_suite.py --split unseen
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import h5py
import numpy as np

import glob

import screen_spatial_pairs as P
from run_pair_oc_demo import build_state, jitter_scene
from build_pair_split import Board
from patch_fixture_objects import (FIXTURES, FIXTURE_BODY, DRAWER_BODY,
                                   DRAWER_JOINT, fixture_poses)
from demogen_libero.convert import load_demo
from demogen_libero import libero_replay as R, oc_obs, spatial_scene as S


def robot_joint_addrs(env):
    """qpos addresses of the arm joints only. The gripper fingers are left alone:
    perturbing them changes whether the hand is open, which is a different initial
    condition from 'the arm starts somewhere slightly different'."""
    import mujoco
    mdl = env.sim.model._model
    addrs = []
    for j in range(mdl.njnt):
        nm = mujoco.mj_id2name(mdl, mujoco.mjtObj.mjOBJ_JOINT, j)
        if nm and nm.startswith("robot0_joint"):
            addrs.append(int(mdl.jnt_qposadr[j]))
    return sorted(addrs)


def perturb_robot(state, addrs, rng, amp):
    """A copy of the scene with the arm displaced slightly in JOINT space.

    Joint space, not end-effector space: an EE offset has to be solved back
    through IK and can land in a different arm configuration entirely, whereas a
    small joint perturbation is guaranteed reachable and keeps the arm in the same
    posture family. The scene itself -- every object, both bowls, the fixtures --
    is untouched, so these variants test robustness to the starting pose and
    nothing else."""
    s = np.asarray(state, dtype=np.float64).copy()
    for a in addrs:
        s[1 + a] += rng.uniform(-amp, amp)       # state = [time, qpos..., qvel...]
    return s


def emit(env, obs, state, e, base, task, obj_order, lut, f_bids, d_bid, d_adr,
         meta, scene_log, hdf5_out, kept, args, variant=0, scene_uid=None):
    """Write one T=1 eval scene."""
    # obj_pos comes from robosuite's per-object observations, which exist only
    # for free-jointed objects -- so ask for the five free ones (the seg LUT
    # already covers all seven) and append the fixtures from the sim, where
    # their pose actually lives.
    #
    # The five must be in the TASK's object_order, which is what the
    # object_instances attr written below declares. P.FREE_INST is a DIFFERENT
    # order -- bowl,bowl,plate vs bowl,plate,bowl -- so using it silently
    # transposes rows 1 and 2 while the attr keeps claiming plate is row 1.
    row = oc_obs.extract_oc_frame(env, obs, lut, object_order=obj_order[:5],
                                  depth_mm=not args.no_depth_mm)
    row["actions"] = np.zeros(7)
    row["phase_id"] = np.int32(0)
    rollout = {k: np.asarray(v)[None] for k, v in row.items()}
    if args.with_fixtures:
        fp = np.zeros((1, len(FIXTURES), 3))
        fq = np.zeros((1, len(FIXTURES), 4))
        for i, fb in enumerate(FIXTURES):
            fp[0, i] = env.sim.data.body_xpos[f_bids[fb]]
            q = env.sim.data.body_xquat[f_bids[fb]]          # wxyz
            fq[0, i] = [q[1], q[2], q[3], q[0]]              # -> xyzw
        rollout["obj_pos"] = np.concatenate([rollout["obj_pos"], fp], axis=1)
        rollout["obj_quat"] = np.concatenate([rollout["obj_quat"], fq], axis=1)
    rollout["drawer_pos"] = env.sim.data.body_xpos[d_bid][None].copy()
    rollout["drawer_qpos"] = np.array(
        [[env.sim.data.qpos[d_adr] if d_adr is not None else 0.0]])
    rollout["states"] = np.asarray(state)[None]

    name = f"eval_{kept}"
    oc_obs.write_oc_demo(hdf5_out, name, rollout, success=True,
                         object_order=obj_order)
    meta[task][name] = oc_obs.metainfo_entry(task, rollout, obj_order, state)
    scene_log.append({
        "demo_name": name, "split": args.split, "task": task,
        "pair": e["pair"], "target_role": e["target_role"],
        # provenance: which planning batch this scene came from, so eval scenes can
        # be sliced the same way the train demos can
        "batch": e.get("batch"),
        "scene_index": e["scene_index"],
        # variants 0..n-1 share one geometry and differ only in the arm's starting
        # joint angles; group by scene_uid to score per scene rather than per state
        "scene_uid": scene_uid if scene_uid is not None else name,
        "robot_variant": int(variant),
        "ee_pos": [float(v) for v in rollout["ee_pos"][0]],
        "target_xy": e["target_xy"], "target_cell": e["target_cell"],
        "target_in_unseen_cell": bool(e["target_in_unseen_cell"]),
        # read from the PLAN's scene: a jittered copy carries geometry only
        "counterfact_partner_cell":
            base[f"cell_{'b' if e['target_role'] == 'a' else 'a'}"],
        "drawer_qpos": float(rollout["drawer_qpos"][0, 0]),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="output/pair_split.json")
    ap.add_argument("--split", default="counterfact",
                    choices=["counterfact", "unseen", "train"])
    ap.add_argument("--task", default=None, help="short task name; default = all")
    ap.add_argument("--min-px", type=int, default=100,
                    help="visibility gate on the five free objects")
    ap.add_argument("--min-px-bowl", type=int, default=250,
                    help="stricter floor for BOTH bowls, matching run_pair_oc_demo")
    ap.add_argument("--batch", default=None,
                    help="only build episodes carrying this `batch` tag")
    ap.add_argument("--robot-variants", type=int, default=None,
                    help="initial arm poses per geometry (default: 5 for unseen, "
                         "2 for counterfact). Variant 0 is always unperturbed")
    ap.add_argument("--robot-jitter", type=float, default=0.03,
                    help="uniform +/- radians added to each of the 7 arm joints")
    ap.add_argument("--robot-tries", type=int, default=8,
                    help="re-draws allowed when a perturbed arm occludes an object")
    ap.add_argument("--jitter-tries", type=int, default=12,
                    help="re-draws allowed to find a fully visible variant")
    ap.add_argument("--jitter-pos", type=float, default=0.02)
    ap.add_argument("--jitter-yaw", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-plan", action="store_true",
                    help="counterfact: re-sample from the plan instead of reusing the "
                         "generated train geometry (breaks pixel-identity)")
    ap.add_argument("--with-fixtures", action="store_true", default=True,
                    help="record stove and cabinet as objects (seg 110 / 120)")
    ap.add_argument("--out-root", default="output/libero_spatial_pairs")
    ap.add_argument("--no-depth-mm", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    if args.robot_variants is None:
        args.robot_variants = {"unseen": 5, "counterfact": 2}.get(args.split, 1)

    import mujoco
    plan = json.load(open(args.plan))
    scenes = plan["scenes"]
    cfg_all = S.load_spatial_config()
    rel = P.relations_from_config(cfg_all)
    board = Board(plan["board"]["n"])
    rng = np.random.default_rng(args.seed)

    eps = defaultdict(list)
    for e in plan["episodes"]:
        if e["split"] == args.split and (args.batch is None
                                        or e.get("batch") == args.batch):
            eps[e["task"]].append(e)
    tasks = ([t for t in eps if P.short(t) == args.task] if args.task else list(eps))
    if not tasks:
        raise SystemExit(f"no episodes for split={args.split} task={args.task}"
                         + (f" batch={args.batch}" if args.batch else ""))

    # For counterfact the image must be the SAME one the model trained on -- only
    # the instruction changes. The geometry lives in the train directory of the
    # PARTNER task, not this one: a scene is used once, as one task's training
    # episode, and its counterfactual is issued against the OTHER task. Looking
    # only in the current task's own directory finds nothing (measured: 0 of 263
    # episodes, against 255 when every directory is searched), and every scene
    # then falls back to re-sampling from the plan -- which is exactly the path
    # documented as breaking pixel-identity. So index every train directory once.
    train_state = {}
    if args.split == "counterfact" and not args.from_plan:
        tr_root = os.path.join(args.out_root, "train")
        for d in (sorted(os.listdir(tr_root)) if os.path.isdir(tr_root) else []):
            tr_h5 = glob.glob(os.path.join(tr_root, d, "*_demo.hdf5"))
            tr_log = os.path.join(tr_root, d, "scene_log.json")
            if not tr_h5 or not os.path.exists(tr_log):
                continue
            with h5py.File(tr_h5[0], "r") as tf:
                for r in json.load(open(tr_log)):
                    if r["demo_name"] not in tf["data"]:
                        continue
                    g = tf["data"][r["demo_name"]]
                    # Take the fixture poses the demo RECORDED rather than
                    # reconstructing them from the bowl. Reconstruction infers the
                    # group's translation from where the bowl ended up at frame 0,
                    # but recording starts after 5 warmup steps, so a bowl that
                    # settles a few mm drags the reconstructed fixture with it --
                    # measured as a ~1 px shift of the cabinet's silhouette (1274
                    # of its 6208 pixels differing, with its AREA unchanged). The
                    # recorded pose has no such inference in it. Only available
                    # since obj_pos gained slots 5 and 6.
                    fxp = None
                    op, oq = g["obs"]["obj_pos"], g["obs"]["obj_quat"]
                    if op.shape[1] >= 5 + len(FIXTURES):
                        fxp = {}
                        for i, fb in enumerate(FIXTURES):
                            q = np.array(oq[0, 5 + i])            # stored xyzw
                            fxp[FIXTURE_BODY[fb]] = {
                                "pos": np.array(op[0, 5 + i]),
                                "quat": np.array([q[3], q[0], q[1], q[2]]),  # -> wxyz
                            }
                    train_state.setdefault(
                        r["scene_index"], (np.array(g["states"][0]), fxp))
        print(f"[eval] {len(train_state)} train geometries indexed across "
              f"{len(os.listdir(tr_root)) if os.path.isdir(tr_root) else 0} task dirs "
              f"for counterfactuals reusing train geometry", flush=True)

    grand = [0, 0]
    for task in tasks:
        short = P.short(task)
        out_dir = os.path.join(args.out_root, args.split, short)
        os.makedirs(out_dir, exist_ok=True)
        hdf5_final = os.path.join(out_dir, f"{task}_demo.hdf5")
        # Build into a temp file and swap it in only once the task is complete.
        # Deleting the real file up front and rebuilding in place means any
        # interruption -- a kill, a crash, a machine going down -- leaves the split
        # destroyed rather than merely stale, and this has already cost one full
        # eval run: a hard kill mid-rebuild left one task with 14 scenes in the
        # hdf5 and no scene_log at all (the JSONs are written at the end), and
        # wiped the tasks it had not reached yet.
        hdf5_out = hdf5_final + ".tmp"
        if os.path.exists(hdf5_out):
            os.remove(hdf5_out)
        meta = {task: {}}
        scene_log = []

        obj_order = list(cfg_all[task]["object_order"])
        if args.with_fixtures:
            obj_order += FIXTURES

        demo0 = load_demo(os.path.join(P.DATA_DIR, f"{task}_demo.hdf5"), "demo_0")
        env = oc_obs.make_oc_env(demo0.bddl_file)
        env.reset()
        R.reset_to_init_state(env, demo0.init_state)
        lut = oc_obs.build_seg_lut(env, obj_order)
        lay = S.read_layout(env, demo0.init_state, P.FREE_JOINTS, S.ALL_FIXTURES)
        addrs = {j: lay["free"][j]["addr"] for j in P.FREE_JOINTS}
        nq = env.sim.model.nq
        mdl = env.sim.model._model
        jid = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_JOINT, P.DRAWER_JOINT)
        drawer_adr = int(mdl.jnt_qposadr[jid]) if jid >= 0 else None
        d_bid = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_BODY, DRAWER_BODY)
        d_jid = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_JOINT, DRAWER_JOINT)
        d_adr = int(mdl.jnt_qposadr[d_jid]) if d_jid >= 0 else None
        f_bids = {f: env.sim.model.body_name2id(FIXTURE_BODY[f]) for f in FIXTURES}
        seg_ids = {inst: 60 + 10 * i for i, inst in enumerate(obj_order)}

        # Re-sampling from the plan cannot promise pixel-identity: where the base
        # geometry failed the visibility gate, training silently fell back to a
        # jittered variant that was never recorded. So counterfact scenes are taken
        # from the generated TRAIN demos' frame-0 state (indexed above), with the
        # two identical bowls swapped -- pixel-exact by construction and already
        # known to pass the gate.
        r_addrs = robot_joint_addrs(env)
        n_var = args.robot_variants

        def emit_variants(state, fx, e, base, kept, ee_log):
            """One geometry, `n_var` slightly different arm starting poses.

            Variant 0 is always the UNPERTURBED state. For counterfact that is not
            cosmetic: its premise is the training scene with a different correct
            answer, and only the nominal arm pose reproduces that frame.

            "Reproduces" is exact in the OBJECT STATE, not bit-exact in pixels.
            Measured control -- re-rendering a training frame from its own state,
            with no bowl swap and no fixture reconstruction involved -- still moves
            179-350 pixels, 60-80% of them on the robot arm's silhouette (seg 50)
            and the rest background, with objects contributing 6-19. Restoring a
            mid-episode state through a model rebuild does not reproduce the arm's
            rasterization to the pixel. So the honest claim is: identical object
            poses, visually identical scene, ~0.3-0.5% of pixels differing on the
            arm's edge.

            Each variant is re-rendered and re-gated, because moving the arm moves
            what it occludes -- the reason the gate exists at all.
            """
            made = 0
            base_state = np.asarray(state, dtype=np.float64).copy()
            base_state[1 + nq:] = 0.0    # an init state starts at rest; the unseen
            #                              path captures this after a settle, which
            #                              can leave small residual velocities
            for v in range(n_var):
                got = None
                for _ in range(args.robot_tries if v else 1):
                    st_v = (base_state if v == 0 else
                            perturb_robot(base_state, r_addrs, rng, args.robot_jitter))
                    R.reset_to_init_state(env, st_v)
                    S.apply_fixture_edits(env, fx)
                    env.sim.forward()
                    o = env.env._get_observations(force_update=True)
                    sg = lut[np.clip(o["agentview_segmentation_instance"][..., 0],
                                     0, len(lut) - 1)]
                    if all(int((sg == 60 + 10 * i).sum())
                           >= (args.min_px_bowl if inst.startswith("akita_black_bowl")
                               else args.min_px)
                           for i, inst in enumerate(P.FREE_INST)):
                        got = (st_v, o)
                        break
                if got is None:
                    continue
                st_v, o = got
                emit(env, o, st_v, e, base, task, obj_order, lut, f_bids, d_bid,
                     d_adr, meta, scene_log, hdf5_out, kept + made, args,
                     variant=v, scene_uid=f"scene{e['scene_index']}_{e['target_role']}")
                ee_log.append(np.asarray(o["robot0_eef_pos"], dtype=float))
                made += 1
            return made

        kept = n_dim = n_exact = 0
        ee_log = []
        try:
            for e in eps[task]:
                base = scenes[e["scene_index"]]
                if e["scene_index"] in train_state:
                    tr_st, tr_fx = train_state[e["scene_index"]]
                    st = tr_st.copy()
                    a, b = addrs[P.BOWL_A], addrs[P.BOWL_B]
                    st[a:a + 7], st[b:b + 7] = tr_st[b:b + 7], tr_st[a:a + 7]
                    # `fixture_poses` needs the role the state it is handed was
                    # BUILT with, because that is what tells it which slot holds
                    # which of the scene's two bowls -- and it recovers the fixture
                    # pose from the bowl's displacement off its scene position.
                    # `st` is the train state with the two slots swapped, which is
                    # exactly a state built with the counterfactual's own role. The
                    # opposite role makes it measure the displacement of bowl A
                    # against bowl B's scene position, so the group fixture is
                    # translated by the whole bowl-to-bowl offset. Measured: with
                    # the wrong role the plate rendered 0 px in 24 of 24
                    # on_the_wooden_cabinet counterfactuals -- the displaced cabinet
                    # was sitting on top of it. Tasks with no group fixture were
                    # unaffected, which is why the drop rate tracked exactly which
                    # tasks are fixture-anchored.
                    fx2 = (tr_fx if tr_fx is not None
                           else fixture_poses(base, rel, st, addrs, e["target_role"]))
                    made = emit_variants(st, fx2, e, base, kept, ee_log)
                    if made:
                        n_exact += 1
                    else:
                        n_dim += 1
                    kept += made
                    continue
                want_cell = tuple(e["target_cell"])
                bowl = P.BOWL_A if e["target_role"] == "a" else P.BOWL_B
                # the split plan is geometry only -- it never rendered anything, so
                # an independent object can sit behind a fixture. Training recovers
                # by re-drawing the jitter until every object is visible; do the
                # same here, keeping the target inside its assigned cell so the
                # seen/unseen label still holds.
                sc = state = fx = None
                for k in range(args.jitter_tries):
                    cand = base if k == 0 else jitter_scene(
                        base, rel, rng, args.jitter_pos, np.deg2rad(args.jitter_yaw))
                    if board.cell(cand["free"][bowl]["xy"]) != want_cell:
                        continue
                    st, f2 = build_state(cand, e["target_role"], demo0.init_state,
                                         addrs, nq, drawer_adr)
                    R.reset_to_init_state(env, st)
                    S.apply_fixture_edits(env, f2)
                    S.settle(env, S.SpatialSpec().settle_steps)
                    obs = env.env._get_observations(force_update=True)
                    seg = lut[np.clip(obs["agentview_segmentation_instance"][..., 0],
                                      0, len(lut) - 1)]
                    # both bowls carry the stricter floor, same rule the training
                    # gate applies: an eval scene where the distractor is hidden
                    # does not test the choice the instruction is asking for
                    if all(int((seg == 60 + 10 * i).sum())
                           >= (args.min_px_bowl if inst.startswith("akita_black_bowl")
                               else args.min_px)
                           for i, inst in enumerate(P.FREE_INST)):
                        sc, state, fx = cand, st, f2
                        break
                if sc is None:
                    n_dim += 1
                    continue

                # the settle above is what makes this geometry valid; the variants
                # re-render from the SETTLED state, so they do not re-drop objects
                made = emit_variants(np.asarray(env.sim.get_state().flatten()),
                                     fx, e, base, kept, ee_log)
                if not made:
                    n_dim += 1
                kept += made
        except BaseException:
            # includes KeyboardInterrupt: leave whatever was already on disk alone
            env.close()
            if os.path.exists(hdf5_out):
                os.remove(hdf5_out)
            print(f"[{short}] interrupted -- {hdf5_final} left untouched", flush=True)
            raise
        env.close()
        # commit: the hdf5 and the two JSONs that describe it land together, so a
        # directory is never left holding scenes it has no scene_log for
        if os.path.exists(hdf5_out):
            os.replace(hdf5_out, hdf5_final)
        else:                      # every episode was dropped; nothing to swap in
            print(f"[{short}] no scene survived the gates -- "
                  f"{hdf5_final} left as it was", flush=True)
        json.dump(meta, open(os.path.join(out_dir, "metainfo.json"), "w"), indent=2)
        json.dump(scene_log, open(os.path.join(out_dir, "scene_log.json"), "w"),
                  indent=2)
        json.dump(oc_obs.PHASE_MAP,
                  open(os.path.join(out_dir, "phase_map.json"), "w"), indent=2)
        grand[0] += kept
        grand[1] += n_dim
        spread = ""
        if len(ee_log) > 1:
            byu = defaultdict(list)
            for r, p in zip(scene_log, ee_log):
                byu[r["scene_uid"]].append(p)
            d = [float(np.linalg.norm(p - v[0])) for v in byu.values() for p in v[1:]]
            if d:
                spread = (f", arm start offset med={np.median(d)*100:.1f} cm "
                          f"max={max(d)*100:.1f} cm")
        print(f"[{short}] {kept} eval scenes from {len(eps[task])} geometries "
              f"x{n_var} arm poses"
              + (f", {n_exact} from train geometry" if n_exact else "")
              + (f", {n_dim} geometries dropped by the visibility gate" if n_dim else "")
              + spread, flush=True)

    print(f"\nTOTAL {grand[0]} eval scenes for split={args.split}"
          + (f" ({grand[1]} dropped)" if grand[1] else ""))


if __name__ == "__main__":
    main()
