"""Full-grid trajectory generation for the goal 12-layout suite
(output/goal_suite_12/manifest.json; plan docs/LIBERO_goal_12布局_轨迹生成计划.md §6).

Cells: 8 non-push tasks x (4 train + 4 cf + 4 unseen) + push x its 4 verified
train cells = 100 cells x --demos-per-cell trajectories. cf/unseen
trajectories are the D6 feasibility proof and are written to QUARANTINED
files (train/, quarantine_cf/, quarantine_unseen/).

Per-demo protocol:
  - layout jitter: objects +-2.5cm, fixtures +-1cm, fixture yaw 0 (the plan's
    +-5 deg was never gate-validated; recorded as 0 in the manifest)
  - leakage guard (round-2 B5): every jittered entity keeps >=0.06m from the
    same entity in every layout of the OPPOSITE split side
    (train/cf demos guard against the 4 unseen layouts; unseen demos guard
    against the 8 seen layouts) -- jitter is resampled until it passes
  - source selection (plan §6.1): healthy pool only (goal_source_screening),
    top-k=5 by summed anchor-entity distance to the jittered placement,
    uniform random inside the top-k; source id recorded per demo
  - robot init perturbation (plan §6.2): arm qpos noise sigma=0.02 rad,
    accepted only if the warmed-up EE lands within 2.5cm of the unperturbed
    EE (resampled up to 5x, else no noise)
  - acceptance: END-STATE task predicate; failures are discarded and retried
    (attempts recorded -> per-cell yield in the generation log)

Format: standard LIBERO-style rollout (obs pre-step; states are the full
79-dim flattened sim state) so OC observations (seg/depth/bbox, plan §7) can
be re-rendered later by state replay without re-running physics. Per-demo
attrs record layout_id, split, source, jitter, robot noise, fixture edits
(pos + wxyz quat).

Resumable: per-(split,task) hdf5s are append-only; a cell with enough demos
is skipped on restart.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_goal_generation.py --out-dir output/goal_gen_500
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np

from demogen_libero.convert import load_demo
from demogen_libero.trajectory import synthesize_uniform
from demogen_libero import libero_replay as R
from demogen_libero import oc_obs
from demogen_libero import spatial_scene as S
from demogen_libero import goal_scene as G

from libero.libero import get_libero_path
from smoke_goal_traj import frames_for

PUSH = "push_the_plate_to_the_front_of_the_stove"

# anchor entities per task, used for §6.1 source-proximity selection
TASK_ANCHORS = {
    "open_the_middle_drawer_of_the_cabinet": ["wooden_cabinet_1_main"],
    "turn_on_the_stove": ["flat_stove_1_main"],
    "put_the_bowl_on_the_plate": ["akita_black_bowl_1_joint0", "plate_1_joint0"],
    "put_the_bowl_on_the_stove": ["akita_black_bowl_1_joint0", "flat_stove_1_main"],
    "put_the_bowl_on_top_of_the_cabinet": ["akita_black_bowl_1_joint0",
                                           "wooden_cabinet_1_main"],
    "put_the_cream_cheese_in_the_bowl": ["cream_cheese_1_joint0",
                                         "akita_black_bowl_1_joint0"],
    "put_the_wine_bottle_on_the_rack": ["wine_bottle_1_joint0", "wine_rack_1_main"],
    "put_the_wine_bottle_on_top_of_the_cabinet": ["wine_bottle_1_joint0",
                                                  "wooden_cabinet_1_main"],
    PUSH: ["plate_1_joint0"],
}


# Final-pose acceptance, per task (measured 2026-08-24 against the source
# demos; the goal predicates are region-containment only and pass a fallen
# object). Quaternion is wxyz; upright = body z-axis' world z-component.
#   bottle->cabinet-top: originals end 0.98 (min 0.81); v1 generated ended 0.78
#     (min -0.31) with 20/60 tilted >25deg -- bottles lying on the cabinet top.
#   push: originals 0.99 (min 0.93); v1 had 3/20 down to 0.61.
#   NOT gated: bottle->RACK (originals lie at 0.51 by design -- a blanket
#     upright rule would wrongly reject all 60), the bowl tasks (already 1.00),
#     and cheese->bowl (the box tumbles into the bowl in the originals too).
POSE_CRITERIA = {
    "put_the_wine_bottle_on_top_of_the_cabinet": ("wine_bottle_1_joint0", 0.80),
    "push_the_plate_to_the_front_of_the_stove": ("plate_1_joint0", 0.90),
}
QUAT_COL = {"akita_black_bowl_1_joint0": 13, "cream_cheese_1_joint0": 20,
            "wine_bottle_1_joint0": 27, "plate_1_joint0": 34}


# Joint-limit acceptance. The reach envelope above is prophylactic; this is the
# direct measurement -- a replayed trajectory that drives any arm joint to its
# hard limit is rejected. EXCEPTION: put_the_wine_bottle_on_top_of_the_cabinet,
# where 30/30 SOURCE demos already run at the joint-4 limit (margin 0.007), so
# the saturation is inherited from the human demonstrations and gating it would
# reject every candidate. The per-demo margin is stored either way.
JOINT_LIMITS = [(-2.897, 2.897), (-1.763, 1.763), (-2.897, 2.897), (-3.072, -0.070),
                (-2.897, 2.897), (-0.018, 3.752), (-2.897, 2.897)]
JOINT_MARGIN_FLOOR = 0.05
JOINT_GATE_EXEMPT = {"put_the_wine_bottle_on_top_of_the_cabinet"}


def joint_margin(joint_states):
    js = np.asarray(joint_states)
    return float(min(min(abs(js[:, j].min() - JOINT_LIMITS[j][0]),
                         abs(js[:, j].max() - JOINT_LIMITS[j][1])) for j in range(7)))


def pose_ok(task, final_state):
    """True if the manipulated object's final orientation is acceptable."""
    if task not in POSE_CRITERIA:
        return True, None
    key, floor = POSE_CRITERIA[task]
    c = QUAT_COL[key]
    q = np.asarray(final_state)[c:c + 4]
    upright = float(1 - 2 * (q[1] ** 2 + q[2] ** 2))
    return upright >= floor, round(upright, 3)


def jitter_layout(rng, layout, obj_j=0.025, fix_j=0.01):
    out = {"objects": {}, "fixtures": {}}
    for jn, xy in layout["objects"].items():
        out["objects"][jn] = (np.asarray(xy) + rng.uniform(-obj_j, obj_j, 2)).tolist()
    for fb, xy in layout["fixtures"].items():
        out["fixtures"][fb] = (np.asarray(xy) + rng.uniform(-fix_j, fix_j, 2)).tolist()
    return out


def guard_ok(jl, opposite_layouts, floor=0.06):
    # the jittered placement must ALSO stay inside the arm's demonstrated reach
    # envelope: jitter can push an in-limit layout point past it (2026-08-24)
    if G.layout_reach_violations(jl):
        return False
    for key in list(G.GOAL_JOINTS):
        if not G.jitter_ok(key, jl["objects"][key], opposite_layouts, floor):
            return False
    for key in list(G.GOAL_FIXTURES):
        if not G.jitter_ok(key, jl["fixtures"][key], opposite_layouts, floor):
            return False
    return True


def entity_pos(layout, key):
    return np.asarray(layout["objects" if key.endswith("_joint0") else "fixtures"][key])


def demo_entity_pos(demo_layout, key):
    if key.endswith("_joint0"):
        return demo_layout["free"][key]["pos"][:2]
    return demo_layout["fixtures"][key]["pos"][:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join("output", "goal_suite_12", "manifest.json"))
    ap.add_argument("--source-base-dir", default="D:/Data/LingLing/libero/hf/libero_goal")
    ap.add_argument("--screening", default=os.path.join("output", "goal_source_screening.json"))
    ap.add_argument("--out-dir", default=os.path.join("output", "goal_gen_500"))
    ap.add_argument("--demos-per-cell", type=int, default=5)
    ap.add_argument("--max-attempts-per-cell", type=int, default=15)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--robot-noise", type=float, default=0.02)
    ap.add_argument("--jitter-floor", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="subset of tasks (default: all in the manifest matrices)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    man = json.load(open(args.manifest))
    screening = json.load(open(args.screening))
    by_id = {l["id"]: l for l in man["layouts"]}
    unseen_layouts = [by_id[i] for i in man["unseen_ids"]]
    seen_layouts = [by_id[i] for i in man["seen_ids"]]
    os.makedirs(args.out_dir, exist_ok=True)

    # cells: (task, layout_id, split)
    cells = []
    for task, tl in man["train_matrix"].items():
        cells += [(task, lid, "train") for lid in tl]
    for task, cl in man["cf_matrix"].items():
        cells += [(task, lid, "quarantine_cf") for lid in cl]
    for task in man["train_matrix"]:
        if task != PUSH:
            cells += [(task, lid, "quarantine_unseen") for lid in man["unseen_ids"]]
    if args.tasks:
        cells = [c for c in cells if c[0] in args.tasks]
    print(f"{len(cells)} cells x {args.demos_per_cell} demos target", flush=True)

    gen_log_path = os.path.join(args.out_dir, "generation_log.json")
    gen_log = json.load(open(gen_log_path)) if os.path.exists(gen_log_path) else {}

    def h5path(split, task):
        d = os.path.join(args.out_dir, split)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{task}_demo.hdf5")

    def demos_in_cell(split, task, lid):
        p = h5path(split, task)
        if not os.path.exists(p):
            return 0
        with h5py.File(p, "r") as f:
            if "data" not in f:
                return 0
            return sum(1 for k in f["data"] if f["data"][k].attrs.get("layout_id") == lid)

    # group cells by task so each env is built once
    by_task = {}
    for c in cells:
        by_task.setdefault(c[0], []).append(c)

    for task, task_cells in by_task.items():
        cfg = G.GOAL_TASKS[task]
        h5src = os.path.join(args.source_base_dir, f"{task}_demo.hdf5")
        pool = screening[task]["healthy"]
        env = oc_obs.make_oc_env(os.path.join(
            get_libero_path("bddl_files"), "libero_goal", f"{task}.bddl"))
        env.reset()
        fixture_ref = G.capture_fixture_ref(env)
        # preload sources + frames + per-source layouts
        sources = {}
        for key in pool:
            d = load_demo(h5src, key)
            try:
                frames = frames_for(cfg, d, h5src)
            except AssertionError as exc:
                print(f"[{task}] skip source {key}: {exc}", flush=True)
                continue
            dl = G.read_demo_layout(env, d.init_state, fixture_ref)
            sources[key] = (d, frames, dl)
        print(f"[{task}] {len(sources)} usable sources; {len(task_cells)} cells",
              flush=True)

        for (_, lid, split) in task_cells:
            have = demos_in_cell(split, task, lid)
            if have >= args.demos_per_cell:
                continue
            layout = by_id[lid]["layout"]
            opposite = seen_layouts if split == "quarantine_unseen" else unseen_layouts
            attempts = 0
            got = have
            cell_rec = gen_log.setdefault(f"{task}|{lid}|{split}", {"attempts": []})
            while got < args.demos_per_cell and attempts < args.max_attempts_per_cell:
                attempts += 1
                # jitter + leakage guard
                jl = None
                for _ in range(60):
                    cand = jitter_layout(rng, layout)
                    if guard_ok(cand, opposite, args.jitter_floor):
                        jl = cand
                        break
                if jl is None:
                    print(f"[{task}|{lid}] jitter guard unsatisfiable", flush=True)
                    break
                # source selection: top-k by anchor distance, random within
                def src_dist(key):
                    _, _, dl = sources[key]
                    return sum(np.linalg.norm(entity_pos(jl, a) - demo_entity_pos(dl, a))
                               for a in TASK_ANCHORS[task])
                ranked = sorted(sources, key=src_dist)
                src_key = ranked[int(rng.integers(0, min(args.top_k, len(ranked))))]
                demo, frames, demo_layout = sources[src_key]
                try:
                    obj_t, tar_t = G.anchor_deltas(cfg, jl, demo_layout)
                    ref, base_actions, new_frames = synthesize_uniform(
                        demo.state, demo.action, frames, obj_t, tar_t)
                    new_init, fx = G.apply_goal_layout(jl, demo.init_state, demo_layout)
                    # robot init perturbation with EE-displacement gate
                    noise_used = np.zeros(7)
                    base_ee = np.asarray(demo.state[0])
                    for _n in range(5):
                        noise = rng.normal(0, args.robot_noise, 7)
                        pert = new_init.copy()
                        pert[1:8] = pert[1:8] + noise
                        R.reset_to_init_state(env, pert)
                        S.apply_fixture_edits(env, fx)
                        obs0 = env.env._get_observations(force_update=True)
                        if np.linalg.norm(np.asarray(obs0["robot0_eef_pos"]) - base_ee) < 0.025:
                            noise_used = noise
                            new_init = pert
                            break
                    else:
                        R.reset_to_init_state(env, new_init)
                        S.apply_fixture_edits(env, fx)
                    success, obs, rollout = R.replay_uniform(
                        env, base_actions, ref, new_frames, collect=True)
                    end = bool(env.check_success())
                except Exception as exc:
                    cell_rec["attempts"].append({"source": src_key, "error": repr(exc)})
                    print(f"[{task}|{lid}|{split}] ERROR {exc!r}", flush=True)
                    continue
                ok_pose, upright = pose_ok(task, rollout["states"][-1])
                jm = joint_margin(rollout["joint_states"])
                ok_joint = (task in JOINT_GATE_EXEMPT) or (jm >= JOINT_MARGIN_FLOOR)
                cell_rec["attempts"].append({"source": src_key, "success_end": end,
                                             "pose_ok": ok_pose, "upright": upright,
                                             "joint_margin": round(jm, 4),
                                             "joint_ok": ok_joint})
                if not (end and ok_pose and ok_joint):
                    if end and not ok_pose:
                        print(f"[{task}|{lid}|{split}] predicate OK but POSE REJECT "
                              f"(upright={upright})", flush=True)
                    elif end and not ok_joint:
                        print(f"[{task}|{lid}|{split}] predicate OK but JOINT-LIMIT "
                              f"REJECT (margin={jm:.3f})", flush=True)
                    continue
                p = h5path(split, task)
                with h5py.File(p, "a") as f:
                    data = f.require_group("data")
                    if "bddl_file_name" not in f["data"].attrs:
                        f["data"].attrs["bddl_file_name"] = (
                            f"libero/libero/bddl_files/libero_goal/{task}.bddl")
                        f["data"].attrs["manifest"] = os.path.abspath(args.manifest)
                    key = f"demo_{len(data.keys())}"
                    ep = data.create_group(key)
                    ep.create_dataset("actions", data=np.asarray(rollout["actions"]))
                    ep.create_dataset("states", data=np.asarray(rollout["states"]))
                    og = ep.create_group("obs")
                    for k in ("ee_pos", "gripper_states", "joint_states",
                              "agentview_rgb", "eye_in_hand_rgb"):
                        og.create_dataset(k, data=np.asarray(
                            rollout[{"agentview_rgb": "agentview_rgb",
                                     "eye_in_hand_rgb": "eye_in_hand_rgb"}.get(k, k)]))
                    ep.create_dataset("phase_id", data=np.asarray(rollout["phase_id"]))
                    ep.attrs["layout_id"] = lid
                    ep.attrs["split"] = split
                    ep.attrs["source_demo"] = src_key
                    ep.attrs["robot_noise"] = noise_used
                    if upright is not None:
                        ep.attrs["final_upright"] = float(upright)
                    ep.attrs["joint_margin"] = jm
                    ep.attrs["jitter_objects"] = json.dumps(jl["objects"])
                    ep.attrs["jitter_fixtures"] = json.dumps(jl["fixtures"])
                    ep.attrs["fixture_edits"] = json.dumps(
                        {fb: {"pos": e["pos"].tolist(), "quat_wxyz": e["quat"].tolist()}
                         for fb, e in fx.items()})
                got += 1
                print(f"[{task}|{lid}|{split}] demo {got}/{args.demos_per_cell} "
                      f"(src={src_key}, attempt {attempts})", flush=True)
            cell_rec["yield"] = f"{got - have}/{attempts}"
            cell_rec["total"] = got
            with open(gen_log_path, "w") as f:
                json.dump(gen_log, f, indent=2)
        env.close()

    n_cells_done = sum(1 for k, v in gen_log.items() if v.get("total", 0) >= args.demos_per_cell)
    print(f"\nDONE this pass: {n_cells_done} cells at target; log: {gen_log_path}")


if __name__ == "__main__":
    main()
