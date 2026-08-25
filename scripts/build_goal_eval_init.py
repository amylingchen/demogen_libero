"""Build evaluation init states for the goal 12-layout suite (plan §8.2).

The init states MUST come out of the same protocol as the training demos --
plan §6.2 records the failure mode: an eval side regenerated with a different
protocol drifted the arm 6.6 cm out of the training distribution and depressed
the unseen readings. So this reuses run_goal_generation's exact pipeline:

    layout -> per-demo jitter (objects +-2.5cm, fixtures +-1cm)
           -> leakage guard (>=0.06 m from every layout on the opposite split
              side) + reach envelope
           -> robot arm qpos noise (sigma 0.02 rad, accepted only if the warmed
              up EE lands within 2.5 cm of the unperturbed EE)
           -> 5 un-recorded warmup steps (same as replay_uniform)
           -> settle + visibility check
           -> save the flattened sim state

No trajectory is replayed, so this is cheap; the states are the same
distribution the policy was trained from.

Fixtures are welded bodies whose poses live in model.body_pos, NOT in the
state vector, and env.reset() wipes them -- every init therefore carries its
`fixture_edits` and the evaluator MUST re-apply them after each reset
(spatial_scene.apply_fixture_edits).

Usage:
    .venv\\Scripts\\python.exe scripts\\build_goal_eval_init.py \\
        --manifest output/goal_suite_12_v3/manifest.json \\
        --out-dir output/goal_eval_init_v3 --n-per-cell 20
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np

from demogen_libero.convert import load_demo
from demogen_libero import libero_replay as R
from demogen_libero import oc_obs
from demogen_libero import spatial_scene as S
from demogen_libero import goal_scene as G

from libero.libero import get_libero_path
from run_goal_generation import jitter_layout, guard_ok

PUSH = "push_the_plate_to_the_front_of_the_stove"
ENTITIES = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1",
            "wooden_cabinet_1", "flat_stove_1", "wine_rack_1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join("output", "goal_suite_12_v3", "manifest.json"))
    ap.add_argument("--source-base-dir", default="D:/Data/LingLing/libero/hf/libero_goal")
    ap.add_argument("--out-dir", default=os.path.join("output", "goal_eval_init_v3"))
    ap.add_argument("--n-per-cell", type=int, default=20)
    ap.add_argument("--splits", nargs="+",
                    default=["train", "quarantine_cf", "quarantine_unseen"],
                    help="train inits are the reference control for cf/unseen readings")
    ap.add_argument("--robot-noise", type=float, default=0.02)
    ap.add_argument("--jitter-floor", type=float, default=0.06)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=307)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    man = json.load(open(args.manifest))
    by_id = {l["id"]: l for l in man["layouts"]}
    unseen_layouts = [by_id[i] for i in man["unseen_ids"]]
    seen_layouts = [by_id[i] for i in man["seen_ids"]]
    spec = G.GoalSpec()
    os.makedirs(args.out_dir, exist_ok=True)

    cells = []
    for task, ls in man["train_matrix"].items():
        if "train" in args.splits:
            cells += [(task, l, "train") for l in ls]
    for task, ls in man["cf_matrix"].items():
        if "quarantine_cf" in args.splits:
            cells += [(task, l, "quarantine_cf") for l in ls]
    if "quarantine_unseen" in args.splits:
        for task in man["train_matrix"]:
            if task != PUSH:
                cells += [(task, l, "quarantine_unseen") for l in man["unseen_ids"]]
    print(f"{len(cells)} cells x {args.n_per_cell} init states", flush=True)

    by_task = {}
    for c in cells:
        by_task.setdefault(c[0], []).append(c)

    log = {}
    for task, task_cells in by_task.items():
        h5src = os.path.join(args.source_base_dir, f"{task}_demo.hdf5")
        demo = load_demo(h5src, "demo_0")
        env = oc_obs.make_oc_env(demo.bddl_file)
        env.reset()
        fixture_ref = G.capture_fixture_ref(env)
        demo_layout = G.read_demo_layout(env, demo.init_state, fixture_ref)
        base_ee = np.asarray(demo.state[0])
        names = list(env.env.model.instances_to_ids.keys())

        for (_, lid, split) in task_cells:
            layout = by_id[lid]["layout"]
            opposite = seen_layouts if split == "quarantine_unseen" else unseen_layouts
            out = os.path.join(args.out_dir, split)
            os.makedirs(out, exist_ok=True)
            path = os.path.join(out, f"{task}_init.hdf5")
            got, tried = 0, 0
            while got < args.n_per_cell and tried < args.n_per_cell * 12:
                tried += 1
                jl = None
                for _ in range(60):
                    cand = jitter_layout(rng, layout)
                    if guard_ok(cand, opposite, args.jitter_floor):
                        jl = cand
                        break
                if jl is None:
                    break
                new_init, fx = G.apply_goal_layout(jl, demo.init_state, demo_layout)
                noise_used = np.zeros(7)
                for _n in range(5):
                    noise = rng.normal(0, args.robot_noise, 7)
                    pert = new_init.copy()
                    pert[1:8] = pert[1:8] + noise
                    R.reset_to_init_state(env, pert)
                    S.apply_fixture_edits(env, fx)
                    o = env.env._get_observations(force_update=True)
                    if np.linalg.norm(np.asarray(o["robot0_eef_pos"]) - base_ee) < 0.025:
                        noise_used, new_init = noise, pert
                        break
                else:
                    R.reset_to_init_state(env, new_init)
                    S.apply_fixture_edits(env, fx)
                # identical warmup to replay_uniform, so the saved state sits at
                # the same point of the episode as a training demo's states[0]
                warm = np.zeros(7)
                warm[6] = -1.0
                for _ in range(max(args.warmup, 1)):
                    env.step(warm)
                rep = S.settle(env, spec.settle_steps)
                obs = env.env._get_observations(force_update=True)
                raw = obs["agentview_segmentation_instance"][..., 0]
                px = {nm: int((raw == names.index(nm) + 1).sum()) for nm in ENTITIES}
                if not (rep["converged"] and rep["max_disp_cm"] < 1.0
                        and all(v >= spec.min_px for v in px.values())):
                    continue
                st = env.sim.get_state().flatten()
                with h5py.File(path, "a") as f:
                    data = f.require_group("data")
                    if "bddl_file_name" not in data.attrs:
                        data.attrs["bddl_file_name"] = (
                            f"libero/libero/bddl_files/libero_goal/{task}.bddl")
                        data.attrs["manifest"] = os.path.abspath(args.manifest)
                        data.attrs["protocol"] = (
                            "same as run_goal_generation: jitter + leakage guard + "
                            "reach envelope + robot noise (EE<2.5cm) + 5 warmup steps")
                    key = f"init_{len(data.keys())}"
                    g = data.create_group(key)
                    g.create_dataset("state", data=st)
                    g.attrs["layout_id"] = lid
                    g.attrs["split"] = split
                    g.attrs["robot_noise"] = noise_used
                    g.attrs["jitter_objects"] = json.dumps(jl["objects"])
                    g.attrs["jitter_fixtures"] = json.dumps(jl["fixtures"])
                    g.attrs["fixture_edits"] = json.dumps(
                        {fb: {"pos": e["pos"].tolist(), "quat_wxyz": e["quat"].tolist()}
                         for fb, e in fx.items()})
                got += 1
            log[f"{task}|{lid}|{split}"] = {"n": got, "tried": tried}
            print(f"[{task[:38]:38s}|{lid}|{split.replace('quarantine_','')}] "
                  f"{got}/{args.n_per_cell} ({tried} tried)", flush=True)
        env.close()

    with open(os.path.join(args.out_dir, "init_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    total = sum(v["n"] for v in log.values())
    short = {k: v["n"] for k, v in log.items() if v["n"] < args.n_per_cell}
    print(f"\nDONE: {total} init states over {len(log)} cells; short cells: {len(short)}")
    for k, v in short.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
