"""Build a held-out EVALUATION suite of scenes for a task: sample init states
that are DISJOINT from the training scenes (no source demos rolled out, no
trajectories) so evaluation is out-of-distribution w.r.t. the training set.

Each eval scene is applied onto a source demo's init_state to get a valid full
110-d state, then the env is reset to it and frame-0 OC observations + GT object
poses are rendered (T=1 per scene). Disjointness is enforced by requiring each
eval target position to be >= --min-train-dist from every training target
position (read from the training scene_log.json).

Usage:
    .venv\Scripts\python.exe scripts\build_eval_suite.py --task salad_dressing \
        --train-scene-log output\grid_oc_salad_v2\scene_log.json \
        --n-scenes 30 --out-dir output\eval_salad
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
from demogen_libero.gridscene import PlacementSpec, sample_scene, apply_scene, object_state_slices

from demogen_libero.config import DATA_DIR as SOURCE_BASE_DIR


def resolve_task(name):
    return name if name.startswith("pick_up_the_") else f"pick_up_the_{name}_and_place_it_in_the_basket"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--source-base-dir", type=str, default=SOURCE_BASE_DIR)
    parser.add_argument("--sources-json", type=str, default="output/source_screening.json")
    parser.add_argument("--train-scene-log", type=str, default=None,
                        help="training scene_log.json; eval targets kept away from its targets")
    parser.add_argument("--min-train-dist", type=float, default=0.05,
                        help="min xy distance (m) from any training target position")
    parser.add_argument("--n-scenes", type=int, default=30)
    parser.add_argument("--n-distractors", type=int, default=-1)
    parser.add_argument("--min-spacing", type=float, default=0.10)
    parser.add_argument("--x-range", type=float, nargs=2, default=[-0.26, 0.18])
    parser.add_argument("--y-range", type=float, nargs=2, default=[-0.24, 0.18])
    parser.add_argument("--cell-size", type=float, default=0.11)
    parser.add_argument("--basket-clearance", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--no-depth-mm", action="store_true")
    args = parser.parse_args()

    TASK_KEY = resolve_task(args.task)
    cfg = oc_obs.load_task_config()[TASK_KEY]
    obj_order = cfg["object_order"]
    hdf5_src = os.path.join(args.source_base_dir, f"{TASK_KEY}_demo.hdf5")

    # one healthy source provides the base state layout + robot home
    screened = json.load(open(args.sources_json))[TASK_KEY]
    base_key = screened["healthy"][0]
    base = load_demo(hdf5_src, base_key)

    train_targets = []
    if args.train_scene_log and os.path.exists(args.train_scene_log):
        for r in json.load(open(args.train_scene_log)):
            xy = (r.get("scene") or {}).get("target_xy") or r.get("target_new_xy")
            if xy is not None:
                train_targets.append(np.asarray(xy))
    train_targets = np.array(train_targets) if train_targets else np.zeros((0, 2))

    os.makedirs(args.out_dir, exist_ok=True)
    hdf5_out = os.path.join(args.out_dir, f"{TASK_KEY}_eval.hdf5")
    if os.path.exists(hdf5_out):
        os.remove(hdf5_out)
    meta_out = os.path.join(args.out_dir, "metainfo.json")

    spec = PlacementSpec(x_range=tuple(args.x_range), y_range=tuple(args.y_range),
                         cell_size=args.cell_size, min_spacing=args.min_spacing,
                         basket_clearance=args.basket_clearance)
    rng = np.random.default_rng(args.seed)

    env = oc_obs.make_oc_env(base.bddl_file)
    env.reset()
    lut = oc_obs.build_seg_lut(env, obj_order)
    R.reset_to_init_state(env, base.init_state)
    b = object_state_slices(env, [cfg["basket_joint"]])[cfg["basket_joint"]]
    ref_basket_xy = base.init_state[b:b + 2].copy()
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    cam_xy = get_camera_extrinsic_matrix(env.sim, "agentview")[:2, 3].copy()

    meta = {TASK_KEY: {}}
    scene_log = []
    kept = 0
    used_target_xy = [t for t in train_targets]  # spread eval away from train too
    # adaptive disjointness: when training targets densely cover the workspace
    # the requested min-train-dist may be unsatisfiable; halve until the suite
    # fills (the ID-eval trap being avoided is eval == exact training inits,
    # and distractor layouts / basket jitter differ regardless)
    min_dist = args.min_train_dist
    try:
        for i in range(args.n_scenes):
            # sample a scene whose target is far enough from all training targets
            scene = None
            while scene is None and min_dist > 0.005:
                for _ in range(200):
                    nd = args.n_distractors if args.n_distractors >= 0 else int(rng.integers(0, 6))
                    cand = None
                    while cand is None:
                        try:
                            cand = sample_scene(rng, spec, nd, ref_basket_xy, used_target_xy,
                                                cam_xy=cam_xy, distractor_pool=cfg["distractor_joints"])
                        except RuntimeError:
                            nd -= 1
                    txy = np.asarray(cand["target_xy"])
                    if train_targets.shape[0] == 0 or \
                            np.linalg.norm(train_targets - txy, axis=1).min() >= min_dist:
                        scene = cand
                        break
                if scene is None:
                    min_dist /= 2.0
                    print(f"[eval] relaxing min-train-dist to {min_dist*100:.2f} cm", flush=True)
            if scene is None:
                continue
            used_target_xy.append(np.asarray(scene["target_xy"]))

            new_init, info = apply_scene(scene, base.init_state, env, cfg["target_joint"],
                                         cfg["distractor_joints"], cfg["basket_joint"], rng, spec)
            obs = env.set_init_state(new_init)
            row = oc_obs.extract_oc_frame(env, obs, lut, object_order=obj_order,
                                          depth_mm=not args.no_depth_mm)
            row["actions"] = np.zeros(7)
            row["phase_id"] = np.int32(0)
            rollout = {k: np.asarray(v)[None] for k, v in row.items()}  # T=1
            rollout["states"] = new_init[None]
            demo_name = f"eval_{kept}"
            oc_obs.write_oc_demo(hdf5_out, demo_name, rollout, success=True, object_order=obj_order)
            meta[TASK_KEY][demo_name] = oc_obs.metainfo_entry(TASK_KEY, rollout, obj_order, new_init)
            scene_log.append({"demo_name": demo_name, "scene": scene, "eval": True, **info})
            kept += 1
    finally:
        env.close()

    json.dump(meta, open(meta_out, "w", encoding="utf-8"), indent=2)
    json.dump(scene_log, open(os.path.join(args.out_dir, "scene_log.json"), "w"), indent=2)
    # min distance achieved between eval and train targets
    if kept and train_targets.shape[0]:
        ev = np.array([s["scene"]["target_xy"] for s in scene_log])
        dmin = min(np.linalg.norm(train_targets - e, axis=1).min() for e in ev)
        print(f"eval targets min distance to any train target: {dmin*100:.1f} cm (threshold {args.min_train_dist*100:.0f})")
    print(f"wrote {kept} eval scenes -> {hdf5_out}")


if __name__ == "__main__":
    main()
