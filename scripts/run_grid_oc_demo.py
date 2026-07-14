"""Scene-based OC data generation: sample N scenes (target placed continuously
with farthest-point spread, distractors on spaced grid cells, basket ~fixed),
and for each scene synthesize + replay trajectories from `--demos-per-scene`
different source demos. Successful episodes are saved in the output/demo
(LIBERO-OC) format with depth/seg/bbox, per-frame phase_id, pre-step obs/action
alignment, warmup before recording, and post-trajectory hold frames.

Usage:
    .venv\Scripts\python.exe scripts\run_grid_oc_demo.py --n-scenes 6 --demos-per-scene 2 --out-dir output\grid_oc_validate
    .venv\Scripts\python.exe scripts\run_grid_oc_demo.py --n-scenes 70 --demos-per-scene 2 --out-dir output\grid_oc_salad_v2
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from demogen_libero.convert import load_demo
from demogen_libero.trajectory import auto_segment, segment_regrasp, synthesize_uniform
from demogen_libero import libero_replay as R
from demogen_libero import oc_obs
from demogen_libero.gridscene import PlacementSpec, sample_scene, apply_scene, object_state_slices

from demogen_libero.config import DATA_DIR as SOURCE_BASE_DIR


def resolve_task(name: str) -> str:
    """Accept a short target name ('salad_dressing') or a full task key."""
    if name.startswith("pick_up_the_"):
        return name
    return f"pick_up_the_{name}_and_place_it_in_the_basket"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="salad_dressing",
                        help="target name (e.g. butter) or full task key")
    parser.add_argument("--source-base-dir", type=str, default=SOURCE_BASE_DIR)
    parser.add_argument("--source-demos", nargs="+", default=None,
                        help="source demo keys; if omitted, read from --sources-json or use demo_0..N")
    parser.add_argument("--sources-json", type=str, default=None,
                        help="per-task screened sources json (from screen_sources.py)")
    parser.add_argument("--n-scenes", type=int, default=6)
    parser.add_argument("--demos-per-scene", type=int, default=1,
                        help="how many different source demos to roll out on each scene")
    parser.add_argument("--n-distractors", type=int, default=-1,
                        help="-1 = random 0..5 per scene")
    parser.add_argument("--min-spacing", type=float, default=0.10)
    parser.add_argument("--x-range", type=float, nargs=2, default=[-0.26, 0.18])
    parser.add_argument("--y-range", type=float, nargs=2, default=[-0.24, 0.18])
    parser.add_argument("--cell-size", type=float, default=0.11)
    parser.add_argument("--basket-clearance", type=float, default=0.12)
    parser.add_argument("--primary-mode", choices=["score", "random"], default="score",
                        help="target placement: score = farthest-from-used spread, "
                             "random = uniform (use for top-up when spread saturates at edges)")
    parser.add_argument("--scene-retries", type=int, default=3,
                        help="if a rollout fails, retry that slot with another source demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--segment", choices=["auto", "regrasp"], default="auto",
                        help="regrasp folds grasp-release-regrasp cycles into skill_1")
    parser.add_argument("--out-dir", type=str, default=os.path.join("output", "grid_oc_validate"))
    parser.add_argument("--no-depth-mm", action="store_true")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--no-dump-aux", action="store_true",
                        help="skip auto-exporting camera_params.json + object_geometry.json")
    parser.add_argument("--no-viz", action="store_true",
                        help="skip auto-generating init-state mosaic + sample phase videos")
    args = parser.parse_args()

    TASK_KEY = resolve_task(args.task)
    cfg = oc_obs.load_task_config()[TASK_KEY]
    TARGET_JOINT = cfg["target_joint"]
    BASKET_JOINT = cfg["basket_joint"]
    DISTRACTOR_JOINTS = cfg["distractor_joints"]
    HDF5_PATH = os.path.join(args.source_base_dir, f"{TASK_KEY}_demo.hdf5")

    # resolve source demo list: explicit > sources-json > all demos in the file
    if args.source_demos:
        source_demos = args.source_demos
    elif args.sources_json and os.path.exists(args.sources_json):
        screened = json.load(open(args.sources_json))[TASK_KEY]
        source_demos = screened["healthy"] if args.segment == "auto" else screened["regrasp"]
    else:
        import h5py
        with h5py.File(HDF5_PATH, "r") as f:
            source_demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    args.source_demos = source_demos

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    hdf5_out = os.path.join(args.out_dir, f"{TASK_KEY}_demo.hdf5")
    meta_out = os.path.join(args.out_dir, "metainfo.json")
    with open(os.path.join(args.out_dir, "phase_map.json"), "w") as f:
        json.dump(oc_obs.PHASE_MAP, f, indent=2)

    spec = PlacementSpec(
        x_range=tuple(args.x_range), y_range=tuple(args.y_range),
        cell_size=args.cell_size, min_spacing=args.min_spacing,
        basket_clearance=args.basket_clearance, primary_mode=args.primary_mode,
    )

    segment_fn = segment_regrasp if args.segment == "regrasp" else auto_segment
    sources = {}
    for key in source_demos:
        d = load_demo(HDF5_PATH, key)
        sources[key] = (d, segment_fn(d.state, d.action))
    any_demo = next(iter(sources.values()))[0]
    env = oc_obs.make_oc_env(any_demo.bddl_file)
    env.reset()
    obj_order = oc_obs.OBJECT_ORDER[TASK_KEY]
    lut = oc_obs.build_seg_lut(env, obj_order)
    extract = lambda e, obs: oc_obs.extract_oc_frame(
        e, obs, lut, object_order=obj_order, depth_mm=not args.no_depth_mm)

    # reference basket position from the first source's init state
    R.reset_to_init_state(env, any_demo.init_state)
    b = object_state_slices(env, [BASKET_JOINT])[BASKET_JOINT]
    ref_basket_xy = any_demo.init_state[b:b + 2].copy()

    # agentview camera ground position for the anti-occlusion ray filter
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    cam_xy = get_camera_extrinsic_matrix(env.sim, "agentview")[:2, 3].copy()

    seg_ids_for = {j: 60 + 10 * obj_order.index(j.replace("_joint0", ""))
                   for j in [TARGET_JOINT, BASKET_JOINT, *DISTRACTOR_JOINTS]}

    # size-aware visibility thresholds: flat/small objects (butter, cream
    # cheese, chocolate pudding: z-extent < 5 cm) legitimately render only
    # ~80-200 px unoccluded, so a flat 150 px gate rejects them at far
    # positions ("small" is not "occluded")
    sys.path.insert(0, os.path.dirname(__file__))
    from dump_object_geometry import measure_task
    _geom = measure_task(env, obj_order)
    min_px_for = {}
    for j in seg_ids_for:
        inst = j.replace("_joint0", "")
        ext = _geom.get(oc_obs.display_name(inst), {}).get("extents", [1, 1, 1])
        min_px_for[j] = 60 if ext[2] < 0.05 else 150

    def scene_fully_visible(new_init, info):
        """Render frame 0 and require every placed object's seg blob to exceed
        its size-aware pixel threshold -- catches any residual occlusion the
        geometric ray filter missed."""
        obs0 = env.set_init_state(new_init)
        seg = oc_obs.extract_oc_frame(env, obs0, lut, depth_mm=False)["agentview_seg"]
        need = [TARGET_JOINT, BASKET_JOINT, *info.get("distractor_joints", [])]
        return all(int((seg == seg_ids_for[j]).sum()) >= min_px_for[j] for j in need)

    # resume/append: continue numbering after demos already in the output file
    # (used by the regrasp batch appending into the same task directory)
    n_success = 0
    meta = {TASK_KEY: {}}
    scene_log = []
    if os.path.exists(hdf5_out):
        import h5py
        with h5py.File(hdf5_out, "r") as f:
            n_success = len(f["data"].keys()) if "data" in f else 0
        if os.path.exists(meta_out):
            meta = json.load(open(meta_out, encoding="utf-8"))
            meta.setdefault(TASK_KEY, {})
        log_path = os.path.join(args.out_dir, "scene_log.json")
        if os.path.exists(log_path):
            scene_log = json.load(open(log_path, encoding="utf-8"))
        print(f"[resume] {n_success} existing demos in {hdf5_out}; appending", flush=True)

    def dump_meta():
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    used_target_xy = [np.array(r["scene"]["target_xy"]) for r in scene_log if "scene" in r]
    try:
        for scene_i in range(args.n_scenes):
            n_dist = args.n_distractors if args.n_distractors >= 0 else int(rng.integers(0, 6))
            # sample until the scene passes both the geometric anti-occlusion
            # filter (inside sample_scene) and the rendered seg-visibility check
            scene = None
            for _try in range(10):
                cand = None
                nd = n_dist
                while cand is None:
                    try:
                        cand = sample_scene(rng, spec, nd, ref_basket_xy, used_target_xy,
                                            cam_xy=cam_xy, distractor_pool=DISTRACTOR_JOINTS)
                    except RuntimeError:
                        nd -= 1  # tight bounds may not fit many distractors
                probe_init, probe_info = apply_scene(cand, any_demo.init_state, env,
                                                     TARGET_JOINT, DISTRACTOR_JOINTS,
                                                     BASKET_JOINT, rng, spec)
                if scene_fully_visible(probe_init, probe_info):
                    scene = cand
                    n_dist = nd
                    break
                print(f"[scene {scene_i}] visibility check failed (try {_try + 1}); resampling", flush=True)
            if scene is None:
                print(f"[scene {scene_i}] could not find a fully visible scene; skipping", flush=True)
                continue
            used_target_xy.append(np.array(scene["target_xy"]))

            pool = list(rng.permutation(args.source_demos))
            wanted = args.demos_per_scene
            got = 0
            tried = []
            while pool and got < wanted and len(tried) < wanted + args.scene_retries:
                src_key = pool.pop(0)
                tried.append(src_key)
                demo, frames = sources[src_key]
                try:
                    R.reset_to_init_state(env, demo.init_state)
                    new_init, info = apply_scene(scene, demo.init_state, env, TARGET_JOINT,
                                                 DISTRACTOR_JOINTS, BASKET_JOINT, rng, spec)
                    obj_t = np.array([*(np.array(info["target_new_xy"]) - np.array(info["target_old_xy"])), 0.0])
                    tar_t = np.array([*info["basket_delta"], 0.0])
                    ref, base_actions, new_frames = synthesize_uniform(
                        demo.state, demo.action, frames, obj_t, tar_t)
                    R.reset_to_init_state(env, new_init)
                    success, obs, rollout = R.replay_uniform(
                        env, base_actions, ref, new_frames, collect=True, extract=extract)
                except Exception as exc:
                    print(f"[scene {scene_i}] src={src_key} ERROR: {exc!r}", flush=True)
                    continue
                print(f"[scene {scene_i}] src={src_key} n_dist={n_dist} "
                      f"target={np.round(scene['target_xy'], 3)} -> success={success} "
                      f"(total {n_success})", flush=True)
                if not success:
                    continue
                demo_name = f"demo_{n_success}"
                rollout = {k: np.asarray(v) for k, v in rollout.items()}
                target_nm = oc_obs.display_name(obj_order[0])
                goal_nm = oc_obs.display_name(obj_order[1])
                rollout["subtask_id"], subtasks = oc_obs.annotate_subtasks(
                    rollout["actions"], target_nm, goal_nm)
                oc_obs.write_oc_demo(hdf5_out, demo_name, rollout, success=True,
                                     object_order=obj_order, subtasks=subtasks)
                meta[TASK_KEY][demo_name] = oc_obs.metainfo_entry(
                    TASK_KEY, rollout, obj_order, new_init, subtasks=subtasks)
                if n_success % 10 == 0:
                    dump_meta()
                if args.save_videos:
                    R.save_video(os.path.join(args.out_dir, f"{demo_name}.mp4"),
                                 rollout["agentview_rgb"][:, ::-1])
                scene_log.append({"demo_name": demo_name, "scene_id": scene_i,
                                  "source_demo": src_key, "seed": args.seed,
                                  "scene": scene, **info})
                n_success += 1
                got += 1
    finally:
        env.close()
        dump_meta()
        with open(os.path.join(args.out_dir, "scene_log.json"), "w") as f:
            json.dump(scene_log, f, indent=2)

    # auto-export camera params + object geometry alongside the dataset so it is
    # self-contained (env already closed; the dumpers open their own)
    import subprocess
    here = os.path.dirname(__file__)
    py = sys.executable
    if not args.no_dump_aux and n_success > 0:
        for script, out_name in [("dump_camera_params.py", "camera_params.json"),
                                 ("dump_object_geometry.py", "object_geometry.json")]:
            try:
                subprocess.run([py, os.path.join(here, script), "--task", args.task,
                                "--out", os.path.join(args.out_dir, out_name)], check=True)
            except Exception as exc:
                print(f"[warn] {script} failed: {exc!r}")

    # auto-visualize the task's dataset: init-state mosaic + sample phase videos
    if not args.no_viz and n_success > 0:
        try:
            subprocess.run([py, os.path.join(here, "visualize_init_states.py"),
                            "--dir", args.out_dir], check=True)
        except Exception as exc:
            print(f"[warn] init-state viz failed: {exc!r}")
        samples = sorted({0, n_success // 2, n_success - 1})
        try:
            subprocess.run([py, os.path.join(here, "visualize_phases.py"),
                            "--dir", args.out_dir,
                            "--demos", *[f"demo_{i}" for i in samples]], check=True)
        except Exception as exc:
            print(f"[warn] phase viz failed: {exc!r}")

    print(f"\nCollected {n_success} episodes over {args.n_scenes} scenes "
          f"({args.demos_per_scene} demos/scene requested).")
    print(f"HDF5:     {hdf5_out}\nmetainfo: {meta_out}")


if __name__ == "__main__":
    main()
