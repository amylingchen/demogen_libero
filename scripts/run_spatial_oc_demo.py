"""DemoGen-style OC data generation for LIBERO_Spatial with the relational
rigid-group scheme (src/demogen_libero/spatial_scene.py).

Each scene rigidly relocates {target bowl + spatial-relation anchors + anchor
fixture} together (planar translate + yaw), so the instruction stays true, then
synthesizes + replays each source demo onto it. Fixture-anchored tasks (stove /
cabinet / drawer) move the fixture via model.body_pos, re-applied after every
env reset. Successful episodes are written in the output/demo (LIBERO-OC) format
with depth/seg/bbox, per-frame phase_id, and subtask annotation -- identical
layout to the libero_object datasets.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_spatial_oc_demo.py \
        --task pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate \
        --n-scenes 6 --demos-per-scene 2 --out-dir output\\spatial_validate --save-videos
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
from demogen_libero import spatial_scene as S


def segment_for(demo):
    """Pick a segmentation per source demo (mixed single-grasp + regrasp batch):
    a demo with >1 gripper-close cycle is a REGRASP (grab-release-regrasp) -> fold
    the whole recovery into skill_1 via segment_regrasp (replayed verbatim, so the
    recovery behavior is preserved and is translation-invariant). Fall back to
    auto_segment when segment_regrasp yields invalid frames (e.g. a 1-frame end
    twitch after release, which is not a real regrasp). Raises AssertionError for
    demos with no clean grasp cycle (caller skips them)."""
    grip = demo.action[:, 6]
    n_close = int(np.sum((grip[:-1] <= 0) & (grip[1:] > 0)))
    if n_close > 1:
        try:
            fr = segment_regrasp(demo.state, demo.action)
            f1, f2, f3 = fr.as_tuple()
            if 0 < f1 < f2 < f3 < len(demo.action):
                return fr, True
        except AssertionError:
            pass
    return auto_segment(demo.state, demo.action), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="full libero_spatial task key")
    ap.add_argument("--source-base-dir", default="D:/Data/LingLing/libero/hf/libero_spatial")
    ap.add_argument("--source-demos", nargs="+", default=None,
                    help="source demo keys; default = all demos in the file")
    ap.add_argument("--sources-json", default=None,
                    help="screened sources json {task: {healthy: [...]}}")
    ap.add_argument("--n-scenes", type=int, default=20)
    ap.add_argument("--demos-per-scene", type=int, default=5)
    ap.add_argument("--scene-retries", type=int, default=4)
    ap.add_argument("--jitter-pos", type=float, default=0.02,
                    help="per-demo planar position jitter within a scene (m)")
    ap.add_argument("--jitter-yaw", type=float, default=3.0,
                    help="per-demo yaw jitter within a scene (degrees)")
    ap.add_argument("--min-px", type=int, default=100,
                    help="min seg pixels per object for the visibility gate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join("output", "spatial_validate"))
    ap.add_argument("--no-depth-mm", action="store_true")
    ap.add_argument("--save-videos", action="store_true")
    args = ap.parse_args()

    TASK = args.task
    cfg = S.load_spatial_config()[TASK]
    all_joints = list(dict.fromkeys(
        [cfg["target_joint"], cfg["dest_joint"], *cfg["group_joints"], *cfg["indep_joints"]]))
    obj_order = cfg["object_order"]
    HDF5_PATH = os.path.join(args.source_base_dir, f"{TASK}_demo.hdf5")

    # resolve source demos
    if args.source_demos:
        source_demos = args.source_demos
    elif args.sources_json and os.path.exists(args.sources_json):
        source_demos = json.load(open(args.sources_json))[TASK]["healthy"]
    else:
        import h5py
        with h5py.File(HDF5_PATH, "r") as f:
            source_demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    hdf5_out = os.path.join(args.out_dir, f"{TASK}_demo.hdf5")
    meta_out = os.path.join(args.out_dir, "metainfo.json")
    with open(os.path.join(args.out_dir, "phase_map.json"), "w") as f:
        json.dump(oc_obs.PHASE_MAP, f, indent=2)

    spec = S.SpatialSpec()
    jitter_yaw_rad = np.deg2rad(args.jitter_yaw)

    # preload sources with their segmentation frames + per-demo layout; mixed
    # single-grasp + regrasp (segment chosen per demo), skipping demos with no
    # clean grasp cycle
    sources = {}
    n_regrasp = 0
    for key in source_demos:
        d = load_demo(HDF5_PATH, key)
        try:
            frames, is_regrasp = segment_for(d)
            sources[key] = (d, frames)
            n_regrasp += int(is_regrasp)
        except AssertionError as exc:
            print(f"[skip source] {key}: {exc}", flush=True)
    source_demos = [k for k in source_demos if k in sources]
    print(f"[sources] {len(source_demos)} usable ({n_regrasp} regrasp, "
          f"{len(source_demos) - n_regrasp} single-grasp)", flush=True)
    if not source_demos:
        raise RuntimeError("no usable source demos after segmentation screening")
    any_demo = sources[source_demos[0]][0]

    env = oc_obs.make_oc_env(any_demo.bddl_file)
    env.reset()
    lut = oc_obs.build_seg_lut(env, obj_order)
    extract = lambda e, obs: oc_obs.extract_oc_frame(
        e, obs, lut, object_order=obj_order, depth_mm=not args.no_depth_mm)
    seg_ids = {j: 60 + 10 * obj_order.index(j.replace("_joint0", ""))
               for j in all_joints if j.replace("_joint0", "") in obj_order}

    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    cam_xy = get_camera_extrinsic_matrix(env.sim, "agentview")[:2, 3].copy()

    # per-demo reference layouts (read once)
    layouts = {}
    for key, (d, _) in sources.items():
        R.reset_to_init_state(env, d.init_state)
        layouts[key] = S.read_layout(env, d.init_state, all_joints, S.ALL_FIXTURES)
    ref_layout = layouts[source_demos[0]]

    def scene_visible(new_init, fx):
        R.reset_to_init_state(env, new_init)
        S.apply_fixture_edits(env, fx)
        S.settle(env, spec.settle_steps)
        # force_update is required, not optional (same fix as
        # run_pair_oc_demo.py's `usable`): settle() drives env.sim.step()
        # directly and never touches robosuite's observable cache, which
        # reset_to_init_state() left holding the env's DEFAULT reset scene.
        # Without it this gate renders a layout that is not the one being
        # tested, and the default layout has every object in plain view -- so
        # it passes every candidate. Measured on libero_spatial_pairs demos
        # that are known-occluded: without force_update the gate returns
        # near-constant counts across totally different scenes (ramekin
        # 499-525 px in every one) and passed 8/8; with it, it returns the
        # true counts (ramekin 0-38 px) and rejected 8/8.
        obs = env.env._get_observations(force_update=True)
        raw = obs["agentview_segmentation_instance"][..., 0]
        segids = lut[np.clip(raw, 0, len(lut) - 1)]
        return all(int((segids == sid).sum()) >= args.min_px for sid in seg_ids.values())

    n_success = 0
    meta = {TASK: {}}
    scene_log = []
    if os.path.exists(hdf5_out):
        import h5py
        with h5py.File(hdf5_out, "r") as f:
            n_success = len(f["data"].keys()) if "data" in f else 0
        if os.path.exists(meta_out):
            meta = json.load(open(meta_out, encoding="utf-8")); meta.setdefault(TASK, {})
        print(f"[resume] {n_success} existing demos; appending", flush=True)

    def dump_meta():
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # drive to an ABSOLUTE target demo COUNT: a scene that can't be placed/made
    # visible is skipped and more scenes attempted; on resume (n_success already
    # >0 from a prior crashed run) this tops up to the same target, not beyond.
    # OOD cell coverage comes from uniform-random target placement (no global
    # farthest-point spread -- that saturates and can't reach the target count).
    target_demos = args.n_scenes * args.demos_per_scene
    try:
        for scene_i in range(args.n_scenes * 6):
            if n_success >= target_demos:
                break
            # sample a visible scene; on a visibility miss, add the failed target
            # to a local avoid list so the next try picks a different spot
            scene = None
            avoid = []
            for _try in range(20):
                try:
                    cand = S.sample_scene(rng, spec, cfg, ref_layout, cam_xy, avoid)
                    probe_init, probe_fx, *_ = S.apply_scene(cand, any_demo.init_state, env, cfg, ref_layout)
                except RuntimeError:
                    continue
                if scene_visible(probe_init, probe_fx):
                    scene = cand
                    break
                avoid.append(np.array(cand["target_new_xy"]))
            if scene is None:
                print(f"[scene {scene_i}] no visible scene found; skipping", flush=True)
                continue

            pool = list(rng.permutation(source_demos))
            got = 0
            tried = 0
            while pool and got < args.demos_per_scene and tried < args.demos_per_scene + args.scene_retries:
                src_key = pool.pop(0)
                tried += 1
                demo, frames = sources[src_key]
                layout = layouts[src_key]
                try:
                    # slight per-demo offset within the scene (visibility-checked;
                    # fall back to the base scene if a jitter goes occluded)
                    dscene = scene
                    for _j in range(4):
                        cand = S.jitter_scene(scene, rng, args.jitter_pos, jitter_yaw_rad)
                        ni, fxx, *_ = S.apply_scene(cand, demo.init_state, env, cfg, layout)
                        if scene_visible(ni, fxx):
                            dscene = cand
                            break
                    new_init, fx, obj_t, tar_t, info = S.apply_scene(dscene, demo.init_state, env, cfg, layout)
                    ref, base_actions, new_frames = synthesize_uniform(
                        demo.state, demo.action, frames, obj_t, tar_t)
                    R.reset_to_init_state(env, new_init)
                    S.apply_fixture_edits(env, fx)
                    success, obs, rollout = R.replay_uniform(
                        env, base_actions, ref, new_frames, collect=True, extract=extract)
                except Exception as exc:
                    print(f"[scene {scene_i}] src={src_key} ERROR: {exc!r}", flush=True)
                    continue
                print(f"[scene {scene_i}] src={src_key} yaw={np.rad2deg(info['theta']):+.0f} "
                      f"obj_t={np.round(obj_t[:2],3)} tar_t={np.round(tar_t[:2],3)} "
                      f"-> success={success} (total {n_success})", flush=True)
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
                meta[TASK][demo_name] = oc_obs.metainfo_entry(
                    TASK, rollout, obj_order, new_init, subtasks=subtasks)
                if args.save_videos:
                    R.save_video(os.path.join(args.out_dir, f"{demo_name}.mp4"),
                                 rollout["agentview_rgb"][:, ::-1])
                tnew = np.asarray(info["target_new_xy"])
                cell = [int(np.floor(tnew[0] / spec.target_cell)),
                        int(np.floor(tnew[1] / spec.target_cell))]
                scene_log.append({"demo_name": demo_name, "scene_id": scene_i,
                                  "source_demo": src_key, "seed": args.seed,
                                  "target_cell": cell,       # per-demo OOD checkerboard cell
                                  "scene": scene, **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                     for k, v in info.items()}})
                n_success += 1
                got += 1
                if n_success % 10 == 0:
                    dump_meta()
    finally:
        env.close()
        dump_meta()
        with open(os.path.join(args.out_dir, "scene_log.json"), "w") as f:
            json.dump(scene_log, f, indent=2)

    print(f"\nCollected {n_success} episodes over {args.n_scenes} scenes.")
    print(f"HDF5:     {hdf5_out}\nmetainfo: {meta_out}")


if __name__ == "__main__":
    main()
