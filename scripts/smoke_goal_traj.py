"""Smoke for the LIBERO-Goal 12-layout plan: sample ONE new layout (shared by
all tasks, matching the plan's layout-is-global design), then generate ONE
trajectory per task on it via segment-transform replay, judged by each task's
own goal predicate at the END state (latched success recorded separately --
latched-vs-end divergence was a real bug class in the object counterfactual
set).

Outputs to --out-dir: per-task mp4 + init png, a 3x3 init mosaic annotated
with results, layout.json, results.json.

Usage:
    .venv\\Scripts\\python.exe scripts\\smoke_goal_traj.py --seed 1 --out-dir output/goal_smoke
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from demogen_libero.convert import load_demo, list_demo_keys
from demogen_libero.trajectory import auto_segment, segment_regrasp, synthesize_uniform
from demogen_libero import libero_replay as R
from demogen_libero import oc_obs
from demogen_libero import spatial_scene as S
from demogen_libero import goal_scene as G

from libero.libero import get_libero_path


def frames_for(task_cfg, demo, hdf5_path):
    """Segmentation per task kind. pick_place: gripper-cycle based (regrasp
    folded into skill_1 when present). fixture_op / push: contact-frame based
    whole-trajectory translation (drawer demos never close the gripper, stove
    demos never re-open it -- probed 2026-08-22)."""
    if task_cfg["kind"] == "pick_place":
        grip = demo.action[:, 6]
        n_close = int(np.sum((grip[:-1] <= 0) & (grip[1:] > 0)))
        if n_close > 1:
            try:
                fr = segment_regrasp(demo.state, demo.action)
                f1, f2, f3 = fr.as_tuple()
                if 0 < f1 < f2 < f3 < len(demo.action):
                    return fr
            except AssertionError:
                pass
        return auto_segment(demo.state, demo.action)
    f1 = G.contact_frame(hdf5_path, demo.demo_key, task_cfg["contact"])
    if task_cfg["kind"] == "fixture_op":
        # keep any gripper-close (stove knob) inside the verbatim part
        grip = demo.action[:, 6]
        close = np.where(grip > 0)[0]
        if close.size:
            f1 = min(f1, max(int(close[0]) - 2, 1))
    return G.whole_traj_frames(f1, len(demo.action))


def render_init_px(env, lut_names):
    """Rendered per-entity visible pixel counts (force_update: settle drives
    sim.step directly and leaves the observable cache stale -- same fix as
    run_spatial_oc_demo's gate)."""
    obs = env.env._get_observations(force_update=True)
    raw = obs["agentview_segmentation_instance"][..., 0]
    names = list(env.env.model.instances_to_ids.keys())
    counts = {}
    for nm in lut_names:
        counts[nm] = int((raw == names.index(nm) + 1).sum())
    return counts, obs["agentview_image"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-base-dir", default="D:/Data/LingLing/libero/hf/libero_goal")
    ap.add_argument("--out-dir", default=os.path.join("output", "goal_smoke"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-sources", type=int, default=8,
                    help="source demos tried per task before giving up")
    ap.add_argument("--layout-tries", type=int, default=30,
                    help="sampled layouts tried against the settle+visibility gates")
    ap.add_argument("--layout-file", default=None,
                    help="reuse a saved layout.json instead of sampling a new one")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="subset of tasks to run (default: all 9)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tasks = list(G.GOAL_TASKS)
    entities = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1",
                "wooden_cabinet_1", "flat_stove_1", "wine_rack_1"]
    spec = G.GoalSpec()

    # ---- stage 1: sample ONE layout, gated on settle + rendered visibility ----
    layout = None
    if args.layout_file:
        layout = json.load(open(args.layout_file))["layout"]
        print(f"[layout] reusing {args.layout_file}", flush=True)
    else:
        probe_task = tasks[0]
        probe_h5 = os.path.join(args.source_base_dir, f"{probe_task}_demo.hdf5")
        probe_demo = load_demo(probe_h5, "demo_0")
        env = oc_obs.make_oc_env(probe_demo.bddl_file)
        env.reset()
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
        cam_xy = get_camera_extrinsic_matrix(env.sim, "agentview")[:2, 3].copy()
        ref_layout = S.read_layout(env, probe_demo.init_state, G.GOAL_JOINTS, G.GOAL_FIXTURES)
        for li in range(args.layout_tries):
            cand = G.sample_goal_layout(rng, spec, ref_layout, cam_xy)
            new_init, fx = G.apply_goal_layout(cand, probe_demo.init_state, ref_layout)
            R.reset_to_init_state(env, new_init)
            S.apply_fixture_edits(env, fx)
            rep = S.settle(env, spec.settle_steps)
            px, img = render_init_px(env, entities)
            ok_px = all(v >= spec.min_px for v in px.values())
            print(f"[layout {li}] settle={rep['converged']} ({rep['steps']} steps, "
                  f"moved {rep['max_disp_cm']:.1f}cm) px={px} -> "
                  f"{'ACCEPT' if (rep['converged'] and ok_px) else 'reject'}", flush=True)
            if rep["converged"] and ok_px:
                layout = cand
                from PIL import Image
                Image.fromarray(np.asarray(img)[::-1]).save(
                    os.path.join(args.out_dir, "layout_probe.png"))
                break
        env.close()
        if layout is None:
            raise RuntimeError("no layout passed the settle+visibility gates")
        with open(os.path.join(args.out_dir, "layout.json"), "w") as f:
            json.dump({"layout": layout, "seed": args.seed,
                       "ref_fixtures": {fb: ref_layout["fixtures"][fb]["pos"].tolist()
                                        for fb in G.GOAL_FIXTURES}}, f, indent=2)

    # ---- stage 2: one trajectory per task on that layout ----
    res_path = os.path.join(args.out_dir, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    init_frames = {}
    run_tasks = args.tasks or tasks
    for task in run_tasks:
        cfg = G.GOAL_TASKS[task]
        h5 = os.path.join(args.source_base_dir, f"{task}_demo.hdf5")
        keys = list_demo_keys(h5)
        order = list(rng.permutation(keys))[:args.max_sources]
        env = oc_obs.make_oc_env(os.path.join(
            get_libero_path("bddl_files"), "libero_goal", f"{task}.bddl"))
        env.reset()
        # pristine fixture poses, captured BEFORE any apply_fixture_edits --
        # reading them from the live model between attempts returns the
        # already-moved pose and zeroes every fixture delta
        fixture_ref = G.capture_fixture_ref(env)
        task_res = {"attempts": []}
        for src_key in order:
            demo = load_demo(h5, src_key)
            try:
                frames = frames_for(cfg, demo, h5)
                demo_layout = G.read_demo_layout(env, demo.init_state, fixture_ref)
                obj_t, tar_t = G.anchor_deltas(cfg, layout, demo_layout)
                ref, base_actions, new_frames = synthesize_uniform(
                    demo.state, demo.action, frames, obj_t, tar_t)
                new_init, fx = G.apply_goal_layout(layout, demo.init_state, demo_layout)
                R.reset_to_init_state(env, new_init)
                S.apply_fixture_edits(env, fx)
                latched, obs, rollout = R.replay_uniform(
                    env, base_actions, ref, new_frames, collect=True)
                end = bool(env.check_success())
            except Exception as exc:
                print(f"[{task}] src={src_key} ERROR: {exc!r}", flush=True)
                task_res["attempts"].append({"source": src_key, "error": repr(exc)})
                continue
            att = {"source": src_key, "success_end": end, "success_latched": bool(latched),
                   "obj_t": np.round(obj_t, 4).tolist(), "tar_t": np.round(tar_t, 4).tolist(),
                   "frames": list(frames.as_tuple()), "T_src": len(demo.action),
                   "T_out": len(rollout["actions"])}
            task_res["attempts"].append(att)
            print(f"[{task}] src={src_key} end={end} latched={latched} "
                  f"obj_t={att['obj_t'][:2]} tar_t={att['tar_t'][:2]}", flush=True)
            if end:
                R.save_video(os.path.join(args.out_dir, f"{task}.mp4"),
                             rollout["agentview_rgb"])
                init_frames[task] = np.asarray(rollout["agentview_rgb"][0])[::-1]
                break
            elif "fail_video_saved" not in task_res:
                R.save_video(os.path.join(args.out_dir, f"{task}_FAIL_{src_key}.mp4"),
                             rollout["agentview_rgb"])
                init_frames.setdefault(task, np.asarray(rollout["agentview_rgb"][0])[::-1])
                task_res["fail_video_saved"] = src_key
        task_res["success"] = any(a.get("success_end") for a in task_res["attempts"])
        task_res["n_tried"] = len(task_res["attempts"])
        results[task] = task_res
        env.close()
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)

    # ---- mosaic ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2

    def frame0_from_video(task):
        for suffix in ("", "_FAIL_*"):
            import glob as _glob
            hits = _glob.glob(os.path.join(args.out_dir, f"{task}{suffix}.mp4"))
            if hits:
                cap = cv2.VideoCapture(hits[0])
                ok, fr = cap.read()
                cap.release()
                if ok:
                    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        return None

    fig, axes = plt.subplots(3, 3, figsize=(3 * 3.4, 3 * 3.7))
    for ax, task in zip(axes.flat, tasks):
        img = init_frames.get(task)
        if img is None:
            img = frame0_from_video(task)
        if img is not None:
            ax.imshow(img)
        if task not in results:
            ax.set_title(task.replace("_", " ") + "\n(not run)", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ok = results[task]["success"]
        n = results[task]["n_tried"]
        ax.set_title(f"{task.replace('_', ' ')}\n"
                     f"{'SUCCESS' if ok else 'FAIL'} ({n} tried)",
                     fontsize=8, color="green" if ok else "red")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("goal smoke: 1 new layout, 1 trajectory per task", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(args.out_dir, "smoke_mosaic.png"), dpi=110,
                bbox_inches="tight")

    n_ok = sum(int(r["success"]) for r in results.values())
    print(f"\n== smoke done: {n_ok}/{len(results)} tasks succeeded ==")
    for task, r in results.items():
        print(f"  {'OK ' if r['success'] else 'FAIL'} {task} ({r['n_tried']} tried)")


if __name__ == "__main__":
    main()
