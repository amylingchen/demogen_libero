"""Smoke-test the spatial relational-group scene sampler across tasks WITHOUT
running any trajectory: for each task, sample N init scenes, apply them, render
agentview frame 0, and save a per-task mosaic + a visibility report. Lets the
whole set of tasks be eyeballed (relation preserved, objects seated, nothing
occluded) before committing to full episode generation.

Usage:
    .venv\\Scripts\\python.exe scripts\\smoke_spatial_inits.py --n 30 --out-dir output/spatial_smoke
    (default tasks = all non-drawer libero_spatial tasks)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np
from PIL import Image

from demogen_libero.convert import resolve_bddl_file
from demogen_libero import libero_replay as R
from demogen_libero import oc_obs
from demogen_libero import spatial_scene as S

DRAWER = "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"


def mosaic(imgs, cols):
    rows = (len(imgs) + cols - 1) // cols
    h, w, _ = imgs[0].shape
    canvas = np.full((rows * h, cols * w, 3), 40, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return canvas


def smoke_task(task, cfg, data_dir, n, seed, out_dir, min_px_gate=100, cols=6):
    all_joints = list(dict.fromkeys([cfg["target_joint"], cfg["dest_joint"],
                                     *cfg["group_joints"], *cfg["indep_joints"]]))
    obj_order = cfg["object_order"]
    hdf5 = os.path.join(data_dir, f"{task}_demo.hdf5")
    with h5py.File(hdf5, "r") as f:
        bddl = resolve_bddl_file(f["data"].attrs["bddl_file_name"])
        init_state = np.array(f["data"]["demo_0"]["states"][0], dtype=np.float64)

    env = oc_obs.make_oc_env(bddl)
    env.reset()
    lut = oc_obs.build_seg_lut(env, obj_order)
    seg_ids = {j: 60 + 10 * obj_order.index(j.replace("_joint0", ""))
               for j in all_joints if j.replace("_joint0", "") in obj_order}
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    cam_xy = get_camera_extrinsic_matrix(env.sim, "agentview")[:2, 3].copy()

    R.reset_to_init_state(env, init_state)
    layout = S.read_layout(env, init_state, all_joints, S.ALL_FIXTURES)
    spec = S.SpatialSpec()
    rng = np.random.default_rng(seed)

    imgs, min_pxs = [], []
    used_target_xy = []
    tries = 0
    while len(imgs) < n and tries < n * 4:
        tries += 1
        try:
            scene = S.sample_scene(rng, spec, cfg, layout, cam_xy, used_target_xy)
            new_init, fx, *_ = S.apply_scene(scene, init_state, env, cfg, layout)
        except RuntimeError:
            continue
        used_target_xy.append(np.array(scene["target_new_xy"]))
        R.reset_to_init_state(env, new_init)
        S.apply_fixture_edits(env, fx)
        S.settle(env, spec.settle_physics_steps)
        obs = env.env._get_observations()
        raw = obs["agentview_segmentation_instance"][..., 0]
        segids = lut[np.clip(raw, 0, len(lut) - 1)]
        min_px = min(int((segids == sid).sum()) for sid in seg_ids.values())
        if min_px < min_px_gate:      # mirror the generation visibility gate
            continue
        imgs.append(env.sim.render(camera_name="agentview", height=256, width=256)[::-1])
        min_pxs.append(min_px)
    env.close()

    short = task.replace("pick_up_the_black_bowl_", "").replace("_and_place_it_on_the_plate", "")
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(mosaic(imgs, cols)).save(os.path.join(out_dir, f"{short}.png"))
    report = {"task": task, "rendered": len(imgs), "tries": tries,
              "min_px_min": int(min(min_pxs)) if min_pxs else -1,
              "min_px_mean": float(np.mean(min_pxs)) if min_pxs else -1}
    print(f"{short}: {len(imgs)}/{n} scenes (tries={tries}) "
          f"min_px[min={report['min_px_min']} mean={report['min_px_mean']:.0f}]", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=None, help="full task keys; default = all non-drawer")
    ap.add_argument("--data-dir", default="D:/Data/LingLing/libero/hf/libero_spatial")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-px", type=int, default=100,
                    help="visibility gate: drop scenes with any object below this many seg px")
    ap.add_argument("--out-dir", default="output/spatial_smoke")
    args = ap.parse_args()

    cfg_all = S.load_spatial_config()
    tasks = args.tasks or [t for t in cfg_all if t != DRAWER]
    reports = []
    for task in tasks:
        reports.append(smoke_task(task, cfg_all[task], args.data_dir, args.n,
                                  args.seed, args.out_dir, min_px_gate=args.min_px))
    json.dump(reports, open(os.path.join(args.out_dir, "smoke_report.json"), "w"), indent=2)
    print(f"\nwrote {len(reports)} task mosaics -> {args.out_dir}")


if __name__ == "__main__":
    main()
