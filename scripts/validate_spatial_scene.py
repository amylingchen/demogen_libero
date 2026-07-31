"""Sanity-check the spatial relational-group scene sampler WITHOUT running any
trajectory: sample N scenes for a task, apply them to a source init state, and
render agentview frame 0 into a mosaic so the placement (relation preserved,
objects seated, nothing occluded) can be eyeballed.

Usage:
    .venv\\Scripts\\python.exe scripts\\validate_spatial_scene.py \
        --task pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate \
        --n 9 --out output/spatial_scene_check/next_to_ramekin.png
"""
import argparse
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


def mosaic(imgs, cols):
    n = len(imgs)
    rows = (n + cols - 1) // cols
    h, w, _ = imgs[0].shape
    canvas = np.full((rows * h, cols * w, 3), 40, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--data-dir", default="D:/Data/LingLing/libero/hf/libero_spatial")
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg_all = S.load_spatial_config()
    cfg = cfg_all[args.task]
    all_joints = [cfg["target_joint"], cfg["dest_joint"], *cfg["group_joints"],
                  *cfg["indep_joints"]]
    all_joints = list(dict.fromkeys(all_joints))
    fixtures = S.ALL_FIXTURES

    hdf5 = os.path.join(args.data_dir, f"{args.task}_demo.hdf5")
    with h5py.File(hdf5, "r") as f:
        bddl = resolve_bddl_file(f["data"].attrs["bddl_file_name"])
        init_state = np.array(f["data"][args.demo]["states"][0], dtype=np.float64)

    env = oc_obs.make_oc_env(bddl)
    env.reset()
    obj_order = cfg["object_order"]
    lut = oc_obs.build_seg_lut(env, obj_order)
    seg_ids = {j: 60 + 10 * obj_order.index(j.replace("_joint0", ""))
               for j in all_joints if j.replace("_joint0", "") in obj_order}

    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    cam_xy = get_camera_extrinsic_matrix(env.sim, "agentview")[:2, 3].copy()

    R.reset_to_init_state(env, init_state)
    layout = S.read_layout(env, init_state, all_joints, fixtures)
    spec = S.SpatialSpec()
    rng = np.random.default_rng(args.seed)

    # frame 0 of the untouched source, for reference
    def render():
        return env.sim.render(camera_name="agentview", height=256, width=256)[::-1]

    imgs = [render()]
    labels = ["SOURCE"]
    ok = 0
    for i in range(args.n):
        try:
            scene = S.sample_scene(rng, spec, cfg, layout, cam_xy)
            new_init, fx, obj_t, tar_t, info = S.apply_scene(scene, init_state, env, cfg, layout)
        except RuntimeError as e:
            print(f"scene {i}: sample failed: {e}")
            continue
        R.reset_to_init_state(env, new_init)
        S.apply_fixture_edits(env, fx)
        S.settle(env, spec.settle_steps)
        img = render()
        # rendered seg visibility check via the env observation dict
        obs = env.env._get_observations()
        raw = obs["agentview_segmentation_instance"][..., 0]
        segids = lut[np.clip(raw, 0, len(lut) - 1)]
        px = {j: int((segids == sid).sum()) for j, sid in seg_ids.items()}
        min_px = min(px.values()) if px else -1
        theta_deg = np.rad2deg(info["theta"])
        print(f"scene {i}: obj_t={np.round(obj_t[:2],3)} tar_t={np.round(tar_t[:2],3)} "
              f"yaw={theta_deg:+.0f} min_px={min_px} px={px}")
        imgs.append(img)
        labels.append(f"#{i} yaw{theta_deg:+.0f} px{min_px}")
        ok += 1

    env.close()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Image.fromarray(mosaic(imgs, cols=4)).save(args.out)
    print(f"\n{ok}/{args.n} scenes rendered -> {args.out}")


if __name__ == "__main__":
    main()
