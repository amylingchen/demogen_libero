"""Sample N goal layouts (settle+visibility gated, no trajectories) and render
them as a mosaic + a placement scatter, to show how much the sampler actually
shuffles object positions across layouts.

Usage:
    .venv\\Scripts\\python.exe scripts\\preview_goal_layouts.py --n 12
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demogen_libero.convert import load_demo
from demogen_libero import libero_replay as R
from demogen_libero import oc_obs
from demogen_libero import spatial_scene as S
from demogen_libero import goal_scene as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--source-base-dir", default="D:/Data/LingLing/libero/hf/libero_goal")
    ap.add_argument("--out", default=os.path.join("output", "goal_smoke_v2", "layout_preview.png"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    spec = G.GoalSpec()
    entities = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1",
                "wooden_cabinet_1", "flat_stove_1", "wine_rack_1"]

    task = list(G.GOAL_TASKS)[0]
    demo = load_demo(os.path.join(args.source_base_dir, f"{task}_demo.hdf5"), "demo_0")
    env = oc_obs.make_oc_env(demo.bddl_file)
    env.reset()
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    cam_xy = get_camera_extrinsic_matrix(env.sim, "agentview")[:2, 3].copy()
    ref_layout = S.read_layout(env, demo.init_state, G.GOAL_JOINTS, G.GOAL_FIXTURES)

    layouts, imgs = [], []
    tries = 0
    while len(layouts) < args.n and tries < args.n * 5:
        tries += 1
        cand = G.sample_goal_layout(rng, spec, ref_layout, cam_xy)
        new_init, fx = G.apply_goal_layout(cand, demo.init_state, ref_layout)
        R.reset_to_init_state(env, new_init)
        S.apply_fixture_edits(env, fx)
        rep = S.settle(env, spec.settle_steps)
        obs = env.env._get_observations(force_update=True)
        raw = obs["agentview_segmentation_instance"][..., 0]
        names = list(env.env.model.instances_to_ids.keys())
        ok_px = all(int((raw == names.index(nm) + 1).sum()) >= spec.min_px for nm in entities)
        if rep["converged"] and ok_px:
            layouts.append(cand)
            imgs.append(np.asarray(obs["agentview_image"])[::-1])
        print(f"try {tries}: settle={rep['converged']} px_ok={ok_px} "
              f"kept={len(layouts)}", flush=True)
    env.close()

    cols = 4
    rows = int(np.ceil(len(imgs) / cols))
    fig = plt.figure(figsize=(cols * 3.2, rows * 3.3 + 4.2))
    gs = fig.add_gridspec(rows + 2, cols, hspace=0.15, wspace=0.05)
    for i, im in enumerate(imgs):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.imshow(im)
        ax.set_title(f"L{i:02d}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[rows:, :])
    markers = {"akita_black_bowl_1_joint0": ("o", "tab:blue", "bowl"),
               "cream_cheese_1_joint0": ("s", "tab:orange", "cheese"),
               "wine_bottle_1_joint0": ("^", "tab:green", "bottle"),
               "plate_1_joint0": ("D", "tab:red", "plate")}
    fmarkers = {"wooden_cabinet_1_main": ("P", "k", "cabinet"),
                "flat_stove_1_main": ("X", "gray", "stove"),
                "wine_rack_1_main": ("*", "brown", "rack")}
    for jn, (m, c, lbl) in markers.items():
        xy = np.array([l["objects"][jn] for l in layouts])
        ax.scatter(xy[:, 0], xy[:, 1], marker=m, c=c, s=48, label=lbl, edgecolors="k", lw=0.4)
    for fb, (m, c, lbl) in fmarkers.items():
        xy = np.array([l["fixtures"][fb] for l in layouts])
        ax.scatter(xy[:, 0], xy[:, 1], marker=m, c=c, s=90, label=lbl)
    ax.legend(fontsize=9, ncol=4)
    ax.set_xlim(-0.58, 0.25); ax.set_ylim(-0.42, 0.42)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"placements across {len(layouts)} sampled layouts "
                 "(objects free in workspace; plate/bowl in central reach band; "
                 "fixtures uniform over probed corridors)", fontsize=10)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    with open(args.out.replace(".png", ".json"), "w") as f:
        json.dump(layouts, f, indent=2)
    print(f"wrote {args.out} ({len(layouts)} layouts)")


if __name__ == "__main__":
    main()
