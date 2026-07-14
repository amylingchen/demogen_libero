"""Tile every demo's initial state (agentview frame 0) into one figure, with
the target object's bbox highlighted on each thumbnail and a bottom panel
summarizing all target placements over the workspace grid.

Usage:
    .venv\Scripts\python.exe scripts\visualize_init_states.py --dir output\grid_oc_salad_150
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=os.path.join("output", "libero_object_3"))
    parser.add_argument("--cols", type=int, default=15)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    hdf5_path = glob.glob(os.path.join(args.dir, "*_demo.hdf5"))[0]
    meta_path = os.path.join(args.dir, "metainfo.json")
    log_path = os.path.join(args.dir, "scene_log.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    task_key = next(iter(meta)) if meta else None
    scenes = {}
    if os.path.exists(log_path):
        scenes = {r["demo_name"]: r for r in json.load(open(log_path, encoding="utf-8"))}

    with h5py.File(hdf5_path, "r") as f:
        names = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        thumbs = {k: np.array(f["data"][k]["obs"]["agentview_rgb"][0]) for k in names}

    n = len(names)
    cols = args.cols
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(cols * 1.35, rows * 1.35 + 3.6))
    gs = fig.add_gridspec(rows + 3, cols, hspace=0.25, wspace=0.03)
    fig.suptitle(f"initial states of all {n} demos (red box = target object)", fontsize=13, y=0.995)

    for i, k in enumerate(names):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        img = thumbs[k]
        ax.imshow(img)
        # highlight the target from metainfo frame-0 exo boxes
        try:
            entry = meta[task_key][k]
            target_name = entry.get("target_object") or entry["task_nouns"][1]
            box = entry["exo_boxes"][0].get(target_name)
            if box:
                _, (x, y, bw, bh) = box
                h, w = img.shape[:2]
                ax.add_patch(mpatches.Rectangle((x * w, y * h), bw * w, bh * h,
                                                fill=False, edgecolor="red", lw=1.2))
        except Exception:
            pass
        rec = scenes.get(k, {})
        nd = rec.get("n_distractors", "?")
        src = str(rec.get("source_demo", "")).replace("demo_", "s")
        ax.set_title(f"{k.split('_')[-1]} {src}·d{nd}", fontsize=6, pad=1.5)
        ax.set_xticks([]); ax.set_yticks([])

    # bottom summary: target placements over the workspace
    ax = fig.add_subplot(gs[rows:rows + 3, 0:cols])
    if scenes:
        xy = np.array([scenes[k]["target_new_xy"] for k in names if k in scenes])
        nd = np.array([scenes[k]["n_distractors"] for k in names if k in scenes])
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=nd, cmap="viridis", s=42, edgecolors="k", lw=0.4)
        plt.colorbar(sc, ax=ax, label="n_distractors", fraction=0.025)
        ax.plot(0.0, 0.26, "o", ms=16, mfc="none", mec="#f39c12", mew=3, label="basket")
        ax.legend(fontsize=9)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("target object placements across all demos (color = distractor count)", fontsize=10)
    ax.set_xlim(-0.32, 0.32); ax.set_ylim(-0.32, 0.32)
    ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.3)

    out = args.out or os.path.join(args.dir, "init_states_all.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    print(f"wrote {out}  ({n} demos, {rows}x{cols} grid)")


if __name__ == "__main__":
    main()
