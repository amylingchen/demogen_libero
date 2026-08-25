"""Export videos from the generated goal dataset (frames are already stored;
no simulation needed).

Per task: a 2x3 grid mp4 showing the SAME instruction executed on 6 different
layouts (2 train / 2 cf / 2 unseen), each panel labeled with layout id + split
so the layout variation is visible at a glance. Shorter clips freeze on their
last frame until the longest one ends. Also dumps a few single-demo mp4s.

Usage:
    .venv\\Scripts\\python.exe scripts\\dump_goal_videos.py --dir output/goal_gen_500
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import cv2
import h5py
import numpy as np

SPLITS = ["train", "quarantine_cf", "quarantine_unseen"]
SHORT = {"train": "train", "quarantine_cf": "cf", "quarantine_unseen": "UNSEEN"}
COLOR = {"train": (90, 200, 90), "quarantine_cf": (230, 180, 60),
         "quarantine_unseen": (90, 90, 240)}


def load_demo_frames(path, key):
    with h5py.File(path, "r") as f:
        g = f["data"][key]
        rgb = np.array(g["obs"]["agentview_rgb"])[:, ::-1]   # GL -> upright
        a = dict(g.attrs)
    return rgb, a


def label(img, text, color):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (25, 25, 25), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1,
                cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("output", "goal_gen_500"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--per-split", type=int, default=2)
    ap.add_argument("--panels", type=int, default=6,
                    help="total panels in the grid, filled round-robin over splits")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--singles", type=int, default=1,
                    help="single-demo mp4s per task (one per split)")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.dir, "videos")
    os.makedirs(out_dir, exist_ok=True)

    # index demos: task -> split -> [(path, key, layout_id)]
    index = defaultdict(lambda: defaultdict(list))
    for split in SPLITS:
        for p in sorted(glob.glob(os.path.join(args.dir, split, "*.hdf5"))):
            task = os.path.basename(p).replace("_demo.hdf5", "")
            with h5py.File(p, "r") as f:
                for k in f["data"]:
                    index[task][split].append((p, k, f["data"][k].attrs["layout_id"]))

    summary = {}
    for task, by_split in sorted(index.items()):
        # Round-robin across splits, taking a NEW layout each time, until the
        # grid is full. A per-split quota (the previous rule) starved tasks that
        # only exist in one split: push has 4 train cells and no cf/unseen, so it
        # rendered just 2 panels while holding 4 distinct layouts.
        pools = {sp: list(by_split.get(sp, [])) for sp in SPLITS}
        picks, used = [], set()
        target = args.panels or (args.per_split * len(SPLITS))
        progress = True
        while len(picks) < target and progress:
            progress = False
            for sp in SPLITS:
                if len(picks) >= target:
                    break
                for r in pools[sp]:
                    if (sp, r[2]) in used:
                        continue
                    picks.append(r)
                    used.add((sp, r[2]))
                    progress = True
                    break
        if not picks:
            continue

        clips = []
        for p, k, lid in picks:
            rgb, a = load_demo_frames(p, k)
            split = a["split"]
            txt = f"{lid} {SHORT[split]}  src={a['source_demo']}"
            clips.append(([label(f, txt, COLOR[split]) for f in rgb], lid, split))
        T = max(len(c[0]) for c in clips)
        h, w = clips[0][0][0].shape[:2]
        cols = min(args.cols, len(clips))
        rows_n = int(np.ceil(len(clips) / cols))
        vpath = os.path.join(out_dir, f"{task}__grid.mp4")
        vw = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (cols * w, rows_n * h))
        for t in range(T):
            canvas = np.full((rows_n * h, cols * w, 3), 30, np.uint8)
            for i, (frames, lid, split) in enumerate(clips):
                fr = frames[min(t, len(frames) - 1)]
                r, c = divmod(i, cols)
                canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = fr
            vw.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        vw.release()

        # a couple of single-demo videos (one per split)
        singles = []
        for split in SPLITS:
            rows = by_split.get(split, [])
            for p, k, lid in rows[:args.singles]:
                rgb, a = load_demo_frames(p, k)
                sp = os.path.join(out_dir, f"{task}__{lid}_{SHORT[split]}.mp4")
                vw = cv2.VideoWriter(sp, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                     (rgb.shape[2], rgb.shape[1]))
                for f in rgb:
                    vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                vw.release()
                singles.append(os.path.basename(sp))
        summary[task] = {"grid": os.path.basename(vpath),
                         "panels": [f"{l} {SHORT[s]}" for _, l, s in clips],
                         "singles": singles, "frames": T}
        print(f"{task[:44]:44s} grid={len(clips)} panels, {T} frames -> "
              f"{os.path.basename(vpath)}", flush=True)

    with open(os.path.join(out_dir, "videos_index.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {len(summary)} grid videos + singles to {out_dir}")


if __name__ == "__main__":
    main()
