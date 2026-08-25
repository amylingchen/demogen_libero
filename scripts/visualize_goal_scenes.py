"""Render the default init scene of every libero_goal task into one 2x5 figure.

For each bddl file in the libero_goal suite: build the env, reset() (samples
the bddl's own placement regions), render agentview frame 0. No hdf5 needed.

Usage:
    .venv\\Scripts\\python.exe scripts\\visualize_goal_scenes.py --out output/goal_scenes/goal_tasks_2x5.png
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from libero.libero import get_libero_path
from demogen_libero import oc_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str,
                        default=os.path.join("output", "goal_scenes", "goal_tasks_2x5.png"))
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bddls = sorted(glob.glob(os.path.join(get_libero_path("bddl_files"),
                                          "libero_goal", "*.bddl")))
    assert len(bddls) == 10, f"expected 10 libero_goal bddl files, got {len(bddls)}"

    frames = []
    for i, bddl in enumerate(bddls):
        task = os.path.splitext(os.path.basename(bddl))[0]
        print(f"[{i + 1}/10] {task}", flush=True)
        env = oc_obs.make_oc_env(bddl, height=args.size, width=args.size)
        env.seed(args.seed)
        obs = env.reset()
        img = np.array(obs["agentview_image"])[::-1]  # render is upside down
        frames.append((task, img))
        env.close()

    fig, axes = plt.subplots(2, 5, figsize=(5 * 3.2, 2 * 3.55))
    for ax, (task, img) in zip(axes.flat, frames):
        ax.imshow(img)
        ax.set_title(task.replace("_", " "), fontsize=8, wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("libero_goal: default init scene of each task (agentview)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
