"""Plot where the two bowls actually start, per task: LIBERO's own source demos
against the generated libero_spatial_100 dataset.

Positions come from each demo's obs/obj_pos frame 0 (the pose that was actually
simulated, jitter included) rather than from scene_log's pre-jitter parameters.

Usage:
    .venv\\Scripts\\python.exe scripts\\plot_bowl_distribution.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demogen_libero import spatial_scene as S

SOURCE_DIR = "D:/Data/LingLing/libero/hf/libero_spatial"
GEN_DIR = "output/libero_spatial_100"
BOWL1, BOWL2 = "akita_black_bowl_1", "akita_black_bowl_2"


def short(t):
    return (t.replace("pick_up_the_black_bowl_", "")
             .replace("_and_place_it_on_the_plate", ""))


def source_positions(task, limit=None):
    """bowl_1 / bowl_2 xy from every source demo's init state."""
    from demogen_libero.convert import resolve_bddl_file
    from demogen_libero import libero_replay as R
    path = os.path.join(SOURCE_DIR, f"{task}_demo.hdf5")
    with h5py.File(path, "r") as f:
        bddl = resolve_bddl_file(f["data"].attrs["bddl_file_name"])
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:limit]
        states = [np.array(f["data"][k]["states"][0], dtype=np.float64) for k in keys]
    env = R.make_env(bddl)
    env.reset()
    joints = [f"{BOWL1}_joint0", f"{BOWL2}_joint0"]
    lay = S.read_layout(env, states[0], joints, [])
    addr = {j: lay["free"][j]["addr"] for j in joints}
    env.close()
    out = {}
    for j, b in zip(joints, (BOWL1, BOWL2)):
        out[b] = np.array([s[addr[j]:addr[j] + 2] for s in states])
    return out


def generated_positions(task):
    """bowl_1 / bowl_2 xy at frame 0 of every generated demo."""
    d = os.path.join(GEN_DIR, short(task))
    path = os.path.join(d, f"{task}_demo.hdf5")
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        data = f["data"]
        keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))
        order = json.loads(data[keys[0]].attrs["object_instances"])
        i1, i2 = order.index(BOWL1), order.index(BOWL2)
        pos = np.array([data[k]["obs"]["obj_pos"][0] for k in keys])   # (N, 5, 3)
    return {BOWL1: pos[:, i1, :2], BOWL2: pos[:, i2, :2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/bowl_distribution.png")
    ap.add_argument("--source-limit", type=int, default=50)
    args = ap.parse_args()

    tasks = list(S.load_spatial_config().keys())
    spec = S.SpatialSpec()

    fig, axes = plt.subplots(2, 5, figsize=(23, 10), sharex=True, sharey=True)
    for ax, task in zip(axes.ravel(), tasks):
        src = source_positions(task, args.source_limit)
        gen = generated_positions(task)

        # the sampler's target workspace and the reachable place zone, for scale
        ax.add_patch(plt.Rectangle((spec.x_range[0], spec.y_range[0]),
                                   spec.x_range[1] - spec.x_range[0],
                                   spec.y_range[1] - spec.y_range[0],
                                   fill=False, ec="0.55", ls="--", lw=1.0))
        ax.add_patch(plt.Rectangle((spec.dest_x[0], spec.dest_y[0]),
                                   spec.dest_x[1] - spec.dest_x[0],
                                   spec.dest_y[1] - spec.dest_y[0],
                                   fill=False, ec="seagreen", ls=":", lw=1.2))

        if gen is not None:
            ax.scatter(*gen[BOWL1].T, s=16, c="#1f77b4", alpha=.55,
                       lw=0, label=f"bowl_1 generated ({len(gen[BOWL1])})")
            ax.scatter(*gen[BOWL2].T, s=16, c="#d62728", alpha=.55,
                       lw=0, label=f"bowl_2 generated ({len(gen[BOWL2])})")
        ax.scatter(*src[BOWL1].T, s=46, marker="x", c="#0b3d64", lw=1.4,
                   label=f"bowl_1 source ({len(src[BOWL1])})")
        ax.scatter(*src[BOWL2].T, s=46, marker="x", c="#7a1416", lw=1.4,
                   label=f"bowl_2 source ({len(src[BOWL2])})")

        sp1 = gen[BOWL1].ptp(0) if gen is not None else np.zeros(2)
        sp2 = gen[BOWL2].ptp(0) if gen is not None else np.zeros(2)
        ax.set_title(f"{short(task)}\ngen spread  b1 {sp1[0]:.2f}x{sp1[1]:.2f}   "
                     f"b2 {sp2[0]:.2f}x{sp2[1]:.2f} m", fontsize=9)
        ax.axhline(0, c="0.85", lw=.6, zorder=0)
        ax.axvline(0, c="0.85", lw=.6, zorder=0)
        ax.set_aspect("equal")
        ax.grid(alpha=.15)

    axes[0, 0].set_xlim(-0.52, 0.36)
    axes[0, 0].set_ylim(-0.46, 0.44)
    for ax in axes[1]:
        ax.set_xlabel("x (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend([*h[:2], *h[2:]],
               ["bowl_1 generated", "bowl_2 generated",
                "bowl_1 source (LIBERO)", "bowl_2 source (LIBERO)"],
               loc="lower center", ncol=4, frameon=False, fontsize=11)
    fig.suptitle("Initial bowl placement per libero_spatial task — LIBERO source vs "
                 "generated libero_spatial_100\n"
                 "dashed grey = sampler target workspace,  dotted green = reachable "
                 "place zone (plate)", fontsize=13)
    fig.tight_layout(rect=[0, 0.045, 1, 0.93])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
