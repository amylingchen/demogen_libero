"""Visualize an OC-format episode: RGB with bounding boxes, depth, segmentation
(both cameras), plus action curves with frame markers. Outputs per-frame panel
PNGs, one action-curve PNG, and a bbox-overlay mp4.

Usage:
    .venv\Scripts\python.exe scripts\visualize_oc_demo.py --dir output\grid_oc_salad
    .venv\Scripts\python.exe scripts\visualize_oc_demo.py --dir output\grid_oc_salad --demo demo_0 --frames 0 40 80 120
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
import cv2

SEG_COLORS = {  # seg id -> RGB
    0: (30, 30, 30), 50: (150, 150, 150), 60: (231, 76, 60), 70: (241, 196, 15),
    80: (46, 204, 113), 90: (52, 152, 219), 100: (155, 89, 182), 110: (26, 188, 156),
    120: (230, 126, 34),
}


def seg_to_rgb(seg):
    out = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for sid, c in SEG_COLORS.items():
        out[seg == sid] = c
    return out


def draw_boxes(rgb, frame_boxes):
    img = rgb.copy()
    h, w = img.shape[:2]
    for name, (sid, (x, y, bw, bh)) in frame_boxes.items():
        c = SEG_COLORS.get(sid, (255, 255, 255))
        p0 = (int(x * w), int(y * h))
        p1 = (int((x + bw) * w), int((y + bh) * h))
        cv2.rectangle(img, p0, p1, c, 1)
        cv2.putText(img, name, (p0[0], max(p0[1] - 3, 8)), cv2.FONT_HERSHEY_PLAIN, 0.7, c, 1)
    return img


def panel_figure(d, meta_entry, t, out_path):
    exo_boxes = meta_entry["exo_boxes"][t]
    ego_boxes = meta_entry["ego_boxes"][t]

    def depth_cm_float(cam):
        """Prefer the lossless mm channel (as float cm) when present."""
        mm_key = f"obs/{cam}_depth_mm"
        if mm_key in d:
            return np.asarray(d[mm_key][t], dtype=np.float64) / 10.0
        return np.asarray(d[f"obs/{cam}_depth"][t], dtype=np.float64)

    rows = [
        ("agentview", d["obs/agentview_rgb"][t], depth_cm_float("agentview"),
         d["obs/agentview_seg"][t], exo_boxes),
        ("eye_in_hand", d["obs/eye_in_hand_rgb"][t], depth_cm_float("eye_in_hand"),
         d["obs/eye_in_hand_seg"][t], ego_boxes),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.4))
    fig.suptitle(f"frame {t}   action={np.round(np.array(d['actions'][t]), 2)}", fontsize=10)
    for r, (cam, rgb, dep, seg, boxes) in enumerate(rows):
        axes[r, 0].imshow(rgb); axes[r, 0].set_title(f"{cam} RGB", fontsize=9)
        axes[r, 1].imshow(draw_boxes(np.array(rgb), boxes)); axes[r, 1].set_title("RGB + bbox", fontsize=9)
        dep_arr = np.asarray(dep)
        vmin, vmax = np.percentile(dep_arr, [2, 98])  # stretch: thin objects differ
        im = axes[r, 2].imshow(dep_arr, cmap="turbo", vmin=vmin, vmax=vmax)  # by only a few cm
        axes[r, 2].set_title(f"depth cm (stretch {int(vmin)}-{int(vmax)})", fontsize=9)
        plt.colorbar(im, ax=axes[r, 2], fraction=0.046)
        # local-contrast residual: depth minus its local median background, so
        # flat thin objects (e.g. cream cheese, ~3 cm) stand out from the
        # smooth floor-depth gradient that dominates the global range
        # (cv2.medianBlur needs uint8 for large kernels; sub-cm detail survives
        # in dep_arr itself when the mm channel is available)
        dep_clip = np.minimum(dep_arr, 255.0)  # match the cm channel's far cap
        bg = cv2.medianBlur(np.rint(dep_clip).astype(np.uint8), 31).astype(np.float64)
        resid = dep_clip - bg
        im2 = axes[r, 3].imshow(resid, cmap="RdBu", vmin=-6, vmax=6)
        axes[r, 3].set_title("depth - local median (+-6 cm)", fontsize=9)
        plt.colorbar(im2, ax=axes[r, 3], fraction=0.046)
        axes[r, 4].imshow(seg_to_rgb(np.array(seg))); axes[r, 4].set_title("segmentation", fontsize=9)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    names = {50: "robot", 60: "salad dressing", 70: "basket", 80: "ketchup",
             90: "alphabet soup", 100: "cream cheese", 110: "milk", 120: "tomato sauce"}
    present = sorted(set(np.unique(np.array(d["obs/agentview_seg"][t]))) - {0})
    handles = [mpatches.Patch(color=np.array(SEG_COLORS[s]) / 255, label=f"{s}: {names.get(s, '?')}")
               for s in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def action_figure(d, marks, out_path):
    act = np.array(d["actions"])
    dims = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "gripper"]
    fig, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(act[:, i], lw=0.9, color="steelblue")
        ax.axhline(0, color="k", ls="--", lw=0.5, alpha=0.5)
        for m in marks:
            ax.axvline(m, color="crimson", ls=":", lw=0.8, alpha=0.7)
        ax.set_ylabel(dims[i], fontsize=9)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("frame (red dotted = visualized frames)")
    fig.suptitle("actions", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def bbox_video(d, meta_entry, out_path, fps=20):
    rgbs = np.array(d["obs/agentview_rgb"])
    T, H, W, _ = rgbs.shape
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    try:
        for t in range(T):
            img = draw_boxes(rgbs[t], meta_entry["exo_boxes"][t])
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=os.path.join("output", "grid_oc_salad"))
    parser.add_argument("--hdf5", type=str, default=None, help="defaults to the *_demo.hdf5 in --dir")
    parser.add_argument("--metainfo", type=str, default=None)
    parser.add_argument("--demo", type=str, default="demo_0")
    parser.add_argument("--frames", type=int, nargs="*", default=None,
                        help="frame indices; default = 6 evenly spaced")
    args = parser.parse_args()

    hdf5_path = args.hdf5 or glob.glob(os.path.join(args.dir, "*_demo.hdf5"))[0]
    meta_path = args.metainfo or os.path.join(args.dir, "metainfo.json")
    out_dir = os.path.join(args.dir, f"viz_{args.demo}")
    os.makedirs(out_dir, exist_ok=True)

    meta = json.load(open(meta_path, encoding="utf-8"))
    task_key = next(k for k, v in meta.items() if args.demo in v)
    entry = meta[task_key][args.demo]

    with h5py.File(hdf5_path, "r") as f:
        d = f["data"][args.demo]
        T = int(d.attrs["num_samples"])
        marks = args.frames if args.frames else np.linspace(0, T - 1, 6).astype(int).tolist()
        for t in marks:
            panel_figure(d, entry, int(t), os.path.join(out_dir, f"frame_{int(t):03d}.png"))
        action_figure(d, marks, os.path.join(out_dir, "actions.png"))
        bbox_video(d, entry, os.path.join(out_dir, "agentview_bbox.mp4"))

    print(f"wrote {len(marks)} frame panels + actions.png + agentview_bbox.mp4 -> {out_dir}")


if __name__ == "__main__":
    main()
