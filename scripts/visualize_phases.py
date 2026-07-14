"""Per-demo phase visualization for OC episodes with phase_id: agentview frames
at each phase start, the top-down EE path colored by phase (with object/basket
markers), and a timeline with phase bands + gripper/speed curves.

Usage:
    .venv\Scripts\python.exe scripts\visualize_phases.py --dir output\grid_oc_salad_150 --demos demo_0 demo_42 demo_105
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PHASE_NAMES = {0: "motion_1 reach", 1: "skill_1 grasp", 2: "motion_2 transport", 3: "skill_2 place"}
PHASE_COLORS = {0: "#3498db", 1: "#e74c3c", 2: "#2ecc71", 3: "#9b59b6"}


def phase_starts(phase):
    return {int(p): int(np.argmax(phase == p)) for p in np.unique(phase)}


def visualize(d, scene, demo_name, out_path):
    phase = np.array(d["phase_id"])
    ee = np.array(d["obs"]["ee_pos"])
    act = np.array(d["actions"])
    T = len(phase)
    starts = phase_starts(phase)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.1, 1.4], hspace=0.25, wspace=0.15)
    src = scene.get("source_demo", "?") if scene else "?"
    fig.suptitle(f"{demo_name}  (source {src}, T={T})", fontsize=12)

    # top row: frame at the start of each phase + final frame
    marks = [(starts[p], f"t={starts[p]}  {PHASE_NAMES[p]}", PHASE_COLORS[p]) for p in sorted(starts)]
    marks.append((T - 1, f"t={T-1}  final", "#7f8c8d"))
    for i, (t, title, color) in enumerate(marks):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(d["obs"]["agentview_rgb"][t])
        ax.set_title(title, fontsize=9, color=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2.5)
        ax.set_xticks([]); ax.set_yticks([])

    # bottom left: top-down EE path colored by phase
    ax = fig.add_subplot(gs[1, 0:2])
    for p in sorted(starts):
        m = phase == p
        ax.plot(ee[m, 0], ee[m, 1], ".", ms=3, color=PHASE_COLORS[p], label=PHASE_NAMES[p])
    ax.plot(ee[0, 0], ee[0, 1], "k^", ms=9, label="EE start")
    tgt = None
    if scene:
        tgt = scene.get("target_new_xy") or (scene.get("scene") or {}).get("target_xy")
    if tgt is not None:
        ax.plot(tgt[0], tgt[1], "r*", ms=16, mec="k", label="target object")
        for dxy in (scene.get("scene") or {}).get("distractor_xys", []):
            ax.plot(dxy[0], dxy[1], "s", ms=8, color="#95a5a6")
    ax.plot(0.0, 0.26, "o", ms=14, mfc="none", mec="#f39c12", mew=2.5, label="basket")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("top-down EE path by phase", fontsize=10)
    ax.axis("equal"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    # bottom right: timeline with phase bands + curves
    ax = fig.add_subplot(gs[1, 2:5])
    speed = np.concatenate([[0], np.linalg.norm(np.diff(ee, axis=0), axis=1)]) * 1000
    for p in sorted(starts):
        m = phase == p
        idx = np.where(m)[0]
        ax.axvspan(idx[0], idx[-1] + 1, color=PHASE_COLORS[p], alpha=0.15)
        ax.text((idx[0] + idx[-1]) / 2, 1.06, PHASE_NAMES[p].split()[0], ha="center",
                transform=ax.get_xaxis_transform(), fontsize=8, color=PHASE_COLORS[p])
    ax.plot(speed, color="k", lw=1.0, label="EE speed (mm/step)")
    ax.plot(ee[:, 2] * 100, color="#16a085", lw=1.0, label="EE z (cm)")
    ax.plot(act[:, 6] * 5 + 20, color="#e67e22", lw=1.2, label="gripper cmd (scaled)")
    for t, _, color in marks[:-1]:
        ax.axvline(t, color=color, ls=":", lw=1.2)
    ax.set_xlabel("frame"); ax.set_title("timeline: phase bands, speed, height, gripper", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="upper left")

    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


PHASE_BGR = {p: tuple(int(c.lstrip("#")[i:i+2], 16) for i in (4, 2, 0)) for p, c in PHASE_COLORS.items()}


def phase_video(d, demo_name, out_path, fps=20, scale=2):
    import cv2

    phase = np.array(d["phase_id"])
    rgbs = np.array(d["obs"]["agentview_rgb"])
    grip = np.array(d["actions"][:, 6])
    T, H, W, _ = rgbs.shape
    H2, W2 = H * scale, W * scale
    banner = 34
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W2, H2 + banner))
    try:
        for t in range(T):
            img = cv2.cvtColor(cv2.resize(rgbs[t], (W2, H2), interpolation=cv2.INTER_NEAREST),
                               cv2.COLOR_RGB2BGR)
            p = int(phase[t])
            color = PHASE_BGR[p]
            frame = np.zeros((H2 + banner, W2, 3), dtype=np.uint8)
            frame[:banner] = (40, 40, 40)
            frame[banner:] = img
            cv2.rectangle(frame, (0, 0), (W2 - 1, banner - 1), color, 2)
            cv2.putText(frame, f"{PHASE_NAMES[p]}", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            grip_txt = "closed" if grip[t] > 0 else "open"
            cv2.putText(frame, f"t={t:3d}  grip:{grip_txt}", (W2 - 190, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            # phase progress strip along the bottom
            for tt in range(T):
                x0 = int(tt / T * W2); x1 = int((tt + 1) / T * W2)
                cv2.rectangle(frame, (x0, H2 + banner - 6), (x1, H2 + banner - 1),
                              PHASE_BGR[int(phase[tt])], -1)
            xm = int(t / T * W2)
            cv2.rectangle(frame, (xm - 1, H2 + banner - 10), (xm + 1, H2 + banner - 1), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=os.path.join("output", "grid_oc_salad_150"))
    parser.add_argument("--demos", nargs="+", default=["demo_0", "demo_42", "demo_105"])
    args = parser.parse_args()

    hdf5_path = glob.glob(os.path.join(args.dir, "*_demo.hdf5"))[0]
    log_path = os.path.join(args.dir, "scene_log.json")
    scenes = {}
    if os.path.exists(log_path):
        scenes = {r["demo_name"]: r for r in json.load(open(log_path, encoding="utf-8"))}

    out_dir = os.path.join(args.dir, "viz_phases")
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(hdf5_path, "r") as f:
        for name in args.demos:
            out = os.path.join(out_dir, f"{name}_phases.png")
            visualize(f["data"][name], scenes.get(name), name, out)
            vid = os.path.join(out_dir, f"{name}_phases.mp4")
            phase_video(f["data"][name], name, vid)
            print("wrote", out, "+", vid)


if __name__ == "__main__":
    main()
