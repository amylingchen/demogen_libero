"""Compare two generated goal datasets (v1 vs v3) on the axes where v1 was
found defective, plus the standard packaging checks.

Every number is measured from the written hdf5s, and where a reference exists
the ORIGINAL human demos are measured the same way and reported alongside --
"better than v1" is not the bar; "matches what the source demos do" is.

Axes:
  1. joint-limit saturation (min margin over the episode; <0.05 rad = saturated)
  2. final object pose (upright of the manipulated object, per task)
  3. EE max distance from the robot base (the reach envelope)
  4. task-goal reach points per demo (max over the layout's points)
  5. cell/demo counts and per-task yields (from the generation logs)

Usage:
    .venv\\Scripts\\python.exe scripts\\compare_goal_versions.py \\
        --a output/goal_gen_500 --b output/goal_gen_v3
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np

from demogen_libero import goal_scene as G

SPLITS = ["train", "quarantine_cf", "quarantine_unseen"]
LIM = [(-2.897, 2.897), (-1.763, 1.763), (-2.897, 2.897), (-3.072, -0.070),
       (-2.897, 2.897), (-0.018, 3.752), (-2.897, 2.897)]
BASE = np.array([-0.66, 0.0])
QUAT = {"akita_black_bowl_1_joint0": 13, "cream_cheese_1_joint0": 20,
        "wine_bottle_1_joint0": 27, "plate_1_joint0": 34}
# which object each task manipulates (for the pose axis)
MANIP = {
    "put_the_bowl_on_the_plate": "akita_black_bowl_1_joint0",
    "put_the_bowl_on_the_stove": "akita_black_bowl_1_joint0",
    "put_the_bowl_on_top_of_the_cabinet": "akita_black_bowl_1_joint0",
    "put_the_cream_cheese_in_the_bowl": "cream_cheese_1_joint0",
    "put_the_wine_bottle_on_the_rack": "wine_bottle_1_joint0",
    "put_the_wine_bottle_on_top_of_the_cabinet": "wine_bottle_1_joint0",
    "push_the_plate_to_the_front_of_the_stove": "plate_1_joint0",
}


def margin(js):
    js = np.asarray(js)
    return float(min(min(abs(js[:, j].min() - LIM[j][0]),
                         abs(js[:, j].max() - LIM[j][1])) for j in range(7)))


def upright(q):
    return float(1 - 2 * (q[1] ** 2 + q[2] ** 2))


def scan(root):
    """task -> list of per-demo measurements"""
    out = defaultdict(list)
    for split in SPLITS:
        for p in sorted(glob.glob(os.path.join(root, split, "*.hdf5"))):
            task = os.path.basename(p).replace("_demo.hdf5", "")
            with h5py.File(p, "r") as f:
                for k in f["data"]:
                    g = f["data"][k]
                    rec = {"split": split, "key": k,
                           "layout": g.attrs.get("layout_id"),
                           "source": g.attrs.get("source_demo"),
                           "margin": margin(g["obs"]["joint_states"]),
                           "ee_max": float(np.linalg.norm(
                               np.array(g["obs"]["ee_pos"])[:, :2] - BASE, axis=1).max())}
                    if task in MANIP:
                        c = QUAT[MANIP[task]]
                        rec["upright"] = upright(np.array(g["states"][-1])[c:c + 4])
                    if "jitter_objects" in g.attrs:
                        jl = {"objects": json.loads(g.attrs["jitter_objects"]),
                              "fixtures": json.loads(g.attrs["jitter_fixtures"])}
                        rec["reach_viol"] = len(G.layout_reach_violations(jl))
                    out[task].append(rec)
    return out


def scan_source(task, base_dir):
    p = os.path.join(base_dir, task + "_demo.hdf5")
    if not os.path.exists(p):
        return None
    ms, us, ee = [], [], []
    with h5py.File(p, "r") as f:
        for k in list(f["data"])[:50]:
            g = f["data"][k]
            ms.append(margin(g["obs"]["joint_states"]))
            ee.append(float(np.linalg.norm(
                np.array(g["obs"]["ee_pos"])[:, :2] - BASE, axis=1).max()))
            if task in MANIP:
                c = QUAT[MANIP[task]]
                us.append(upright(np.array(g["states"][-1])[c:c + 4]))
    return {"margin": np.array(ms), "upright": np.array(us), "ee_max": np.array(ee)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=os.path.join("output", "goal_gen_500"))
    ap.add_argument("--b", default=os.path.join("output", "goal_gen_v3"))
    ap.add_argument("--label-a", default="v1")
    ap.add_argument("--label-b", default="v3")
    ap.add_argument("--source-base-dir", default="D:/Data/LingLing/libero/hf/libero_goal")
    ap.add_argument("--out", default=os.path.join("output", "goal_v1_vs_v3.json"))
    args = ap.parse_args()

    A, B = scan(args.a), scan(args.b)
    tasks = sorted(set(A) | set(B))
    report = {"a": args.a, "b": args.b, "tasks": {}}

    print("=" * 100)
    print("1. JOINT-LIMIT SATURATION  (min margin over episode; <0.05 rad = at a hard limit)")
    print(f"{'task':42s}{args.label_a:>14}{args.label_b:>14}{'SOURCE demos':>18}")
    tot = {"a": [0, 0], "b": [0, 0]}
    for t in tasks:
        src = scan_source(t, args.source_base_dir)
        a = np.array([r["margin"] for r in A.get(t, [])])
        b = np.array([r["margin"] for r in B.get(t, [])])
        sa = f"{(a < 0.05).sum()}/{len(a)}" if len(a) else "-"
        sb = f"{(b < 0.05).sum()}/{len(b)}" if len(b) else "-"
        ss = f"{(src['margin'] < 0.05).sum()}/{len(src['margin'])}" if src else "-"
        tot["a"][0] += int((a < 0.05).sum()); tot["a"][1] += len(a)
        tot["b"][0] += int((b < 0.05).sum()); tot["b"][1] += len(b)
        print(f"  {t[:40]:40s}{sa:>14}{sb:>14}{ss:>18}")
        report["tasks"].setdefault(t, {})["saturated"] = {
            args.label_a: sa, args.label_b: sb, "source": ss}
    print(f"  {'TOTAL':40s}{tot['a'][0]}/{tot['a'][1]:<9}{tot['b'][0]}/{tot['b'][1]:<9}")

    print("\n" + "=" * 100)
    print("2. FINAL POSE of the manipulated object (upright: 1=standing, 0=on its side)")
    print(f"{'task':42s}{'v1 mean/min':>22}{'v3 mean/min':>22}{'SOURCE mean/min':>22}")
    for t in tasks:
        if t not in MANIP:
            continue
        src = scan_source(t, args.source_base_dir)
        a = np.array([r["upright"] for r in A.get(t, []) if "upright" in r])
        b = np.array([r["upright"] for r in B.get(t, []) if "upright" in r])
        fa = f"{a.mean():+.2f}/{a.min():+.2f}" if len(a) else "-"
        fb = f"{b.mean():+.2f}/{b.min():+.2f}" if len(b) else "-"
        fs = f"{src['upright'].mean():+.2f}/{src['upright'].min():+.2f}" if src and len(src["upright"]) else "-"
        print(f"  {t[:40]:40s}{fa:>22}{fb:>22}{fs:>22}")
        report["tasks"].setdefault(t, {})["upright"] = {
            args.label_a: fa, args.label_b: fb, "source": fs}

    print("\n" + "=" * 100)
    print("3. EE MAX DISTANCE from the robot base (the reach envelope the arm actually used)")
    print(f"{'task':42s}{'v1 mean/max':>22}{'v3 mean/max':>22}{'SOURCE mean/max':>22}")
    for t in tasks:
        src = scan_source(t, args.source_base_dir)
        a = np.array([r["ee_max"] for r in A.get(t, [])])
        b = np.array([r["ee_max"] for r in B.get(t, [])])
        fa = f"{a.mean():.3f}/{a.max():.3f}" if len(a) else "-"
        fb = f"{b.mean():.3f}/{b.max():.3f}" if len(b) else "-"
        fs = f"{src['ee_max'].mean():.3f}/{src['ee_max'].max():.3f}" if src else "-"
        print(f"  {t[:40]:40s}{fa:>22}{fb:>22}{fs:>22}")
        report["tasks"].setdefault(t, {})["ee_max"] = {
            args.label_a: fa, args.label_b: fb, "source": fs}

    print("\n" + "=" * 100)
    print("4. REACH-ENVELOPE VIOLATIONS per demo (task goal points outside the limits)")
    for lbl, D in ((args.label_a, A), (args.label_b, B)):
        n = sum(1 for t in D for r in D[t] if r.get("reach_viol", 0) > 0)
        tt = sum(len(D[t]) for t in D)
        print(f"  {lbl}: {n}/{tt} demos have at least one out-of-envelope goal point")
        report.setdefault("reach_violating_demos", {})[lbl] = f"{n}/{tt}"

    print("\n" + "=" * 100)
    print("5. COUNTS")
    for lbl, D, root in ((args.label_a, A, args.a), (args.label_b, B, args.b)):
        n = sum(len(D[t]) for t in D)
        cells = len({(t, r["layout"], r["split"]) for t in D for r in D[t]})
        sz = sum(os.path.getsize(p) for p in glob.glob(os.path.join(root, "*", "*.hdf5"))) / 1e9
        print(f"  {lbl}: {n} demos, {cells} cells, {sz:.1f} GB")
        report.setdefault("counts", {})[lbl] = {"demos": n, "cells": cells, "gb": round(sz, 1)}

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
