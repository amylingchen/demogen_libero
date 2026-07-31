"""Label every generated demo with whether it contains a failed grasp followed
by a recovery regrasp, derived from the recorded actions.

The label is not something the generator can know up front: whether a rollout
regrasps depends on which source demo happened to be drawn for that scene, and
`segment_for` preserves the source's grasp-release-regrasp block verbatim. So it
is recovered here the same way source screening detects it -- by counting
gripper CLOSE transitions in the action stream, where >1 means the gripper let
go and took hold again.

Written into metainfo.json and scene_log.json (not the hdf5, so a running
generator is never contended with). Idempotent: re-running only refreshes.

Usage:
    .venv\\Scripts\\python.exe scripts\\patch_regrasp_label.py
    .venv\\Scripts\\python.exe scripts\\patch_regrasp_label.py --root output/libero_spatial_100
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np


def close_cycles(actions):
    """Gripper close transitions. LIBERO's action[:, 6] is the gripper command:
    <=0 open, >0 closed, so a rising edge is one grasp attempt."""
    g = np.asarray(actions)[:, 6]
    return int(np.sum((g[:-1] <= 0) & (g[1:] > 0)))


def task_dirs(root):
    for meta in sorted(glob.glob(os.path.join(root, "**", "metainfo.json"),
                                 recursive=True)):
        d = os.path.dirname(meta)
        h5 = glob.glob(os.path.join(d, "*_demo.hdf5"))
        if h5:
            yield d, h5[0], meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/libero_spatial_pairs")
    ap.add_argument("--only-finished", action="store_true",
                    help="skip a directory whose generator has not logged 'done:' "
                         "yet, so a live run is never patched mid-write")
    ap.add_argument("--log-dir", default=None,
                    help="where per-task generator logs live, for --only-finished")
    args = ap.parse_args()

    grand = [0, 0]
    for d, h5, meta_path in task_dirs(args.root):
        name = os.path.basename(d)
        if args.only_finished and args.log_dir:
            lg = os.path.join(args.log_dir, f"{name}.log")
            if not (os.path.exists(lg)
                    and any("done:" in l for l in open(lg, encoding="utf-8",
                                                       errors="replace"))):
                print(f"  {name:<42} still generating -- skipped")
                continue
        try:
            with h5py.File(h5, "r") as f:
                cycles = {k: close_cycles(f["data"][k]["actions"])
                          for k in f["data"].keys()}
        except Exception as exc:
            print(f"  {name:<42} unreadable ({exc!r}) -- skipped")
            continue

        meta = json.load(open(meta_path))
        n_patched = 0
        for task, demos in meta.items():
            if not isinstance(demos, dict):
                continue
            for dn, entry in demos.items():
                if dn in cycles and isinstance(entry, dict):
                    entry["n_grasp_cycles"] = cycles[dn]
                    entry["is_regrasp"] = bool(cycles[dn] > 1)
                    n_patched += 1
        json.dump(meta, open(meta_path, "w"), indent=2)

        log_path = os.path.join(d, "scene_log.json")
        if os.path.exists(log_path):
            log = json.load(open(log_path))
            for row in log:
                dn = row.get("demo_name")
                if dn in cycles:
                    row["n_grasp_cycles"] = cycles[dn]
                    row["is_regrasp"] = bool(cycles[dn] > 1)
            json.dump(log, open(log_path, "w"), indent=2)

        rg = sum(1 for v in cycles.values() if v > 1)
        grand[0] += rg
        grand[1] += len(cycles)
        print(f"  {name:<42}{len(cycles):>4} demos  {rg:>3} regrasp "
              f"({100 * rg / max(len(cycles), 1):>3.0f}%)  patched {n_patched}")

    print(f"\ntotal {grand[0]}/{grand[1]} = "
          f"{100 * grand[0] / max(grand[1], 1):.0f}% regrasp")


if __name__ == "__main__":
    main()
