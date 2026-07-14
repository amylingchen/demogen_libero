"""Post-hoc phase_id labeling for already-generated OC episodes: derive the
DemoGen two-stage boundaries from each saved episode's own actions + ee_pos
(same gripper-command + EE-speed logic as trajectory.auto_segment) and write a
per-frame int32 `phase_id` dataset into the HDF5.

Phases: 0=motion_1_reach, 1=skill_1_grasp, 2=motion_2_transport, 3=skill_2_place.

Usage:
    .venv\Scripts\python.exe scripts\patch_phase_id.py --dir output\grid_oc_salad_150
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np

from demogen_libero.trajectory import auto_segment
from demogen_libero.oc_obs import PHASE_MAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=os.path.join("output", "grid_oc_salad_150"))
    parser.add_argument("--hdf5", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute even if phase_id already exists")
    args = parser.parse_args()

    hdf5_path = args.hdf5 or glob.glob(os.path.join(args.dir, "*_demo.hdf5"))[0]
    with open(os.path.join(args.dir, "phase_map.json"), "w") as f:
        json.dump(PHASE_MAP, f, indent=2)

    n_done, n_skip, n_fail = 0, 0, 0
    lengths = {0: [], 1: [], 2: [], 3: []}
    with h5py.File(hdf5_path, "a") as f:
        for demo_name in sorted(f["data"].keys()):
            grp = f["data"][demo_name]
            if "phase_id" in grp and not args.overwrite:
                n_skip += 1
                continue
            actions = np.asarray(grp["actions"], dtype=np.float64)
            ee_pos = np.asarray(grp["obs"]["ee_pos"], dtype=np.float64)
            try:
                frames = auto_segment(ee_pos, actions)
            except AssertionError as exc:
                print(f"{demo_name}: segmentation failed ({exc})")
                n_fail += 1
                continue
            f1, f2, f3 = frames.as_tuple()
            T = actions.shape[0]
            phase = np.empty(T, dtype=np.int32)
            phase[:f1] = 0
            phase[f1:f2] = 1
            phase[f2:f3] = 2
            phase[f3:] = 3
            if "phase_id" in grp:
                del grp["phase_id"]
            grp.create_dataset("phase_id", data=phase)
            for p in range(4):
                lengths[p].append(int((phase == p).sum()))
            n_done += 1

    print(f"patched {n_done}, skipped {n_skip} (already labeled), failed {n_fail} -> {hdf5_path}")
    if n_done:
        for p, name in PHASE_MAP.items():
            arr = np.array(lengths[p])
            print(f"  phase {p} {name:20s} frames/demo: {arr.mean():.0f} (min {arr.min()}, max {arr.max()})")


if __name__ == "__main__":
    main()
