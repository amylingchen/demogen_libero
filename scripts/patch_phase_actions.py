"""Backfill the fine-grained `phases` annotation (action word + per-stage
target object) onto existing OC datasets, derived deterministically from the
stored per-frame phase_id and the task's target/goal objects -- no
re-simulation. Writes the `phases` JSON attr on every demo group and the
"phases" field into metainfo.json.

Usage:
    .venv\Scripts\python.exe scripts\patch_phase_actions.py --base output\libero_object_100
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np

from demogen_libero.oc_obs import phase_stages, display_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=os.path.join("output", "libero_object_100"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dirs = [d for d in sorted(glob.glob(os.path.join(args.base, "*"))) if os.path.isdir(d)]
    for d in dirs:
        h5s = glob.glob(os.path.join(d, "*.hdf5"))
        if not h5s:
            continue
        meta_path = os.path.join(d, "metainfo.json")
        meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else None
        task_key = next(iter(meta)) if meta else None
        n_done = n_skip = 0
        with h5py.File(h5s[0], "a") as f:
            for k in f["data"].keys():
                g = f["data"][k]
                if "phase_id" not in g:
                    n_skip += 1
                    continue
                if "phases" in g.attrs and not args.overwrite:
                    n_skip += 1
                    continue
                order = json.loads(g.attrs["object_instances"]) if "object_instances" in g.attrs else None
                if order is None:
                    n_skip += 1
                    continue
                stages = phase_stages(np.array(g["phase_id"]),
                                      display_name(order[0]), display_name(order[1]))
                g.attrs["phases"] = json.dumps(stages)
                if meta and k in meta.get(task_key, {}):
                    meta[task_key][k]["phases"] = stages
                n_done += 1
        if meta is not None and n_done:
            json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=2)
        print(f"{os.path.basename(d)}: patched {n_done}, skipped {n_skip}")


if __name__ == "__main__":
    main()
