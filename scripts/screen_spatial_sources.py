"""Per-task source-demo screening for LIBERO_Spatial, mirroring
screen_sources.py (libero_object):
  - healthy: single grasp cycle AND open-loop zero-offset replay reaches success
  - regrasp: >1 gripper close cycle (breaks auto_segment) -> excluded for now
  - bad:     single grasp but open-loop replay fails (drifts) -> excluded
Writes a JSON consumed by run_spatial_oc_demo.py (--sources-json).

Usage:
    .venv\\Scripts\\python.exe scripts\\screen_spatial_sources.py            # all 10 tasks
    .venv\\Scripts\\python.exe scripts\\screen_spatial_sources.py --tasks pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np

from demogen_libero.convert import load_demo
from demogen_libero import libero_replay as R
from demogen_libero import spatial_scene as S


def n_close_cycles(actions):
    grip = actions[:, 6]
    return int(np.sum((grip[:-1] <= 0) & (grip[1:] > 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=None, help="full task keys; default all")
    ap.add_argument("--source-base-dir", default="D:/Data/LingLing/libero/hf/libero_spatial")
    ap.add_argument("--demo-limit", type=int, default=0, help="0 = all demos")
    ap.add_argument("--out", default="output/spatial_source_screening.json")
    args = ap.parse_args()

    tasks = args.tasks or list(S.load_spatial_config().keys())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result = json.load(open(args.out)) if os.path.exists(args.out) else {}
    env = None
    try:
        for task in tasks:
            hdf5 = os.path.join(args.source_base_dir, f"{task}_demo.hdf5")
            with h5py.File(hdf5, "r") as f:
                demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            if args.demo_limit:
                demos = demos[:args.demo_limit]
            healthy, regrasp, bad = [], [], []
            for key in demos:
                d = load_demo(hdf5, key)
                if env is None or getattr(env, "_task_bddl", None) != d.bddl_file:
                    if env is not None:
                        env.close()
                    env = R.make_env(d.bddl_file)
                    env._task_bddl = d.bddl_file
                if n_close_cycles(d.action) > 1:
                    regrasp.append(key)
                    continue
                R.reset_to_init_state(env, d.init_state)
                success, _, _ = R.replay_actions(env, d.action)
                (healthy if success else bad).append(key)
            result[task] = {"healthy": healthy, "regrasp": regrasp, "bad": bad,
                            "n_healthy": len(healthy), "n_regrasp": len(regrasp), "n_bad": len(bad)}
            json.dump(result, open(args.out, "w"), indent=2)
            short = task.replace("pick_up_the_black_bowl_", "").replace("_and_place_it_on_the_plate", "")
            print(f"{short}: healthy={len(healthy)} regrasp={len(regrasp)} bad={len(bad)}", flush=True)
    finally:
        if env is not None:
            env.close()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
