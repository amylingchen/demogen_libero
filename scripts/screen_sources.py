"""Per-task source-demo screening: for every libero_object task, classify each
source demo as
  - healthy: single grasp cycle AND open-loop zero-offset replay reaches success
  - regrasp: >1 gripper close cycle (kept for --segment regrasp augmentation)
  - bad: single grasp but open-loop replay fails (drifts) -> excluded
Writes a JSON consumed by run_grid_oc_demo.py (--sources-json).

Usage:
    .venv\Scripts\python.exe scripts\screen_sources.py --tasks salad_dressing butter ...
    .venv\Scripts\python.exe scripts\screen_sources.py              # all 10 tasks
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
from demogen_libero import oc_obs

from demogen_libero.config import DATA_DIR as SOURCE_BASE_DIR


def n_close_cycles(actions):
    grip = actions[:, 6]
    return int(np.sum((grip[:-1] <= 0) & (grip[1:] > 0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=None, help="short names; default all")
    parser.add_argument("--source-base-dir", type=str, default=SOURCE_BASE_DIR)
    parser.add_argument("--demo-limit", type=int, default=0, help="0 = all demos")
    parser.add_argument("--out", type=str, default="output/source_screening.json")
    args = parser.parse_args()

    all_tasks = list(oc_obs.load_task_config().keys())
    if args.tasks:
        want = {f"pick_up_the_{t}_and_place_it_in_the_basket" if not t.startswith("pick_up") else t
                for t in args.tasks}
        tasks = [t for t in all_tasks if t in want]
    else:
        tasks = all_tasks

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
                if env is None:
                    env = R.make_env(d.bddl_file)
                else:
                    # rebuild env if the task (bddl) changed
                    if getattr(env, "_task_bddl", None) != d.bddl_file:
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
            print(f"{task.replace('pick_up_the_','').replace('_and_place_it_in_the_basket','')}: "
                  f"healthy={len(healthy)} regrasp={len(regrasp)} bad={len(bad)}", flush=True)
    finally:
        if env is not None:
            env.close()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
