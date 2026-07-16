"""Re-execute original LIBERO source demos in the simulator and re-record them in
OC format (rgb + depth cm/mm + instance seg + obj_pos/obj_quat), by STATE replay:
each stored sim state is set and OC observations are rendered fresh, so the
trajectory matches the human demo exactly (open-loop action replay drifts).

This is the faithful "replay an existing LIBERO demo and collect seg/depth"
path -- a generalized standalone version of append_regrasp_demos.state_replay_oc
that takes ALL (or a chosen subset of) demos rather than only regrasp ones.

Usage:
    .venv\\Scripts\\python.exe scripts\\replay_source_oc.py ^
        --task-key pick_up_the_salad_dressing_and_place_it_in_the_basket ^
        --out-dir output\\replay_salad
    # subset + drop the lossless mm depth:
    .venv\\Scripts\\python.exe scripts\\replay_source_oc.py --task-key <key> ^
        --demos demo_0 demo_1 --no-depth-mm
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np

from demogen_libero.convert import load_demo
from demogen_libero.trajectory import auto_segment, segment_regrasp
from demogen_libero import oc_obs
from demogen_libero.config import source_hdf5


def state_replay_oc(env, lut, states, actions, frames, object_order=None, depth_mm=True):
    """Set each stored sim state and render one OC frame; returns (rollout, success).
    Pre-step aligned: obs[t] rendered from state[t] paired with action[t]."""
    f1, f2, f3 = frames.as_tuple()
    n = min(len(states), len(actions))
    rollout = {}
    env.reset()
    for t in range(n):
        obs = env.set_init_state(states[t])
        row = oc_obs.extract_oc_frame(env, obs, lut, object_order=object_order, depth_mm=depth_mm)
        row["actions"] = actions[t]
        row["phase_id"] = np.int32(0 if t < f1 else (1 if t < f2 else (2 if t < f3 else 3)))
        for k, v in row.items():
            rollout.setdefault(k, []).append(v)
    success = bool(env.check_success())
    return {k: np.asarray(v) for k, v in rollout.items()}, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-key", type=str, required=True,
                        help="task name; keys oc_obs.OBJECT_ORDER and source_hdf5()")
    parser.add_argument("--src", type=str, default=None,
                        help="source HDF5 (default: resolved from --task-key)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output dir (default: output/replay_<task-key>)")
    parser.add_argument("--demos", type=str, nargs="*", default=None,
                        help="demo keys to replay (default: all)")
    parser.add_argument("--bddl", type=str, default=None,
                        help="bddl file override (for sources lacking a bddl_file_name attr, "
                             "e.g. replaying an already-generated OC dataset)")
    parser.add_argument("--segment", choices=["auto", "regrasp"], default="regrasp")
    parser.add_argument("--no-depth-mm", action="store_true")
    args = parser.parse_args()

    src = args.src or source_hdf5(args.task_key)
    out_dir = args.out_dir or os.path.join("output", f"replay_{args.task_key}")
    os.makedirs(out_dir, exist_ok=True)
    hdf5_out = os.path.join(out_dir, f"{args.task_key}_demo.hdf5")
    meta_out = os.path.join(out_dir, "metainfo.json")

    with h5py.File(src, "r") as f:
        all_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    keys = args.demos if args.demos else all_keys
    print(f"replaying {len(keys)} demos from {src}")

    bddl_file = args.bddl if args.bddl else load_demo(src, keys[0]).bddl_file
    env = oc_obs.make_oc_env(bddl_file)
    env.reset()
    obj_order = oc_obs.OBJECT_ORDER[args.task_key]
    lut = oc_obs.build_seg_lut(env, obj_order)
    segment_fn = segment_regrasp if args.segment == "regrasp" else auto_segment

    meta = {args.task_key: {}}
    n_ok = 0
    try:
        with h5py.File(src, "r") as f:
            for i, key in enumerate(keys):
                states = np.array(f["data"][key]["states"], dtype=np.float64)
                actions = np.array(f["data"][key]["actions"], dtype=np.float64)
                ee = np.array(f["data"][key]["obs"]["ee_pos"], dtype=np.float64)
                frames = segment_fn(ee, actions)
                rollout, success = state_replay_oc(
                    env, lut, states, actions, frames,
                    object_order=obj_order, depth_mm=not args.no_depth_mm)
                if not success:
                    print(f"  {key}: state-replay did NOT reach success; skipping")
                    continue
                demo_name = f"demo_{n_ok}"
                target_nm = oc_obs.display_name(obj_order[0])
                goal_nm = oc_obs.display_name(obj_order[1])
                rollout["subtask_id"], subtasks = oc_obs.annotate_subtasks(
                    rollout["actions"], target_nm, goal_nm, post_steps=0)
                oc_obs.write_oc_demo(hdf5_out, demo_name, rollout, success=True,
                                     object_order=obj_order, subtasks=subtasks)
                meta[args.task_key][demo_name] = oc_obs.metainfo_entry(
                    args.task_key, rollout, obj_order, states[0], subtasks=subtasks)
                print(f"  {key} -> {demo_name}  T={rollout['actions'].shape[0]}")
                n_ok += 1
    finally:
        env.close()

    json.dump(meta, open(meta_out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {n_ok} demos -> {hdf5_out}")


if __name__ == "__main__":
    main()
