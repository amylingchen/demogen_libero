"""Find regrasp source trajectories (>1 gripper close cycle) and append them to
an existing OC dataset via STATE replay: each recorded sim state is set and the
OC observations (rgb/depth/seg/bbox + mm depth) are rendered fresh, so the
trajectory matches the human demo exactly (open-loop action replay drifts and
fails on these). Phases use segment_regrasp, which folds the whole regrasp
attempt into skill_1. Obs are pre-step aligned (obs rendered from state[t]
paired with action[t]).

Usage:
    .venv\Scripts\python.exe scripts\append_regrasp_demos.py --dir output\grid_oc_salad_v2
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np

from demogen_libero.convert import load_demo
from demogen_libero.trajectory import segment_regrasp
from demogen_libero import oc_obs

from demogen_libero.config import source_hdf5
HDF5_PATH = source_hdf5("pick_up_the_salad_dressing_and_place_it_in_the_basket")
TASK_KEY = "pick_up_the_salad_dressing_and_place_it_in_the_basket"


def find_regrasp_demos(src_path):
    out = []
    with h5py.File(src_path, "r") as f:
        for k in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1])):
            grip = np.array(f["data"][k]["actions"][:, 6])
            n_close = int(np.sum((grip[:-1] <= 0) & (grip[1:] > 0)))
            if n_close > 1:
                out.append(k)
    return out


def state_replay_oc(env, lut, states, actions, frames, object_order=None, depth_mm=True):
    """Render OC observations by setting each sim state; returns rollout dict +
    success. Pre-step aligned: obs[t] rendered from state[t] paired with
    action[t]. phase_id from `frames`."""
    f1, f2, f3 = frames.as_tuple()
    n = min(len(states), len(actions))
    rollout = {}
    env.reset()
    for t in range(n):
        obs = env.set_init_state(states[t])          # sim now at state[t]
        row = oc_obs.extract_oc_frame(env, obs, lut, object_order=object_order, depth_mm=depth_mm)
        row["actions"] = actions[t]
        row["phase_id"] = np.int32(0 if t < f1 else (1 if t < f2 else (2 if t < f3 else 3)))
        for k, v in row.items():
            rollout.setdefault(k, []).append(v)
    success = bool(env.check_success())
    return {k: np.asarray(v) for k, v in rollout.items()}, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=os.path.join("output", "grid_oc_salad_v2"))
    parser.add_argument("--src", type=str, default=HDF5_PATH)
    parser.add_argument("--no-depth-mm", action="store_true")
    args = parser.parse_args()

    hdf5_out = glob.glob(os.path.join(args.dir, "*_demo.hdf5"))[0]
    meta_out = os.path.join(args.dir, "metainfo.json")
    log_out = os.path.join(args.dir, "scene_log.json")

    regrasp = find_regrasp_demos(args.src)
    print(f"regrasp source demos ({len(regrasp)}): {regrasp}")

    with h5py.File(hdf5_out, "r") as f:
        existing = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        next_idx = int(existing[-1].split("_")[-1]) + 1
    meta = json.load(open(meta_out, encoding="utf-8"))
    scene_log = json.load(open(log_out, encoding="utf-8")) if os.path.exists(log_out) else []

    demo0 = load_demo(args.src, regrasp[0])
    env = oc_obs.make_oc_env(demo0.bddl_file)
    env.reset()
    obj_order = oc_obs.OBJECT_ORDER[TASK_KEY]
    lut = oc_obs.build_seg_lut(env, obj_order)

    added = 0
    try:
        with h5py.File(args.src, "r") as src:
            for src_key in regrasp:
                states = np.array(src["data"][src_key]["states"], dtype=np.float64)
                actions = np.array(src["data"][src_key]["actions"], dtype=np.float64)
                ee = np.array(src["data"][src_key]["obs"]["ee_pos"], dtype=np.float64)
                frames = segment_regrasp(ee, actions)
                rollout, success = state_replay_oc(env, lut, states, actions, frames,
                                                   object_order=obj_order, depth_mm=not args.no_depth_mm)
                if not success:
                    print(f"  {src_key}: state-replay did NOT reach success; skipping")
                    continue
                demo_name = f"demo_{next_idx + added}"
                target_nm = oc_obs.display_name(obj_order[0])
                goal_nm = oc_obs.display_name(obj_order[1])
                rollout["subtask_id"], subtasks = oc_obs.annotate_subtasks(
                    rollout["actions"], target_nm, goal_nm, post_steps=0)
                oc_obs.write_oc_demo(hdf5_out, demo_name, rollout, success=True,
                                     object_order=obj_order, subtasks=subtasks)
                meta[TASK_KEY][demo_name] = oc_obs.metainfo_entry(
                    TASK_KEY, rollout, obj_order, states[0], subtasks=subtasks)
                scene_log.append({
                    "demo_name": demo_name, "source_demo": src_key,
                    "kind": "regrasp_state_replay", "frames": list(frames.as_tuple()),
                    "n_frames": int(rollout["actions"].shape[0]),
                })
                ph = rollout["phase_id"]
                print(f"  {src_key} -> {demo_name}  T={len(ph)}  "
                      f"phases {{0:{int((ph==0).sum())},1:{int((ph==1).sum())},"
                      f"2:{int((ph==2).sum())},3:{int((ph==3).sum())}}}")
                added += 1
    finally:
        env.close()

    json.dump(meta, open(meta_out, "w", encoding="utf-8"), indent=2)
    json.dump(scene_log, open(log_out, "w"), indent=2)
    with h5py.File(hdf5_out, "r") as f:
        n = len(f["data"].keys())
    print(f"\nappended {added} regrasp demos -> {hdf5_out}  (now {n} total)")


if __name__ == "__main__":
    main()
