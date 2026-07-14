"""Shuffle-placement generation: place the target object at each of the scene's
existing object slots (its own + the 5 distractor positions read from the source
demo's init state, swapping the displaced distractor to the target's old spot so
nothing overlaps), synthesize a retargeted trajectory per slot from each source
demo, replay it in sim, and save successful and failed results (HDF5 episodes for
successes + a 256x256 agentview mp4 for every attempt).

Usage:
    .venv\Scripts\python.exe scripts\run_shuffle_positions.py
    .venv\Scripts\python.exe scripts\run_shuffle_positions.py --demos demo_0 demo_1 demo_2 demo_3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from demogen_libero.convert import load_demo
from demogen_libero.trajectory import auto_segment, synthesize_uniform
from demogen_libero import libero_replay as R

from demogen_libero.config import source_hdf5
DEFAULT_HDF5 = source_hdf5("pick_up_the_salad_dressing_and_place_it_in_the_basket")
TARGET_JOINT = "salad_dressing_1_joint0"
DISTRACTOR_JOINTS = [
    "ketchup_1_joint0",
    "alphabet_soup_1_joint0",
    "cream_cheese_1_joint0",
    "milk_1_joint0",
    "tomato_sauce_1_joint0",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=str, default=DEFAULT_HDF5)
    parser.add_argument("--demos", nargs="+", default=["demo_0", "demo_1", "demo_2", "demo_3"])
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default=os.path.join("output", "shuffle_salad_dressing"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, "generated.hdf5")

    env = None
    results = []
    try:
        for demo_key in args.demos:
            demo = load_demo(args.hdf5, demo_key)
            if env is None:
                env = R.make_env(demo.bddl_file, camera_height=args.camera_size,
                                 camera_width=args.camera_size)
            frames = auto_segment(demo.state, demo.action)
            print(f"\n=== {demo_key}: T={demo.action.shape[0]} frames={frames.as_tuple()} ===")

            # Slot positions come from this demo's own init state.
            R.reset_to_init_state(env, demo.init_state)
            target_xy = R.get_body_pos(env, TARGET_JOINT)[:2]
            slots = {"original": None}  # zero-offset baseline slot
            for j in DISTRACTOR_JOINTS:
                slots[j.replace("_1_joint0", "")] = R.get_body_pos(env, j)[:2]

            for slot_name, slot_xy in slots.items():
                if slot_xy is None:
                    obj_t = np.zeros(3)
                    swap_joint = None
                else:
                    obj_t = np.array([slot_xy[0] - target_xy[0], slot_xy[1] - target_xy[1], 0.0])
                    swap_joint = f"{slot_name}_1_joint0"

                ref, base_actions, new_frames = synthesize_uniform(
                    demo.state, demo.action, frames, obj_t, np.zeros(3))
                R.reset_to_init_state(env, demo.init_state)
                if swap_joint is not None:
                    R.swap_object_xy(env, TARGET_JOINT, swap_joint)
                success, obs, rollout = R.replay_uniform(
                    env, base_actions, ref, new_frames, collect=True)

                tag = f"{demo_key}_slot_{slot_name}"
                video_path = os.path.join(args.out_dir, f"{tag}{'_ok' if success else '_fail'}.mp4")
                R.save_video(video_path, rollout["agentview_rgb"])
                if success:
                    R.save_episode(save_path, tag, rollout, attrs={
                        "source_demo": demo_key,
                        "source_hdf5": args.hdf5,
                        "slot": slot_name,
                        "obj_t": obj_t,
                        "frames": np.array(frames.as_tuple()),
                    })
                results.append((demo_key, slot_name, success, obj_t[:2].copy()))
                print(f"  [{tag}] obj_t={np.round(obj_t[:2], 3)} -> success={success}  video={os.path.basename(video_path)}")
    finally:
        if env is not None:
            env.close()

    n_ok = sum(1 for r in results if r[2])
    print(f"\n=== Summary: {n_ok}/{len(results)} succeeded ===")
    for demo_key, slot_name, success, obj_xy in results:
        print(f"  {demo_key:8s} {slot_name:14s} obj_t={np.round(obj_xy, 3)} {'OK' if success else 'FAIL'}")
    print(f"\nEpisodes: {save_path}\nVideos:   {args.out_dir}")


if __name__ == "__main__":
    main()
