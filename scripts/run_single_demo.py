"""Randomly re-place the LIBERO target object, synthesize a new pick-place
trajectory from a single source demo (DemoGen two-stage retargeting), replay it
open-loop in the real LIBERO simulator, and report success. Retries with a new
random offset until --num-success successes are collected (or --max-attempts is
exhausted). Successful episodes are saved to an output HDF5 (mirroring the
source LIBERO demo layout) plus one agentview mp4 per episode.

Usage:
    .venv\Scripts\python.exe scripts\run_single_demo.py
    .venv\Scripts\python.exe scripts\run_single_demo.py ^
        --hdf5 data\libero_object\pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5 ^
        --demo demo_1 --object-joint salad_dressing_1_joint0 --num-success 10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from demogen_libero.convert import load_demo
from demogen_libero.trajectory import auto_segment, sample_object_offset, synthesize_two_stage
from demogen_libero import libero_replay as R

from demogen_libero.config import source_hdf5
DEFAULT_HDF5 = source_hdf5("pick_up_the_alphabet_soup_and_place_it_in_the_basket")


def run_attempt(env, demo, frames, obj_t, tar_t, attempt, object_joint, collect):
    new_action, _ = synthesize_two_stage(demo.state, demo.action, frames, obj_t, tar_t)

    R.reset_to_init_state(env, demo.init_state)
    R.translate_body_joint(env, object_joint, obj_t)
    success, obs, rollout = R.replay_actions(env, new_action, collect=collect)

    final_obj_pos = R.get_body_pos(env, object_joint)
    print(
        f"[attempt {attempt}] obj_t={np.round(obj_t, 3)} -> "
        f"success={success} final_object_xyz={np.round(final_obj_pos, 3)}"
    )
    return success, rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=str, default=DEFAULT_HDF5)
    parser.add_argument("--demo", type=str, default="demo_0")
    parser.add_argument("--object-joint", type=str, default="alphabet_soup_1_joint0",
                        help="free-joint name of the object to randomly re-place")
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--num-success", type=int, default=1,
                        help="stop after collecting this many successful episodes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius-min", type=float, default=0.02)
    parser.add_argument("--radius-max", type=float, default=0.06)
    parser.add_argument("--out-dir", type=str, default="output",
                        help="directory for the generated HDF5 + videos ('' to disable saving)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    demo = load_demo(args.hdf5, args.demo)
    frames = auto_segment(demo.state, demo.action)
    print(f"{args.demo}: T={demo.action.shape[0]} frames={frames.as_tuple()}")

    task_name = os.path.splitext(os.path.basename(args.hdf5))[0].replace("_demo", "")
    out_dir = args.out_dir or None
    if out_dir:
        out_dir = os.path.join(out_dir, f"{task_name}_{args.demo}")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "generated.hdf5")
    else:
        save_path = None

    env = R.make_env(demo.bddl_file)

    tar_t = np.zeros(3)  # only the target object is randomly re-placed, not the basket
    n_success = 0
    attempt = 0
    try:
        for attempt in range(1, args.max_attempts + 1):
            obj_t = sample_object_offset(rng, radius_xy=(args.radius_min, args.radius_max))
            success, rollout = run_attempt(env, demo, frames, obj_t, tar_t, attempt,
                                           args.object_joint, collect=save_path is not None)
            if not success:
                continue
            n_success += 1
            if save_path:
                episode_key = f"{args.demo}_aug_{n_success - 1}"
                R.save_episode(
                    save_path, episode_key, rollout,
                    attrs={
                        "source_demo": args.demo,
                        "source_hdf5": args.hdf5,
                        "obj_t": obj_t,
                        "tar_t": tar_t,
                        "frames": np.array(frames.as_tuple()),
                        "seed": args.seed,
                        "attempt": attempt,
                    },
                )
                video_path = os.path.join(out_dir, f"{episode_key}.mp4")
                R.save_video(video_path, rollout["agentview_rgb"])
                print(f"  saved -> {save_path}:data/{episode_key} + {video_path}")
            if n_success >= args.num_success:
                break
    finally:
        env.close()

    print(f"\nCollected {n_success}/{args.num_success} successful episode(s) "
          f"in {attempt} attempt(s).")
    sys.exit(0 if n_success >= args.num_success else 1)


if __name__ == "__main__":
    main()
