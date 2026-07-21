"""Run every libero_spatial task once by state-replaying a source demo and
recording an agentview mp4 per task.

State replay (set each recorded sim state + forward + render) reproduces the
human demonstration exactly, independent of OSC controller dynamics, so the
video shows the full pick-and-place and check_success() reflects the demo.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_spatial_replay.py
    .venv\\Scripts\\python.exe scripts\\run_spatial_replay.py --demo demo_0 ^
        --out-dir output\\libero_spatial_replay --height 256 --width 256
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np

from libero.libero import benchmark
from demogen_libero.convert import resolve_bddl_file
from demogen_libero import libero_replay as R


def load_states(hdf5_path, demo_key):
    with h5py.File(hdf5_path, "r") as f:
        bddl = resolve_bddl_file(f["data"].attrs["bddl_file_name"])
        states = np.array(f["data"][demo_key]["states"], dtype=np.float64)
    return bddl, states


def state_replay(env, states, camera, height, width):
    """Set each recorded sim state, render agentview, track success."""
    frames = []
    success = False
    success_step = -1
    for t, s in enumerate(states):
        env.sim.set_state_from_flattened(s)
        env.sim.forward()
        img = env.sim.render(camera_name=camera, height=height, width=width)
        frames.append(img)
        if env.check_success():
            success = True
            if success_step < 0:
                success_step = t
    return success, success_step, np.asarray(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=str, default="demo_0")
    parser.add_argument("--data-dir", type=str,
                        default="D:/Data/LingLing/libero/hf/libero_spatial")
    parser.add_argument("--out-dir", type=str, default="output/libero_spatial_replay")
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bm = benchmark.get_benchmark_dict()["libero_spatial"]()

    results = []
    for i in range(bm.n_tasks):
        task = bm.get_task(i)
        name = task.name
        hdf5 = os.path.join(args.data_dir, f"{name}_demo.hdf5")
        if not os.path.exists(hdf5):
            print(f"[{i}] MISSING {hdf5}")
            results.append((i, name, "missing", -1, 0))
            continue

        bddl, states = load_states(hdf5, args.demo)
        env = R.make_env(bddl, camera_height=args.height, camera_width=args.width)
        try:
            env.reset()
            success, sstep, frames = state_replay(
                env, states, args.camera, args.height, args.width)
        finally:
            env.close()

        video_path = os.path.join(args.out_dir, f"{i:02d}_{name}.mp4")
        R.save_video(video_path, frames, fps=args.fps)
        status = "SUCCESS" if success else "FAIL"
        print(f"[{i}] {status} T={len(states)} success@{sstep} -> {video_path}")
        results.append((i, name, status, sstep, len(states)))

    print("\n=== summary ===")
    n_ok = sum(1 for r in results if r[2] == "SUCCESS")
    for i, name, status, sstep, T in results:
        print(f"  [{i}] {status:8s} T={T:4d} success@{sstep:4d}  {name}")
    print(f"{n_ok}/{len(results)} tasks succeeded")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
