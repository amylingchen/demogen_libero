"""Render OC observations for the generated goal dataset by STATE REPLAY, and
fix the stored image orientation at the same time (plan §7).

Why this can be done without re-running physics: every demo stores the full
79-dim sim state per frame, and the fixture poses live in per-demo attrs. The
fixture pixel check measured that state + attrs rebuild the scene (0.23% of
pixels differ from the stored frame, against 9.8-28.7% when the fixture edits
are skipped), and a per-frame check across a trajectory puts the residual at
0.003-0.8% -- rendering noise, not drift.

Two things are corrected relative to goal_gen_v3:

1. ORIENTATION. goal_gen_v3 stores raw GL frames (row 0 = bottom) because it
   used replay_uniform's default recording path, while the object and spatial
   suites store frames flipped upright. Measured, not assumed: the object and
   spatial suites' gripper masks sit at rows 46-49 against 139-140 for the
   table objects, i.e. gripper on top; goal's stored frames match a raw GL
   render. Everything written here is flipped upright, matching the other
   suites, so downstream code needs no per-suite special case.
2. FIELD PARITY. goal_gen_v3 carries 5 obs fields; object/spatial carry 15.
   This adds depth (uint8 cm and lossless uint16 mm), segmentation, ee
   orientation and the per-frame object pose GT.

Segmentation uses the goal entity table (goal_scene.GOAL_SEG_IDS), which is
built from ELEMENT segmentation regrouped by body so the cabinet's middle and
top drawers get their own ids -- an instance mask cannot ground "the middle
drawer".

Resumable: demos already present in the output file are skipped, so a run
interrupted by a full disk or a crash continues where it stopped.

Usage:
    .venv\\Scripts\\python.exe scripts\\render_goal_oc.py \\
        --src output/goal_gen_v3 --out output/goal_oc_v3
"""
import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np
import robosuite.utils.transform_utils as TU
from robosuite.utils.camera_utils import get_real_depth_map

from demogen_libero import libero_replay as R
from demogen_libero import spatial_scene as S
from demogen_libero import goal_scene as G

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SPLITS = ["train", "quarantine_cf", "quarantine_unseen"]


def flip(a):
    return np.ascontiguousarray(a[::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join("output", "goal_gen_v3"))
    ap.add_argument("--out", default=os.path.join("output", "goal_oc_v3"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--no-depth-mm", action="store_true",
                    help="skip the lossless uint16 mm depth (~20GB smaller)")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--min-free-gb", type=float, default=8.0,
                    help="stop cleanly when the disk drops below this")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = []
    for split in SPLITS:
        for p in sorted(glob.glob(os.path.join(args.src, split, "*.hdf5"))):
            task = os.path.basename(p).replace("_demo.hdf5", "")
            if args.tasks and task not in args.tasks:
                continue
            jobs.append((split, task, p))

    total_done = 0
    for split, task, src_path in jobs:
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{task}_demo.hdf5")
        with h5py.File(src_path, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        have = set()
        if os.path.exists(out_path):
            with h5py.File(out_path, "r") as f:
                have = set(f["data"].keys()) if "data" in f else set()
        todo = [k for k in keys if k not in have]
        if not todo:
            print(f"[{split}/{task}] already complete ({len(have)})", flush=True)
            continue

        env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                        "libero_goal", f"{task}.bddl"),
            camera_heights=args.size, camera_widths=args.size,
            camera_depths=True, camera_segmentations="element")
        env.reset()
        lut = G.build_goal_seg_lut(env)
        # orientation guard: the LUT-mapped RAW render must read as "gl", so a
        # flip is genuinely needed. If this ever reads "upright" the render
        # convention changed and flipping would double-invert.
        fp = G.flip_fingerprint(env, lut)
        assert fp["verdict_raw_render"] == "gl" and fp["discriminative"], \
            f"orientation guard failed: {fp['verdict_raw_render']}, " \
            f"discriminative={fp['discriminative']}"

        for key in todo:
            free_gb = shutil.disk_usage(os.path.abspath(args.out)).free / 1e9
            if free_gb < args.min_free_gb:
                print(f"\nSTOPPING: only {free_gb:.1f} GB free "
                      f"(--min-free-gb {args.min_free_gb}). Re-run to continue.",
                      flush=True)
                env.close()
                return
            with h5py.File(src_path, "r") as f:
                src_manifest = f["data"].attrs.get("manifest", "")
                g = f["data"][key]
                states = np.array(g["states"], dtype=np.float64)
                actions = np.array(g["actions"])
                phase = np.array(g["phase_id"])
                attrs = dict(g.attrs)
            fx = {fb: {"pos": np.asarray(e["pos"]),
                       "quat": np.asarray(e["quat_wxyz"])}
                  for fb, e in json.loads(attrs["fixture_edits"]).items()}

            rows = {k: [] for k in
                    ["agentview_rgb", "agentview_depth", "agentview_seg",
                     "eye_in_hand_rgb", "eye_in_hand_depth", "eye_in_hand_seg",
                     "ee_pos", "ee_ori", "ee_states", "gripper_states",
                     "joint_states", "obj_pos", "obj_quat"]}
            if not args.no_depth_mm:
                rows["agentview_depth_mm"] = []
                rows["eye_in_hand_depth_mm"] = []

            # ONE reset per demo, then set the state directly. Calling
            # reset_to_init_state per frame costs 2202 ms/frame because it does a
            # full env.reset(); setting the state and forwarding is 74 ms, a 30x
            # speedup (46 h -> 1.5 h for the dataset) at identical fidelity
            # (max 0.70% of pixels differing by >8 levels either way). The reset
            # is still needed once, to place the fixtures before the loop.
            env.reset()
            S.apply_fixture_edits(env, fx)
            for t in range(len(states)):
                env.sim.set_state_from_flattened(states[t])
                env.sim.forward()
                obs = env.env._get_observations(force_update=True)
                a_m = get_real_depth_map(env.sim, obs["agentview_depth"])[..., 0]
                h_m = get_real_depth_map(env.sim, obs["robot0_eye_in_hand_depth"])[..., 0]
                a_seg = lut[np.clip(obs["agentview_segmentation_element"][..., 0],
                                    0, len(lut) - 1)]
                h_seg = lut[np.clip(
                    obs["robot0_eye_in_hand_segmentation_element"][..., 0],
                    0, len(lut) - 1)]
                rows["agentview_rgb"].append(flip(obs["agentview_image"]))
                rows["agentview_depth"].append(
                    flip(np.clip(np.rint(a_m * 100.0), 0, 255).astype(np.uint8)))
                rows["agentview_seg"].append(flip(a_seg))
                rows["eye_in_hand_rgb"].append(flip(obs["robot0_eye_in_hand_image"]))
                rows["eye_in_hand_depth"].append(
                    flip(np.clip(np.rint(h_m * 100.0), 0, 255).astype(np.uint8)))
                rows["eye_in_hand_seg"].append(flip(h_seg))
                if not args.no_depth_mm:
                    rows["agentview_depth_mm"].append(
                        flip(np.clip(np.rint(a_m * 1000.0), 0, 65535).astype(np.uint16)))
                    rows["eye_in_hand_depth_mm"].append(
                        flip(np.clip(np.rint(h_m * 1000.0), 0, 65535).astype(np.uint16)))
                ee_p = np.asarray(obs["robot0_eef_pos"])
                ee_q = np.asarray(obs["robot0_eef_quat"])
                rows["ee_pos"].append(ee_p)
                rows["ee_ori"].append(TU.quat2axisangle(ee_q))
                rows["ee_states"].append(np.concatenate([ee_p, TU.quat2axisangle(ee_q)]))
                rows["gripper_states"].append(np.asarray(obs["robot0_gripper_qpos"]))
                rows["joint_states"].append(np.asarray(obs["robot0_joint_pos"]))
                # per-frame GT pose of the 4 movable objects, in GOAL_SEG_IDS order
                op, oq = [], []
                for jn in G.GOAL_JOINTS:
                    q = env.sim.data.get_joint_qpos(jn)
                    op.append(np.asarray(q[:3]))
                    oq.append(np.asarray(q[3:7]))
                rows["obj_pos"].append(np.stack(op))
                rows["obj_quat"].append(np.stack(oq))

            with h5py.File(out_path, "a") as f:
                data = f.require_group("data")
                if "bddl_file_name" not in data.attrs:
                    data.attrs["bddl_file_name"] = (
                        f"libero/libero/bddl_files/libero_goal/{task}.bddl")
                    data.attrs["manifest"] = src_manifest
                    data.attrs["source_dataset"] = os.path.abspath(args.src)
                    data.attrs["images_vertically_flipped"] = True
                    data.attrs["seg_ids"] = json.dumps(G.GOAL_SEG_IDS)
                    data.attrs["obj_order"] = json.dumps(
                        [j.replace("_joint0", "") for j in G.GOAL_JOINTS])
                ep = data.create_group(key)
                ep.create_dataset("actions", data=actions)
                ep.create_dataset("states", data=states)
                ep.create_dataset("phase_id", data=phase)
                og = ep.create_group("obs")
                for k, v in rows.items():
                    og.create_dataset(k, data=np.asarray(v))
                for ak, av in attrs.items():
                    ep.attrs[ak] = av
                ep.attrs["num_samples"] = len(states)
            total_done += 1
            print(f"[{split}/{task}] {key} ({len(states)} frames)  "
                  f"free {free_gb:.0f}GB  done {total_done}", flush=True)
        env.close()

    print(f"\nDONE: rendered {total_done} demos into {args.out}")


if __name__ == "__main__":
    main()
