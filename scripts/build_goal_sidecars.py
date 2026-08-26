"""Write the sidecar json files for a generated goal dataset.

Produces, next to the split directories:

  phase_map.json    what phase_id means -- NOT the libero_object mapping. The
                    goal suite segments two ways: pick-and-place tasks get the
                    canonical four phases, while the fixture-operation tasks
                    (drawer, knob) and push are replayed as ONE rigidly
                    translated trajectory, so their phase 2 is a single nominal
                    frame and phase 3 holds the entire verbatim operation.
                    Measured spans are included per task so the file describes
                    the data rather than an intention.
  scene_log.json    one record per demo: layout, split, source demo, jitter,
                    robot noise, fixture poses, joint margin, final pose --
                    the same values carried in the hdf5 attrs, consolidated.
  camera_params.json  agentview intrinsics/extrinsics + the eye-in-hand
                    hand-eye transform FOR THIS SUITE. Per-suite camera params
                    differ (plan §7.2: the spatial suite once reused the object
                    suite's and landed 320 px off), so this is dumped from a
                    goal env, and a reprojection self-check is run and recorded.

NOT produced here: metainfo.json. That file is OC-format metadata (per-frame
segmentation boxes, subtask annotation) and only becomes meaningful once the
OC observations are rendered by state replay (plan §7); writing a placeholder
would claim coverage this dataset does not have.

Usage:
    .venv\\Scripts\\python.exe scripts\\build_goal_sidecars.py --dir output/goal_gen_v3
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np
import robosuite.utils.transform_utils as TU
from robosuite.utils.camera_utils import (get_camera_extrinsic_matrix,
                                          get_camera_intrinsic_matrix,
                                          get_camera_transform_matrix)

from demogen_libero import oc_obs
from demogen_libero import goal_scene as G

from libero.libero import get_libero_path

SPLITS = ["train", "quarantine_cf", "quarantine_unseen"]
PUSH = "push_the_plate_to_the_front_of_the_stove"
KIND_NOTE = {
    "pick_place": ("canonical four phases: reach, grasp, transport, place"),
    "fixture_op": ("ONE rigidly translated trajectory: phase 0 is the approach, "
                   "phase 1 the pre-contact settle, phase 2 a single nominal "
                   "frame, phase 3 the entire verbatim operation (pull / turn)"),
    "push": ("ONE rigidly translated trajectory anchored on the plate: phase 3 "
             "holds the whole contact push"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("output", "goal_gen_v3"))
    ap.add_argument("--manifest", default=None,
                    help="defaults to the manifest recorded in the hdf5 attrs")
    args = ap.parse_args()

    # ---- scan the dataset ----
    records = []
    spans = defaultdict(Counter)
    manifest_path = args.manifest
    for split in SPLITS:
        for p in sorted(glob.glob(os.path.join(args.dir, split, "*.hdf5"))):
            task = os.path.basename(p).replace("_demo.hdf5", "")
            with h5py.File(p, "r") as f:
                manifest_path = manifest_path or f["data"].attrs.get("manifest")
                for k in f["data"]:
                    g = f["data"][k]
                    a = dict(g.attrs)
                    ph = np.array(g["phase_id"])
                    c = Counter(ph.tolist())
                    for i in range(4):
                        spans[task][i] += c.get(i, 0)
                    spans[task]["n_demos"] += 1
                    spans[task]["n_frames"] += len(ph)
                    rec = {
                        "split": split, "task": task, "demo": k,
                        "layout_id": a.get("layout_id"),
                        "source_demo": a.get("source_demo"),
                        "T": int(len(ph)),
                        "joint_margin": float(a["joint_margin"]) if "joint_margin" in a else None,
                        "robot_noise": np.asarray(a["robot_noise"]).round(5).tolist()
                        if "robot_noise" in a else None,
                        "jitter_objects": json.loads(a["jitter_objects"]),
                        "jitter_fixtures": json.loads(a["jitter_fixtures"]),
                        "fixture_edits": json.loads(a["fixture_edits"]),
                    }
                    if "final_upright" in a:
                        rec["final_upright"] = float(a["final_upright"])
                    records.append(rec)
    print(f"scanned {len(records)} demos", flush=True)

    # ---- phase_map.json, described from the measured spans ----
    per_task = {}
    for task, c in spans.items():
        kind = G.GOAL_TASKS[task]["kind"]
        n = c["n_demos"]
        per_task[task] = {
            "segmentation_kind": kind,
            "meaning": KIND_NOTE[kind],
            "mean_frames_per_phase": {str(i): round(c[i] / max(n, 1), 1) for i in range(4)},
            "n_demos": n,
        }
    phase_map = {
        "phase_id": {str(k): v for k, v in oc_obs.PHASE_MAP.items()},
        "warning": ("phase names come from the pick-and-place convention; for "
                    "fixture_op and push tasks phase 2 is a single nominal "
                    "frame and phase 3 carries the operation -- read "
                    "per_task.meaning before using these labels"),
        "per_task": per_task,
    }
    with open(os.path.join(args.dir, "phase_map.json"), "w") as f:
        json.dump(phase_map, f, indent=2, ensure_ascii=False)

    # ---- scene_log.json ----
    with open(os.path.join(args.dir, "scene_log.json"), "w") as f:
        json.dump(records, f, indent=2)

    # ---- camera_params.json (dumped from a goal env + reprojection check) ----
    env = oc_obs.make_oc_env(os.path.join(
        get_libero_path("bddl_files"), "libero_goal", "put_the_bowl_on_the_plate.bddl"))
    obs = env.reset()
    H = W = 256
    K = get_camera_intrinsic_matrix(env.sim, "agentview", H, W)
    E = get_camera_extrinsic_matrix(env.sim, "agentview")
    P = get_camera_transform_matrix(env.sim, "agentview", H, W)
    Kh = get_camera_intrinsic_matrix(env.sim, "robot0_eye_in_hand", H, W)
    Eh = get_camera_extrinsic_matrix(env.sim, "robot0_eye_in_hand")
    Te = np.eye(4)
    Te[:3, :3] = TU.quat2mat(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    Te[:3, 3] = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    T_ee_cam = np.linalg.inv(Te) @ Eh

    # reprojection self-check: project each object's true position and confirm
    # the pixel lands on that object in the segmentation image
    from robosuite.utils.camera_utils import project_points_from_world_to_camera
    raw = obs["agentview_segmentation_instance"][..., 0]
    names = list(env.env.model.instances_to_ids.keys())
    check = {}
    for jn in G.GOAL_JOINTS:
        inst = jn.replace("_joint0", "")
        xyz = np.asarray(env.sim.data.get_joint_qpos(jn)[:3])
        rc = project_points_from_world_to_camera(xyz[None, :], P, H, W)[0]
        r, c = int(round(rc[0])), int(round(rc[1]))
        # MEASURED 2026-08-25: the projection row is in the UPRIGHT (top-left
        # origin) convention, so indexing this dataset's stored images -- which
        # keep the raw GL orientation, bottom row first -- needs H-1-r. Verified
        # against segmentation centroids: cheese proj row 154 vs centroid 100
        # (H-1-154=101), plate 171 vs 86 (H-1-171=84), columns matching exactly.
        gl_r = H - 1 - r
        v = int(raw[gl_r, c]) if (0 <= gl_r < H and 0 <= c < W) else 0
        hit = names[v - 1] if 0 < v <= len(names) else None
        check[inst] = {"world": xyz.round(4).tolist(),
                       "pixel_rowcol_upright": [r, c],
                       "pixel_rowcol_stored_gl": [gl_r, c],
                       "seg_hit": hit, "ok": hit == inst}
    env.close()

    cam = {
        "note": ("images in this dataset are stored in the RAW GL orientation "
                 "(bottom row first); flip vertically for viewing. "
                 "project_points_from_world_to_camera returns [row, col] in the "
                 "UPRIGHT convention: index a viewer-upright image with that row "
                 "directly, and the stored GL image with H-1-row. This was "
                 "measured against segmentation centroids, not assumed."),
        "image_size": [H, W],
        "agentview": {"intrinsic": K.tolist(), "extrinsic_world": E.tolist(),
                      "world_to_pixel": P.tolist()},
        "eye_in_hand": {"intrinsic": Kh.tolist(),
                        "T_ee_cam": T_ee_cam.tolist()},
        "reprojection_self_check": check,
        "self_check_passed": all(v["ok"] for v in check.values()),
    }
    with open(os.path.join(args.dir, "camera_params.json"), "w") as f:
        json.dump(cam, f, indent=2)

    print(f"wrote phase_map.json, scene_log.json ({len(records)} records), "
          f"camera_params.json")
    print(f"  camera reprojection self-check passed: {cam['self_check_passed']}")
    for k, v in check.items():
        print(f"    {k:22s} upright={v['pixel_rowcol_upright']} stored={v['pixel_rowcol_stored_gl']} hit={v['seg_hit']} ok={v['ok']}")
    print("  NOT written: metainfo.json (OC-format metadata; requires the "
          "deferred OC observation pass, plan §7)")


if __name__ == "__main__":
    main()
