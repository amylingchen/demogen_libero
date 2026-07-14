"""Dump the exact camera parameters used by the grid_oc_salad_150 dataset:
agentview intrinsics/extrinsics and the eye_in_hand hand-eye transform
(T_ee_cam), in OpenCV convention (x right, y down, z forward), for the
UPRIGHT (vertically flipped) images as stored in the OC hdf5.

Run with the demogen venv:
    .venv\\Scripts\\python.exe scripts\\dump_camera_params.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import (get_camera_extrinsic_matrix,
                                          get_camera_intrinsic_matrix,
                                          get_camera_transform_matrix)

from demogen_libero.convert import load_demo
from demogen_libero import oc_obs

from demogen_libero.config import DATA_DIR as SOURCE_BASE_DIR


def ee_pose_T(obs):
    Te = np.eye(4)
    Te[:3, :3] = T.quat2mat(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    Te[:3, 3] = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    return Te


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="salad_dressing")
    parser.add_argument("--source-base-dir", type=str, default=SOURCE_BASE_DIR)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--out", type=str, default="output/camera_params.json")
    args = parser.parse_args()
    H = W = args.height
    W = args.width
    tk = args.task if args.task.startswith("pick_up") else f"pick_up_the_{args.task}_and_place_it_in_the_basket"

    demo = load_demo(os.path.join(args.source_base_dir, f"{tk}_demo.hdf5"), "demo_1")
    env = oc_obs.make_oc_env(demo.bddl_file, height=H, width=W)
    obs = env.reset()
    sim = env.sim

    K_agent = get_camera_intrinsic_matrix(sim, "agentview", H, W)
    T_agent = get_camera_extrinsic_matrix(sim, "agentview")
    K_hand = get_camera_intrinsic_matrix(sim, "robot0_eye_in_hand", H, W)

    # hand-eye transform, checked for constancy across two different ee poses
    T_ec_list = []
    for _ in range(2):
        T_wc = get_camera_extrinsic_matrix(sim, "robot0_eye_in_hand")
        T_we = ee_pose_T(obs)
        T_ec_list.append(np.linalg.inv(T_we) @ T_wc)
        obs, *_ = env.step([0.3, 0.2, -0.3, 0.1, 0.1, 0.1, -1.0])
        for _ in range(4):
            obs, *_ = env.step([0.3, 0.2, -0.3, 0.1, 0.1, 0.1, -1.0])
    dev = np.abs(T_ec_list[0] - T_ec_list[1]).max()

    # mujoco fovy per camera (for reference)
    fovy = {name: float(sim.model.cam_fovy[sim.model.camera_name2id(name)])
            for name in ("agentview", "robot0_eye_in_hand")}

    M_agent = get_camera_transform_matrix(sim, "agentview", H, W)  # world->pixel
    out = {
        "note": ("OpenCV convention from robosuite camera_utils; images stored "
                 "vertically flipped (upright): after projecting with world_to_pixel "
                 "(-> [row,col]) apply row -> H-1-row"),
        "image_hw": [H, W],
        "images_vertically_flipped": True,
        "fovy_deg": fovy,
        "agentview": {"K": K_agent.tolist(), "T_world_cam": T_agent.tolist(),
                      "world_to_pixel": M_agent.tolist(), "fixed": True},
        "eye_in_hand": {"K": K_hand.tolist(),
                        "T_ee_cam": T_ec_list[0].tolist(),
                        "hand_eye_consistency_dev": float(dev),
                        "note": "world cam pose = T_world_ee @ T_ee_cam (per frame)"},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}  hand-eye dev={dev:.2e}")
    env.close()


if __name__ == "__main__":
    main()
