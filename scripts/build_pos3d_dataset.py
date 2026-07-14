"""Build a compact object-3D-position dataset from existing OC datasets.

For every episode in each source dataset, subsample <=N evenly spaced frames and
attach ground-truth object 3D world positions (from the sim state's free-joint
qpos) plus camera intrinsics/extrinsics, so a model can be trained to regress
object 3D positions from the (exocentric) image. Both source datasets are merged
into one output HDF5.

Per-frame additions:
    obs/object_positions   (7, 3)  world xyz of each object (OBJECT_ORDER)
    obs/object_present     (7,)    bool, False for parked (off-table) distractors
    obs/eye_in_hand_extrinsic (4,4) wrist-cam pose (moves per frame)
Top-level attrs: object_names, object_seg_ids, agentview_intrinsic/extrinsic,
    eye_in_hand_intrinsic.

Usage:
    .venv\Scripts\python.exe scripts\build_pos3d_dataset.py --max-steps 10
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np
from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_camera_transform_matrix,
)

from demogen_libero.convert import load_demo
from demogen_libero import oc_obs
from demogen_libero.gridscene import object_state_slices

from demogen_libero.config import source_hdf5
HDF5_PATH = source_hdf5("pick_up_the_salad_dressing_and_place_it_in_the_basket")
TASK_KEY = "pick_up_the_salad_dressing_and_place_it_in_the_basket"
OBJ_JOINTS = ["salad_dressing_1_joint0", "basket_1_joint0", "ketchup_1_joint0",
              "alphabet_soup_1_joint0", "cream_cheese_1_joint0", "milk_1_joint0",
              "tomato_sauce_1_joint0"]
OBJ_NAMES = [oc_obs.display_name(j.replace("_joint0", "")) for j in OBJ_JOINTS]
SEG_IDS = [60 + 10 * i for i in range(len(OBJ_JOINTS))]
PARKED_X = 1.0  # objects at x >= this are parked off-table

# passthrough obs keys copied (subsampled) from the source demos
COPY_OBS = list(oc_obs.OBS_KEYS) + list(oc_obs.EXTRA_OBS_KEYS)


def subsample_indices(T, n):
    if T <= n:
        return np.arange(T)
    return np.unique(np.linspace(0, T - 1, n).round().astype(int))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+",
                        default=["output/grid_oc_salad_v2", "output/grid_oc_regrasp"])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default=os.path.join("output", "salad_pos3d"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_hdf5 = os.path.join(args.out_dir, f"{TASK_KEY}_pos3d.hdf5")
    if os.path.exists(out_hdf5):
        os.remove(out_hdf5)

    # env only for camera params + object qpos slices (states are self-contained)
    demo = load_demo(HDF5_PATH, "demo_1")
    env = oc_obs.make_oc_env(demo.bddl_file)
    env.reset()
    slices = object_state_slices(env, OBJ_JOINTS)
    K_exo = get_camera_intrinsic_matrix(env.sim, "agentview", 256, 256)
    E_exo = get_camera_extrinsic_matrix(env.sim, "agentview")   # world-fixed
    M_exo = get_camera_transform_matrix(env.sim, "agentview", 256, 256)  # world->pixel, fixed
    K_ego = get_camera_intrinsic_matrix(env.sim, "robot0_eye_in_hand", 256, 256)

    def object_positions(state):
        pos = np.array([state[slices[j]:slices[j] + 3] for j in OBJ_JOINTS], dtype=np.float64)
        present = pos[:, 0] < PARKED_X
        return pos, present

    out = h5py.File(out_hdf5, "w")
    data = out.create_group("data")
    idx = 0
    per_source = {}
    try:
        for src_dir in args.sources:
            src_path = glob.glob(os.path.join(src_dir, "*_demo.hdf5"))[0]
            with h5py.File(src_path, "r") as src:
                names = sorted(src["data"].keys(), key=lambda k: int(k.split("_")[-1]))
                for name in names:
                    g = src["data"][name]
                    T = int(g.attrs["num_samples"])
                    sel = subsample_indices(T, args.max_steps)
                    states = np.array(g["states"])[sel]

                    obj_pos = np.zeros((len(sel), len(OBJ_JOINTS), 3))
                    obj_present = np.zeros((len(sel), len(OBJ_JOINTS)), dtype=bool)
                    ego_ext = np.zeros((len(sel), 4, 4))
                    ego_w2p = np.zeros((len(sel), 4, 4))
                    for i, s in enumerate(states):
                        obj_pos[i], obj_present[i] = object_positions(s)
                        env.set_init_state(s)
                        ego_ext[i] = get_camera_extrinsic_matrix(env.sim, "robot0_eye_in_hand")
                        ego_w2p[i] = get_camera_transform_matrix(env.sim, "robot0_eye_in_hand", 256, 256)

                    dg = data.create_group(f"demo_{idx}")
                    dg.attrs["num_samples"] = len(sel)
                    dg.attrs["source_dataset"] = os.path.basename(src_dir)
                    dg.attrs["source_demo"] = name
                    dg.attrs["orig_num_samples"] = T
                    dg.create_dataset("frame_indices", data=sel.astype(np.int32))
                    og = dg.create_group("obs")
                    for k in COPY_OBS:
                        if k in g["obs"]:
                            og.create_dataset(k, data=np.array(g["obs"][k])[sel],
                                              compression="gzip", compression_opts=4)
                    og.create_dataset("object_positions", data=obj_pos)
                    og.create_dataset("object_present", data=obj_present)
                    og.create_dataset("eye_in_hand_extrinsic", data=ego_ext)
                    og.create_dataset("eye_in_hand_world_to_pixel", data=ego_w2p)
                    for k in ("actions", "phase_id", "states"):
                        if k in g:
                            dg.create_dataset(k, data=np.array(g[k])[sel])
                    idx += 1
                    per_source[os.path.basename(src_dir)] = per_source.get(os.path.basename(src_dir), 0) + 1

        data.attrs["num_demos"] = idx
        data.attrs["total"] = int(sum(data[k].attrs["num_samples"] for k in data.keys()))
        data.attrs["object_names"] = json.dumps(OBJ_NAMES)
        data.attrs["object_seg_ids"] = np.array(SEG_IDS, dtype=np.int32)
        data.attrs["agentview_intrinsic"] = K_exo
        data.attrs["agentview_extrinsic"] = E_exo
        data.attrs["agentview_world_to_pixel"] = M_exo  # world-fixed; pixel=[row,col]
        data.attrs["eye_in_hand_intrinsic"] = K_ego
        data.attrs["parked_x_threshold"] = PARKED_X
        # stored images are vertically flipped (upright); apply row -> H-1-row after
        # projecting with world_to_pixel. project_points_from_world_to_camera returns [row, col].
        data.attrs["images_vertically_flipped"] = True
        data.attrs["image_hw"] = np.array([256, 256], dtype=np.int32)
    finally:
        out.close()
        env.close()

    # subsampled metainfo: pull per-frame boxes from each source metainfo at the
    # kept frame indices, re-keyed to the merged demo names
    src_metas = {}
    for src_dir in args.sources:
        mp = os.path.join(src_dir, "metainfo.json")
        if os.path.exists(mp):
            src_metas[os.path.basename(src_dir)] = json.load(open(mp, encoding="utf-8"))
    out_meta = {TASK_KEY: {}}
    with h5py.File(out_hdf5, "r") as f:
        for dk in f["data"].keys():
            g = f["data"][dk]
            sd = g.attrs["source_dataset"]
            src_demo = g.attrs["source_demo"]
            sel = list(np.array(g["frame_indices"]))
            src_entry = src_metas.get(sd, {}).get(TASK_KEY, {}).get(src_demo)
            if src_entry is None:
                continue
            out_meta[TASK_KEY][dk] = {
                "success": True,
                "initial_state": src_entry["initial_state"],
                "task_nouns": src_entry["task_nouns"],
                "task_description": src_entry["task_description"],
                "source_dataset": sd, "source_demo": src_demo,
                "frame_indices": [int(i) for i in sel],
                "exo_boxes": [src_entry["exo_boxes"][i] for i in sel],
                "ego_boxes": [src_entry["ego_boxes"][i] for i in sel],
            }
    meta_path = os.path.join(args.out_dir, f"{TASK_KEY}_pos3d_metainfo.json")
    json.dump(out_meta, open(meta_path, "w", encoding="utf-8"), indent=2)
    print(f"wrote {out_hdf5}")
    print(f"wrote {meta_path}  ({len(out_meta[TASK_KEY])} entries)")
    print(f"  demos: {idx}  per source: {per_source}")
    with h5py.File(out_hdf5, "r") as f:
        Ts = [int(f["data"][k].attrs["num_samples"]) for k in f["data"].keys()]
        print(f"  frames/demo: min {min(Ts)} max {max(Ts)}  total {f['data'].attrs['total']}")
        print(f"  file size: {os.path.getsize(out_hdf5)/1e6:.0f} MB")
        print(f"  object order: {OBJ_NAMES}")


if __name__ == "__main__":
    main()
