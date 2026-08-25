"""Measure, from the libero_goal source demos, the EE position at each task's
goal event (drawer start/end of travel, knob turn, each place-release), plus
the middle-drawer body geometry from the compiled model. This is the evidence
file behind goal_scene.py's goal-point offsets and the drawer-sweep keep-out
(adversarial review 2026-08-23 flagged those numbers as having no producing
script). Writes output/goal_geometry/goal_event_ee.json.

Usage:
    .venv\\Scripts\\python.exe scripts\\measure_goal_event_ee.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np

BASE = "D:/Data/LingLing/libero/hf/libero_goal/"
COL_MID, COL_KNOB = 39, 41
FIXTURE_NOMINAL = {"cabinet": (0.031, -0.236), "stove": (-0.405, 0.201),
                   "rack": (-0.265, -0.268)}


def ee_at(task, cond, n=8):
    out = []
    with h5py.File(BASE + task + "_demo.hdf5", "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:n]
        for k in keys:
            st = np.array(f["data"][k]["states"])
            ee = np.array(f["data"][k]["obs"]["ee_pos"])
            act = np.array(f["data"][k]["actions"])
            idx = cond(st, act)
            if idx is not None:
                out.append(ee[idx])
    return np.array(out)


def drawer_start(st, act):
    m = np.where(np.abs(st[:, COL_MID] - st[0, COL_MID]) > 0.003)[0]
    return int(m[0]) if m.size else None


def drawer_end(st, act):
    m = np.where(np.abs(st[:, COL_MID] - st[0, COL_MID]) > 0.003)[0]
    return int(m[-1]) if m.size else None


def knob_start(st, act):
    m = np.where(np.abs(st[:, COL_KNOB] - st[0, COL_KNOB]) > 0.05)[0]
    return int(m[0]) if m.size else None


def release(st, act):
    grip = act[:, 6]
    m = np.where((grip[:-1] > 0) & (grip[1:] <= 0))[0]
    return int(m[-1]) if m.size else None


def push_final_plate(n=8):
    """Final plate xy of push source demos + the initial plate xy (for the
    per-layout feasibility prediction: final = source_final + plate_delta)."""
    out = []
    with h5py.File(BASE + "push_the_plate_to_the_front_of_the_stove_demo.hdf5", "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:n]
        for k in keys:
            st = np.array(f["data"][k]["states"])
            out.append({"demo": k, "plate_init": st[0, 31:33].tolist(),
                        "plate_final": st[-1, 31:33].tolist()})
    return out


def main():
    os.makedirs("output/goal_geometry", exist_ok=True)
    result = {"n_demos_each": 8, "fixture_nominal": FIXTURE_NOMINAL,
              "note": "offsets = EE mean minus fixture nominal; ee frame = world"}
    for name, task, conds in [
        ("drawer", "open_the_middle_drawer_of_the_cabinet",
         [("start", drawer_start), ("end", drawer_end)]),
        ("knob", "turn_on_the_stove", [("start", knob_start)]),
        ("bowl_on_stove", "put_the_bowl_on_the_stove", [("release", release)]),
        ("bowl_on_cabtop", "put_the_bowl_on_top_of_the_cabinet", [("release", release)]),
        ("bottle_on_rack", "put_the_wine_bottle_on_the_rack", [("release", release)]),
        ("bottle_on_cabtop", "put_the_wine_bottle_on_top_of_the_cabinet",
         [("release", release)]),
    ]:
        result[name] = {}
        for lbl, cond in conds:
            pts = ee_at(task, cond)
            result[name][lbl] = {"ee_mean": np.round(pts.mean(0), 4).tolist(),
                                 "ee_std": np.round(pts.std(0), 4).tolist(),
                                 "n": len(pts)}
    result["push_source_finals"] = push_final_plate()

    # middle-drawer geometry from the compiled model (sweep rectangle evidence)
    from libero.libero import get_libero_path
    from demogen_libero import oc_obs
    env = oc_obs.make_oc_env(os.path.join(
        get_libero_path("bddl_files"), "libero_goal",
        "open_the_middle_drawer_of_the_cabinet.bddl"))
    env.reset()
    m = env.sim.model
    drawer_geoms = []
    for g in range(m.ngeom):
        n = m.geom_id2name(g)
        if n and "cabinet_middle" in n:
            drawer_geoms.append({"geom": n,
                                 "xpos": np.round(env.sim.data.geom_xpos[g], 4).tolist(),
                                 "size": np.round(m.geom_size[g], 4).tolist()})
    jid = m.joint_name2id("wooden_cabinet_1_middle_level")
    result["middle_drawer"] = {
        "geoms": drawer_geoms,
        "joint_axis_local": np.round(m.jnt_axis[jid], 3).tolist(),
        "joint_range": np.round(m.jnt_range[jid], 3).tolist(),
    }
    env.close()

    with open("output/goal_geometry/goal_event_ee.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["middle_drawer"], indent=1))
    print("wrote output/goal_geometry/goal_event_ee.json")


if __name__ == "__main__":
    main()
