"""Round-2 follow-ups on the goal suite (user decisions 2026-08-23):

(a) B7: replay-TEST the 8 cells labeled push-infeasible (the label was
    asserted from the r=0.03 disk rule, never measured). 2 healthy sources
    per cell, end-state predicate.
(b) B6: part-level visibility audit of all 12 layouts: from the agentview
    camera, cast rays to a 5-point cross around each task-relevant part
    (middle-drawer handle, stove knob, burner, rack slot, cabinet top) and
    record what the ray hits first. A part is occluded if most rays hit a
    DIFFERENT entity first.

Results are patched into manifest.json (push_cell_test, part_visibility).

Usage:
    .venv\\Scripts\\python.exe scripts\\audit_goal_suite.py --dir output/goal_suite_12
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from demogen_libero.convert import load_demo, list_demo_keys
from demogen_libero.trajectory import synthesize_uniform
from demogen_libero import libero_replay as R
from demogen_libero import oc_obs
from demogen_libero import spatial_scene as S
from demogen_libero import goal_scene as G

from libero.libero import get_libero_path
from smoke_goal_traj import frames_for

PUSH = "push_the_plate_to_the_front_of_the_stove"

# part points: (label, anchor fixture, offset xy, z)
PART_POINTS = [
    ("drawer_handle", "wooden_cabinet_1_main", (0.0, 0.10), 1.03),
    ("stove_knob", "flat_stove_1_main", (0.0, 0.0), 0.95),
    ("burner", "flat_stove_1_main", (0.15, 0.0), 0.93),
    ("rack_slot", "wine_rack_1_main", (0.083, 0.0), 1.14),
    ("cabinet_top", "wooden_cabinet_1_main", (0.0, 0.0), 1.13),
]
CROSS = [(0, 0), (0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)]


def ray_first_body(env, cam_pos, target):
    """Body name first hit by the camera->target ray (mujoco mj_ray)."""
    import mujoco
    vec = np.asarray(target) - np.asarray(cam_pos)
    dist = float(np.linalg.norm(vec))
    geomid = np.array([-1], dtype=np.int32)
    m = env.sim.model._model
    d = env.sim.data._data
    t = mujoco.mj_ray(m, d, np.asarray(cam_pos), vec / dist,
                      None, 1, -1, geomid)
    if geomid[0] < 0 or t >= dist - 0.015:
        return None, dist   # nothing (or nothing closer than the point itself)
    body = m.geom(geomid[0]).bodyid
    return env.sim.model.body_id2name(int(body)), float(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("output", "goal_suite_12"))
    ap.add_argument("--source-base-dir", default="D:/Data/LingLing/libero/hf/libero_goal")
    ap.add_argument("--seed", type=int, default=61)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    man_path = os.path.join(args.dir, "manifest.json")
    man = json.load(open(man_path))
    by_id = {l["id"]: l for l in man["layouts"]}
    screening = json.load(open("output/goal_source_screening.json"))

    # ---- (a) push replay on the 8 asserted-infeasible cells ----
    cfg = G.GOAL_TASKS[PUSH]
    h5 = os.path.join(args.source_base_dir, f"{PUSH}_demo.hdf5")
    env = oc_obs.make_oc_env(os.path.join(
        get_libero_path("bddl_files"), "libero_goal", f"{PUSH}.bddl"))
    env.reset()
    fixture_ref = G.capture_fixture_ref(env)
    push_test = {}
    for lid in man["push_infeasible_cells"]:
        layout = by_id[lid]["layout"]
        pool = screening[PUSH]["healthy"]
        tried = []
        for src_key in list(rng.permutation(pool))[:2]:
            d = load_demo(h5, src_key)
            try:
                frames = frames_for(cfg, d, h5)
                demo_layout = G.read_demo_layout(env, d.init_state, fixture_ref)
                obj_t, tar_t = G.anchor_deltas(cfg, layout, demo_layout)
                ref, base_actions, new_frames = synthesize_uniform(
                    d.state, d.action, frames, obj_t, tar_t)
                new_init, fx = G.apply_goal_layout(layout, d.init_state, demo_layout)
                R.reset_to_init_state(env, new_init)
                S.apply_fixture_edits(env, fx)
                R.replay_uniform(env, base_actions, ref, new_frames, collect=False)
                end = bool(env.check_success())
            except Exception as exc:
                tried.append({"source": src_key, "error": repr(exc)})
                continue
            tried.append({"source": src_key, "success_end": end})
            if end:
                break
        push_test[lid] = {"attempts": tried,
                          "any_success": any(a.get("success_end") for a in tried)}
        print(f"[push test] {lid}: {push_test[lid]['any_success']} {tried}", flush=True)
    env.close()

    # ---- (b) part-level ray visibility on all 12 layouts ----
    demo = load_demo(os.path.join(args.source_base_dir,
                                  "open_the_middle_drawer_of_the_cabinet_demo.hdf5"),
                     "demo_0")
    env = oc_obs.make_oc_env(demo.bddl_file)
    env.reset()
    ref_layout = S.read_layout(env, demo.init_state, G.GOAL_JOINTS, G.GOAL_FIXTURES)
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    cam_pos = get_camera_extrinsic_matrix(env.sim, "agentview")[:3, 3].copy()
    OWN = {"drawer_handle": "wooden_cabinet", "stove_knob": "flat_stove",
           "burner": "flat_stove", "rack_slot": "wine_rack",
           "cabinet_top": "wooden_cabinet"}
    part_vis = {}
    for rec in man["layouts"]:
        layout = rec["layout"]
        new_init, fx = G.apply_goal_layout(layout, demo.init_state, ref_layout)
        R.reset_to_init_state(env, new_init)
        S.apply_fixture_edits(env, fx)
        S.settle(env, 300)
        row = {}
        for label, fb, off, z in PART_POINTS:
            base = np.asarray(layout["fixtures"][fb])
            hits = []
            for dx, dy in CROSS:
                pt = np.array([base[0] + off[0] + dx, base[1] + off[1] + dy, z])
                body, _ = ray_first_body(env, cam_pos, pt)
                own = body is None or (body and OWN[label] in body)
                hits.append((body, own))
            n_clear = sum(own for _, own in hits)
            blockers = sorted({b for b, own in hits if not own and b})
            row[label] = {"clear": f"{n_clear}/5", "blockers": blockers}
        part_vis[rec["id"]] = row
        worst = min(int(v["clear"][0]) for v in row.values())
        print(f"[vis] {rec['id']}: worst part {worst}/5 " +
              "; ".join(f"{k}:{v['clear']}" + (f"({','.join(v['blockers'])})" if v["blockers"] else "")
                        for k, v in row.items()), flush=True)
    env.close()

    man["push_cell_test"] = {"seed": args.seed, "results": push_test,
                             "note": "replay test of the cells previously labeled "
                                     "infeasible by the disk rule alone (review B7)"}
    man["part_visibility_audit"] = {
        "method": "mujoco mj_ray from agentview camera to a 5-point cross "
                  "around each part point; a part counts occluded when rays "
                  "first hit a different entity (review B6; plan gate 2)",
        "results": part_vis}
    with open(man_path, "w") as f:
        json.dump(man, f, indent=2)
    print("manifest patched with push_cell_test + part_visibility_audit")


if __name__ == "__main__":
    main()
