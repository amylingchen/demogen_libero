"""Probe how far each goal fixture can move while its tasks still replay:
force the fixture to candidate-corridor CORNER positions (far beyond the ±12cm
plan default), sample objects around it, run 1-2 source demos per case, judge
by end-state predicate. The corridors these corners span are the proposed
WIDENED fixture zones needed so unseen layouts differ meaningfully in the
fixture dimension (6 of 9 tasks anchor on a fixture).

Usage:
    .venv\\Scripts\\python.exe scripts\\probe_goal_fixture_extremes.py --out-dir output/goal_fixture_probe
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

# proposed corridors (base-body xy); corners probed here, nominal for reference:
#   cabinet_main nominal (0.031,-0.236)  -> corridor x(-0.17,0.13) y(-0.38,-0.08)
#   stove_main   nominal (-0.405,0.201)  -> corridor x(-0.50,-0.15) y(0.08,0.33)
#                (burner = base + 0.15x, so base -0.15 puts the burner at x=0.0)
#   rack_main    nominal (-0.265,-0.268) -> corridor x(-0.42,-0.10) y(-0.38,-0.14)
CASES = [
    # (task, fixture_body, forced_xy)
    ("open_the_middle_drawer_of_the_cabinet", "wooden_cabinet_1_main", (0.13, -0.08)),
    ("open_the_middle_drawer_of_the_cabinet", "wooden_cabinet_1_main", (-0.17, -0.08)),
    ("open_the_middle_drawer_of_the_cabinet", "wooden_cabinet_1_main", (-0.17, -0.38)),
    ("open_the_middle_drawer_of_the_cabinet", "wooden_cabinet_1_main", (0.13, -0.38)),
    ("put_the_bowl_on_top_of_the_cabinet", "wooden_cabinet_1_main", (0.13, -0.08)),
    ("put_the_bowl_on_top_of_the_cabinet", "wooden_cabinet_1_main", (-0.17, -0.38)),
    ("turn_on_the_stove", "flat_stove_1_main", (-0.15, 0.08)),
    ("turn_on_the_stove", "flat_stove_1_main", (-0.15, 0.33)),
    ("turn_on_the_stove", "flat_stove_1_main", (-0.50, 0.08)),
    ("turn_on_the_stove", "flat_stove_1_main", (-0.50, 0.33)),
    ("put_the_bowl_on_the_stove", "flat_stove_1_main", (-0.15, 0.08)),
    ("put_the_bowl_on_the_stove", "flat_stove_1_main", (-0.50, 0.33)),
    ("put_the_wine_bottle_on_the_rack", "wine_rack_1_main", (-0.10, -0.14)),
    ("put_the_wine_bottle_on_the_rack", "wine_rack_1_main", (-0.10, -0.38)),
    ("put_the_wine_bottle_on_the_rack", "wine_rack_1_main", (-0.42, -0.14)),
    ("put_the_wine_bottle_on_the_rack", "wine_rack_1_main", (-0.42, -0.38)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-base-dir", default="D:/Data/LingLing/libero/hf/libero_goal")
    ap.add_argument("--out-dir", default=os.path.join("output", "goal_fixture_probe"))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--max-sources", type=int, default=2)
    ap.add_argument("--cases", nargs="+", default=None,
                    help='override cases as "task:fixture_body:x,y" strings')
    args = ap.parse_args()

    global CASES
    if args.cases:
        CASES = []
        for c in args.cases:
            task, fb, xy = c.split(":")
            x, y = map(float, xy.split(","))
            CASES.append((task, fb, (x, y)))

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    spec = G.GoalSpec()
    entities = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1",
                "wooden_cabinet_1", "flat_stove_1", "wine_rack_1"]

    by_task = {}
    for case in CASES:
        by_task.setdefault(case[0], []).append(case)

    results = []
    for task, cases in by_task.items():
        cfg = G.GOAL_TASKS[task]
        h5 = os.path.join(args.source_base_dir, f"{task}_demo.hdf5")
        keys = list_demo_keys(h5)
        env = oc_obs.make_oc_env(os.path.join(
            get_libero_path("bddl_files"), "libero_goal", f"{task}.bddl"))
        env.reset()
        fixture_ref = G.capture_fixture_ref(env)
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
        cam_xy = get_camera_extrinsic_matrix(env.sim, "agentview")[:2, 3].copy()

        for (_, fb, xy) in cases:
            # fixtures: forced one at xy, the others at nominal
            fixtures = {b: fixture_ref[b]["pos"][:2].copy() for b in G.GOAL_FIXTURES}
            fixtures[fb] = np.asarray(xy, dtype=np.float64)
            rec = {"task": task, "fixture": fb, "xy": list(xy), "attempts": []}
            objects = sample_objects_retry(rng, spec, fixtures, cam_xy)
            if objects is None:
                rec["placement"] = "FAILED (no object placement fits)"
                results.append(rec)
                print(f"[{task}] {fb}@{xy}: object placement FAILED", flush=True)
                continue
            layout = {"fixtures": {b: list(map(float, v)) for b, v in fixtures.items()},
                      "objects": {j: v.tolist() for j, v in objects.items()}}
            for src_key in list(rng.permutation(keys))[:args.max_sources]:
                demo = load_demo(h5, src_key)
                try:
                    frames = frames_for(cfg, demo, h5)
                    demo_layout = G.read_demo_layout(env, demo.init_state, fixture_ref)
                    obj_t, tar_t = G.anchor_deltas(cfg, layout, demo_layout)
                    ref, base_actions, new_frames = synthesize_uniform(
                        demo.state, demo.action, frames, obj_t, tar_t)
                    new_init, fx = G.apply_goal_layout(layout, demo.init_state, demo_layout)
                    R.reset_to_init_state(env, new_init)
                    S.apply_fixture_edits(env, fx)
                    latched, obs, rollout = R.replay_uniform(
                        env, base_actions, ref, new_frames, collect=True)
                    end = bool(env.check_success())
                except Exception as exc:
                    rec["attempts"].append({"source": src_key, "error": repr(exc)})
                    print(f"[{task}] {fb}@{xy} src={src_key} ERROR {exc!r}", flush=True)
                    continue
                rec["attempts"].append({"source": src_key, "success_end": end,
                                        "success_latched": bool(latched)})
                print(f"[{task}] {fb}@{xy} src={src_key} end={end}", flush=True)
                if end:
                    R.save_video(os.path.join(
                        args.out_dir, f"{task}__{xy[0]:+.2f}_{xy[1]:+.2f}.mp4"),
                        rollout["agentview_rgb"])
                    break
            rec["success"] = any(a.get("success_end") for a in rec["attempts"])
            results.append(rec)
            with open(os.path.join(args.out_dir, "probe_results.json"), "w") as f:
                json.dump(results, f, indent=2)
        env.close()

    print("\n== fixture-extreme probe ==")
    for r in results:
        tag = "OK  " if r.get("success") else "FAIL"
        print(f"  {tag} {r['task'][:40]:40s} {r['fixture'].split('_')[1]:8s} @ {r['xy']}"
              + ("  " + r.get("placement", "") if "placement" in r else ""))


def sample_objects_retry(rng, spec, fixtures, cam_xy, tries=40):
    for _ in range(tries):
        objects = G.sample_objects(rng, spec, fixtures, cam_xy)
        if objects is not None:
            return objects
    return None


if __name__ == "__main__":
    main()
