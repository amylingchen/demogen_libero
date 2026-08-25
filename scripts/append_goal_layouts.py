"""Append targeted extra layouts to an existing goal-layout manifest: force
one fixture into an under-covered corridor region (the first 12 layouts left
the cabinet front band y(-0.22,-0.10) and the rack front band nearly empty,
capping the task-aware unseen floor at 0.049m), run the SAME gates
(geometry -> settle -> visibility -> dedup -> 6-task replay), and append the
passers to manifest["layouts"]. Re-run optimize_goal_split.py afterwards.

Usage:
    .venv\\Scripts\\python.exe scripts\\append_goal_layouts.py --dir output/goal_layouts_12 \\
        --zone wooden_cabinet_1_main:-0.17,-0.02,-0.20,-0.10 --n 2 --seed 31
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
from sample_goal_layouts import GATE_TASKS, ENTITIES, entity_dists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("output", "goal_layouts_12"))
    ap.add_argument("--zone", action="append", required=True,
                    help="fixture zone override 'body:x0,x1,y0,y1' (repeatable; "
                         "each --zone batch samples --n layouts)")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--gate-sources", type=int, default=2)
    ap.add_argument("--source-base-dir", default="D:/Data/LingLing/libero/hf/libero_goal")
    args = ap.parse_args()

    man_path = os.path.join(args.dir, "manifest.json")
    man = json.load(open(man_path))
    rng = np.random.default_rng(args.seed)
    spec = G.GoalSpec()

    demo = load_demo(os.path.join(args.source_base_dir,
                                  "open_the_middle_drawer_of_the_cabinet_demo.hdf5"), "demo_0")
    env = oc_obs.make_oc_env(demo.bddl_file)
    env.reset()
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    cam_xy = get_camera_extrinsic_matrix(env.sim, "agentview")[:2, 3].copy()
    ref_layout = S.read_layout(env, demo.init_state, G.GOAL_JOINTS, G.GOAL_FIXTURES)

    gate_envs = {}
    for task in GATE_TASKS:
        genv = oc_obs.make_oc_env(os.path.join(
            get_libero_path("bddl_files"), "libero_goal", f"{task}.bddl"))
        genv.reset()
        gate_envs[task] = (genv, G.capture_fixture_ref(genv),
                          os.path.join(args.source_base_dir, f"{task}_demo.hdf5"))

    def full_gates(layout):
        new_init, fx = G.apply_goal_layout(layout, demo.init_state, ref_layout)
        R.reset_to_init_state(env, new_init)
        S.apply_fixture_edits(env, fx)
        rep = S.settle(env, spec.settle_steps)
        obs = env.env._get_observations(force_update=True)
        raw = obs["agentview_segmentation_instance"][..., 0]
        names = list(env.env.model.instances_to_ids.keys())
        px = {nm: int((raw == names.index(nm) + 1).sum()) for nm in ENTITIES}
        return (rep["converged"] and all(v >= spec.min_px for v in px.values()),
                {"settle": rep, "px": px})

    def replay_gate(layout):
        report = {}
        for task in GATE_TASKS:
            cfg = G.GOAL_TASKS[task]
            genv, fixture_ref, h5 = gate_envs[task]
            keys = list(rng.permutation(list_demo_keys(h5)))[:args.gate_sources]
            ok = False
            tried = []
            for src_key in keys:
                d = load_demo(h5, src_key)
                try:
                    frames = frames_for(cfg, d, h5)
                    demo_layout = G.read_demo_layout(genv, d.init_state, fixture_ref)
                    obj_t, tar_t = G.anchor_deltas(cfg, layout, demo_layout)
                    ref, base_actions, new_frames = synthesize_uniform(
                        d.state, d.action, frames, obj_t, tar_t)
                    new_init, fx = G.apply_goal_layout(layout, d.init_state, demo_layout)
                    R.reset_to_init_state(genv, new_init)
                    S.apply_fixture_edits(genv, fx)
                    R.replay_uniform(genv, base_actions, ref, new_frames, collect=False)
                    end = bool(genv.check_success())
                except Exception as exc:
                    tried.append({"source": src_key, "error": repr(exc)})
                    continue
                tried.append({"source": src_key, "success_end": end})
                if end:
                    ok = True
                    break
            report[task] = tried
            if not ok:
                return False, report
        return True, report

    appended = 0
    for zone_spec in args.zone:
        body, nums = zone_spec.split(":")
        x0, x1, y0, y1 = map(float, nums.split(","))
        sub = G.GoalSpec()
        sub.fixture_corridor = dict(spec.fixture_corridor)
        sub.fixture_corridor[body] = ((x0, x1), (y0, y1))
        got = 0
        for tried in range(args.n * 40):
            if got >= args.n:
                break
            try:
                cand = G.sample_goal_layout(rng, sub, ref_layout, cam_xy)
            except RuntimeError:
                continue
            if G._corridor_cut(body, cand["fixtures"][body]):
                continue
            if any(all(v < 0.05 for v in entity_dists(cand, acc["layout"]).values())
                   for acc in man["layouts"]):
                continue
            ok, rep = full_gates(cand)
            if not ok:
                continue
            ok, gate_rep = replay_gate(cand)
            print(f"[zone {body} try {tried}] replay gate "
                  f"{'PASS' if ok else 'FAIL'} "
                  f"{ {t.split('_')[-1]: any(a.get('success_end') for a in v) for t, v in gate_rep.items()} }",
                  flush=True)
            if not ok:
                continue
            lid = f"L{len(man['layouts']):02d}"
            man["layouts"].append({"id": lid, "layout": cand, "gates": rep,
                                   "replay_gate": gate_rep,
                                   "zone_override": zone_spec})
            got += 1
            appended += 1
            with open(man_path, "w") as f:
                json.dump(man, f, indent=2)
    env.close()
    for genv, *_ in gate_envs.values():
        genv.close()
    print(f"appended {appended} layouts -> {len(man['layouts'])} total; "
          f"re-run optimize_goal_split.py")


if __name__ == "__main__":
    main()
