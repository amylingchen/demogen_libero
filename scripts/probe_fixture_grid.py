"""Grid-probe the feasible region of each fixture BY REPLAY (2026-08-24).

Motivation: the corridors in GoalSpec rest on 4 corner probes per fixture, and
v3 sampling found a band INSIDE the cabinet corridor where the drawer task
failed 22/22 -- "inside the corridor" is not evidence of feasibility. This
sweeps a grid over each fixture corridor PLUS a proposed enlargement, replays
that fixture's tasks at every grid point, and writes a feasibility map, so the
corridor can be defined from measurement instead of from four corners.

Objects are re-sampled around the probed fixture (nominal fixtures elsewhere),
so the reading isolates the fixture position.

Usage:
    .venv\\Scripts\\python.exe scripts\\probe_fixture_grid.py --out output/goal_fixture_grid
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from demogen_libero.convert import load_demo
from demogen_libero.trajectory import synthesize_uniform
from demogen_libero import libero_replay as R
from demogen_libero import oc_obs
from demogen_libero import spatial_scene as S
from demogen_libero import goal_scene as G

from libero.libero import get_libero_path
from smoke_goal_traj import frames_for
from run_goal_generation import (joint_margin, JOINT_MARGIN_FLOOR,
                                 JOINT_GATE_EXEMPT, pose_ok)

# fixture -> (tasks reaching it, x range, y range, grid step)
SWEEP = {
    "wooden_cabinet_1_main": (
        ["open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_top_of_the_cabinet"],
        (-0.24, 0.13), (-0.40, -0.08), 0.055),
    "flat_stove_1_main": (
        ["turn_on_the_stove", "put_the_bowl_on_the_stove"],
        (-0.52, -0.13), (0.06, 0.38), 0.055),
    "wine_rack_1_main": (
        ["put_the_wine_bottle_on_the_rack"],
        (-0.46, -0.08), (-0.40, -0.12), 0.055),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("output", "goal_fixture_grid"))
    ap.add_argument("--source-base-dir", default="D:/Data/LingLing/libero/hf/libero_goal")
    ap.add_argument("--screening", default=os.path.join("output", "goal_source_screening.json"))
    ap.add_argument("--sources", type=int, default=2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--fixtures", nargs="+", default=list(SWEEP))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    screening = json.load(open(args.screening))
    spec = G.GoalSpec()

    probe = load_demo(os.path.join(args.source_base_dir,
                                   "open_the_middle_drawer_of_the_cabinet_demo.hdf5"),
                      "demo_0")
    env0 = oc_obs.make_oc_env(probe.bddl_file)
    env0.reset()
    ref = S.read_layout(env0, probe.init_state, G.GOAL_JOINTS, G.GOAL_FIXTURES)
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    cam_xy = get_camera_extrinsic_matrix(env0.sim, "agentview")[:2, 3].copy()
    env0.close()
    nominal_fix = {fb: ref["fixtures"][fb]["pos"][:2].tolist() for fb in G.GOAL_FIXTURES}

    results = {}
    out_path = os.path.join(args.out, "fixture_grid.json")
    for fb in args.fixtures:
        tasks, xr, yr, step = SWEEP[fb]
        xs = np.round(np.arange(xr[0], xr[1] + 1e-9, step), 3)
        ys = np.round(np.arange(yr[0], yr[1] + 1e-9, step), 3)
        envs = {}
        for t in tasks:
            e = oc_obs.make_oc_env(os.path.join(
                get_libero_path("bddl_files"), "libero_goal", f"{t}.bddl"))
            e.reset()
            envs[t] = (e, G.capture_fixture_ref(e),
                       os.path.join(args.source_base_dir, f"{t}_demo.hdf5"))
        grid = {}
        for x in xs:
            for y in ys:
                key = "%+.3f,%+.3f" % (x, y)
                reach_bad = bool(G.fixture_reach_violations(fb, [x, y]))
                # The OTHER two fixtures must be re-sampled subject to their own
                # corridors, reach limits and the fixture-fixture clearance. An
                # earlier version pinned them at nominal, which forced the probed
                # cabinet INTO the wine rack for every x <= -0.02 and produced a
                # spurious 0/12 "infeasible" verdict there (2026-08-24).
                fixtures = {fb: np.array([float(x), float(y)])}
                for other in G.GOAL_FIXTURES:
                    if other == fb:
                        continue
                    (ox, oy) = spec.fixture_corridor[other]
                    for _ in range(500):
                        c = np.array([rng.uniform(*ox), rng.uniform(*oy)])
                        if G._corridor_cut(other, c) or G.fixture_reach_violations(other, c):
                            continue
                        clash = False
                        for have, hxy in fixtures.items():
                            for c1, r1 in G._circles(other, c):
                                for c2, r2 in G._circles(have, hxy):
                                    if np.linalg.norm(c1 - c2) < r1 + r2 + spec.fixture_clearance:
                                        clash = True
                        if not clash:
                            fixtures[other] = c
                            break
                if len(fixtures) < 3:
                    grid[key] = {"status": "no_fixture_config", "reach_bad": reach_bad}
                    print("[%s] (%+.3f,%+.3f) NO-FIXTURE-CONFIG" %
                          (fb.replace("_1_main", ""), x, y), flush=True)
                    results[fb] = {"xs": xs.tolist(), "ys": ys.tolist(), "grid": grid}
                    with open(out_path, "w") as f:
                        json.dump(results, f, indent=2)
                    continue
                fixtures = {k: (v.tolist() if hasattr(v, "tolist") else list(v))
                            for k, v in fixtures.items()}
                objs = G.sample_objects(
                    rng, spec, {k: np.asarray(v) for k, v in fixtures.items()}, cam_xy)
                if objs is None:
                    grid[key] = {"status": "no_object_placement", "reach_bad": reach_bad}
                    print("[%s] (%+.3f,%+.3f) NO-OBJ-PLACEMENT" %
                          (fb.replace("_1_main", ""), x, y), flush=True)
                    results[fb] = {"xs": xs.tolist(), "ys": ys.tolist(), "grid": grid}
                    with open(out_path, "w") as f:
                        json.dump(results, f, indent=2)
                    continue
                layout = {"fixtures": {k: [float(v[0]), float(v[1])]
                                       for k, v in fixtures.items()},
                          "objects": {k: v.tolist() for k, v in objs.items()}}
                per = {}
                for t in tasks:
                    e, fref, h5 = envs[t]
                    pool = screening[t]["healthy"]
                    ok = False
                    detail = []
                    for src in list(rng.permutation(pool))[:args.sources]:
                        d = load_demo(h5, src)
                        try:
                            frames = frames_for(G.GOAL_TASKS[t], d, h5)
                            dl = G.read_demo_layout(e, d.init_state, fref)
                            obj_t, tar_t = G.anchor_deltas(G.GOAL_TASKS[t], layout, dl)
                            rp, ba, nf = synthesize_uniform(d.state, d.action, frames,
                                                            obj_t, tar_t)
                            ni, fx = G.apply_goal_layout(layout, d.init_state, dl)
                            R.reset_to_init_state(e, ni)
                            S.apply_fixture_edits(e, fx)
                            _, _, roll = R.replay_uniform(e, ba, rp, nf, collect=True)
                            end = bool(e.check_success())
                            jm = joint_margin(roll["joint_states"])
                            pk = pose_ok(t, roll["states"][-1])[0]
                            good = (end and pk and
                                    (t in JOINT_GATE_EXEMPT or jm >= JOINT_MARGIN_FLOOR))
                            detail.append({"src": src, "end": end,
                                           "margin": round(jm, 3), "pose": pk})
                        except Exception as exc:
                            detail.append({"src": src, "error": repr(exc)[:60]})
                            continue
                        if good:
                            ok = True
                            break
                    per[t] = {"ok": ok, "attempts": detail}
                status = "ok" if all(v["ok"] for v in per.values()) else "fail"
                grid[key] = {"status": status, "reach_bad": reach_bad, "tasks": per}
                print("[%s] (%+.3f,%+.3f) %s%s  %s" %
                      (fb.replace("_1_main", ""), x, y,
                       "OK  " if status == "ok" else "FAIL",
                       "  reach_flagged" if reach_bad else "",
                       ",".join("%s:%s" % (t.split("_")[-1], "Y" if v["ok"] else "n")
                                for t, v in per.items())), flush=True)
                results[fb] = {"xs": xs.tolist(), "ys": ys.tolist(), "grid": grid}
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
        for e, _fr, _h in envs.values():
            e.close()
    n_ok = sum(1 for r in results.values() for v in r["grid"].values()
               if v["status"] == "ok")
    n = sum(len(r["grid"]) for r in results.values())
    print("\nDONE: %d/%d grid points feasible; wrote %s" % (n_ok, n, out_path))


if __name__ == "__main__":
    main()
