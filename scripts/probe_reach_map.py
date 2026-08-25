"""DROPPED INSTRUMENT -- kept for the record, do NOT use as a reachability gate.

This servo-based reach map failed cross-validation against known-good replays
TWICE (2026-08-23): v1 (straight line from home) marked the stove-knob region
unreachable where smoke replays had succeeded, and v2 (hover-then-descend)
marked mid-table grasp points and a cabinet-top pocket unreachable where
generation demonstrably succeeded. Replaying the stored map against the 12
replay-verified layouts vetoed a task on all 12, including the official
nominal layout. Reachability is now decided BY REPLAY (the gate in
scripts/sample_goal_suite.py) plus the measured reach envelope in
goal_scene.REACH_*. The original docstring follows.

Measure the arm's EMPIRICAL reach map with the real OSC controller: servo
the EE to every grid point on 3 z-planes (table-level manipulation, rack/drawer
level, cabinet-top level) in an emptied scene, record the converged residual.
The map becomes the sampling-time reachability gate for goal layouts (both
object positions and per-task goal points must be reachable).

Servo (not geometry) because reach is NOT radius-monotone: the fixture-extreme
probe had failures both too far (cabinet back-left) and too close to the base
(stove knob at 10cm from the base column).

Usage:
    .venv\\Scripts\\python.exe scripts\\probe_reach_map.py --out output/goal_reach
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from libero.libero import get_libero_path
from demogen_libero import oc_obs
from demogen_libero import goal_scene as G

Z_PLANES = [1.00, 1.15, 1.26]   # table grasp/place + knob/burner; rack slot/drawer handle; cabinet top
GRID_X = (-0.55, 0.20)
GRID_Y = (-0.45, 0.45)
STEP = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("output", "goal_reach"))
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--tol", type=float, default=0.015)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bddl = os.path.join(get_libero_path("bddl_files"), "libero_goal",
                        "put_the_bowl_on_the_plate.bddl")
    env = oc_obs.make_oc_env(bddl)
    steps_used = [0]

    def fresh_episode():
        """(Re)start an episode with an emptied scene: park free objects far
        away, sink fixtures below the table. Needed both at start and whenever
        the episode horizon runs out mid-sweep."""
        env.reset()
        sim = env.sim
        m = sim.model
        for jn in G.GOAL_JOINTS:
            a = m.get_joint_qpos_addr(jn)[0]
            sim.data.qpos[a:a + 2] = [5.0 + 0.5 * G.GOAL_JOINTS.index(jn), 5.0]
        for fb in G.GOAL_FIXTURES:
            bid = m.body_name2id(fb)
            m.body_pos[bid][2] -= 5.0
        sim.forward()
        steps_used[0] = 0
        return sim.data.qpos[:9].copy()

    home_qpos = fresh_episode()
    HORIZON_GUARD = 860   # robosuite horizon minus two-phase servo budget (2*40+1)

    def reset_arm():
        nonlocal home_qpos
        if steps_used[0] > HORIZON_GUARD:
            home_qpos = fresh_episode()
        sim = env.sim
        sim.data.qpos[:9] = home_qpos
        sim.data.qvel[:9] = 0
        sim.forward()

    def servo(target):
        """Two-phase: hover just ABOVE the target (target z + 0.12), then
        descend -- matches how every task trajectory actually approaches
        (vertical final descent). A straight line from home wedges the arm at
        low-z near-base targets (false unreachable pillars over spots where
        smoke replays succeeded, e.g. the knob at (-0.48,0.30)); a FIXED high
        hover (1.30) is itself unreachable at far x and poisoned the descent
        there -- hence relative hover."""
        HOVER_Z = min(float(target[2]) + 0.12, 1.32)
        reset_arm()
        obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
        steps_used[0] += 1

        def drive(goal, n):
            nonlocal obs
            for _ in range(n):
                ee = np.asarray(obs["robot0_eef_pos"])
                if float(np.linalg.norm(goal - ee)) < args.tol * 0.5:
                    break
                act = np.zeros(7)
                act[:3] = np.clip((goal - ee) / 0.05, -1.0, 1.0)
                act[6] = -1.0
                obs, *_ = env.step(act)
                steps_used[0] += 1

        drive(np.array([target[0], target[1], HOVER_Z]), args.max_steps)
        drive(np.asarray(target), args.max_steps)
        ee = np.asarray(obs["robot0_eef_pos"])
        return float(np.linalg.norm(target - ee))

    xs = np.round(np.arange(GRID_X[0], GRID_X[1] + 1e-9, STEP), 3)
    ys = np.round(np.arange(GRID_Y[0], GRID_Y[1] + 1e-9, STEP), 3)
    maps = {}
    for z in Z_PLANES:
        res = np.full((len(xs), len(ys)), np.nan)
        for i, x in enumerate(xs):
            # serpentine sweep to keep consecutive targets close
            cols = list(enumerate(ys)) if i % 2 == 0 else list(enumerate(ys))[::-1]
            for j, y in cols:
                res[i, j] = servo(np.array([x, y, z]))
            print(f"z={z} row x={x}: reachable {(res[i] < args.tol).sum()}/{len(ys)}",
                  flush=True)
        maps[str(z)] = res.tolist()
    env.close()

    out = {"xs": xs.tolist(), "ys": ys.tolist(), "z_planes": Z_PLANES,
           "tol": args.tol, "residual": maps}
    with open(os.path.join(args.out, "reach_map.json"), "w") as f:
        json.dump(out, f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(Z_PLANES), figsize=(6 * len(Z_PLANES), 5.5))
    for ax, z in zip(np.atleast_1d(axes), Z_PLANES):
        r = np.array(maps[str(z)])
        im = ax.pcolormesh(xs, ys, r.T, cmap="RdYlGn_r", vmin=0, vmax=0.10, shading="nearest")
        ax.contour(xs, ys, (r.T < args.tol).astype(float), levels=[0.5], colors="k", linewidths=2)
        ax.plot(-0.66, 0, "k^", ms=14)
        ax.set_title(f"EE servo residual @ z={z} (black = reach boundary, tol {args.tol}m)")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "reach_map.png"), dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}/reach_map.json + reach_map.png")


if __name__ == "__main__":
    main()
