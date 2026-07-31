"""Plot where the two bowls land in PAIRED (two-relation) scenes, one panel per
candidate pair.

bowl_1 carries task A's relation, bowl_2 carries task B's. The screen itself
stores only statistics, so scenes are re-sampled here; sampling is deterministic
given --seed. Only the geometric gates are applied (no physics/render), which
the full screen shows keeps 92% of geometrically valid scenes, so the placement
distribution is faithful while the plot stays cheap.

Usage:
    .venv\\Scripts\\python.exe scripts\\plot_pair_distribution.py
    .venv\\Scripts\\python.exe scripts\\plot_pair_distribution.py --max-yaw-object 90
"""
import argparse
import itertools
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import screen_spatial_pairs as P
from demogen_libero.convert import resolve_bddl_file
from demogen_libero import libero_replay as R, oc_obs, spatial_scene as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="scenes to plot per pair")
    ap.add_argument("--max-draws", type=int, default=120000,
                    help="per-pair sampling budget; pairs that run out are plotted "
                         "with however many scenes they produced")
    ap.add_argument("--max-yaw-object", type=float, default=None,
                    help="group yaw limit in degrees for object-anchored relations "
                         "(default: the single-task +-30)")
    ap.add_argument("--exclusion", type=float, default=0.24)
    ap.add_argument("--exclusivity", choices=["pair", "global"], default="pair")
    ap.add_argument("--cache", default="output/spatial_pairs/layout_cache.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-json", default=None,
                    help="also write the sampled bowl/plate positions, so a "
                         "seen/unseen split can be analysed without re-sampling")
    ap.add_argument("--out", default="output/pair_distribution.png")
    args = ap.parse_args()

    cfg = S.load_spatial_config()
    tasks = list(cfg)
    rel = P.relations_from_config(cfg)
    cache = json.load(open(args.cache))["cache"]

    with h5py.File(os.path.join(P.DATA_DIR, f"{P.HOST_TASK}_demo.hdf5"), "r") as f:
        bddl = resolve_bddl_file(f["data"].attrs["bddl_file_name"])
        init = np.array(f["data"]["demo_0"]["states"][0], dtype=np.float64)
    env = oc_obs.make_oc_env(bddl)
    env.reset()
    R.reset_to_init_state(env, init)
    geom, _owner, table = P.measure_scene_geometry(env)
    env.close()

    rel_points = P.measure_relation_points(cache, rel)
    spec = P.Spec(exclusion=args.exclusion, exclusivity=args.exclusivity,
                  max_yaw_object=args.max_yaw_object)
    spec.set_table(table)

    pairs = [(a, b) for a, b in itertools.combinations(tasks, 2)
             if P.compat(a, b, rel, rel_points, args.exclusion)[0] == "candidate"]

    data = {}
    for ta, tb in pairs:
        rng = np.random.default_rng(args.seed)
        stats, b1, b2, plate = Counter(), [], [], []
        while len(b1) < args.n and stats["draws"] < args.max_draws:
            sc, _w = P.sample_pair_geometry(
                rng, ta, tb, cache, rel, geom, spec, rel_points,
                int(rng.integers(len(cache[ta]))), int(rng.integers(len(cache[tb]))),
                stats)
            if sc is None:
                continue
            b1.append(sc["gA"][P.BOWL_A]["xy"])
            b2.append(sc["gB"][P.BOWL_B]["xy"])
            g = sc["gA"].get(P.PLATE) or sc["gB"].get(P.PLATE) or sc["indep"].get(P.PLATE)
            plate.append(g["xy"])
        data[(ta, tb)] = (np.array(b1), np.array(b2), np.array(plate), stats["draws"])
        print(f"  {P.short(ta)} + {P.short(tb)}: {len(b1)} scenes "
              f"({stats['draws']} draws)", flush=True)

    if args.dump_json:
        blob = {f"{P.short(a)}|{P.short(b)}": {
            "bowl_1": data[(a, b)][0].tolist(), "bowl_2": data[(a, b)][1].tolist(),
            "plate": data[(a, b)][2].tolist(), "draws": int(data[(a, b)][3])}
            for a, b in pairs}
        json.dump({"pairs": blob, "params": vars(args)},
                  open(args.dump_json, "w"))
        print(f"wrote {args.dump_json}")

    ncol, nrow = 5, int(np.ceil(len(pairs) / 5))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.1 * nrow),
                             sharex=True, sharey=True)
    sp = S.SpatialSpec()
    for ax, key in zip(axes.ravel(), pairs):
        b1, b2, pl, draws = data[key]
        ax.add_patch(plt.Rectangle((sp.x_range[0], sp.y_range[0]),
                                   sp.x_range[1] - sp.x_range[0],
                                   sp.y_range[1] - sp.y_range[0],
                                   fill=False, ec="0.6", ls="--", lw=.9))
        ax.add_patch(plt.Rectangle((sp.dest_x[0], sp.dest_y[0]),
                                   sp.dest_x[1] - sp.dest_x[0],
                                   sp.dest_y[1] - sp.dest_y[0],
                                   fill=False, ec="seagreen", ls=":", lw=1.1))
        if len(pl):
            ax.scatter(*pl.T, s=13, c="seagreen", alpha=.35, lw=0, marker="s")
        if len(b1):
            ax.scatter(*b1.T, s=20, c="#1f77b4", alpha=.65, lw=0)
            ax.scatter(*b2.T, s=20, c="#d62728", alpha=.65, lw=0)
            s1, s2 = b1.ptp(0), b2.ptp(0)
            sub = f"b1 {s1[0]:.2f}x{s1[1]:.2f}   b2 {s2[0]:.2f}x{s2[1]:.2f}"
        else:
            sub = "no valid scene found"
        ax.set_title(f"{P.short(key[0])[:26]}\n+ {P.short(key[1])[:26]}\n"
                     f"n={len(b1)}  {sub}", fontsize=7.5)
        ax.set_aspect("equal")
        ax.grid(alpha=.15)
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    axes.ravel()[0].set_xlim(-0.52, 0.36)
    axes.ravel()[0].set_ylim(-0.46, 0.44)

    yaw = args.max_yaw_object if args.max_yaw_object is not None else 30
    fig.suptitle(
        f"Paired-scene bowl placement, one panel per candidate pair  "
        f"(blue = bowl_1 holding the FIRST relation, red = bowl_2 holding the SECOND, "
        f"green = destination plate)\n"
        f"object-group yaw +-{yaw:.0f} deg,  exclusion {args.exclusion:.2f} m,  "
        f"exclusivity '{args.exclusivity}'   |   dashed = target workspace, "
        f"dotted = reachable place zone", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
