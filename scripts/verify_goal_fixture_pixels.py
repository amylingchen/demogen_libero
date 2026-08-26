"""Pixel verification of the fixture poses carried in each demo's sidecar
attrs (plan §7.6). Two independent checks, both against rendered pixels:

A. REPRODUCTION -- reset the sim to the demo's stored states[0], re-apply
   `fixture_edits`, render agentview, and compare with the demo's stored
   agentview_rgb[0]. If the fixture poses were wrong or missing, the fixtures
   would render somewhere else and the frames would not match. This is the
   check that matters for downstream use: it proves state + attrs are enough
   to rebuild the scene, which is what the OC re-render pass will rely on.

B. PROJECTION -- project each fixture's recorded position, and its task goal
   points (cabinet top / drawer handle, rack slot, burner, knob), through the
   goal camera matrix and confirm the pixel lands on that fixture in the
   segmentation image. The earlier camera self-check only covered the four
   movable objects.

Both are run on demos drawn from DIFFERENT layouts, since a check on the
nominal layout alone cannot distinguish "poses are recorded correctly" from
"the fixtures happen to be where the bddl put them".

Usage:
    .venv\\Scripts\\python.exe scripts\\verify_goal_fixture_pixels.py --dir output/goal_gen_v3
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np
from robosuite.utils.camera_utils import (get_camera_transform_matrix,
                                          project_points_from_world_to_camera)

from demogen_libero import libero_replay as R
from demogen_libero import oc_obs
from demogen_libero import spatial_scene as S
from demogen_libero import goal_scene as G

from libero.libero import get_libero_path

SPLITS = ["train", "quarantine_cf", "quarantine_unseen"]
# fixture -> its task goal points, as offsets in the fixture frame + a z
GOAL_POINTS = {
    "wooden_cabinet_1_main": [("cabinet_top", (-0.03, 0.05), 1.127),
                              ("drawer_handle", (0.0, 0.10), 1.03)],
    "flat_stove_1_main": [("burner", (0.16, 0.05), 0.93),
                          ("knob", (0.0, 0.0), 0.95)],
    "wine_rack_1_main": [("rack_slot", (0.083, 0.0), 1.14)],
}
INSTANCE = {"wooden_cabinet_1_main": "wooden_cabinet_1",
            "flat_stove_1_main": "flat_stove_1",
            "wine_rack_1_main": "wine_rack_1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("output", "goal_gen_v3"))
    ap.add_argument("--per-layout", type=int, default=1,
                    help="demos checked per distinct layout")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.dir, "fixture_pixel_check.json")

    # one demo per distinct layout, preferring a task whose env we build once
    task = "open_the_middle_drawer_of_the_cabinet"
    picks = []
    seen = defaultdict(int)
    for split in SPLITS:
        p = os.path.join(args.dir, split, f"{task}_demo.hdf5")
        if not os.path.exists(p):
            continue
        with h5py.File(p, "r") as f:
            for k in f["data"]:
                lid = f["data"][k].attrs["layout_id"]
                if seen[lid] < args.per_layout:
                    seen[lid] += 1
                    picks.append((p, k, lid, split))
    print(f"checking {len(picks)} demos over {len(seen)} distinct layouts", flush=True)

    env = oc_obs.make_oc_env(os.path.join(
        get_libero_path("bddl_files"), "libero_goal", f"{task}.bddl"))
    env.reset()
    H = W = 256
    P = get_camera_transform_matrix(env.sim, "agentview", H, W)
    names = list(env.env.model.instances_to_ids.keys())

    report = {"reproduction": [], "projection": []}
    for path, key, lid, split in picks:
        with h5py.File(path, "r") as f:
            g = f["data"][key]
            st0 = np.array(g["states"][0], dtype=np.float64)
            stored = np.array(g["obs"]["agentview_rgb"][0])
            fx_json = json.loads(g.attrs["fixture_edits"])
        fx = {fb: {"pos": np.asarray(e["pos"]), "quat": np.asarray(e["quat_wxyz"])}
              for fb, e in fx_json.items()}

        # --- A. reproduction: with the recorded fixture poses ---
        R.reset_to_init_state(env, st0)
        S.apply_fixture_edits(env, fx)
        obs = env.env._get_observations(force_update=True)
        re_rgb = np.asarray(obs["agentview_image"])
        diff_ok = float(np.abs(re_rgb.astype(int) - stored.astype(int)).mean())
        pct_ok = float((np.abs(re_rgb.astype(int) - stored.astype(int)).max(2) > 8).mean() * 100)

        # --- A'. NEGATIVE CONTROL: skip the fixture edits ---
        R.reset_to_init_state(env, st0)
        obs_n = env.env._get_observations(force_update=True)
        re_no = np.asarray(obs_n["agentview_image"])
        diff_no = float(np.abs(re_no.astype(int) - stored.astype(int)).mean())
        pct_no = float((np.abs(re_no.astype(int) - stored.astype(int)).max(2) > 8).mean() * 100)

        report["reproduction"].append(
            {"layout": lid, "split": split, "demo": key,
             "with_fixture_edits": {"mean_abs_diff": round(diff_ok, 3),
                                    "pct_pixels_differing": round(pct_ok, 2)},
             "without_fixture_edits": {"mean_abs_diff": round(diff_no, 3),
                                       "pct_pixels_differing": round(pct_no, 2)}})
        print(f"  [{lid} {split:17s}] reproduce: {pct_ok:5.2f}% px differ  |  "
              f"WITHOUT edits: {pct_no:5.2f}%", flush=True)

        # --- B. projection of fixture body + goal points ---
        R.reset_to_init_state(env, st0)
        S.apply_fixture_edits(env, fx)
        obs = env.env._get_observations(force_update=True)
        raw = obs["agentview_segmentation_instance"][..., 0]
        for fb, pts in GOAL_POINTS.items():
            base = np.asarray(fx_json[fb]["pos"])
            targets = [("body_origin", base)]
            targets += [(nm, np.array([base[0] + o[0], base[1] + o[1], z]))
                        for nm, o, z in pts]
            for nm, xyz in targets:
                rc = project_points_from_world_to_camera(
                    np.asarray(xyz)[None, :], P, H, W)[0]
                r, c = int(round(rc[0])), int(round(rc[1]))
                gl_r = H - 1 - r
                v = int(raw[gl_r, c]) if (0 <= gl_r < H and 0 <= c < W) else 0
                hit = names[v - 1] if 0 < v <= len(names) else None
                report["projection"].append(
                    {"layout": lid, "fixture": INSTANCE[fb], "point": nm,
                     "world": np.round(xyz, 4).tolist(),
                     "pixel_stored_gl": [gl_r, c], "seg_hit": hit,
                     "ok": hit == INSTANCE[fb]})
    env.close()

    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    rep = report["reproduction"]
    worst_ok = max(r["with_fixture_edits"]["pct_pixels_differing"] for r in rep)
    best_no = min(r["without_fixture_edits"]["pct_pixels_differing"] for r in rep)
    print(f"\nA. reproduction over {len(rep)} layouts:")
    print(f"   with fixture_edits    : worst {worst_ok:.2f}% of pixels differ")
    print(f"   WITHOUT (neg control) : best  {best_no:.2f}% of pixels differ")
    proj = report["projection"]
    by_pt = defaultdict(lambda: [0, 0])
    for x in proj:
        by_pt[(x["fixture"], x["point"])][1] += 1
        by_pt[(x["fixture"], x["point"])][0] += int(x["ok"])
    print(f"\nB. projection hits ({len(proj)} points over {len(seen)} layouts):")
    for (fbn, pt), (ok, n) in sorted(by_pt.items()):
        flag = "" if ok == n else "   <-- misses"
        print(f"   {fbn:18s} {pt:15s} {ok}/{n}{flag}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
