"""Can the cabinet's drawers be addressed separately in segmentation? (plan §7.3)

The plan treats part-level drawer ids as decided -- "open the middle drawer"
cannot ground its noun on a single whole-cabinet mask -- but the INSTANCE
segmentation this repo uses returns one `wooden_cabinet_1` covering the whole
unit. This probe checks whether the ELEMENT (per-geom) segmentation can be
regrouped by body into per-drawer masks, and whether those masks are big
enough to be usable.

Measured at three drawer openings (closed / half / fully open) because a mask
that only exists when the drawer is pulled out is not a usable grounding
target for the initial frame.

Usage:
    .venv\\Scripts\\python.exe scripts\\probe_drawer_seg.py --out output/goal_drawer_seg
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from demogen_libero import goal_scene as G
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

CABINET_PARTS = ["wooden_cabinet_1_cabinet_top", "wooden_cabinet_1_cabinet_middle",
                 "wooden_cabinet_1_cabinet_bottom", "wooden_cabinet_1_main"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("output", "goal_drawer_seg"))
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bddl = os.path.join(get_libero_path("bddl_files"), "libero_goal",
                        "open_the_middle_drawer_of_the_cabinet.bddl")
    report = {}

    # --- what does each segmentation level actually return? ---
    for level in ["instance", "class", "element"]:
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.size,
                                 camera_widths=args.size, camera_depths=True,
                                 camera_segmentations=level)
        obs = env.reset()
        seg = obs[f"agentview_segmentation_{level}"][..., 0]
        m = env.sim.model
        uniq = sorted(set(int(v) for v in np.unique(seg)))
        report[level] = {"n_unique_ids": len(uniq)}
        print(f"[{level}] {len(uniq)} unique ids in the frame", flush=True)
        if level == "element":
            # element ids index geoms (+1); regroup them by the body they
            # belong to, which is what a part-level mask would be built from
            by_body = defaultdict(int)
            for v in uniq:
                if v <= 0 or v > m.ngeom:
                    continue
                bid = int(m.geom_bodyid[v - 1])
                by_body[m.body_id2name(bid)] += int((seg == v).sum())
            report["element_by_body"] = {k: int(n) for k, n in
                                         sorted(by_body.items(), key=lambda x: -x[1])}
            print("  visible bodies (px):", flush=True)
            for k, n in sorted(by_body.items(), key=lambda x: -x[1])[:12]:
                mark = "  <-- cabinet part" if k in CABINET_PARTS else ""
                print(f"    {k:42s} {n:6d}{mark}", flush=True)
        env.close()

    # --- per-drawer pixel counts at three openings ---
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.size,
                             camera_widths=args.size, camera_depths=True,
                             camera_segmentations="element")
    env.reset()
    m = env.sim.model
    openings = {"closed": 0.0, "half": -0.08, "open": -0.16}
    per_open = {}
    for name, qpos in openings.items():
        env.reset()
        for j in ("wooden_cabinet_1_top_level", "wooden_cabinet_1_middle_level"):
            adr = m.get_joint_qpos_addr(j)
            env.sim.data.qpos[adr] = qpos if "middle" in j else 0.0
        env.sim.forward()
        obs = env.env._get_observations(force_update=True)
        seg = obs["agentview_segmentation_element"][..., 0]
        counts = Counter()
        for v in set(int(x) for x in np.unique(seg)):
            if v <= 0 or v > m.ngeom:
                continue
            counts[m.body_id2name(int(m.geom_bodyid[v - 1]))] += int((seg == v).sum())
        per_open[name] = {p: counts.get(p, 0) for p in CABINET_PARTS}
        print(f"[middle drawer {name:6s} qpos={qpos:+.2f}] " +
              "  ".join(f"{p.split('cabinet_')[-1]}={counts.get(p,0)}"
                        for p in CABINET_PARTS), flush=True)
    env.close()
    report["per_opening_px"] = per_open

    # --- joint <-> body naming check: move ONE slide joint, see which body
    # moves. Without this the part masks could be labelled off-by-one, which
    # would silently mis-ground "the middle drawer". ---
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.size,
                             camera_widths=args.size, camera_depths=True,
                             camera_segmentations="element")
    env.reset()
    m = env.sim.model
    naming = {}
    for lvl in ["top", "middle", "bottom"]:
        env.reset()
        env.sim.data.qpos[m.get_joint_qpos_addr(f"wooden_cabinet_1_{lvl}_level")] = -0.16
        env.sim.forward()
        moved = {}
        for part in CABINET_PARTS[:3]:
            moved[part] = float(env.sim.data.get_body_xpos(part)[1])
        # the body that travelled in +y is the one this joint drives
        driven = max(moved, key=lambda k: moved[k])
        naming[f"{lvl}_level"] = {"drives_body": driven, "body_y": moved,
                                  "ok": driven.endswith(lvl)}
        print(f"  joint {lvl:6s}_level drives {driven.split('cabinet_')[-1]:8s} "
              f"{'OK' if driven.endswith(lvl) else 'MISMATCH'}", flush=True)
        # a stationary part's mask can balloon when a NEIGHBOUR opens, because
        # the gap exposes that neighbour's box: record it so nobody uses
        # "largest cabinet part mask" as the referent heuristic
        obs = env.env._get_observations(force_update=True)
        seg = obs["agentview_segmentation_element"][..., 0]
        px = {}
        for part in CABINET_PARTS[:3]:
            bid = m.body_name2id(part)
            n = sum(int((seg == v).sum()) for v in set(int(x) for x in np.unique(seg))
                    if 0 < v <= m.ngeom and int(m.geom_bodyid[v - 1]) == bid)
            px[part.split("cabinet_")[-1]] = n
        naming[f"{lvl}_level"]["px_while_open"] = px
    env.close()
    report["joint_body_naming"] = naming
    report["caveat"] = ("a STATIONARY drawer's mask grows when a neighbour opens "
                        "(the gap exposes the neighbour's box): opening the middle "
                        "drawer takes cabinet_top from ~469 to ~4184 px. Never use "
                        "'largest cabinet part mask' to pick the referred drawer.")

    with open(os.path.join(args.out, "drawer_seg_probe.json"), "w") as f:
        json.dump(report, f, indent=2)

    mid_closed = per_open["closed"]["wooden_cabinet_1_cabinet_middle"]
    top_closed = per_open["closed"]["wooden_cabinet_1_cabinet_top"]
    print("\nverdict:")
    print(f"  middle drawer visible when CLOSED: {mid_closed} px")
    print(f"  top drawer visible when CLOSED:    {top_closed} px")
    print("  -> part-level drawer masks are "
          + ("FEASIBLE via element seg regrouped by body"
             if min(mid_closed, top_closed) >= 60 else
             "NOT usable from the initial frame at this camera/resolution"))
    print(f"wrote {args.out}/drawer_seg_probe.json")


if __name__ == "__main__":
    main()
