"""SUPERSEDED by scripts/sample_goal_suite.py -- kept for provenance.

This v1 sampler picked the seen/unseen split AFTER sampling and scored leakage
per task against that task's own train cells only; adversarial review showed
that guard is not the one a single multi-task policy needs (it trains on
frames of ALL seen layouts), and the realized unseen-vs-seen distance under it
fell to 1.64 cm. The successor samples unseen FIRST with per-entity exclusion
circles so the floor holds by construction. The original docstring follows.

Sample the 12 goal layouts (8 seen + 4 unseen) with every gate from the
plan doc §4 (2026-08-22 revision), and write the manifest draft.

Gates per layout: geometry (spacing/corridors/receiver zones/drawer-sweep
keep-out/occlusion, goal_scene sampler) -> settle converged -> rendered
per-entity visibility -> near-duplicate rejection -> REACHABILITY BY REPLAY:
the 6 fixture-anchored tasks (drawer, knob, bowl->stove, bowl->cabinet-top,
bottle->rack, bottle->cabinet-top) must each succeed end-state on the layout
within 2 source demos. Replay IS the reach oracle -- the servo reach-map probe
was dropped after twice failing cross-validation against known-good replays
(P-control servo with locked orientation wedges where demo-shaped trajectories
demonstrably pass). Object grasp reach is covered by the workspace bounds
(never a failure inside them across spatial's 1000+ demos and both smokes).

Split: L00 is the original nominal layout (anchor to the official suite,
always seen, replay-gated like the rest). The 4 unseen layouts are chosen by
exhaustive search maximizing the per-anchor-fixture separation floor,
enforcing >= GoalSpec.unseen_anchor_min_dist from every seen layout.

Usage:
    .venv\\Scripts\\python.exe scripts\\sample_goal_layouts.py --seed 21 --out-dir output/goal_layouts_12
"""
import argparse
import itertools
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

ENTITIES = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1",
            "wooden_cabinet_1", "flat_stove_1", "wine_rack_1"]

# reachability-by-replay gate tasks: every fixture-anchored goal, cheapest-to-
# fail first so rejected layouts cost the least sim time
GATE_TASKS = [
    "turn_on_the_stove",
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_the_rack",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_wine_bottle_on_top_of_the_cabinet",
]


def entity_dists(la, lb):
    """Per-entity planar distance between two layouts (7 entries)."""
    out = {}
    for jn in G.GOAL_JOINTS:
        out[jn] = float(np.linalg.norm(np.asarray(la["objects"][jn]) -
                                       np.asarray(lb["objects"][jn])))
    for fb in G.GOAL_FIXTURES:
        out[fb] = float(np.linalg.norm(np.asarray(la["fixtures"][fb]) -
                                       np.asarray(lb["fixtures"][fb])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--n-total", type=int, default=12)
    ap.add_argument("--n-unseen", type=int, default=4)
    ap.add_argument("--source-base-dir", default="D:/Data/LingLing/libero/hf/libero_goal")
    ap.add_argument("--out-dir", default=os.path.join("output", "goal_layouts_12"))
    ap.add_argument("--dedup-min", type=float, default=0.05,
                    help="reject a candidate if EVERY entity is within this of an accepted layout")
    ap.add_argument("--gate-sources", type=int, default=2,
                    help="source demos tried per gate task before rejecting the layout")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    spec = G.GoalSpec()

    task0 = list(G.GOAL_TASKS)[0]
    demo = load_demo(os.path.join(args.source_base_dir, f"{task0}_demo.hdf5"), "demo_0")
    env = oc_obs.make_oc_env(demo.bddl_file)
    env.reset()
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    cam_xy = get_camera_extrinsic_matrix(env.sim, "agentview")[:2, 3].copy()
    ref_layout = S.read_layout(env, demo.init_state, G.GOAL_JOINTS, G.GOAL_FIXTURES)

    def full_gates(layout):
        """settle + rendered visibility; returns (ok, report, img)."""
        new_init, fx = G.apply_goal_layout(layout, demo.init_state, ref_layout)
        R.reset_to_init_state(env, new_init)
        S.apply_fixture_edits(env, fx)
        rep = S.settle(env, spec.settle_steps)
        obs = env.env._get_observations(force_update=True)
        raw = obs["agentview_segmentation_instance"][..., 0]
        names = list(env.env.model.instances_to_ids.keys())
        px = {nm: int((raw == names.index(nm) + 1).sum()) for nm in ENTITIES}
        ok = rep["converged"] and all(v >= spec.min_px for v in px.values())
        return ok, {"settle": rep, "px": px}, np.asarray(obs["agentview_image"])[::-1]

    # replay-gate envs, one per gate task (built once, reused for all layouts)
    gate_envs = {}
    for task in GATE_TASKS:
        genv = oc_obs.make_oc_env(os.path.join(
            get_libero_path("bddl_files"), "libero_goal", f"{task}.bddl"))
        genv.reset()
        gate_envs[task] = (genv, G.capture_fixture_ref(genv),
                          os.path.join(args.source_base_dir, f"{task}_demo.hdf5"))

    def replay_gate(layout, rng):
        """Every fixture-anchored task must succeed (end-state predicate) on
        this layout within --gate-sources source demos. Returns (ok, report)."""
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

    # L00: the original nominal layout (replay-gated like the rest)
    nominal = {"objects": {jn: ref_layout["free"][jn]["pos"][:2].tolist()
                           for jn in G.GOAL_JOINTS},
               "fixtures": {fb: ref_layout["fixtures"][fb]["pos"][:2].tolist()
                            for fb in G.GOAL_FIXTURES}}
    ok, rep0, img0 = full_gates(nominal)
    assert ok, f"nominal layout fails settle/visibility?! {rep0}"
    ok0, gate0 = replay_gate(nominal, rng)
    assert ok0, f"nominal layout fails the replay gate?! {gate0}"

    layouts = [{"id": "L00", "layout": nominal, "gates": rep0,
                "replay_gate": gate0, "nominal": True}]
    imgs = [img0]
    tried = 0
    while len(layouts) < args.n_total and tried < args.n_total * 40:
        tried += 1
        try:
            cand = G.sample_goal_layout(rng, spec, ref_layout, cam_xy)
        except RuntimeError:
            continue
        dup = None
        for acc in layouts:
            d = entity_dists(cand, acc["layout"])
            if all(v < args.dedup_min for v in d.values()):
                dup = acc["id"]
                break
        if dup:
            print(f"[try {tried}] near-duplicate of {dup}", flush=True)
            continue
        ok, rep, img = full_gates(cand)
        if not ok:
            print(f"[try {tried}] settle/visibility reject "
                  f"(settle={rep['settle']['converged']} min_px={min(rep['px'].values())})",
                  flush=True)
            continue
        ok, gate_rep = replay_gate(cand, rng)
        n_replays = sum(len(v) for v in gate_rep.values())
        print(f"[try {tried}] replay gate {'PASS' if ok else 'FAIL'} "
              f"({n_replays} replays; "
              + "; ".join(f"{t.split('_')[-1]}:{'Y' if any(a.get('success_end') for a in v) else 'n'}"
                          for t, v in gate_rep.items())
              + f") ({len(layouts)}/{args.n_total})", flush=True)
        if not ok:
            continue
        layouts.append({"id": f"L{len(layouts):02d}", "layout": cand, "gates": rep,
                        "replay_gate": gate_rep})
        imgs.append(img)
        with open(os.path.join(args.out_dir, "layouts_progress.json"), "w") as f:
            json.dump(layouts, f, indent=2)
    env.close()
    for genv, *_ in gate_envs.values():
        genv.close()
    assert len(layouts) == args.n_total, f"only {len(layouts)} layouts after {tried} tries"

    # unseen selection: exhaustive over non-nominal layouts, maximize the
    # minimum per-fixture separation floor to the seen side; enforce the floor
    cand_idx = [i for i in range(len(layouts)) if not layouts[i].get("nominal")]
    best, best_floor = None, -1.0
    for combo in itertools.combinations(cand_idx, args.n_unseen):
        seen = [i for i in range(len(layouts)) if i not in combo]
        floor = min(
            np.linalg.norm(np.asarray(layouts[u]["layout"]["fixtures"][fb]) -
                           np.asarray(layouts[s]["layout"]["fixtures"][fb]))
            for u in combo for s in seen for fb in G.GOAL_FIXTURES)
        if floor > best_floor:
            best, best_floor = combo, floor
    unseen_ids = [layouts[i]["id"] for i in best]
    print(f"unseen={unseen_ids} fixture-separation floor={best_floor:.3f} m "
          f"(required {spec.unseen_anchor_min_dist})", flush=True)
    floor_ok = best_floor >= spec.unseen_anchor_min_dist

    # per-unseen-layout per-entity minimum distance to the seen side (gate 5)
    seen_idx = [i for i in range(len(layouts)) if layouts[i]["id"] not in unseen_ids]
    leakage = {}
    for u in best:
        per_entity = {}
        for ent in ENTITIES:
            key = next(k for k in list(G.GOAL_JOINTS) + list(G.GOAL_FIXTURES)
                       if k.startswith(ent))
            per_entity[ent] = round(min(
                entity_dists(layouts[u]["layout"], layouts[s]["layout"])[key]
                for s in seen_idx), 4)
        leakage[layouts[u]["id"]] = per_entity

    # task x seen-layout assignment (plan §5): each task 4 train + 4 cf over
    # the 8 seen layouts, constrained so each layout trains 4-5 tasks
    seen_ids = [layouts[i]["id"] for i in seen_idx]
    tasks = list(G.GOAL_TASKS)
    for _ in range(10000):
        matrix = {t: sorted(rng.choice(len(seen_ids), 4, replace=False).tolist())
                  for t in tasks}
        counts = np.zeros(len(seen_ids), int)
        for t in tasks:
            counts[matrix[t]] += 1
        if counts.min() >= 4 and counts.max() <= 5:
            break
    matrix_ids = {t: [seen_ids[j] for j in js] for t, js in matrix.items()}

    manifest = {
        "created": "2026-08-22", "seed": args.seed,
        "reach_gate": {"kind": "replay", "tasks": GATE_TASKS,
                       "sources_per_task": args.gate_sources,
                       "note": "servo reach map dropped: failed cross-validation "
                               "against known-good replays twice"},
        "spec": {
            "fixture_corridor": spec.fixture_corridor,
            "receiver_zone": spec.receiver_zone,
            "unseen_anchor_min_dist": spec.unseen_anchor_min_dist,
            "dedup_min": args.dedup_min, "min_px": spec.min_px},
        "layouts": layouts,
        "unseen_ids": unseen_ids, "seen_ids": seen_ids,
        "unseen_fixture_floor_m": round(best_floor, 4),
        "unseen_floor_ok": bool(floor_ok),
        "leakage_per_unseen_per_entity_min_dist": leakage,
        "train_matrix": matrix_ids,
        "cf_matrix": {t: [sid for sid in seen_ids if sid not in matrix_ids[t]]
                      for t in tasks},
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(4 * 3.3, 3 * 3.5))
    for ax, rec, img in zip(axes.flat, layouts, imgs):
        ax.imshow(img)
        u = rec["id"] in unseen_ids
        ax.set_title(f"{rec['id']}{' (nominal)' if rec.get('nominal') else ''}"
                     f"{' UNSEEN' if u else ''}",
                     fontsize=10, color="tab:red" if u else "black")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"12 goal layouts: 8 seen + 4 unseen "
                 f"(fixture floor {best_floor:.2f}m, required {spec.unseen_anchor_min_dist})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(args.out_dir, "layouts_12.png"), dpi=110, bbox_inches="tight")
    print(f"wrote {args.out_dir}/manifest.json + layouts_12.png; floor_ok={floor_ok}")


if __name__ == "__main__":
    main()
