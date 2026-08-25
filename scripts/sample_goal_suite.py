"""Goal-suite 12-layout sampler, v2 (post review round 1, 2026-08-23).

Changes vs sample_goal_layouts.py (v1, kept for history):
- UNSEEN-FIRST with an all-seen guard: one multi-task policy trains on frames
  of ALL 8 seen layouts, so every unseen layout keeps >=0.08m per-entity
  distance (all 7 entities) to EVERY seen layout -- enforced by construction:
  the 4 unseen are sampled first (>=0.08 to L00, >=0.06 pairwise internally,
  plate >=0.11 from the push disk so later disk plates keep the 0.08 guard),
  then the 7 sampled seen layouts must clear 0.08 to every unseen.
- push kept on feasible layouts only (user decision): 3 sampled seen layouts
  have their plate forced into the push-feasibility disk (r=0.03 around the
  nominal plate; push's table-fixed goal region tolerates only ~3-4cm of
  plate displacement under whole-trajectory translation). push train cells =
  L00 + those 3, each verified by an actual push replay; all other push cells
  are marked infeasible in the manifest.
- replay gate covers all 8 non-push tasks (v1 covered only the 6
  fixture-anchored ones; bowl->plate and cheese->bowl had recorded smoke
  failures and were ungated).
- settle gate with teeth: converged AND max_disp < 1cm AND drawer/knob joints
  did not self-move.
- per-entity dedup/diversity floors instead of the v1 all-7-entities-<5cm
  near-duplicate test (which could not fire: observed min over 120 pairs was
  2.5x the threshold, and it admitted a 1.5mm stove duplicate).
- manifest records the full spec, all seeds, reject counts, the per-unseen
  per-entity leakage table, and per-task replay-verification coverage.

Usage:
    .venv\\Scripts\\python.exe scripts\\sample_goal_suite.py --seed 41 --out-dir output/goal_suite_12
"""
import argparse
import json
import os
import sys
from dataclasses import asdict

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
from run_goal_generation import (joint_margin, JOINT_MARGIN_FLOOR,
                                 JOINT_GATE_EXEMPT, pose_ok)

ENTITIES = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1",
            "wooden_cabinet_1", "flat_stove_1", "wine_rack_1"]
ENTITY_KEYS = list(G.GOAL_JOINTS) + list(G.GOAL_FIXTURES)
PUSH = "push_the_plate_to_the_front_of_the_stove"
GATE_TASKS = [t for t in G.GOAL_TASKS if t != PUSH]  # cheap-to-fail ordering below
GATE_TASKS = ["turn_on_the_stove", "open_the_middle_drawer_of_the_cabinet",
              "put_the_bowl_on_the_stove", "put_the_bowl_on_the_plate",
              "put_the_cream_cheese_in_the_bowl", "put_the_wine_bottle_on_the_rack",
              "put_the_bowl_on_top_of_the_cabinet",
              "put_the_wine_bottle_on_top_of_the_cabinet"]


def entity_dists(la, lb):
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
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--n-unseen", type=int, default=4)
    ap.add_argument("--n-seen-sampled", type=int, default=7)
    ap.add_argument("--source-base-dir", default="D:/Data/LingLing/libero/hf/libero_goal")
    ap.add_argument("--out-dir", default=os.path.join("output", "goal_suite_12"))
    ap.add_argument("--gate-sources", type=int, default=2)
    ap.add_argument("--screening", default=os.path.join("output", "goal_source_screening.json"),
                    help="healthy-source pools from screen_goal_sources.py; the gate "
                         "draws only from these so it measures layout geometry, not "
                         "source fragility (bowl->cabtop is 64%% healthy at nominal)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from layouts_progress.json (already-gated layouts "
                         "are kept verbatim; only the missing ones are sampled)")
    ap.add_argument("--rack-y-band", default=None,
                    help="'y0,y1' rack-corridor y override for the REMAINING sampling "
                         "only: rack-task rejections cluster in y<-0.32 and y>-0.20 "
                         "(22 and 12 of 35 logged rack failures) while the mid band "
                         "is near-clean; use for a final layout squeezed into the "
                         "risky bands by the unseen exclusion circles")
    args = ap.parse_args()

    screening = json.load(open(args.screening)) if os.path.exists(args.screening) else {}

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    spec = G.GoalSpec()
    if args.rack_y_band:
        y0, y1 = map(float, args.rack_y_band.split(","))
        (rx, _ry) = spec.fixture_corridor["wine_rack_1_main"]
        spec.fixture_corridor = dict(spec.fixture_corridor)
        spec.fixture_corridor["wine_rack_1_main"] = (rx, (y0, y1))
    rejects = {"geometry": 0, "distance": 0, "settle": 0, "visibility": 0,
               "replay": 0, "push_replay": 0}

    demo = load_demo(os.path.join(args.source_base_dir,
                                  "open_the_middle_drawer_of_the_cabinet_demo.hdf5"),
                     "demo_0")
    env = oc_obs.make_oc_env(demo.bddl_file)
    env.reset()
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    cam_xy = get_camera_extrinsic_matrix(env.sim, "agentview")[:2, 3].copy()
    ref_layout = S.read_layout(env, demo.init_state, G.GOAL_JOINTS, G.GOAL_FIXTURES)

    gate_envs = {}
    for task in GATE_TASKS + [PUSH]:
        genv = oc_obs.make_oc_env(os.path.join(
            get_libero_path("bddl_files"), "libero_goal", f"{task}.bddl"))
        genv.reset()
        gate_envs[task] = (genv, G.capture_fixture_ref(genv),
                          os.path.join(args.source_base_dir, f"{task}_demo.hdf5"))

    def settle_vis(layout):
        """settle with teeth + rendered visibility; (ok, report, img)."""
        new_init, fx = G.apply_goal_layout(layout, demo.init_state, ref_layout)
        R.reset_to_init_state(env, new_init)
        S.apply_fixture_edits(env, fx)
        art0 = [float(env.sim.data.get_joint_qpos(j)) for j in
                ("wooden_cabinet_1_top_level", "wooden_cabinet_1_middle_level",
                 "wooden_cabinet_1_bottom_level", "flat_stove_1_button")]
        rep = S.settle(env, spec.settle_steps)
        art1 = [float(env.sim.data.get_joint_qpos(j)) for j in
                ("wooden_cabinet_1_top_level", "wooden_cabinet_1_middle_level",
                 "wooden_cabinet_1_bottom_level", "flat_stove_1_button")]
        art_drift = max(abs(a - b) for a, b in zip(art0, art1))
        obs = env.env._get_observations(force_update=True)
        raw = obs["agentview_segmentation_instance"][..., 0]
        names = list(env.env.model.instances_to_ids.keys())
        px = {nm: int((raw == names.index(nm) + 1).sum()) for nm in ENTITIES}
        settle_ok = rep["converged"] and rep["max_disp_cm"] < 1.0 and art_drift < 0.005
        vis_ok = all(v >= spec.min_px for v in px.values())
        report = {"settle": rep, "articulation_drift": art_drift, "px": px,
                  "settle_ok": settle_ok, "vis_ok": vis_ok}
        return settle_ok, vis_ok, report, np.asarray(obs["agentview_image"])[::-1]

    def replay_task(task, layout):
        cfg = G.GOAL_TASKS[task]
        genv, fixture_ref, h5 = gate_envs[task]
        pool = screening.get(task, {}).get("healthy") or list_demo_keys(h5)
        keys = list(rng.permutation(pool))[:args.gate_sources]
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
                _, _, roll = R.replay_uniform(genv, base_actions, ref, new_frames,
                                              collect=True)
                end = bool(genv.check_success())
                # a layout that only works with the arm at a joint limit is not
                # a usable layout (2026-08-24); same exemption as the generator
                # for the task whose SOURCES already saturate
                if (task not in JOINT_GATE_EXEMPT
                        and joint_margin(roll["joint_states"]) < JOINT_MARGIN_FLOOR):
                    end = False
                if end and not pose_ok(task, roll["states"][-1])[0]:
                    end = False
            except Exception as exc:
                tried.append({"source": src_key, "error": repr(exc)})
                continue
            tried.append({"source": src_key, "success_end": end})
            if end:
                return True, tried
        return False, tried

    def replay_gate(layout, tasks):
        report = {}
        for task in tasks:
            ok, tried = replay_task(task, layout)
            report[task] = tried
            if not ok:
                return False, report
        return True, report

    def dist_ok(cand, others, floor):
        """per-entity distance floor against each layout in `others`."""
        for o in others:
            d = entity_dists(cand, o["layout"])
            if any(v < floor for v in d.values()):
                return False
        return True

    def in_push_disk(layout):
        return (np.linalg.norm(np.asarray(layout["objects"]["plate_1_joint0"]) -
                               np.asarray(spec.push_disk_center))
                <= spec.push_disk_radius)

    # ---- L00: nominal, exempt from sampler geometry (documented), all gates ----
    progress_path = os.path.join(args.out_dir, "layouts_progress.json")
    if args.resume and os.path.exists(progress_path):
        layouts = json.load(open(progress_path))
        imgs = []
        for rec in layouts:   # re-render (gates already recorded; not re-run)
            *_, img = settle_vis(rec["layout"])
            imgs.append(img)
        print(f"[resume] {len(layouts)} gated layouts loaded "
              f"({sum(l['role'] == 'unseen' for l in layouts)} unseen, "
              f"{sum(l['push_feasible'] for l in layouts)} push-feasible)", flush=True)
    else:
        nominal = {"objects": {jn: ref_layout["free"][jn]["pos"][:2].tolist()
                               for jn in G.GOAL_JOINTS},
                   "fixtures": {fb: ref_layout["fixtures"][fb]["pos"][:2].tolist()
                                for fb in G.GOAL_FIXTURES}}
        s_ok, v_ok, rep0, img0 = settle_vis(nominal)
        assert s_ok and v_ok, f"nominal fails settle/visibility: {rep0}"
        ok0, gate0 = replay_gate(nominal, GATE_TASKS)
        assert ok0, f"nominal fails replay gate: {gate0}"
        okp, pushrep0 = replay_task(PUSH, nominal)
        assert okp, f"nominal fails PUSH replay: {pushrep0}"
        gate0[PUSH] = pushrep0
        layouts = [{"id": "L00", "layout": nominal, "gates": rep0, "replay_gate": gate0,
                    "nominal": True, "role": "seen", "push_feasible": True,
                    "geometry_exempt": True,
                    "geometry_exempt_note": "official LIBERO placement; violates the "
                    "sampler's spacing margins yet passes every replay gate incl. "
                    "push -- margins are conservative, not necessary"}]
        imgs = [img0]

    def sample_candidate(sub_spec, exclusions=None):
        try:
            return G.sample_goal_layout(rng, sub_spec, ref_layout, cam_xy, exclusions)
        except RuntimeError:
            rejects["geometry"] += 1
            return None

    def layout_positions(l):
        return {**{jn: l["layout"]["objects"][jn] for jn in G.GOAL_JOINTS},
                **{fb: l["layout"]["fixtures"][fb] for fb in G.GOAL_FIXTURES}}

    def build_exclusions(reference_layouts, floor, plate_extra=None):
        """Per-entity exclusion circles from a set of layouts at `floor`;
        plate_extra: optional additional (center, dist) for the plate."""
        exc = {}
        for l in reference_layouts:
            for key, xy in layout_positions(l).items():
                exc.setdefault(key, []).append((list(xy), floor))
        if plate_extra is not None:
            exc.setdefault("plate_1_joint0", []).append(plate_extra)
        return exc

    def try_accept(cand, role, push_feasible, extra_gate_tasks):
        s_ok, v_ok, rep, img = settle_vis(cand)
        if not s_ok:
            rejects["settle"] += 1
            print(f"  settle reject (disp={rep['settle']['max_disp_cm']:.2f}cm)", flush=True)
            return False
        if not v_ok:
            rejects["visibility"] += 1
            print(f"  visibility reject {[k for k, v in rep['px'].items() if v < spec.min_px]}",
                  flush=True)
            return False
        ok, gate_rep = replay_gate(cand, GATE_TASKS)
        if not ok:
            rejects["replay"] += 1
            fail_task = list(gate_rep)[-1]
            print(f"  replay FAIL at {fail_task}; candidate objects="
                  f"{ {k.split('_joint')[0]: [round(v,2) for v in xy] for k, xy in cand['objects'].items()} } "
                  f"fixtures={ {k.split('_1_')[0].split('_')[-1]: [round(v,2) for v in xy] for k, xy in cand['fixtures'].items()} }",
                  flush=True)
            return False
        if push_feasible:
            okp, pushrep = replay_task(PUSH, cand)
            gate_rep[PUSH] = pushrep
            if not okp:
                rejects["push_replay"] += 1
                print("  PUSH replay FAIL", flush=True)
                return False
        lid = f"L{len(layouts):02d}"
        layouts.append({"id": lid, "layout": cand, "gates": rep,
                        "replay_gate": gate_rep, "role": role,
                        "push_feasible": bool(push_feasible)})
        imgs.append(img)
        with open(os.path.join(args.out_dir, "layouts_progress.json"), "w") as f:
            json.dump(layouts, f, indent=2)
        print(f"[{lid}] accepted as {role}"
              + (" (push disk)" if push_feasible and role == "seen" else ""), flush=True)
        return True

    # ---- stage 1: 4 unseen (exclusion circles enforced DURING placement) ----
    unseen = lambda: [l for l in layouts if l["role"] == "unseen"]
    tries = 0
    while len(unseen()) < args.n_unseen and tries < 1200:
        tries += 1
        exc = build_exclusions([layouts[0]], spec.unseen_seen_min_dist,
                               plate_extra=(list(spec.push_disk_center),
                                            spec.unseen_seen_min_dist + spec.push_disk_radius))
        for l in unseen():
            for key, xy in layout_positions(l).items():
                exc.setdefault(key, []).append((list(xy), spec.unseen_internal_min_dist))
        cand = sample_candidate(spec, exc)
        if cand is None:
            continue
        # belt-and-braces: the exclusions should already guarantee these
        assert dist_ok(cand, [layouts[0]], spec.unseen_seen_min_dist)
        assert dist_ok(cand, unseen(), spec.unseen_internal_min_dist)
        try_accept(cand, "unseen", push_feasible=False, extra_gate_tasks=[])
    assert len(unseen()) == args.n_unseen, f"only {len(unseen())} unseen after {tries} tries"

    # ---- stage 2: 7 sampled seen (first n_push_seen with the plate in the disk) ----
    def seen_sampled():
        return [l for l in layouts if l["role"] == "seen" and not l.get("nominal")]

    tries = 0
    while len(seen_sampled()) < args.n_seen_sampled and tries < 2000:
        tries += 1
        need_disk = sum(l["push_feasible"] for l in seen_sampled()) < spec.n_push_seen
        import copy
        sub = copy.deepcopy(spec)
        sub.receiver_zone = dict(spec.receiver_zone)
        if need_disk:
            c = spec.push_disk_center
            r = spec.push_disk_radius / np.sqrt(2)
            sub.receiver_zone["plate_1_joint0"] = ((c[0] - r, c[0] + r),
                                                   (c[1] - r, c[1] + r))
        else:
            # non-disk seen plates get a modestly widened zone: the 4 unseen
            # plates' r=0.08 exclusion circles (0.080 m^2) nearly saturate the
            # default plate zone (0.083 m^2) -- the seed-47 run starved at the
            # 7th sampled seen layout for exactly this reason
            sub.receiver_zone["plate_1_joint0"] = ((-0.12, 0.13), (-0.20, 0.20))
        exc = build_exclusions(unseen(), spec.unseen_seen_min_dist)
        for l in [x for x in layouts if x["role"] == "seen"]:
            for fb in G.GOAL_FIXTURES:
                exc.setdefault(fb, []).append((list(l["layout"]["fixtures"][fb]),
                                               spec.seen_fixture_min_dist))
        cand = sample_candidate(sub, exc)
        if cand is None:
            continue
        assert dist_ok(cand, unseen(), spec.unseen_seen_min_dist)
        try_accept(cand, "seen", push_feasible=need_disk and in_push_disk(cand),
                   extra_gate_tasks=[])
    assert len(seen_sampled()) == args.n_seen_sampled, \
        f"only {len(seen_sampled())} sampled seen after {tries} tries"
    env.close()
    for genv, *_ in gate_envs.values():
        genv.close()

    # ---- split + matrices ----
    seen_ids = [l["id"] for l in layouts if l["role"] == "seen"]
    unseen_ids = [l["id"] for l in layouts if l["role"] == "unseen"]
    push_train = [l["id"] for l in layouts if l["push_feasible"]]

    # 8 non-push tasks x 8 seen layouts: 4 train each, per-layout count exactly 4
    mat_rng = np.random.default_rng(args.seed + 1000)
    for _ in range(100000):
        matrix = {t: sorted(mat_rng.choice(len(seen_ids), 4, replace=False).tolist())
                  for t in GATE_TASKS}
        counts = np.zeros(len(seen_ids), int)
        for t in GATE_TASKS:
            counts[matrix[t]] += 1
        if counts.min() == 4 and counts.max() == 4:
            break
    train_matrix = {t: [seen_ids[j] for j in matrix[t]] for t in GATE_TASKS}
    train_matrix[PUSH] = push_train
    cf_matrix = {t: [sid for sid in seen_ids if sid not in train_matrix[t]]
                 for t in GATE_TASKS}
    cf_matrix[PUSH] = []   # push cf cells are infeasible by construction

    # leakage table: per unseen layout, per entity, min distance to ALL seen
    by_id = {l["id"]: l for l in layouts}
    leakage = {}
    for uid in unseen_ids:
        row = {}
        for key in ENTITY_KEYS:
            row[key] = round(min(entity_dists(by_id[uid]["layout"],
                                              by_id[sid]["layout"])[key]
                                 for sid in seen_ids), 4)
        leakage[uid] = row
    overall_floor = min(min(r.values()) for r in leakage.values())

    manifest = {
        "created": "2026-08-23", "version": "v2-unseen-first",
        "seed": args.seed, "matrix_seed": args.seed + 1000,
        "spec": {**{k: v for k, v in asdict(spec).items()},
                 "OBJECT_RADIUS": G.OBJECT_RADIUS,
                 "FIXTURE_CIRCLES": {k: [[list(c), r] for c, r in v]
                                     for k, v in G.FIXTURE_CIRCLES.items()},
                 "FIXTURE_OCC_LATERAL": G.FIXTURE_OCC_LATERAL,
                 "drawer_sweep_keepout_joints": list(G.DRAWER_SWEEP_KEEPOUT_JOINTS)},
        "gates": {"order": ["geometry", "distance floors", "settle(conv+disp<1cm+no articulation drift)",
                            f"visibility(px>={spec.min_px} x7)",
                            f"replay({len(GATE_TASKS)} tasks x <= {args.gate_sources} sources, end-state)",
                            "push replay (disk layouts only)"],
                  "reject_counts": rejects},
        "replay_coverage": {"verified_per_layout": GATE_TASKS,
                            "push_verified_on": push_train,
                            "note": "all 8 non-push tasks replay-verified on every "
                                    "layout; push verified only on its train cells"},
        "layouts": layouts,
        "seen_ids": seen_ids, "unseen_ids": unseen_ids,
        "leakage_per_unseen_per_entity_min_dist_to_all_seen": leakage,
        "overall_unseen_seen_floor_m": round(overall_floor, 4),
        "floors_spec": {"unseen_seen_min_dist": spec.unseen_seen_min_dist,
                        "unseen_internal_min_dist": spec.unseen_internal_min_dist},
        "train_matrix": train_matrix,
        "cf_matrix": cf_matrix,
        "push_infeasible_cells": [sid for sid in seen_ids + unseen_ids
                                  if sid not in push_train],
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(4 * 3.3, 3 * 3.5))
    for ax, rec, img in zip(axes.flat, layouts, imgs):
        ax.imshow(img)
        u = rec["role"] == "unseen"
        ax.set_title(f"{rec['id']}{' (nominal)' if rec.get('nominal') else ''}"
                     f"{' UNSEEN' if u else ''}{' push' if rec['push_feasible'] else ''}",
                     fontsize=10, color="tab:red" if u else "black")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"goal suite v2: 8 seen + 4 unseen, per-entity all-seen floor "
                 f"{overall_floor:.3f} m (required {spec.unseen_seen_min_dist})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(args.out_dir, "layouts_12.png"), dpi=110,
                bbox_inches="tight")
    print(f"\nDONE: floor={overall_floor:.4f} (required {spec.unseen_seen_min_dist}); "
          f"rejects={rejects}; push train cells={push_train}")


if __name__ == "__main__":
    main()
