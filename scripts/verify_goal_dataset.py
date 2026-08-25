"""Packaging self-checks for the goal 12-layout dataset (plan §9-5).

Checks, all measured from the written hdf5s (never from the generation log):
 1. END-STATE re-verification: re-set the sim to each demo's final stored state
    and evaluate the task predicate. Catches latched-success leakage and any
    state/attr corruption. Sampled (--sample-per-cell) or full (--all).
 2. REALIZED leakage: for every TRAIN demo, min distance from each jittered
    entity to the same entity in every UNSEEN layout (and vice versa for the
    unseen demos). This is the honest replacement for the manifest's
    center-to-center 0.0809 m -- report the per-entity minimum over the actual
    data (round-2 review B5).
 3. Initial EE pose distribution: train vs unseen-eval, per task. History
    (libero_object): a regenerated eval side drifted 6.6 cm out of the training
    distribution and depressed the unseen readings; this reports the gap.
 4. Split hygiene: every demo's layout_id agrees with the manifest matrices;
    no train demo sits on an unseen layout; cf demos are on seen layouts only.
 5. Attr completeness: source_demo / jitter / robot_noise / fixture_edits
    present and parseable on every demo.

Usage:
    .venv\\Scripts\\python.exe scripts\\verify_goal_dataset.py --dir output/goal_gen_500
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

from demogen_libero import libero_replay as R
from demogen_libero import oc_obs
from demogen_libero import spatial_scene as S
from demogen_libero import goal_scene as G

from libero.libero import get_libero_path

SPLITS = ["train", "quarantine_cf", "quarantine_unseen"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("output", "goal_gen_500"))
    ap.add_argument("--manifest", default=os.path.join("output", "goal_suite_12", "manifest.json"))
    ap.add_argument("--sample-per-cell", type=int, default=2,
                    help="demos per cell re-verified in sim (--all overrides)")
    ap.add_argument("--all", action="store_true", help="re-verify every demo")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    man = json.load(open(args.manifest))
    by_id = {l["id"]: l for l in man["layouts"]}
    unseen_ids, seen_ids = man["unseen_ids"], man["seen_ids"]
    report = {"end_state": {}, "leakage": {}, "ee_pose": {}, "hygiene": [],
              "attrs": [], "counts": {}}

    # ---- gather demo index ----
    index = defaultdict(list)   # task -> [(split, key, attrs)]
    for split in SPLITS:
        for p in sorted(glob.glob(os.path.join(args.dir, split, "*.hdf5"))):
            task = os.path.basename(p).replace("_demo.hdf5", "")
            with h5py.File(p, "r") as f:
                for key in f["data"]:
                    a = dict(f["data"][key].attrs)
                    index[task].append((split, p, key, a))
    report["counts"] = {t: {s: sum(1 for x in v if x[0] == s) for s in SPLITS}
                        for t, v in index.items()}
    total = sum(len(v) for v in index.values())
    print(f"indexed {total} demos across {len(index)} tasks", flush=True)

    # ---- 4 & 5: hygiene + attrs (no sim needed) ----
    for task, rows in index.items():
        for split, p, key, a in rows:
            lid = a.get("layout_id")
            if lid is None:
                report["hygiene"].append(f"{task}/{split}/{key}: missing layout_id")
                continue
            if split == "train" and lid not in man["train_matrix"].get(task, []):
                report["hygiene"].append(f"{task}/{key}: train demo on {lid} "
                                         f"not in train_matrix")
            if split == "quarantine_cf" and lid not in man["cf_matrix"].get(task, []):
                report["hygiene"].append(f"{task}/{key}: cf demo on {lid} not in cf_matrix")
            if split == "quarantine_unseen" and lid not in unseen_ids:
                report["hygiene"].append(f"{task}/{key}: unseen demo on seen layout {lid}")
            for field in ("source_demo", "jitter_objects", "jitter_fixtures",
                          "robot_noise", "fixture_edits", "split"):
                if field not in a:
                    report["attrs"].append(f"{task}/{split}/{key}: missing {field}")
            try:
                json.loads(a["jitter_objects"]); json.loads(a["jitter_fixtures"])
                json.loads(a["fixture_edits"])
            except Exception as exc:
                report["attrs"].append(f"{task}/{split}/{key}: unparseable json {exc!r}")
    print(f"hygiene issues: {len(report['hygiene'])}; attr issues: {len(report['attrs'])}",
          flush=True)

    # ---- 2: realized leakage from the actual jittered placements ----
    def ent_pos(layout_dict, key):
        grp = "objects" if key.endswith("_joint0") else "fixtures"
        return np.asarray(layout_dict[grp][key])

    worst = {}   # entity -> (dist, task, split, key, other_lid)
    for task, rows in index.items():
        for split, p, key, a in rows:
            jl = {"objects": json.loads(a["jitter_objects"]),
                  "fixtures": json.loads(a["jitter_fixtures"])}
            others = unseen_ids if split != "quarantine_unseen" else seen_ids
            for ekey in list(G.GOAL_JOINTS) + list(G.GOAL_FIXTURES):
                mine = ent_pos(jl, ekey)
                for oid in others:
                    d = float(np.linalg.norm(mine - ent_pos(by_id[oid]["layout"], ekey)))
                    if ekey not in worst or d < worst[ekey][0]:
                        worst[ekey] = (round(d, 4), task, split, key, oid)
    report["leakage"] = {"note": "min over ALL demos of distance from the demo's "
                                 "jittered entity to the same entity in every "
                                 "layout on the opposite split side; replaces the "
                                 "manifest's center-to-center floor",
                         "per_entity_worst": {k: {"dist_m": v[0], "task": v[1],
                                                  "split": v[2], "demo": v[3],
                                                  "vs_layout": v[4]}
                                              for k, v in worst.items()},
                         "overall_min_m": min(v[0] for v in worst.values())}
    print(f"realized leakage min = {report['leakage']['overall_min_m']} m", flush=True)

    # ---- 3: initial EE pose, train vs unseen ----
    for task, rows in index.items():
        pos = {"train": [], "quarantine_unseen": []}
        for split, p, key, a in rows:
            if split not in pos:
                continue
            with h5py.File(p, "r") as f:
                pos[split].append(np.array(f["data"][key]["obs"]["ee_pos"][0]))
        if pos["train"] and pos["quarantine_unseen"]:
            tr = np.array(pos["train"]); un = np.array(pos["quarantine_unseen"])
            report["ee_pose"][task] = {
                "train_mean": np.round(tr.mean(0), 4).tolist(),
                "train_std": np.round(tr.std(0), 4).tolist(),
                "unseen_mean": np.round(un.mean(0), 4).tolist(),
                "unseen_std": np.round(un.std(0), 4).tolist(),
                "mean_gap_cm": round(float(np.linalg.norm(tr.mean(0) - un.mean(0))) * 100, 2)}
    gaps = {t: v["mean_gap_cm"] for t, v in report["ee_pose"].items()}
    print(f"init-EE train-vs-unseen mean gaps (cm): {gaps}", flush=True)

    # ---- 1: end-state re-verification in sim ----
    for task, rows in index.items():
        cells = defaultdict(list)
        for r in rows:
            cells[(r[0], r[3].get("layout_id"))].append(r)
        pick = []
        for cell_rows in cells.values():
            if args.all:
                pick += cell_rows
            else:
                idx = rng.permutation(len(cell_rows))[:args.sample_per_cell]
                pick += [cell_rows[i] for i in idx]
        env = oc_obs.make_oc_env(os.path.join(
            get_libero_path("bddl_files"), "libero_goal", f"{task}.bddl"))
        env.reset()
        ok = bad = 0
        bad_list = []
        for split, p, key, a in pick:
            with h5py.File(p, "r") as f:
                final = np.array(f["data"][key]["states"][-1], dtype=np.float64)
            fx = {fb: {"pos": np.asarray(e["pos"]),
                       "quat": np.asarray(e["quat_wxyz"])}
                  for fb, e in json.loads(a["fixture_edits"]).items()}
            R.reset_to_init_state(env, final)
            S.apply_fixture_edits(env, fx)
            env.sim.forward()
            env.env._post_process()   # refresh predicate state from the sim
            if bool(env.check_success()):
                ok += 1
            else:
                bad += 1
                bad_list.append(f"{split}/{key}@{a.get('layout_id')}")
        env.close()
        report["end_state"][task] = {"checked": len(pick), "pass": ok, "fail": bad,
                                     "failures": bad_list[:20]}
        print(f"[end-state] {task[:40]:40s} {ok}/{len(pick)} pass"
              + (f"  FAIL: {bad_list[:6]}" if bad_list else ""), flush=True)

    out = os.path.join(args.dir, "verification_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    n_fail = sum(v["fail"] for v in report["end_state"].values())
    n_chk = sum(v["checked"] for v in report["end_state"].values())
    print(f"\n== end-state {n_chk - n_fail}/{n_chk} pass; hygiene "
          f"{len(report['hygiene'])} issues; attrs {len(report['attrs'])} issues; "
          f"realized leakage min {report['leakage']['overall_min_m']} m ==")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
