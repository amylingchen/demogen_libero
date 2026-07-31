"""Add episodes for extra task pairs to an EXISTING paired split plan.

The first plan was built in one shot, so adding a pair by re-running
build_pair_split.py would renumber every scene and invalidate the
`scene_index` recorded in the scene_log of the ~485 demos already generated
(patch_fixture_objects.py and build_pair_eval_suite.py both dereference it).
This script is append-only instead: new geometries go on the end of
`plan["scenes"]`, so every existing index still points at the same geometry, and
the new episodes carry a `batch` tag so generation can be told to use only them.

Its default target is the pairs whose two relations name the SAME anchor in
different ways -- one bowl ON the ramekin / cookie box, the other NEXT TO it.
The planar exclusion rule that built the first plan rejected those (naming points
0.126 m and 0.117 m apart, under the 0.24 m radius) because it compares
positions, and on-vs-beside differs in SUPPORT, not position. The two bowls still
end up 0.117-0.126 m apart, which is more than their measured 0.056 m radii.

Usage:
    .venv\\Scripts\\python.exe scripts\\add_pair_episodes.py --batch shared_anchor
    .venv\\Scripts\\python.exe scripts\\add_pair_episodes.py --batch shared_anchor \\
        --train-scenes 10 --unseen-scenes 6 --dry-run
"""
import argparse
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import h5py
import numpy as np

import screen_spatial_pairs as P
from build_pair_split import Board, scene_to_dict
from demogen_libero.convert import resolve_bddl_file
from demogen_libero import libero_replay as R, oc_obs, spatial_scene as S


def pick(pool, task, role_of, quota, want_unseen, used, rng, cell_n=None):
    """Choose `quota` geometries for one task, spread over checkerboard cells.

    Cell spread is the objective that matters here: a pair confines both bowls to
    a fixed offset from one shared anchor, so uniform draws of the anchor pile the
    target into a few cells, and "10 different positions" would quietly become
    three positions and seven near-duplicates.

    want_unseen None means either parity, used for the spare positions.
    """
    cell_n = Counter() if cell_n is None else cell_n
    taken = []
    cand = [i for i in range(len(pool)) if i not in used]
    rng.shuffle(cand)
    while len(taken) < quota:
        best, best_key = None, None
        for i in cand:
            if i in used:
                continue
            s = pool[i]
            r = role_of[task]
            if want_unseen is not None and s[f"unseen_{r}"] != want_unseen:
                continue
            key = cell_n[tuple(s[f"cell_{r}"])]
            if best_key is None or key < best_key:
                best, best_key = (i, r), key
                if key == 0:
                    break
        if best is None:
            break
        i, r = best
        used.add(i)
        cell_n[tuple(pool[i][f"cell_{r}"])] += 1
        taken.append({"scene_local": i, "target_role": r, "task": task})
    return taken


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="output/pair_split.json")
    ap.add_argument("--batch", default="shared_anchor",
                    help="tag written on every added episode; pass the same value to "
                         "run_pair_oc_demo.py --batch")
    ap.add_argument("--pair", action="append", default=None, metavar="A+B",
                    help="explicit short-name pair, repeatable; default = every pair "
                         "the on-vs-beside rule admits")
    ap.add_argument("--positions", type=int, default=10,
                    help="distinct placements sampled per task; whatever is left "
                         "after train and unseen is recorded as `spare`")
    ap.add_argument("--train-scenes", type=int, default=3, help="per task")
    ap.add_argument("--unseen-scenes", type=int, default=3, help="per task")
    ap.add_argument("--pool-per-pair", type=int, default=400)
    ap.add_argument("--max-draws", type=int, default=400000, help="per-pair budget")
    ap.add_argument("--exclusion", type=float, default=0.24)
    ap.add_argument("--max-yaw-object", type=float, default=None)
    ap.add_argument("--cache", default="output/spatial_pairs/layout_cache.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--promote", default=None, metavar="TASK",
                    help="promote spare positions of this short task name to --split "
                         "instead of sampling anything new; use when a planned scene "
                         "turned out to be ungeneratable (e.g. its anchor is occluded "
                         "from every jitter)")
    ap.add_argument("--promote-n", type=int, default=1)
    ap.add_argument("--split", default="train", choices=["train", "unseen"],
                    help="target split for --promote")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be added without touching the plan")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    plan = json.load(open(args.plan))
    cfg = S.load_spatial_config()
    tasks = list(cfg)
    rel = P.relations_from_config(cfg)
    cache = json.load(open(args.cache))["cache"]
    board = Board(plan["board"]["n"])

    already = {e["batch"] for e in plan["episodes"] if e.get("batch")}

    # ------------------------------------------------------ promote spares only
    if args.promote:
        want_unseen = args.split == "unseen"
        spares = [e for e in plan["episodes"]
                  if e.get("split") == "spare" and e.get("batch") == args.batch
                  and P.short(e["task"]) == args.promote
                  and bool(e["target_in_unseen_cell"]) == want_unseen]
        if not spares:
            raise SystemExit(f"no spare {args.split}-eligible position left for "
                             f"{args.promote} in batch {args.batch!r}")
        for e in spares[:args.promote_n]:
            e["split"] = args.split
            print(f"promoted scene {e['scene_index']} cell {e['target_cell']} "
                  f"-> {args.split} for {args.promote}")
            if args.split == "train":
                # a training geometry always supplies one counterfactual episode:
                # same scene, other bowl as target, the other task's instruction.
                # Every target-dependent field has to be re-read for the OTHER role
                # -- copying them would label the counterfactual with the training
                # bowl's position and cell.
                s = plan["scenes"][e["scene_index"]]
                r = "b" if e["target_role"] == "a" else "a"
                other = [t for t in e["pair"] if t != e["task"]][0]
                plan["episodes"].append({
                    **e, "split": "counterfact", "task": other, "target_role": r,
                    "target_xy": s[f"pos_{r}"], "target_cell": s[f"cell_{r}"],
                    "target_in_unseen_cell": s[f"unseen_{r}"],
                    "boundary_margin": s[f"boundary_margin_{r}"],
                })
                print(f"  + counterfactual episode for {P.short(other)} "
                      f"(cell {s[f'cell_{r}']})")
        if args.dry_run:
            print("--dry-run: plan not modified")
            return
        json.dump(plan, open(args.plan, "w"))
        print(f"wrote {args.plan}")
        return

    if args.batch in already:
        raise SystemExit(
            f"plan already contains a batch named {args.batch!r} -- pick another name, "
            f"or generation would count those episodes as this batch's progress")

    # naming points measured the same way the screen measures them, so this script
    # can never disagree with it about where a relation holds
    rel_points = P.measure_relation_points(cache, rel)
    short2task = {P.short(t): t for t in tasks}
    if args.pair:
        pairs = []
        for spec in args.pair:
            a, b = [s.strip() for s in spec.replace("+", " ").split()]
            pairs.append((short2task[a], short2task[b]))
    else:
        pairs = [(a, b) for a, b in itertools.combinations(tasks, 2)
                 if P.compat(a, b, rel, rel_points, args.exclusion,
                             allow_shared_anchor=True)[0] == "stacked_vs_beside"]
    if not pairs:
        raise SystemExit("no pairs to add")
    print("pairs to add:")
    for a, b in pairs:
        print(f"  {P.short(a)} + {P.short(b)}")

    # the sampler needs the same measured geometry and relation points the screen used
    with h5py.File(os.path.join(P.DATA_DIR, f"{P.HOST_TASK}_demo.hdf5"), "r") as f:
        bddl = resolve_bddl_file(f["data"].attrs["bddl_file_name"])
        init = np.array(f["data"]["demo_0"]["states"][0], dtype=np.float64)
    env = oc_obs.make_oc_env(bddl)
    env.reset()
    R.reset_to_init_state(env, init)
    # measured live, never read back from pair_screen.json: that file was written by
    # an earlier definition of the prefilter radius (bowl 0.0789 m, the AABB
    # half-diagonal, against 0.0562 m tight), and the loose value exceeds the
    # 0.117-0.126 m at which two bowls sharing one anchor sit -- it would veto every
    # scene this script exists to build, for a reason that is not physical
    geom, _own, table = P.measure_scene_geometry(env)
    env.close()
    spec = P.Spec(exclusion=args.exclusion, max_yaw_object=args.max_yaw_object)
    spec.set_table(table)

    base_n = len(plan["scenes"])
    new_scenes, new_eps = [], []
    for ta, tb in pairs:
        rng = np.random.default_rng(args.seed)
        pool, stats = [], Counter()
        while len(pool) < args.pool_per_pair and stats["draws"] < args.max_draws:
            sc, _w = P.sample_pair_geometry(
                rng, ta, tb, cache, rel, geom, spec, rel_points,
                int(rng.integers(len(cache[ta]))), int(rng.integers(len(cache[tb]))),
                stats)
            if sc is None:
                continue
            pool.append(scene_to_dict(sc, board))
        acc = stats["draw_accept"] / max(stats["draws"], 1)
        print(f"\n{P.short(ta)} + {P.short(tb)}: {len(pool)} geometries "
              f"from {stats['draws']} draws (accept {100*acc:.2f}%)")
        if not pool:
            print("  ! no geometry sampled -- skipping this pair")
            continue
        seen_a = sum(not s["unseen_a"] for s in pool)
        seen_b = sum(not s["unseen_b"] for s in pool)
        print(f"  cells: role a {len({tuple(s['cell_a']) for s in pool})} distinct "
              f"({seen_a} seen), role b {len({tuple(s['cell_b']) for s in pool})} "
              f"distinct ({seen_b} seen)")

        role_of = {ta: "a", tb: "b"}
        used = set()
        rng2 = np.random.default_rng(args.seed + 1)
        # Per task: `--positions` distinct placements, cell-spread, of which
        # train_scenes go to train (seen cells) and unseen_scenes to unseen
        # (unseen cells). The remainder are recorded as `spare` -- never generated,
        # but already sampled and validated, so a train scene that no source demo
        # can solve can be replaced without re-planning.
        train, unseen, spare = [], [], []
        for task in (ta, tb):
            cell_n = Counter()          # shared, so train/unseen/spare do not collide
            got = pick(pool, task, role_of, args.train_scenes, False, used, rng2, cell_n)
            if len(got) < args.train_scenes:
                print(f"  ! {P.short(task)}: only {len(got)}/{args.train_scenes} "
                      f"train scenes (seen cells are the constraint)")
            train += got
            got_u = pick(pool, task, role_of, args.unseen_scenes, True, used, rng2, cell_n)
            if len(got_u) < args.unseen_scenes:
                print(f"  ! {P.short(task)}: only {len(got_u)}/{args.unseen_scenes} "
                      f"unseen scenes")
            unseen += got_u
            n_spare = max(args.positions - len(got) - len(got_u), 0)
            spare += pick(pool, task, role_of, n_spare, None, used, rng2, cell_n)
        # every training geometry also supplies one counterfactual episode: same
        # scene, other bowl as target, the other task's instruction
        cfact = [{"scene_local": t["scene_local"],
                  "target_role": "b" if t["target_role"] == "a" else "a",
                  "task": tb if t["task"] == ta else ta}
                 for t in train]

        # only geometries actually used are appended, so the plan does not grow by
        # the whole sampling pool
        keep = sorted({t["scene_local"] for t in train + unseen + spare})
        remap = {loc: base_n + len(new_scenes) + k for k, loc in enumerate(keep)}
        new_scenes += [pool[loc] for loc in keep]

        for items, split in ((train, "train"), (cfact, "counterfact"),
                             (unseen, "unseen"), (spare, "spare")):
            for it in items:
                s = pool[it["scene_local"]]
                r = it["target_role"]
                new_eps.append({
                    "split": split, "task": it["task"], "target_role": r,
                    "pair": [ta, tb],
                    "target_xy": s[f"pos_{r}"], "target_cell": s[f"cell_{r}"],
                    "target_in_unseen_cell": s[f"unseen_{r}"],
                    "boundary_margin": s[f"boundary_margin_{r}"],
                    "scene_index": remap[it["scene_local"]],
                    "batch": args.batch,
                })

    if not new_eps:
        raise SystemExit("nothing to add")

    print(f"\n{'task':<42}{'train':>7}{'cfact':>7}{'unseen':>8}{'spare':>7}"
          f"{'positions':>11}{'cells':>7}")
    by = defaultdict(lambda: defaultdict(list))
    for e in new_eps:
        by[e["task"]][e["split"]].append(e)
    for task in tasks:
        if task not in by:
            continue
        b = by[task]
        own = b["train"] + b["unseen"] + b["spare"]        # this task's own placements
        cells = len({tuple(e["target_cell"]) for e in own})
        print(f"{P.short(task):<42}{len(b['train']):>7}{len(b['counterfact']):>7}"
              f"{len(b['unseen']):>8}{len(b['spare']):>7}{len(own):>11}{cells:>7}")
    n_per = plan["conventions"]["demos_per_scene"]
    n_train = sum(1 for e in new_eps if e["split"] == "train")
    print(f"\nadded {len(new_scenes)} scenes, {len(new_eps)} episodes "
          f"(batch={args.batch!r})")
    print(f"train episodes {n_train} x {n_per} demos/scene = up to {n_train * n_per} "
          f"new train demos before the per-task --target-count stops generation")
    risky = sum(e["boundary_margin"] < 0.02 for e in new_eps)
    print(f"targets within 0.02 m of a cell boundary: {risky}/{len(new_eps)} "
          f"-> generation recomputes each demo's cell after jitter")

    if args.dry_run:
        print("\n--dry-run: plan not modified")
        return

    # the plan is load-bearing for demos already on disk (their scene_log stores
    # indices into it), and this rewrite is in place and not idempotent
    bak = args.plan + ".bak"
    if not os.path.exists(bak):
        with open(args.plan, "rb") as src, open(bak, "wb") as dst:
            dst.write(src.read())
        print(f"backed up the pre-existing plan to {bak}")

    plan["scenes"] += new_scenes
    plan["episodes"] += new_eps
    plan.setdefault("batches", {})[args.batch] = {
        "pairs": [[P.short(a), P.short(b)] for a, b in pairs],
        "params": vars(args),
        "scene_index_range": [base_n, base_n + len(new_scenes)],
        "n_episodes": len(new_eps),
        "reason": "on-vs-beside shared-anchor pairs, which the planar exclusion "
                  "rule in the first plan rejected as ambiguous",
    }
    json.dump(plan, open(args.plan, "w"))
    print(f"\nappended to {args.plan} "
          f"(scenes {base_n} -> {len(plan['scenes'])}, existing indices unchanged)")


if __name__ == "__main__":
    main()
