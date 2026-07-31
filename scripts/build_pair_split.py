"""Build the train / counterfactual / unseen split for paired spatial scenes.

One paired geometry yields TWO episodes, because the two bowls are the same
asset: whichever bowl the instruction refers to is placed as `akita_black_bowl_1`
(every task's BDDL goal is `(On akita_black_bowl_1 plate_1)`, and the OC format
keys seg id 60 on it), and the other bowl takes the remaining pose. Swapping the
two poses therefore changes the target without touching a single BDDL, config or
seg-id convention.

Splits produced, all keyed on a SHARED 4x4 checkerboard over the target
workspace (measured: a shared board already lands every task between 41% and 59%
seen, so no per-task board is needed):

  train        target in a seen cell
  counterfact  the SAME geometry, other bowl as target, other task's instruction.
               The scene was in training and only the instruction changes; the
               target's own cell is NOT constrained, but it is recorded so the
               set can be sliced afterwards into "target cell also seen" (~44%)
               and "target cell unseen" (~56%) at no cost.
  unseen       target in an unseen cell, from geometries not used for training

Role assignment (which bowl is the train target) is balanced PER TASK, not left
to pair enumeration order: enumeration order alone starves whole tasks -- it
gives on_the_wooden_cabinet and in_the_top_drawer zero training cells while
next_to_the_ramekin, on_the_ramekin and between get zero counterfactual ones.

This script only plans. It writes scene geometries and split labels; nothing is
rendered and no trajectory is synthesised.

Usage:
    .venv\\Scripts\\python.exe scripts\\build_pair_split.py
    .venv\\Scripts\\python.exe scripts\\build_pair_split.py --train-scenes 25 --unseen-scenes 10
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
from demogen_libero.convert import resolve_bddl_file
from demogen_libero import libero_replay as R, oc_obs, spatial_scene as S


# --------------------------------------------------------------- checkerboard

class Board:
    """Shared 4x4 checkerboard over the sampler's target workspace."""

    def __init__(self, n=4):
        sp = S.SpatialSpec()
        self.lo = np.array([sp.x_range[0], sp.y_range[0]])
        self.cw = (np.array([sp.x_range[1], sp.y_range[1]]) - self.lo) / n
        self.n = n

    def cell(self, xy):
        return tuple(int(v) for v in
                     np.clip(np.floor((np.asarray(xy) - self.lo) / self.cw), 0, self.n - 1))

    def is_unseen(self, xy):
        c = self.cell(xy)
        return (c[0] + c[1]) % 2 == 1

    def boundary_margin(self, xy):
        """Distance from `xy` to the nearest cell boundary. A demo whose jitter
        can carry it across a boundary must have its cell recomputed rather than
        inherit the base scene's label."""
        f = ((np.asarray(xy) - self.lo) / self.cw) % 1.0
        return float(np.min(np.minimum(f, 1 - f) * self.cw))


# ------------------------------------------------------------- serialisation

def scene_to_dict(sc, board):
    """Everything generation needs to rebuild this geometry, plus both target
    options and their cells."""
    def member(m):
        return {"xy": [float(m["xy"][0]), float(m["xy"][1])], "z": float(m["z"]),
                "quat": [float(q) for q in m["quat"]], "is_fixture": bool(m["is_fixture"])}

    free, fixtures = {}, {}
    for g in (sc["gA"], sc["gB"]):
        for k, m in g.items():
            (fixtures if m["is_fixture"] else free)[k] = member(m)
    for k, m in sc["indep"].items():
        free[k] = member(m)
    for st in sc["stationary"]:
        fixtures[st["key"]] = {"xy": [float(st["xy"][0]), float(st["xy"][1])],
                               "z": float(st["z"]),
                               "quat": [float(q) for q in st["quat"]],
                               "is_fixture": True}
    pa = sc["gA"][P.BOWL_A]["xy"]
    pb = sc["gB"][P.BOWL_B]["xy"]
    return {
        "task_a": sc["taskA"], "task_b": sc["taskB"],
        "source_demo_a": int(sc["demo_a"]), "source_demo_b": int(sc["demo_b"]),
        "drawer_qpos": float(sc.get("drawer", 0.0)),
        "free": free, "fixtures": fixtures,
        "pos_a": [float(pa[0]), float(pa[1])],
        "pos_b": [float(pb[0]), float(pb[1])],
        "cell_a": list(board.cell(pa)), "cell_b": list(board.cell(pb)),
        "unseen_a": bool(board.is_unseen(pa)), "unseen_b": bool(board.is_unseen(pb)),
        "boundary_margin_a": board.boundary_margin(pa),
        "boundary_margin_b": board.boundary_margin(pb),
    }


# ----------------------------------------------------------------- selection

def choose_pairs(screen_path, max_cost, drop):
    """Candidate pairs ranked by sampling cost, cheapest first."""
    res = json.load(open(screen_path))["results"]
    out = []
    for k, v in res.items():
        a, u = v["draw_accept"], v["usable_rate"]
        if a <= 0 or u <= 0:
            continue
        cost = (1 / a) / u
        if cost > max_cost:
            continue
        if any(d in k for d in drop):
            continue
        out.append((cost, k))
    return [k for _, k in sorted(out)]


def assign(pool, tasks, board, n_train, n_unseen, rng):
    """Pick train scenes, then unseen scenes, under three balance objectives at
    once. A geometry is used at most once: taking it for training also fixes its
    counterfactual episode, so it can never reappear as an unseen test.

      per pair    a task's scenes stay spread over the pairs it belongs to
      per cell    and over the checkerboard cells it can reach, so 25 training
                  scenes do not pile into two cells
      per partner the OTHER task of each chosen pair receives the counterfactual
                  episode, so partner counts are equalised too -- balancing only
                  the training side leaves tasks with 4 counterfactual episodes
                  and others with 51
    """
    cand = defaultdict(list)                    # task -> [(idx, role, partner)]
    for i, s in enumerate(pool):
        cand[s["task_a"]].append((i, "a", s["task_b"]))
        cand[s["task_b"]].append((i, "b", s["task_a"]))
    for v in cand.values():
        rng.shuffle(v)

    used, cfact_count = set(), Counter()

    def pick(task, quota, want_unseen, per_task_state):
        pair_n, cell_n = per_task_state
        taken = []
        while len(taken) < quota:
            best, best_key = None, None
            for idx, role, partner in cand[task]:
                if idx in used:
                    continue
                s = pool[idx]
                if s[f"unseen_{role}"] != want_unseen:
                    continue
                pr = (s["task_a"], s["task_b"])
                key = (pair_n[pr], cell_n[tuple(s[f"cell_{role}"])],
                       cfact_count[partner] if not want_unseen else 0)
                if best_key is None or key < best_key:
                    best, best_key = (idx, role, partner, pr), key
                    if key == (0, 0, 0):
                        break
            if best is None:
                break
            idx, role, partner, pr = best
            used.add(idx)
            pair_n[pr] += 1
            cell_n[tuple(pool[idx][f"cell_{role}"])] += 1
            if not want_unseen:
                cfact_count[partner] += 1
            taken.append({"scene": idx, "target_role": role, "task": task})
        return taken

    train, unseen = [], []
    # scarcest task first: a task present in few pairs has the least room
    order = sorted(tasks, key=lambda t: len({(pool[i]["task_a"], pool[i]["task_b"])
                                             for i, _, _ in cand[t]}))
    for task in order:
        got = pick(task, n_train, False, (Counter(), Counter()))
        train += got
        if len(got) < n_train:
            print(f"  ! {P.short(task)}: only {len(got)}/{n_train} train scenes")
    for task in order:
        got = pick(task, n_unseen, True, (Counter(), Counter()))
        unseen += got
        if len(got) < n_unseen:
            print(f"  ! {P.short(task)}: only {len(got)}/{n_unseen} unseen scenes")
    return train, unseen


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-scenes", type=int, default=25, help="per task")
    ap.add_argument("--unseen-scenes", type=int, default=10, help="per task")
    ap.add_argument("--demos-per-scene", type=int, default=2)
    ap.add_argument("--pool-per-pair", type=int, default=400,
                    help="geometries sampled per pair to select from")
    ap.add_argument("--max-cost", type=float, default=500.0,
                    help="drop pairs needing more than this many draws per scene "
                         "(costs in the screen were measured WITHOUT reverse plate "
                         "sampling, which speeds plate-owning pairs up 4-7x)")
    ap.add_argument("--drop", nargs="*", default=[],
                    help="substrings; any pair containing one is excluded")
    ap.add_argument("--max-draws", type=int, default=200000, help="per-pair budget")
    ap.add_argument("--screen", default="output/spatial_pairs/pair_screen.json")
    ap.add_argument("--cache", default="output/spatial_pairs/layout_cache.json")
    ap.add_argument("--exclusion", type=float, default=0.24)
    ap.add_argument("--max-yaw-object", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output/pair_split.json")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    cfg = S.load_spatial_config()
    tasks = list(cfg)
    rel = P.relations_from_config(cfg)
    cache = json.load(open(args.cache))["cache"]
    board = Board()

    wanted = choose_pairs(args.screen, args.max_cost, args.drop)
    short2task = {P.short(t): t for t in tasks}
    pairs = []
    for key in wanted:
        a, b = [s.strip() for s in key.split(" + ")]
        if a in short2task and b in short2task:
            pairs.append((short2task[a], short2task[b]))
    print(f"pairs in use: {len(pairs)} (cost <= {args.max_cost:.0f} draws/scene"
          + (f", dropped anything matching {args.drop}" if args.drop else "") + ")")

    with h5py.File(os.path.join(P.DATA_DIR, f"{P.HOST_TASK}_demo.hdf5"), "r") as f:
        bddl = resolve_bddl_file(f["data"].attrs["bddl_file_name"])
        init = np.array(f["data"]["demo_0"]["states"][0], dtype=np.float64)
    env = oc_obs.make_oc_env(bddl)
    env.reset()
    R.reset_to_init_state(env, init)
    geom, _own, table = P.measure_scene_geometry(env)
    env.close()
    rel_points = P.measure_relation_points(cache, rel)
    spec = P.Spec(exclusion=args.exclusion, max_yaw_object=args.max_yaw_object)
    spec.set_table(table)

    print(f"\nsampling up to {args.pool_per_pair} geometries per pair...")
    pool = []
    for ta, tb in pairs:
        rng = np.random.default_rng(args.seed)
        stats, got = Counter(), 0
        while got < args.pool_per_pair and stats["draws"] < args.max_draws:
            sc, _w = P.sample_pair_geometry(
                rng, ta, tb, cache, rel, geom, spec, rel_points,
                int(rng.integers(len(cache[ta]))), int(rng.integers(len(cache[tb]))),
                stats)
            if sc is None:
                continue
            pool.append(scene_to_dict(sc, board))
            got += 1
        print(f"  {P.short(ta)} + {P.short(tb)}: {got}", flush=True)

    print(f"\npool: {len(pool)} geometries -> {2 * len(pool)} candidate episodes")
    rng = np.random.default_rng(args.seed + 1)
    train, unseen = assign(pool, tasks, board, args.train_scenes, args.unseen_scenes, rng)

    # every training geometry also supplies one counterfactual episode: same
    # scene, other bowl as target, the other task's instruction
    counterfact = [{"scene": t["scene"],
                    "target_role": "b" if t["target_role"] == "a" else "a",
                    "task": (pool[t["scene"]]["task_b"] if t["target_role"] == "a"
                             else pool[t["scene"]]["task_a"])}
                   for t in train]

    def enrich(items, split):
        out = []
        for it in items:
            s = pool[it["scene"]]
            r = it["target_role"]
            out.append({
                "split": split, "task": it["task"], "target_role": r,
                "pair": [s["task_a"], s["task_b"]],
                "target_xy": s[f"pos_{r}"], "target_cell": s[f"cell_{r}"],
                "target_in_unseen_cell": s[f"unseen_{r}"],
                "boundary_margin": s[f"boundary_margin_{r}"],
                "scene_index": it["scene"],
            })
        return out

    plan = {
        "board": {"n": board.n, "cell_size": board.cw.tolist(),
                  "origin": board.lo.tolist(),
                  "rule": "unseen iff (cx + cy) is odd; shared across all tasks"},
        "params": vars(args),
        "conventions": {
            "target_role": "'a' = bowl at the first task's relation, 'b' = the second; "
                           "at generation the chosen bowl is written as akita_black_bowl_1 "
                           "and the other takes the remaining pose",
            "demos_per_scene": args.demos_per_scene,
            "jitter_rule": "recompute the checkerboard cell from each demo's own "
                           "jittered target position; if it disagrees with the scene's "
                           "split, redraw the jitter",
        },
        "scenes": pool,
        "episodes": enrich(train, "train") + enrich(counterfact, "counterfact")
                    + enrich(unseen, "unseen"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(plan, open(args.out, "w"))

    # ------------------------------------------------------------- report
    print(f"\n{'task':<42}{'train':>7}{'cfact':>7}{'unseen':>8}"
          f"{'train cells':>13}{'pairs used':>12}")
    by = defaultdict(lambda: defaultdict(list))
    for e in plan["episodes"]:
        by[e["task"]][e["split"]].append(e)
    for t in tasks:
        b = by[t]
        cells = len({tuple(e["target_cell"]) for e in b["train"]})
        prs = len({tuple(e["pair"]) for e in b["train"]})
        print(f"{P.short(t):<42}{len(b['train']):>7}{len(b['counterfact']):>7}"
              f"{len(b['unseen']):>8}{cells:>13}{prs:>12}")
    n_ep = len(plan["episodes"])
    cf = [e for e in plan["episodes"] if e["split"] == "counterfact"]
    cf_unseen = sum(e["target_in_unseen_cell"] for e in cf)
    risky = sum(e["boundary_margin"] < 0.02 for e in plan["episodes"])
    print(f"\nepisodes: {n_ep}  x {args.demos_per_scene} demos = "
          f"{n_ep * args.demos_per_scene} demos to generate")
    print(f"counterfact whose target cell is ALSO unseen: {cf_unseen}/{len(cf)} "
          f"({100 * cf_unseen / max(len(cf), 1):.0f}%) -- labelled, not excluded")
    print(f"targets within 0.02 m of a cell boundary: {risky}/{n_ep} "
          f"-> recompute each demo's cell after jitter")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
