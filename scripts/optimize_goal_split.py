"""Task-aware seen/unseen split + train-matrix optimization over the 12
sampled goal layouts (pure math over manifest.json, no sim).

The v1 guard (every unseen fixture >=0.10m from ALL 8 seen layouts' same
fixture) is geometrically infeasible: four r=0.10 exclusion circles outsize
the cabinet corridor itself (best achievable floor 0.031m). But task T only
TRAINS on its 4 assigned layouts -- the leakage that matters is
unseen-anchor <-> TRAIN-layout-anchor, and the assignment is ours to choose.
So: exhaustively try each 4-of-11 unseen combo (L00 always seen); per task
rank seen layouts by anchor distance to the unseen side and take the top 4 as
train cells; repair the per-layout train-count balance (plan §5: 4~5 tasks
per layout); score = min over tasks of the task's unseen<->train anchor
floor. Rewrites manifest with the optimized split + per-task floors and
re-renders the mosaic.

Usage:
    .venv\\Scripts\\python.exe scripts\\optimize_goal_split.py --dir output/goal_layouts_12
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

# anchor entities per task (layout dict keys). Push anchors only on the plate:
# its goal region is table-fixed (does not follow the stove).
TASK_ANCHORS = {
    "open_the_middle_drawer_of_the_cabinet": [("fixtures", "wooden_cabinet_1_main")],
    "turn_on_the_stove": [("fixtures", "flat_stove_1_main")],
    "put_the_bowl_on_the_plate": [("objects", "akita_black_bowl_1_joint0"),
                                  ("objects", "plate_1_joint0")],
    "put_the_bowl_on_the_stove": [("objects", "akita_black_bowl_1_joint0"),
                                  ("fixtures", "flat_stove_1_main")],
    "put_the_bowl_on_top_of_the_cabinet": [("objects", "akita_black_bowl_1_joint0"),
                                           ("fixtures", "wooden_cabinet_1_main")],
    "put_the_cream_cheese_in_the_bowl": [("objects", "cream_cheese_1_joint0"),
                                         ("objects", "akita_black_bowl_1_joint0")],
    "put_the_wine_bottle_on_the_rack": [("objects", "wine_bottle_1_joint0"),
                                        ("fixtures", "wine_rack_1_main")],
    "put_the_wine_bottle_on_top_of_the_cabinet": [("objects", "wine_bottle_1_joint0"),
                                                  ("fixtures", "wooden_cabinet_1_main")],
    "push_the_plate_to_the_front_of_the_stove": [("objects", "plate_1_joint0")],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("output", "goal_layouts_12"))
    ap.add_argument("--n-unseen", type=int, default=4)
    ap.add_argument("--render", action="store_true", default=True)
    args = ap.parse_args()

    man_path = os.path.join(args.dir, "manifest.json")
    man = json.load(open(man_path))
    all_layouts = man["layouts"]
    tasks = list(TASK_ANCHORS)

    def pos(i, group, key):
        return np.asarray(all_layouts[i]["layout"][group][key])

    def anchor_dist(t, i, j):
        return min(float(np.linalg.norm(pos(i, g, k) - pos(j, g, k)))
                   for g, k in TASK_ANCHORS[t])

    # when more than 12 layouts exist (targeted append), also search over
    # which 12 to keep: drop k of the non-nominal ones, exhaustively
    n_all = len(all_layouts)
    non_nominal = [i for i in range(n_all) if not all_layouts[i].get("nominal")]
    k_drop = n_all - 12
    drop_sets = ([()] if k_drop <= 0
                 else list(itertools.combinations(non_nominal, k_drop)))

    def evaluate(idx12, combo):
        """Assignment + balance repair for one (kept-12, unseen-4) choice.
        Returns (score, floors, train) or None if balance is unrepairable."""
        seen = [i for i in idx12 if i not in combo]
        d = {t: {s: min(anchor_dist(t, u, s) for u in combo) for s in seen}
             for t in tasks}
        train = {t: sorted(sorted(seen, key=lambda s: -d[t][s])[:4]) for t in tasks}

        def counts():
            c = {s: 0 for s in seen}
            for t in tasks:
                for s in train[t]:
                    c[s] += 1
            return c

        # balance repair: every seen layout must train 4-5 tasks (36 = 8*4.5)
        for _ in range(200):
            c = counts()
            over = [s for s in seen if c[s] > 5]
            under = [s for s in seen if c[s] < 4]
            if not over:
                break
            s_over = max(over, key=lambda s: c[s])
            move = None
            for t in tasks:
                if s_over not in train[t]:
                    continue
                for s_new in sorted(seen, key=lambda s: -d[t][s]):
                    if s_new in train[t] or c[s_new] >= 5 or (under and s_new not in under):
                        continue
                    cost = d[t][s_over] - d[t][s_new]
                    if move is None or cost < move[3]:
                        move = (t, s_over, s_new, cost)
                    break
            if move is None:
                break
            t, s_old, s_new, _ = move
            train[t] = sorted([s for s in train[t] if s != s_old] + [s_new])
        c = counts()
        if max(c.values()) > 5 or min(c.values()) < 4:
            return None
        floors = {t: min(d[t][s] for s in train[t]) for t in tasks}
        return min(floors.values()), floors, train

    best = None
    for dropped in drop_sets:
        idx12 = [i for i in range(n_all) if i not in dropped]
        cand = [i for i in idx12 if not all_layouts[i].get("nominal")]
        for combo in itertools.combinations(cand, args.n_unseen):
            res = evaluate(idx12, combo)
            if res is None:
                continue
            score, floors, train = res
            if best is None or score > best["score"]:
                best = {"combo": combo, "idx12": idx12, "score": score,
                        "floors": floors,
                        "train": {t: list(v) for t, v in train.items()}}

    assert best, "no balanced assignment found"
    combo = best["combo"]
    idx12 = best["idx12"]
    layouts = [all_layouts[i] for i in idx12]
    ids = {i: all_layouts[i]["id"] for i in idx12}
    unseen_ids = [ids[i] for i in combo]
    seen_idx = [i for i in idx12 if i not in combo]
    seen_ids = [ids[i] for i in seen_idx]
    dropped_ids = [l["id"] for j, l in enumerate(all_layouts) if j not in idx12]
    print(f"kept 12 (dropped {dropped_ids}), unseen={unseen_ids}  "
          f"overall floor={best['score']:.3f} m")
    for t in tasks:
        print(f"  {t[:45]:45s} floor={best['floors'][t]:.3f} "
              f"train={[ids[s] for s in best['train'][t]]}")

    if "unseen_fixture_floor_m" in man:   # first run: archive the v1 guard fields
        man["unseen_ids_v1_all_seen_guard"] = man.pop("unseen_ids")
        man["unseen_fixture_floor_v1"] = man.pop("unseen_fixture_floor_m")
    man.pop("unseen_floor_ok", None)
    man.pop("train_matrix", None)
    man.pop("cf_matrix", None)
    man.pop("leakage_per_unseen_per_entity_min_dist", None)
    man["active_ids"] = [ids[i] for i in idx12]
    man["dropped_ids"] = dropped_ids
    man["unseen_ids"] = unseen_ids
    man["seen_ids"] = seen_ids
    man["split_objective"] = ("max over splits of min over tasks of "
                              "min dist(unseen anchor, TRAIN-layout anchor)")
    man["per_task_unseen_train_floor_m"] = {t: round(v, 4)
                                            for t, v in best["floors"].items()}
    man["overall_floor_m"] = round(best["score"], 4)
    man["train_matrix"] = {t: [ids[s] for s in best["train"][t]] for t in tasks}
    man["cf_matrix"] = {t: [sid for sid in seen_ids
                            if sid not in man["train_matrix"][t]] for t in tasks}
    with open(man_path, "w") as f:
        json.dump(man, f, indent=2)
    print(f"rewrote {man_path}")

    if args.render:
        from demogen_libero.convert import load_demo
        from demogen_libero import libero_replay as R
        from demogen_libero import oc_obs
        from demogen_libero import spatial_scene as S
        from demogen_libero import goal_scene as G
        demo = load_demo("D:/Data/LingLing/libero/hf/libero_goal/"
                         "open_the_middle_drawer_of_the_cabinet_demo.hdf5", "demo_0")
        env = oc_obs.make_oc_env(demo.bddl_file)
        env.reset()
        ref_layout = S.read_layout(env, demo.init_state, G.GOAL_JOINTS, G.GOAL_FIXTURES)
        imgs = []
        for rec in layouts:
            new_init, fx = G.apply_goal_layout(rec["layout"], demo.init_state, ref_layout)
            R.reset_to_init_state(env, new_init)
            S.apply_fixture_edits(env, fx)
            S.settle(env, 300)
            obs = env.env._get_observations(force_update=True)
            imgs.append(np.asarray(obs["agentview_image"])[::-1])
        env.close()
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
                     f"(task-aware anchor floor {best['score']:.2f} m)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(args.dir, "layouts_12.png"), dpi=110,
                    bbox_inches="tight")
        print("re-rendered layouts_12.png")


if __name__ == "__main__":
    main()
