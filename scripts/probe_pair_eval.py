"""Solvability spot-check on the GENERATED eval scenes, with visualisation.

`probe_pair_trajectories.py` rebuilds scenes from the split plan. That is the
right thing to measure before generating, but it is the wrong thing to measure
afterwards: what a policy is actually scored on is the init state stored in the
eval hdf5, and that differs from the plan in three ways -- some scenes fell back
to a jittered variant, counterfactual scenes reuse the training geometry, and
every scene now exists at several slightly different ARM starting poses. So this
probe restores the stored state verbatim and rolls from there.

Solvable means: some source demo yields a trajectory that ends with the bowl the
instruction names on the plate. The bowl the instruction names is whichever one
occupies the akita_black_bowl_1 slot, which is what `check_success` tests -- so a
pass also confirms the counterfactual is asking for the OTHER bowl than training
did, not merely that some bowl reached the plate.

Outputs (per split, under --out-dir):
    <split>_summary.png    per-task solvability + attempts-to-solve + arm offset
    <split>_scenes.png     mosaic of probed init states, green = solved, red = not
    <split>_roll_*.png     filmstrips, both a solved and an unsolved example
    <split>_probe.json     the raw records

Usage:
    .venv\\Scripts\\python.exe scripts\\probe_pair_eval.py --split unseen --per-task 3
    .venv\\Scripts\\python.exe scripts\\probe_pair_eval.py --split counterfact --per-task 3
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import h5py
import numpy as np

import screen_spatial_pairs as P
from run_spatial_oc_demo import segment_for
from demogen_libero.convert import load_demo
from demogen_libero.trajectory import synthesize_uniform
from demogen_libero import libero_replay as R, oc_obs, spatial_scene as S

FIXTURES = S.FIXTURE_INSTANCES


def fx_from_row(obj_pos0, obj_quat0):
    """Fixture poses as the scene RECORDED them (obj_pos slots 5 and 6), rather
    than reconstructed -- the eval scene stores them, so there is nothing to
    infer."""
    fx = {}
    for i, fb in enumerate(FIXTURES):
        q = obj_quat0[5 + i]                       # stored xyzw
        fx[S.FIXTURE_INSTANCE_BODY[fb]] = {
            "pos": np.asarray(obj_pos0[5 + i], dtype=np.float64),
            "quat": np.array([q[3], q[0], q[1], q[2]], dtype=np.float64),   # -> wxyz
        }
    return fx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="unseen", choices=["unseen", "counterfact"])
    ap.add_argument("--per-task", type=int, default=3, help="scenes probed per task")
    ap.add_argument("--task", default=None,
                    help="short task name; default = every task in the split")
    ap.add_argument("--source-retries", type=int, default=4,
                    help="source demos tried before a scene counts as unsolvable; a "
                         "scene is only unusable when every source fails, which is "
                         "how the generator decides it too")
    ap.add_argument("--root", default="output/libero_spatial_pairs")
    ap.add_argument("--out-dir", default="output/pair_eval_probe")
    ap.add_argument("--films", type=int, default=2,
                    help="filmstrips saved per outcome (solved / unsolved)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cfg = S.load_spatial_config()

    split_dir = os.path.join(args.root, args.split)
    results, films = [], {"solved": [], "unsolved": []}
    for short in sorted(os.listdir(split_dir)):
        if args.task and short != args.task:
            continue
        h5 = glob.glob(os.path.join(split_dir, short, "*_demo.hdf5"))
        log_p = os.path.join(split_dir, short, "scene_log.json")
        if not h5 or not os.path.exists(log_p):
            continue
        rows = {r["demo_name"]: r for r in json.load(open(log_p))}
        task = next(iter(rows.values()))["task"]

        # spread the probe over distinct GEOMETRIES, and within a geometry prefer a
        # perturbed arm pose: variant 0 is the nominal state the generator already
        # rendered, so probing only variant 0 would not test the thing the variants
        # were added for
        by_uid = defaultdict(list)
        for name, r in rows.items():
            by_uid[r["scene_uid"]].append(name)
        uids = sorted(by_uid)
        rng.shuffle(uids)
        pick = []
        for u in uids[:args.per_task]:
            names = sorted(by_uid[u], key=lambda n: rows[n]["robot_variant"])
            pick.append(names[-1] if len(names) > 1 else names[0])

        src_path = os.path.join(P.DATA_DIR, f"{task}_demo.hdf5")
        with h5py.File(src_path, "r") as f:
            src_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        demo0 = load_demo(src_path, src_keys[0])
        env = oc_obs.make_oc_env(demo0.bddl_file)
        env.reset()
        obj_order = list(cfg[task]["object_order"]) + FIXTURES
        lut = oc_obs.build_seg_lut(env, obj_order)
        R.reset_to_init_state(env, demo0.init_state)

        # source bowl/plate positions, for the synthesis offsets
        src_xy, src_frames = {}, {}
        for k in src_keys:
            d = load_demo(src_path, k)
            try:
                src_frames[k] = (d, segment_for(d)[0])
            except Exception:
                continue
            lay = S.read_layout(env, d.init_state, P.FREE_JOINTS, S.ALL_FIXTURES)
            src_xy[k] = (lay["free"][P.BOWL_A]["pos"][:2].copy(),
                         lay["free"][P.PLATE]["pos"][:2].copy())

        try:
            with h5py.File(h5[0], "r") as f:
                for name in pick:
                    g = f["data"][name]
                    st0 = np.array(g["states"][0])
                    op = np.array(g["obs"]["obj_pos"][0])
                    oq = np.array(g["obs"]["obj_quat"][0])
                    rgb0 = np.array(g["obs"]["agentview_rgb"][0])
                    fx = fx_from_row(op, oq)
                    tgt_xy, plate_xy = op[0][:2], op[1][:2]   # slot 0 target, 1 plate

                    order = [k for k in src_keys if k in src_frames]
                    rng.shuffle(order)
                    order = order[:max(args.source_retries, 1)]
                    rec = {"task": task, "demo": name, "split": args.split,
                           "scene_uid": rows[name]["scene_uid"],
                           "robot_variant": rows[name]["robot_variant"],
                           "target_cell": rows[name]["target_cell"],
                           "attempts": 0, "ok": False}
                    frames_kept = None
                    for key in order:
                        d, seg_frames = src_frames[key]
                        sb, sp = src_xy[key]
                        rec["attempts"] += 1
                        want_film = len(films["solved"]) < args.films or \
                            len(films["unsolved"]) < args.films
                        try:
                            ref, base_actions, nf = synthesize_uniform(
                                d.state, d.action, seg_frames,
                                np.array([*(tgt_xy - sb), 0.0]),
                                np.array([*(plate_xy - sp), 0.0]))
                            R.reset_to_init_state(env, st0)
                            S.apply_fixture_edits(env, fx)
                            grab = ([], (lambda e, o: {"agentview_rgb":
                                                       np.ascontiguousarray(
                                                           o["agentview_image"][::-1])})
                                    ) if want_film else (None, None)
                            success, _o, roll = R.replay_uniform(
                                env, base_actions, ref, nf,
                                collect=bool(want_film), extract=grab[1])
                        except Exception as exc:
                            rec["err"] = repr(exc)[:90]
                            continue
                        if want_film and roll:
                            frames_kept = np.asarray(roll["agentview_rgb"])
                        if success:
                            rec.update(ok=True, source_demo=key)
                            break
                    rec["rgb0"] = rgb0
                    bucket = "solved" if rec["ok"] else "unsolved"
                    if frames_kept is not None and len(films[bucket]) < args.films:
                        films[bucket].append((rec, frames_kept))
                    results.append(rec)
                    print(f"  {P.short(task):<38}{name:<10}v{rec['robot_variant']} "
                          f"solvable={rec['ok']} after {rec['attempts']} source(s)",
                          flush=True)
        finally:
            env.close()

    # ------------------------------------------------------------------ report
    ok = sum(r["ok"] for r in results)
    print(f"\nSOLVABLE {ok}/{len(results)} = {100*ok/max(len(results),1):.0f}% "
          f"within {args.source_retries} source demos")
    per = defaultdict(lambda: [0, 0, []])
    for r in results:
        per[r["task"]][0] += r["ok"]
        per[r["task"]][1] += 1
        if r["ok"]:
            per[r["task"]][2].append(r["attempts"])
    for t, (a, b, at) in sorted(per.items()):
        print(f"  {P.short(t):<40}{a}/{b}"
              + (f"   median {np.median(at):.0f} source(s) to solve" if at else ""))

    blob = [{k: v for k, v in r.items() if k != "rgb0"} for r in results]
    out_json = os.path.join(args.out_dir, f"{args.split}_probe.json")
    json.dump(blob, open(out_json, "w"), indent=2)

    # ------------------------------------------------------------- visualise
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    # 1. mosaic of probed init states, bordered by outcome
    cols = 6
    rowsn = (len(results) + cols - 1) // cols
    cell = 256
    canvas = np.full((rowsn * cell, cols * cell, 3), 30, np.uint8)
    for i, r in enumerate(results):
        rr, cc = divmod(i, cols)
        img = np.array(r["rgb0"]).copy()
        col = (60, 200, 90) if r["ok"] else (220, 70, 70)
        b = 6
        img[:b], img[-b:], img[:, :b], img[:, -b:] = col, col, col, col
        canvas[rr*cell:(rr+1)*cell, cc*cell:(cc+1)*cell] = img
    p_scenes = os.path.join(args.out_dir, f"{args.split}_scenes.png")
    Image.fromarray(canvas).save(p_scenes)

    # 2. summary chart
    tasks = sorted(per)
    rate = [100 * per[t][0] / per[t][1] for t in tasks]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2),
                           gridspec_kw={"width_ratios": [2, 1]})
    y = np.arange(len(tasks))
    ax[0].barh(y, rate, color=["#3fa66a" if v >= 99 else "#d9a441" if v >= 50
                               else "#c94f4f" for v in rate])
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([P.short(t) for t in tasks], fontsize=9)
    ax[0].set_xlim(0, 105)
    ax[0].set_xlabel("scenes with a working trajectory (%)")
    ax[0].axvline(100 * ok / max(len(results), 1), ls="--", c="k", lw=1,
                  label=f"overall {100*ok/max(len(results),1):.0f}%")
    ax[0].legend(loc="lower right", fontsize=9)
    for i, t in enumerate(tasks):
        ax[0].text(2, i, f"{per[t][0]}/{per[t][1]}", va="center", fontsize=8,
                   color="white", fontweight="bold")
    ax[0].set_title(f"{args.split}: solvability of the STORED eval init states\n"
                    f"(probed from the hdf5, arm pose included; "
                    f"<= {args.source_retries} source demos)", fontsize=10)
    att = [r["attempts"] for r in results if r["ok"]]
    if att:
        ax[1].hist(att, bins=np.arange(0.5, args.source_retries + 1.5),
                   color="#3fa66a", edgecolor="white")
    ax[1].set_xlabel("source demos tried before success")
    ax[1].set_ylabel("scenes")
    ax[1].set_title("how hard was it to solve", fontsize=10)
    fig.tight_layout()
    p_sum = os.path.join(args.out_dir, f"{args.split}_summary.png")
    fig.savefig(p_sum, dpi=110)
    plt.close(fig)

    # 3. filmstrips
    paths = []
    for bucket, items in films.items():
        for j, (rec, fr) in enumerate(items):
            idx = np.linspace(0, len(fr) - 1, 6).astype(int)
            strip = np.concatenate([fr[i] for i in idx], axis=1)
            fig, a2 = plt.subplots(figsize=(15, 3))
            a2.imshow(strip)
            a2.set_xticks([(i + 0.5) * fr.shape[2] for i in range(len(idx))])
            a2.set_xticklabels([f"t={i}" for i in idx], fontsize=8)
            a2.set_yticks([])
            a2.set_title(f"{args.split} / {P.short(rec['task'])} / {rec['demo']} "
                         f"(arm variant {rec['robot_variant']}) -- "
                         f"{'SOLVED' if rec['ok'] else 'NOT SOLVED'}", fontsize=10)
            fig.tight_layout()
            p = os.path.join(args.out_dir, f"{args.split}_roll_{bucket}_{j}.png")
            fig.savefig(p, dpi=110)
            plt.close(fig)
            paths.append(p)

    print(f"\nwrote {out_json}\n      {p_scenes}\n      {p_sum}")
    for p in paths:
        print(f"      {p}")


if __name__ == "__main__":
    main()
