"""Measure the trajectory-synthesis success rate on PAIRED scenes.

The screen validates init states only. A paired scene carries a whole extra
relation group as obstacles, so the reach and transport paths have more to hit
than in single-task scenes, and the success rate that sizes the whole generation
run is unmeasured. This rolls a handful of scenes from the split plan through the
real synthesis path (synthesize_uniform + replay_uniform) and reports it.

Nothing is written to a dataset; this only reports.

Usage:
    .venv\\Scripts\\python.exe scripts\\probe_pair_trajectories.py --n 10
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import h5py
import numpy as np

import screen_spatial_pairs as P
from run_spatial_oc_demo import segment_for
from demogen_libero.convert import load_demo, resolve_bddl_file
from demogen_libero.trajectory import synthesize_uniform
from demogen_libero import libero_replay as R, oc_obs, spatial_scene as S


def build_state(scene, role, host_init, addrs, nq, drawer_adr):
    """Write a paired geometry into a task's own env. `role` selects which bowl
    the instruction refers to; that bowl is written as akita_black_bowl_1, since
    every task's BDDL goal and the OC seg-id convention are keyed on that name,
    and the other bowl takes the remaining pose."""
    state = np.asarray(host_init, dtype=np.float64).copy()
    free = dict(scene["free"])
    if role == "b":
        free[P.BOWL_A], free[P.BOWL_B] = scene["free"][P.BOWL_B], scene["free"][P.BOWL_A]
    for jn, m in free.items():
        a = addrs[jn]
        state[a], state[a + 1], state[a + 2] = m["xy"][0], m["xy"][1], m["z"]
        q = np.asarray(m["quat"], float)
        state[a + 3:a + 7] = q / max(np.linalg.norm(q), 1e-9)
    if drawer_adr is not None:
        state[1 + drawer_adr] = scene.get("drawer_qpos", 0.0)
    state[1 + nq:] = 0.0
    fx = {k: {"pos": np.array([v["xy"][0], v["xy"][1], v["z"]]),
              "quat": np.asarray(v["quat"], float)}
          for k, v in scene["fixtures"].items()}
    return state, fx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--plan", default="output/pair_split.json")
    ap.add_argument("--split", default="train")
    ap.add_argument("--source-retries", type=int, default=1,
                    help="alternative source demos to try per scene. 1 measures the "
                         "per-ATTEMPT success rate; >1 measures SOLVABILITY, which is "
                         "what an eval scene needs -- a scene only counts as unusable "
                         "when every source fails, exactly as the generator decides it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output/pair_traj_probe.json")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    plan = json.load(open(args.plan))
    scenes = plan["scenes"]
    eps = [e for e in plan["episodes"] if e["split"] == args.split]
    rng = np.random.default_rng(args.seed)
    # spread the probe over distinct tasks and pairs rather than sampling blind
    by_task = defaultdict(list)
    for e in eps:
        by_task[e["task"]].append(e)
    picked, tasks = [], sorted(by_task)
    while len(picked) < args.n:
        for t in tasks:
            if len(picked) >= args.n or not by_task[t]:
                continue
            picked.append(by_task[t].pop(int(rng.integers(len(by_task[t])))))

    cfg = S.load_spatial_config()
    results, env, cur = [], None, None
    import mujoco
    try:
        with h5py.File(os.path.join(P.DATA_DIR,
                                    f"{picked[0]['task']}_demo.hdf5"), "r") as _f:
            pass
        for i, e in enumerate(picked):
            task = e["task"]
            sc = scenes[e["scene_index"]]
            src0 = sc["source_demo_a"] if e["target_role"] == "a" else sc["source_demo_b"]
            hdf5 = os.path.join(P.DATA_DIR, f"{task}_demo.hdf5")
            with h5py.File(hdf5, "r") as f:
                all_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            cand_keys = [f"demo_{src0}"] + [k for k in all_keys if k != f"demo_{src0}"]
            cand_keys = cand_keys[:max(args.source_retries, 1)]
            src_i = src0
            demo = load_demo(hdf5, f"demo_{src_i}")
            if cur != task:
                if env is not None:
                    env.close()
                env = oc_obs.make_oc_env(demo.bddl_file)
                env.reset()
                R.reset_to_init_state(env, demo.init_state)
                lay = S.read_layout(env, demo.init_state, P.FREE_JOINTS, S.ALL_FIXTURES)
                addrs = {j: lay["free"][j]["addr"] for j in P.FREE_JOINTS}
                nq = env.sim.model.nq
                jid = mujoco.mj_name2id(env.sim.model._model, mujoco.mjtObj.mjOBJ_JOINT,
                                        P.DRAWER_JOINT)
                dadr = int(env.sim.model._model.jnt_qposadr[jid]) if jid >= 0 else None
                cur = task
            rec = {"task": task, "pair": e["pair"], "role": e["target_role"],
                   "cell": e["target_cell"], "attempts": 0, "ok": False,
                   "stage": "exhausted"}
            for key in cand_keys:
                d = load_demo(hdf5, key)
                src_lay = S.read_layout(env, d.init_state, P.FREE_JOINTS, S.ALL_FIXTURES)
                src_bowl = src_lay["free"][P.BOWL_A]["pos"][:2]
                src_plate = src_lay["free"][P.PLATE]["pos"][:2]
                try:
                    frames, is_regrasp = segment_for(d)
                except Exception:
                    continue                     # source unusable, not the scene's fault
                state, fx = build_state(sc, e["target_role"], d.init_state, addrs, nq, dadr)
                obj_t = np.array([*(np.array(e["target_xy"]) - src_bowl), 0.0])
                tar_t = np.array([*(np.array(sc["free"][P.PLATE]["xy"]) - src_plate), 0.0])
                rec["attempts"] += 1
                try:
                    ref, base_actions, new_frames = synthesize_uniform(
                        d.state, d.action, frames, obj_t, tar_t)
                    R.reset_to_init_state(env, state)
                    S.apply_fixture_edits(env, fx)
                    success, _obs, _roll = R.replay_uniform(
                        env, base_actions, ref, new_frames, collect=False)
                except Exception as exc:
                    rec["stage"], rec["err"] = "replay", repr(exc)[:90]
                    continue
                if success:
                    rec.update(ok=True, stage="done", source_demo=key,
                               regrasp=bool(is_regrasp),
                               obj_t=[float(x) for x in obj_t[:2]])
                    break
            results.append(rec)
            print(f"[{i+1}/{len(picked)}] {P.short(task):<34} "
                  f"solvable={rec['ok']} after {rec['attempts']} source(s)", flush=True)
    finally:
        if env is not None:
            env.close()

    ok = sum(r.get("ok", False) for r in results)
    att = sum(r.get("attempts", 0) for r in results)
    print(f"\nSOLVABLE: {ok}/{len(results)} = {100 * ok / max(len(results), 1):.0f}% "
          f"of scenes yielded a working trajectory within {args.source_retries} source(s)")
    print(f"per-attempt success: {ok}/{att} = {100 * ok / max(att, 1):.0f}%")
    st = Counter(r["stage"] for r in results if not r.get("ok"))
    for k, v in st.items():
        print(f"  unsolved, last stage {k}: {v}")
    per = defaultdict(lambda: [0, 0])
    for r in results:
        per[r["task"]][0] += bool(r.get("ok"))
        per[r["task"]][1] += 1
    print()
    for t, (a, b) in sorted(per.items()):
        print(f"  {P.short(t):<40}{a}/{b}")
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
