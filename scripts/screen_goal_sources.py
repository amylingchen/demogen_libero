"""Screen libero_goal source demos by replaying each on the NOMINAL layout
(L00): a source that cannot even replay at its own recorded placement is
fragile, and random-drawing it inside the layout replay gate makes the gate
measure source luck instead of layout geometry. Writes per-task healthy pools
to output/goal_source_screening.json.

Usage:
    .venv\\Scripts\\python.exe scripts\\screen_goal_sources.py --n-sources 14
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sources", type=int, default=14,
                    help="sources screened per task (random subset, seeded)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--source-base-dir", default="D:/Data/LingLing/libero/hf/libero_goal")
    ap.add_argument("--out", default=os.path.join("output", "goal_source_screening.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    result = {}
    for task in G.GOAL_TASKS:
        cfg = G.GOAL_TASKS[task]
        h5 = os.path.join(args.source_base_dir, f"{task}_demo.hdf5")
        env = oc_obs.make_oc_env(os.path.join(
            get_libero_path("bddl_files"), "libero_goal", f"{task}.bddl"))
        env.reset()
        fixture_ref = G.capture_fixture_ref(env)
        keys = sorted(rng.permutation(list_demo_keys(h5))[:args.n_sources])
        healthy, fragile, errors = [], [], []
        for src_key in keys:
            d = load_demo(h5, src_key)
            try:
                frames = frames_for(cfg, d, h5)
                demo_layout = G.read_demo_layout(env, d.init_state, fixture_ref)
                # nominal replay: zero deltas -- the source's own placement
                obj_t, tar_t = np.zeros(3), np.zeros(3)
                ref, base_actions, new_frames = synthesize_uniform(
                    d.state, d.action, frames, obj_t, tar_t)
                R.reset_to_init_state(env, d.init_state)
                R.replay_uniform(env, base_actions, ref, new_frames, collect=False)
                end = bool(env.check_success())
            except Exception as exc:
                errors.append({"source": src_key, "error": repr(exc)})
                continue
            (healthy if end else fragile).append(src_key)
        env.close()
        result[task] = {"healthy": healthy, "fragile": fragile, "errors": errors}
        print(f"{task}: healthy {len(healthy)}/{len(keys)} "
              f"(fragile={fragile} errors={[e['source'] for e in errors]})", flush=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
