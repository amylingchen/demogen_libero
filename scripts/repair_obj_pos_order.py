"""Put obj_pos / obj_quat rows back in the order object_instances declares.

Two call sites asked `extract_oc_frame` for the five free objects in
`screen_spatial_pairs.FREE_INST` order (bowl_1, bowl_2, plate, cookies, ramekin)
while writing the task's `object_order` (bowl_1, plate, bowl_2, cookies, ramekin)
into the `object_instances` attr. Those differ by one transposition, so rows 1
and 2 held the wrong objects while the file claimed otherwise. Segmentation is
unaffected -- the LUT was always built from object_order.

Nothing is re-rendered: the rows are correct values in the wrong slots, so the
repair is a permutation. Which permutation is not assumed. Every demo is checked
against its own stored sim state -- each free object's true position is read from
its joint qpos and matched to a row -- so a file already in the right order is
left alone, and one that matches no permutation is reported rather than
"repaired" into something else.

Usage:
    .venv\\Scripts\\python.exe scripts\\repair_obj_pos_order.py --dry-run
    .venv\\Scripts\\python.exe scripts\\repair_obj_pos_order.py
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import h5py
import numpy as np

import screen_spatial_pairs as P
from demogen_libero.convert import load_demo
from demogen_libero import libero_replay as R, oc_obs, spatial_scene as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/libero_spatial_pairs")
    ap.add_argument("--splits", nargs="+", default=["train", "counterfact", "unseen"])
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="metres; a row must match a joint position this closely")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    cfg_all = S.load_spatial_config()
    grand = {"ok": 0, "fixed": 0, "unmatched": 0}
    for split in args.splits:
        sdir = os.path.join(args.root, split)
        if not os.path.isdir(sdir):
            continue
        for short in sorted(os.listdir(sdir)):
            h5 = glob.glob(os.path.join(sdir, short, "*_demo.hdf5"))
            if not h5:
                continue
            task = os.path.basename(h5[0])[:-len("_demo.hdf5")]
            if task not in cfg_all:
                print(f"  ! {split}/{short}: unknown task, skipped")
                continue
            order5 = list(cfg_all[task]["object_order"])[:5]

            # qpos address of each free joint, so the TRUE position of every
            # object can be read out of the state the demo itself stored
            demo0 = load_demo(os.path.join(P.DATA_DIR, f"{task}_demo.hdf5"), "demo_0")
            env = oc_obs.make_oc_env(demo0.bddl_file)
            env.reset()
            R.reset_to_init_state(env, demo0.init_state)
            lay = S.read_layout(env, demo0.init_state, P.FREE_JOINTS, S.ALL_FIXTURES)
            addrs = [lay["free"][f"{inst}_joint0"]["addr"] for inst in order5]
            env.close()

            n_ok = n_fix = n_bad = 0
            with h5py.File(h5[0], "r+") as f:
                for name in sorted(f["data"].keys(),
                                   key=lambda k: int(k.split("_")[-1])):
                    g = f["data"][name]
                    op = np.array(g["obs"]["obj_pos"])
                    st0 = np.array(g["states"][0])
                    truth = np.array([st0[a:a + 3] for a in addrs])   # declared order
                    perm, bad = [], False
                    for i in range(5):
                        d = np.linalg.norm(op[0, :5] - truth[i], axis=1)
                        j = int(np.argmin(d))
                        if d[j] > args.tol or j in perm:
                            bad = True
                            break
                        perm.append(j)
                    if bad:
                        n_bad += 1
                        print(f"  ? {split}/{short}/{name}: no clean permutation "
                              f"(closest residual {float(d.min()):.2e} m)")
                        continue
                    if perm == list(range(5)):
                        n_ok += 1
                        continue
                    n_fix += 1
                    if args.dry_run:
                        continue
                    oq = np.array(g["obs"]["obj_quat"])
                    idx = perm + list(range(5, op.shape[1]))   # fixtures stay put
                    g["obs"]["obj_pos"][...] = op[:, idx]
                    g["obs"]["obj_quat"][...] = oq[:, idx]
            tag = "would fix" if args.dry_run else "fixed"
            print(f"  {split}/{short:<42} ok={n_ok:<4} {tag}={n_fix:<4}"
                  + (f" UNMATCHED={n_bad}" if n_bad else ""))
            grand["ok"] += n_ok
            grand["fixed"] += n_fix
            grand["unmatched"] += n_bad
    print(f"\nalready correct {grand['ok']}, "
          f"{'would fix' if args.dry_run else 'fixed'} {grand['fixed']}, "
          f"unmatched {grand['unmatched']}")


if __name__ == "__main__":
    main()
