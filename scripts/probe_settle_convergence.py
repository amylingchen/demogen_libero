"""Does the new settle() actually seat the objects, and does it say so?

Loads the stored (premature) unseen init states, runs the NEW settle(), and
checks three things the old one could not report:
  - it converges, and at how many physics steps;
  - the bowl ends at train's rest height, not 1.8 cm above it;
  - the old cap of 6 would NOT have converged -- run it with require_converged
    off and confirm it reports converged=False, i.e. the guard would have
    caught the original bug.

That last one is the point: a fix whose test only exercises the good path
proves nothing about whether the failure can recur.
"""
import glob
import os
import sys

os.environ.setdefault("MUJOCO_GL", "wgl")
ROOT = r"D:\Data\LingLing\workdata\claudecode\demogen_libero"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import h5py
import numpy as np

from demogen_libero import libero_replay as R, oc_obs, spatial_scene as S

BASE = os.path.join(ROOT, "output", "libero_spatial_pairs")
BOWL = "akita_black_bowl_1_main"
TASK = "on_the_ramekin"
BDDL = (r"D:\Data\LingLing\workdata\claudecode\demogen_libero\LIBERO\libero"
        r"\libero\bddl_files\libero_spatial"
        r"\pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.bddl")

print(f"SpatialSpec().settle_physics_steps = {S.SpatialSpec().settle_physics_steps}")
print()

env = oc_obs.make_oc_env(BDDL)
env.reset()
m = env.sim.model
bid = m.body_name2id(BOWL)

for split, want_rest in (("train", None), ("unseen", None)):
    f = glob.glob(os.path.join(BASE, split, TASK, "*.hdf5"))[0]
    with h5py.File(f, "r") as h:
        demos = sorted(h["data"], key=lambda d: int(d.rsplit("_", 1)[-1]))[:3]
        states = [np.asarray(h[f"data/{d}/states"][0]) for d in demos]
        # the welded fixtures are NOT in the state vector; leaving them at the
        # MJCF default drops whatever was resting on them and would dirty the
        # train control with motion that has nothing to do with settling
        fx = [{b: {"pos": np.asarray(h[f"data/{d}/obs/obj_pos"][0, r]),
                   "quat": np.asarray(h[f"data/{d}/obs/obj_quat"][0, r])}
               for r, b in ((5, "flat_stove_1_main"), (6, "wooden_cabinet_1_main"))}
              for d in demos]
    print(f"=== {split} ===")
    for i, st in enumerate(states):
        R.reset_to_init_state(env, st)
        S.apply_fixture_edits(env, fx[i])
        env.sim.forward()
        z0 = float(env.sim.data.body_xpos[bid][2])

        # the OLD setting, with the guard disabled so it can be observed
        old = S.settle(env, 6, require_converged=False)

        R.reset_to_init_state(env, st)
        S.apply_fixture_edits(env, fx[i])
        env.sim.forward()
        new = S.settle(env, S.SpatialSpec().settle_physics_steps)
        z1 = float(env.sim.data.body_xpos[bid][2])

        print(f"  demo{i}: stored z={z0:.5f} -> settled z={z1:.5f}  "
              f"drop={100*(z0-z1):.3f} cm")
        print(f"     old cap 6 : converged={old['converged']} "
              f"speed={old['max_speed']:.2e} moved={old['max_disp_cm']:.3f} cm")
        print(f"     new       : converged={new['converged']} "
              f"steps={new['steps']} speed={new['max_speed']:.2e} "
              f"moved={new['max_disp_cm']:.3f} cm")
env.close()
