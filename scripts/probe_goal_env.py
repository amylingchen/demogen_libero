"""Wiring probe for the libero_goal scene (plan doc §7): joint names + qpos
addresses, state dims, fixture free-joint poses, articulation joints (drawer /
knob), instance names for the seg gate, and the bddl nominal placements.

Usage:
    .venv\\Scripts\\python.exe scripts\\probe_goal_env.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from libero.libero import get_libero_path
from demogen_libero import oc_obs


def main():
    bddl = os.path.join(get_libero_path("bddl_files"), "libero_goal",
                        "put_the_bowl_on_the_plate.bddl")
    env = oc_obs.make_oc_env(bddl)
    env.reset()
    sim = env.sim
    m = sim.model

    print("== state dims ==")
    st = sim.get_state().flatten()
    print(f"flattened state: {st.shape}  nq={m.nq} nv={m.nv}")

    print("\n== joints (name, type, qpos_addr) ==  type 0=free 1=ball 2=slide 3=hinge")
    for j in range(m.njnt):
        name = m.joint_id2name(j)
        addr = m.get_joint_qpos_addr(name)
        print(f"  {name:45s} type={m.jnt_type[j]} addr={addr}")

    print("\n== free-joint entity poses (qpos) ==")
    for j in range(m.njnt):
        if m.jnt_type[j] != 0:
            continue
        name = m.joint_id2name(j)
        q = sim.data.get_joint_qpos(name)
        print(f"  {name:45s} pos={np.round(q[:3], 3)} quat(wxyz)={np.round(q[3:], 3)}")

    print("\n== instances_to_ids keys (for seg gate) ==")
    print(list(env.env.model.instances_to_ids.keys()))

    print("\n== obj_of_interest / object states keys ==")
    print("objects:", list(env.env.objects_dict.keys()))
    print("fixtures:", list(env.env.fixtures_dict.keys()))

    env.close()


if __name__ == "__main__":
    main()
