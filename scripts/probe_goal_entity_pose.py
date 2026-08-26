"""Does a per-frame 3D position exist for all nine goal entities?

The four movable objects have free joints, so their pose is already stored. The
cabinet, its two addressable drawers, the stove and the rack do not -- the plan
assumed their pose "lives in a slide joint rather than in body_pos"
(oc_obs.EXTRA_OBS_KEYS comment). data.body_xpos is recomputed by mj_forward from
the joints, so the question is measurable rather than arguable: drive the middle
drawer's slide and see whether the body origin moves with it.

Also dumps the mesh AABB for the two drawer bodies, which object_geometry.json
has no entry for (it enumerates free-joint objects plus the three fixture roots).

Usage:
    .venv\Scripts\python.exe scripts\probe_goal_entity_pose.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

from demogen_libero import goal_scene as G

BODIES = [
    ("akita_black_bowl_1", "akita_black_bowl_1_main"),
    ("cream_cheese_1", "cream_cheese_1_main"),
    ("wine_bottle_1", "wine_bottle_1_main"),
    ("plate_1", "plate_1_main"),
    ("wooden_cabinet_1", "wooden_cabinet_1_main"),
    ("wooden_cabinet_1_middle_drawer", "wooden_cabinet_1_cabinet_middle"),
    ("wooden_cabinet_1_top_drawer", "wooden_cabinet_1_cabinet_top"),
    ("flat_stove_1", "flat_stove_1_main"),
    ("wine_rack_1", "wine_rack_1_main"),
]
TASK = "open_the_middle_drawer_of_the_cabinet"


def main():
    bddl = os.path.join(get_libero_path("bddl_files"), "libero_goal", TASK + ".bddl")
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.reset()
    m, d = env.sim.model, env.sim.data
    ids = {}
    for name, body in BODIES:
        try:
            ids[name] = m.body_name2id(body)
        except Exception as e:
            print(f"  MISSING BODY {body}: {e}")

    adr = m.get_joint_qpos_addr("wooden_cabinet_1_middle_level")
    print("body_xpos as the middle drawer is driven (m):")
    print("  %-34s %-24s %-24s %s" % ("entity", "closed", "open(-0.16)", "delta_cm"))
    poses = {}
    for q in (0.0, -0.16):
        d.qpos[adr] = q
        env.sim.forward()
        poses[q] = {n: np.array(d.body_xpos[i]) for n, i in ids.items()}
    for n in ids:
        a, b = poses[0.0][n], poses[-0.16][n]
        print("  %-34s %-24s %-24s %.2f" % (
            n, np.round(a, 3), np.round(b, 3), np.linalg.norm(b - a) * 100))

    print()
    print("mesh AABB for the two drawer bodies (object_geometry.json has no entry):")
    out = {}
    for n, body in BODIES[5:7]:
        bid = m.body_name2id(body)
        vs = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != bid:
                continue
            dt = int(m.geom_dataid[g])
            if dt < 0:
                continue
            s, num = int(m.mesh_vertadr[dt]), int(m.mesh_vertnum[dt])
            v = m.mesh_vert[s:s + num].reshape(-1, 3)
            # geom frame -> body frame
            from robosuite.utils.transform_utils import quat2mat
            gq = m.geom_quat[g]                       # wxyz
            R = quat2mat(np.array([gq[1], gq[2], gq[3], gq[0]]))
            vs.append(v @ R.T + m.geom_pos[g])
        if not vs:
            print("  %-34s no mesh geoms" % n)
            continue
        v = np.concatenate(vs)
        lo, hi = v.min(0), v.max(0)
        out[n] = {"size": (hi - lo).tolist(), "offset_body": ((hi + lo) / 2).tolist()}
        print("  %-34s size %s  offset_body %s" % (n, np.round(hi - lo, 4), np.round((hi + lo) / 2, 4)))
    print()
    print(json.dumps(out, indent=1))
    env.close()


main()
