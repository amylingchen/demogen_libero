"""Add the remaining robomimic-convention fields to the rendered goal OC
dataset, so its per-demo layout matches the object and spatial suites.

Adds four datasets and the attrs that describe them:
  robot_states (T,8)  [ee_pos(3), ee_ori axis-angle(3), gripper_qpos(2)]
  rewards      (T,)   uint8, 1 on the final frame
  dones        (T,)   uint8, 1 on the final frame
  subtask_id   (T,)   int32 index into oc_obs.SUBTASK_VOCAB

Subtask annotation is NOT the object suite's pick-and-place annotator. That one
asserts a gripper close/open cycle, which the drawer task never has (it never
closes) and the knob task never completes (it never re-opens). The closed
vocabulary already carries the right words -- open, rotate, push -- so the
goal tasks are annotated from their phase spans with the verb their own
instruction uses:

  pick_place  transit -> move -> transit -> idle   (as in the other suites)
  drawer      transit -> open   -> idle
  knob        transit -> rotate -> idle
  push        transit -> push   -> idle

Datasets are appended in place; the 73 GB of image data is not rewritten.
Re-runnable: demos that already carry the fields are skipped.

Usage:
    .venv\\Scripts\\python.exe scripts\\patch_goal_oc_parity.py --dir output/goal_oc_v3
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np

from demogen_libero import goal_scene as G
from demogen_libero.oc_obs import SUBTASK_VOCAB, SUBTASK_IDS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_goal_metainfo import TASK_NOUNS, ENTITY_NAMES, stages

SPLITS = ["train", "quarantine_cf", "quarantine_unseen"]
# the verb each fixture-operation task's own instruction uses, taken from the
# existing closed vocabulary rather than extending it
OPERATION_VERB = {
    "open_the_middle_drawer_of_the_cabinet": "open",
    "turn_on_the_stove": "rotate",
    "push_the_plate_to_the_front_of_the_stove": "push",
}


def subtasks_for(task, phase_id, post_steps=3):
    """Coarse spans from the phase_id, in the closed vocabulary."""
    phase_id = np.asarray(phase_id)
    T = len(phase_id)
    tgt, dst = TASK_NOUNS[task]
    kind = G.GOAL_TASKS[task]["kind"]
    contact = int(np.argmax(phase_id >= 1)) if (phase_id >= 1).any() else T
    idle_start = max(T - post_steps, contact)
    spans = []
    if kind == "pick_place":
        # phase 3 begins the place; the hand empties at the end of it
        rel = int(np.argmax(phase_id >= 3)) if (phase_id >= 3).any() else T
        release = max(min(idle_start, T - post_steps), rel)
        spans.append(("transit", tgt, None,
                      f"move the gripper to the {tgt}", 0, contact))
        spans.append(("move", tgt, dst,
                      f"carry the {tgt} to the {dst}", contact, release))
        if release < idle_start:
            spans.append(("transit", dst, None, "retreat", release, idle_start))
    else:
        verb = OPERATION_VERB[task]
        obj = tgt
        text = (f"push the {tgt} to the front of the {dst}" if verb == "push"
                else f"{verb} the {tgt}")
        spans.append(("transit", obj, None,
                      f"move the gripper to the {obj}", 0, contact))
        spans.append((verb, obj, dst if verb == "push" else None,
                      text, contact, idle_start))
    if idle_start < T:
        spans.append(("idle", None, None, "hold", idle_start, T))

    ids = np.full(T, SUBTASK_IDS["idle"], dtype=np.int32)
    out = []
    for action, obj, dest, text, s, e in spans:
        if e <= s:
            continue
        ids[s:e] = SUBTASK_IDS[action]
        out.append({"action": action, "object": obj, "destination": dest,
                    "instruction": text, "start": int(s), "end": int(e)})
    return ids, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("output", "goal_oc_v3"))
    args = ap.parse_args()

    patched = skipped = 0
    for split in SPLITS:
        for p in sorted(glob.glob(os.path.join(args.dir, split, "*.hdf5"))):
            task = os.path.basename(p).replace("_demo.hdf5", "")
            with h5py.File(p, "a") as f:
                data = f["data"]
                n_total = 0
                for k in data:
                    g = data[k]
                    n_total += int(g["actions"].shape[0])
                    if "robot_states" in g and "phases" in g.attrs:
                        skipped += 1
                        continue
                    obs = g["obs"]
                    T = int(g["actions"].shape[0])
                    rs = np.concatenate([np.array(obs["ee_pos"]),
                                         np.array(obs["ee_ori"]),
                                         np.array(obs["gripper_states"])], axis=1)
                    rew = np.zeros(T, np.uint8); rew[-1] = 1
                    dn = np.zeros(T, np.uint8); dn[-1] = 1
                    sid, spans = subtasks_for(task, np.array(g["phase_id"]))
                    for nm, arr in (("robot_states", rs), ("rewards", rew),
                                    ("dones", dn), ("subtask_id", sid)):
                        if nm in g:
                            del g[nm]
                        g.create_dataset(nm, data=arr)
                    g.attrs["subtasks"] = json.dumps(spans)
                    g.attrs["subtask_vocab"] = json.dumps(SUBTASK_VOCAB)
                    g.attrs["object_names"] = json.dumps(ENTITY_NAMES)
                    g.attrs["object_instances"] = json.dumps(
                        list(G.GOAL_SEG_IDS.keys()))
                    # fine-grained per-phase table, same field the object suite
                    # carries on the demo (metainfo.json repeats it)
                    g.attrs["phases"] = json.dumps(
                        stages(np.array(g["phase_id"]), task))
                    patched += 1
                data.attrs["num_demos"] = len(data)
                data.attrs["total"] = n_total
            print(f"  [{split}] {task}: patched (total so far {patched})", flush=True)
    print(f"\nDONE: patched {patched} demos, skipped {skipped} already done")


if __name__ == "__main__":
    main()
