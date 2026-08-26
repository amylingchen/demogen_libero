"""Write metainfo.json for the rendered goal OC dataset (plan §8).

This is the file that was deliberately left out until the OC observations
existed: its content is per-frame segmentation boxes, which cannot be
fabricated from a trajectory alone.

Two things differ from the object suite's metainfo and are NOT inherited:

- Task nouns. The goal instructions name fixtures and drawer parts, not just
  a pick object and a basket, so target/goal per task come from a table rather
  than from object_order[0]/[1].
- Phase wording. oc_obs.phase_stages phrases every phase as pick-and-place
  ("carry the X to the Y", "place the X into the Y"), which is false for the
  drawer, knob and push tasks: those are replayed as one rigidly translated
  trajectory whose phase 3 IS the operation. Their stage text is written from
  the task kind instead, matching what phase_map.json documents.

One file per split directory, keyed by "<task>/<demo>".

Usage:
    .venv\\Scripts\\python.exe scripts\\build_goal_metainfo.py --dir output/goal_oc_v3
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
from demogen_libero.oc_obs import ROBOT_SEG_ID

SPLITS = ["train", "quarantine_cf", "quarantine_unseen"]

# display names in GOAL_SEG_IDS order (ids 60,70,...,140), gripper excluded --
# boxes_for_seg-style id assignment is 60 + 10*index over this list
ENTITY_NAMES = G.GOAL_ENTITY_DISPLAY
ENTITY_IDS = [60, 70, 80, 90, 100, 110, 120, 130, 140]

# what the INSTRUCTION refers to, per task: (thing acted on, destination/None)
TASK_NOUNS = {
    "open_the_middle_drawer_of_the_cabinet": ("middle drawer", None),
    "turn_on_the_stove": ("stove", None),
    "put_the_bowl_on_the_plate": ("bowl", "plate"),
    "put_the_bowl_on_the_stove": ("bowl", "stove"),
    "put_the_bowl_on_top_of_the_cabinet": ("bowl", "cabinet"),
    "put_the_cream_cheese_in_the_bowl": ("cream cheese", "bowl"),
    "put_the_wine_bottle_on_the_rack": ("wine bottle", "wine rack"),
    "put_the_wine_bottle_on_top_of_the_cabinet": ("wine bottle", "cabinet"),
    "push_the_plate_to_the_front_of_the_stove": ("plate", "stove"),
}


def boxes(seg: np.ndarray) -> list:
    """Per-frame normalized [x, y, w, h] boxes keyed by display name."""
    frames = []
    h, w = int(seg.shape[1]), int(seg.shape[2])
    pairs = [("robot", ROBOT_SEG_ID)] + list(zip(ENTITY_NAMES, ENTITY_IDS))
    for frame in seg:
        fb = {}
        for name, sid in pairs:
            ys, xs = np.where(frame == sid)
            if xs.size == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            fb[name] = [int(sid), [x0 / w, y0 / h,
                                   (x1 - x0 + 1) / w, (y1 - y0 + 1) / h]]
        frames.append(fb)
    return frames


def stages(phase_id: np.ndarray, task: str) -> list:
    """Contiguous phase spans with an action word and instruction text that
    matches how the task was actually segmented."""
    kind = G.GOAL_TASKS[task]["kind"]
    tgt, dst = TASK_NOUNS[task]
    if kind == "pick_place":
        tpl = {0: ("move", tgt, f"move the gripper to the {tgt}"),
               1: ("grasp", tgt, f"grasp the {tgt}"),
               2: ("move", dst, f"carry the {tgt} to the {dst}"),
               3: ("place", dst, f"place the {tgt} on the {dst}")}
    elif kind == "fixture_op":
        verb = "pull" if "drawer" in task else "turn"
        tpl = {0: ("move", tgt, f"move the gripper to the {tgt}"),
               1: ("reach", tgt, f"settle onto the {tgt}"),
               2: ("move", tgt, f"stay at the {tgt}"),
               3: (verb, tgt, f"{verb} the {tgt}")}
    else:  # push
        tpl = {0: ("move", tgt, f"move the gripper to the {tgt}"),
               1: ("reach", tgt, f"settle behind the {tgt}"),
               2: ("move", tgt, f"stay at the {tgt}"),
               3: ("push", dst, f"push the {tgt} to the front of the {dst}")}
    phase_id = np.asarray(phase_id)
    out = []
    start = 0
    for i in range(1, len(phase_id) + 1):
        if i == len(phase_id) or phase_id[i] != phase_id[start]:
            pid = int(phase_id[start])
            action, obj, text = tpl[pid]
            out.append({"phase_id": pid, "start": int(start), "end": int(i - 1),
                        "action": action, "object": obj, "instruction": text})
            start = i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("output", "goal_oc_v3"))
    args = ap.parse_args()

    for split in SPLITS:
        d = os.path.join(args.dir, split)
        if not os.path.isdir(d):
            continue
        meta = {}
        for p in sorted(glob.glob(os.path.join(d, "*.hdf5"))):
            task = os.path.basename(p).replace("_demo.hdf5", "")
            tgt, dst = TASK_NOUNS[task]
            with h5py.File(p, "r") as f:
                for k in f["data"]:
                    g = f["data"][k]
                    entry = {
                        "success": True,
                        "task_description": task.replace("_", " "),
                        "task_nouns": ["robot", tgt] + ([dst] if dst else []),
                        "target_object": tgt,
                        "goal_object": dst,
                        "object_names": ENTITY_NAMES,
                        "seg_ids": dict(zip(ENTITY_NAMES, ENTITY_IDS)),
                        "layout_id": g.attrs.get("layout_id"),
                        "split": split,
                        "source_demo": g.attrs.get("source_demo"),
                        "initial_state": np.array(g["states"][0]).tolist(),
                        "phases": stages(np.array(g["phase_id"]), task),
                        "exo_boxes": boxes(np.array(g["obs"]["agentview_seg"])),
                        "ego_boxes": boxes(np.array(g["obs"]["eye_in_hand_seg"])),
                    }
                    meta[f"{task}/{k}"] = entry
            print(f"  [{split}] {task}: {sum(1 for x in meta if x.startswith(task))} demos",
                  flush=True)
        out = os.path.join(d, "metainfo.json")
        with open(out, "w") as f:
            json.dump(meta, f)
        print(f"[{split}] wrote {out} "
              f"({len(meta)} demos, {os.path.getsize(out)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
