"""Build the `--src-root` shaped view of a rendered goal OC dataset.

precompute_stageb_pack.py resolves a demo file's task directory with
`src_root.glob(f"*/{fname}")` and asserts EXACTLY ONE hit, then reads
`metainfo.json` next to it. The goal dataset stores `<split>/<task>_demo.hdf5`
with one metainfo per split, so the same filename exists in all three splits
and that assert fires. The object and spatial suites store
`<split>/<task>/<task>_demo.hdf5`, which is what this produces.

The hdf5 files are HARD LINKED, not copied: both shapes cost one copy on disk.
The flat original stays the input for precompute_libero.py, whose --demo-dir
glob is non-recursive.

metainfo is re-keyed from goal's flat `"<task>/<demo>"` form into the nested
`{task: {demo: {...}}}` form the pack builder indexes. Same content, same
values -- only the key nesting changes.

NOT fixed here (they are schema decisions, not layout): the pack builder also
asserts `object_names[0] == target_object and object_names[1] == goal_object`,
reads one `offset_body` per name out of object_geometry.json, and indexes
obs/obj_pos with one column per name. goal carries 9 entities in a fixed
canonical order, only 4 of which have free joints (hence 4 obj_pos columns),
its goal_object is None for the drawer and knob tasks, and its geometry file
has no entry for either drawer.

Usage:
    python scripts/build_goal_srcroot_tree.py \
        --oc-dir F:/goal_oc_v4 --geometry output/goal_gen_v3/object_geometry.json \
        --out F:/goal_oc_v4_srcroot
"""
import argparse
import glob
import json
import os
import shutil
from collections import defaultdict

SPLITS = ["train", "quarantine_cf", "quarantine_unseen"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oc-dir", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    n_link = n_meta = 0
    for split in SPLITS:
        src = os.path.join(a.oc_dir, split)
        meta_path = os.path.join(src, "metainfo.json")
        if not os.path.isdir(src):
            continue
        flat = json.load(open(meta_path, encoding="utf-8"))
        nested = defaultdict(dict)
        for key, entry in flat.items():
            task, demo = key.rsplit("/", 1)
            nested[task][demo] = entry

        for f in sorted(glob.glob(os.path.join(src, "*.hdf5"))):
            fname = os.path.basename(f)
            task = fname[:-len("_demo.hdf5")]
            d = os.path.join(a.out, split, task)
            os.makedirs(d, exist_ok=True)
            link = os.path.join(d, fname)
            if not os.path.exists(link):
                os.link(f, link)          # hard link, same volume
                n_link += 1
            assert task in nested, f"{split}: no metainfo entries for {task}"
            with open(os.path.join(d, "metainfo.json"), "w", encoding="utf-8") as fh:
                json.dump({task: nested[task]}, fh)
            shutil.copyfile(a.geometry, os.path.join(d, "object_geometry.json"))
            n_meta += 1
        print(f"  [{split}] {len(list(nested))} tasks")
    print(f"linked {n_link} hdf5, wrote {n_meta} per-task metainfo/geometry")


main()
