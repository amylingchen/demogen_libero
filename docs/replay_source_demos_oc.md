# Replaying source demos into OC format (`scripts/replay_source_oc.py`)

Re-execute existing LIBERO demos in the simulator and re-record them in the
full **OC format** — RGB + depth (cm `uint8` + lossless mm `uint16`) + instance
segmentation + per-frame ground-truth object poses (`obj_pos` / `obj_quat`) +
two-level phase / subtask annotation.

Unlike the augmentation pipeline (`run_grid_oc_demo.py`), this script does **not**
re-place objects or synthesize new trajectories. It reproduces each source demo
*exactly*, only enriching it with the observation channels the original LIBERO
files don't ship (depth, segmentation, GT poses). Use it when you want a
faithful OC-format copy of demos you already have.

## Why STATE replay (not action replay)

The script sets each stored sim state frame-by-frame
(`env.set_init_state(states[t])`) and renders a fresh observation, rather than
re-stepping the recorded actions. LIBERO's OSC controller converges only
partway within one control step, so open-loop action replay drifts and can fail
on contact-rich demos (regrasps especially). State replay is byte-for-byte
faithful to the human trajectory. This is the same mechanism as
`append_regrasp_demos.py`, generalized from "regrasp demos only" to "any demo".

Alignment is **pre-step**: `obs[t]` is rendered from `state[t]` and paired with
`action[t]`, matching the rest of the dataset (a policy trained on
`obs[t] -> action[t]` never re-executes a completed motion).

## Where the data comes from

| channel | source | code |
| --- | --- | --- |
| RGB, depth, seg | rendered by `oc_obs.make_oc_env` (`camera_depths=True`, `camera_segmentations="instance"`) | `oc_obs.extract_oc_frame` |
| depth cm / mm | `get_real_depth_map` → metric metres → `uint8` cm and `uint16` mm | `oc_obs.py` |
| `obj_pos` / `obj_quat` | sim obs `<name>_pos` / `<name>_quat`, ordered by `OBJECT_ORDER` | `oc_obs.extract_oc_frame` |
| `phase_id` | segmentation of the source (`auto` or `regrasp`) | `trajectory.segment_*` |
| `subtask_id` / `subtasks` | closed-vocabulary annotation | `oc_obs.annotate_subtasks` |

Only `states`, `actions`, and `obs/ee_pos` are read from the source file — any
HDF5 with those three keys can be a replay source (the original LIBERO datasets,
or an already-generated OC dataset).

## Usage

```bash
PY=.venv/Scripts/python.exe   # .venv/bin/python on Linux

# original LIBERO source (auto-resolves the bddl from the source file)
$PY scripts/replay_source_oc.py \
    --task-key pick_up_the_salad_dressing_and_place_it_in_the_basket \
    --out-dir output/replay_salad

# a subset of demos, dropping the lossless mm depth
$PY scripts/replay_source_oc.py --task-key <key> --demos demo_0 demo_1 --no-depth-mm

# replay an already-generated OC dataset as the source (it has no
# bddl_file_name attr, so pass --src and --bddl explicitly)
$PY scripts/replay_source_oc.py \
    --task-key pick_up_the_salad_dressing_and_place_it_in_the_basket \
    --src  output/libero_object_100/salad_dressing/..._demo.hdf5 \
    --bddl LIBERO/.../bddl_files/libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket.bddl \
    --out-dir output/replay_salad
```

### Arguments

| flag | default | meaning |
| --- | --- | --- |
| `--task-key` | *(required)* | task name; keys `OBJECT_ORDER` and `source_hdf5()` |
| `--src` | resolved from `--task-key` | source HDF5 (`states` / `actions` / `obs/ee_pos`) |
| `--out-dir` | `output/replay_<task-key>` | output directory |
| `--demos` | all | demo keys to replay, e.g. `demo_0 demo_1` |
| `--segment` | `regrasp` | phase segmentation: `regrasp` folds regrasp cycles into `skill_1`; `auto` for single-grasp demos |
| `--no-depth-mm` | off | skip the `uint16` mm-depth channels (keep only `uint8` cm) |
| `--bddl` | from source attr | bddl override, for sources lacking a `bddl_file_name` attr |

Demos whose state replay does not reach `check_success()` are skipped and
reported. Output episodes are renumbered `demo_0..` in success order.

## Output

`<out-dir>/` contains:

- `<task-key>_demo.hdf5` — OC-format episodes (`write_oc_demo` layout;
  `obs/*` with rgb/depth/seg/mm-depth/obj_pos, plus `actions`, `states`,
  `robot_states`, `rewards`, `dones`, `phase_id`, `subtask_id`).
- `metainfo.json` — per-demo entry (`metainfo_entry`): target/goal objects,
  phase stages, subtask spans, and per-frame 2D boxes (`exo_boxes` / `ego_boxes`).

Visualize the result with the standard viewer:

```bash
$PY scripts/visualize_oc_demo.py --dir output/replay_salad --demo demo_0 --frames 0 40 80 120
```

which writes per-frame panels (RGB + bbox, depth, seg for both cameras),
`actions.png`, and a bbox-overlay mp4.

## Related

- `docs/dataset_format_field_reference.md` — full field reference for the OC HDF5.
- `scripts/append_regrasp_demos.py` — the regrasp-only precursor this generalizes.
- `scripts/run_grid_oc_demo.py` — the augmentation pipeline (re-places objects,
  synthesizes new trajectories) rather than a faithful copy.
