# demogen-libero

DemoGen-style trajectory augmentation for **LIBERO** (libero_object suite):
take a handful of human source demos per task, re-place the objects anywhere in
the workspace, synthesize retargeted trajectories, replay them closed-loop in
the real simulator with success filtering, and record a rich OC-style dataset —
RGB / depth (cm + lossless mm) / instance segmentation / per-frame object GT
poses / two-level subtask annotation / per-frame 2D boxes.

Built and validated on all 10 `libero_object` pick-and-place tasks
(100+ demos per task + a disjoint eval suite each).

## How it works (short version)

LIBERO actions are 7-D OSC deltas — position deltas are translation-invariant,
so the contact-rich **skill segments** (grasp / place) of a source demo replay
verbatim at any object position, while the free-space **motion segments** are
re-planned: the source EE path is shifted by the object offset (lateral shift
completed in the first 70% of the segment), re-sampled by arc length at the
source's own speed profile, and executed with feedforward + position feedback
(LIBERO's OSC undershoots each step, so pure open-loop replay systematically
misses at large offsets). A settle-to-5mm wait before each skill segment
absorbs residual error. Scenes are sampled with pairwise spacing and
**anti-occlusion** guarantees (camera-ray filter + rendered segmentation gate
with size-aware thresholds).

Full write-up: `docs/method_generating_new_position_trajectories.md`;
dataset field reference: `docs/dataset_format_field_reference.md`.

## Setup

Python 3.11. Tested on Windows (offscreen WGL rendering); Linux should work
with `MUJOCO_GL=egl|osmesa`.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows
# .venv/bin/pip install -r requirements.txt          # Linux

# LIBERO from source (pinned deps already installed above)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO
# make it importable, e.g.:
echo <abs-path-to>/LIBERO > .venv/Lib/site-packages/libero_repo.pth
```

**Windows-only patches** (robosuite 1.4.0; re-apply after any reinstall):

1. `robosuite/utils/binding_utils.py` — guard the bundled `mujoco.dll` load:
   ```python
   if _SYSTEM == "Windows":
       _dll = os.path.join(os.path.dirname(__file__), "mujoco.dll")
       if os.path.exists(_dll):
           ctypes.WinDLL(_dll)
   ```
2. `robosuite/macros.py` — set `MUJOCO_GPU_RENDERING = False`
   (1.4.0 otherwise forces `MUJOCO_GL=egl`, invalid on Windows).

First LIBERO import asks for a config path interactively; pre-create
`~/.libero/config.yaml` if running non-interactively.

## Data

Source demos (original LIBERO datasets) are read from `data/libero_object/`
by default; override with the `DEMOGEN_LIBERO_DATA` environment variable
(see `src/demogen_libero/config.py`). Download:

```python
from huggingface_hub import hf_hub_download
for task in [...]:  # e.g. pick_up_the_butter_and_place_it_in_the_basket
    hf_hub_download(repo_id="yifengzhu-hf/LIBERO-datasets", repo_type="dataset",
                    filename=f"libero_object/{task}_demo.hdf5",
                    local_dir="data")   # lands in data/libero_object/
```

## Usage

```bash
PY=.venv/Scripts/python.exe   # .venv/bin/python on Linux

# 1) screen source demos per task (healthy / regrasp / bad)
$PY scripts/screen_sources.py

# 2) generate one task: N scenes x M source demos per scene
$PY scripts/run_grid_oc_demo.py --task butter \
    --sources-json output/source_screening.json \
    --n-scenes 50 --demos-per-scene 2 --out-dir output/butter_100

# regrasp-style trajectories (multi-close grasp sources)
$PY scripts/run_grid_oc_demo.py --task salad_dressing --segment regrasp ...

# 3) all 10 tasks + eval suites, then fill to exactly N
bash scripts/run_all_tasks.sh
bash scripts/topup_all.sh

# 4) held-out eval scenes (init states only, disjoint from training)
$PY scripts/build_eval_suite.py --task butter \
    --train-scene-log output/butter_100/scene_log.json \
    --n-scenes 30 --out-dir output/butter_eval

# visualization
$PY scripts/visualize_init_states.py --dir output/butter_100
$PY scripts/visualize_phases.py --dir output/butter_100 --demos demo_0
$PY scripts/visualize_oc_demo.py --dir output/butter_100 --demo demo_0
```

Each generated task directory is self-contained: episodes HDF5, `metainfo.json`
(per-frame 2D boxes, target/goal objects, subtask spans), `scene_log.json`
(provenance), `camera_params.json` (intrinsics + hand-eye), 
`object_geometry.json` (AABB + free-joint origin offsets), placement mosaic,
and sample annotated videos.

## Layout

```
src/demogen_libero/
  config.py         data locations (env var DEMOGEN_LIBERO_DATA)
  convert.py        HDF5 demo -> state / action / init_state
  trajectory.py     segmentation (auto / regrasp) + synthesis
  gridscene.py      scene sampling (spacing, anti-occlusion, scene reuse)
  libero_replay.py  env helpers + closed-loop executors
  oc_obs.py         OC observation extraction, subtask annotation, writers
  tasks_config.json per-task joints + object order (generated from bddl)
scripts/            generation / screening / eval / visualization / dumps
docs/               method write-up + dataset field reference (EN + 中文)
tests/              unit tests for the synthesis math
```
