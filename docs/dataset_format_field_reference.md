# Dataset Format & Field Reference — Explained with a Real Sample

> Running example: **demo_0** from
> `output/libero_object_100/salad_dressing/...demo.hdf5`
> (T=144 frames, source demo_28, target object moved from its source position
> (0.05, -0.10) to the new position (0.065, 0.087)). All numbers below are the
> actual values of that episode.

---

## 1. HDF5 structure overview

```
data/
└── demo_0/          (attrs: num_samples, object_names, object_instances,
                             phases, subtasks, subtask_vocab)
    ├── actions        (144, 7)   float64   ← executed actions
    ├── states         (144, 110) float64   ← full MuJoCo sim state
    ├── robot_states   (144, 8)   float64   ← compact robot state
    ├── rewards        (144,)     uint8     ← last-frame success flag
    ├── dones          (144,)     uint8     ← last-frame done flag
    ├── phase_id       (144,)     int32     ← fine-grained stage index (pairs with
    │                                         attr `phases`: move/grasp/move/place
    │                                         + per-stage target object)
    ├── subtask_id     (144,)     int32     ← coarse closed-vocabulary label (pairs
    │                                         with attr `subtasks`: transit/move/idle,
    │                                         paper convention)
    └── obs/
        ├── agentview_rgb / eye_in_hand_rgb        (144,256,256,3) uint8
        ├── agentview_depth / eye_in_hand_depth    (144,256,256)   uint8  centimeters
        ├── agentview_depth_mm / eye_in_hand_depth_mm (144,256,256) uint16 millimeters
        ├── agentview_seg / eye_in_hand_seg        (144,256,256)   uint8  instance seg
        ├── ee_pos (144,3) · ee_ori (144,3) · ee_states (144,6)
        ├── gripper_states (144,2) · joint_states (144,7)
        ├── obj_pos  (144,7,3)  ← GT world positions of the 7 objects
        └── obj_quat (144,7,4)  ← object orientation quaternions (xyzw)
```

**Temporal alignment (important)**: `obs/*[t]` is the observation **before**
executing `actions[t]` (pre-step alignment). Hence `states[0]` / `ee_pos[0]` /
the frame-0 images describe the initial scene, and a BC mapping trained on
`obs[t] → actions[t]` never learns to "re-execute a motion it has already
completed".

---

## 2. Field-by-field reference (with demo_0's actual values)

### actions (T,7) — OSC pose-delta commands

```
t=0   : [ 0.067  0.215 -0.117 -0.019  0.061 -0.    -1. ]
t=60  : [ 0.206  0.     0.686  0.     0.079 -0.     1. ]
t=143 : [ 0.     0.     0.     0.     0.     0.    -1. ]
```

| Dims | Meaning | Scale |
|---|---|---|
| [0:3] | EE translation delta dx,dy,dz | normalized, ±1 ↔ ±5 cm/step |
| [3:6] | EE rotation delta (axis-angle) | normalized |
| [6] | gripper command | -1 open / +1 close |

Reading: at t=0 the arm advances in (+x,+y) with a slight descent (toward the
new target at y=0.087), gripper open; at t=60, having grasped, it lifts hard
in +z (0.686 ≈ 3.4 cm/step) with the gripper held at +1; t=143 is a tail hold
frame (all zeros + open).

### obs/ee_pos (T,3), obs/ee_ori (T,3), obs/ee_states (T,6)

EE world position (meters) and orientation (**axis-angle**, rad);
ee_states = [pos, ori] concatenated.

```
t=0   : ee_pos=[-0.140 -0.007  0.255]   ee_ori=[3.135 -0.002 -0.106]
t=60  : ee_pos=[ 0.043  0.087  0.124]   ← above the target, descending
t=143 : ee_pos=[-0.022  0.263  0.206]   ← above the basket
```

`ee_ori ≈ [π, 0, ~0]` means the gripper points straight down (180° about x);
the third component is the wrist yaw.

### obs/gripper_states (T,2) — two-finger opening qpos

```
t=0: [0.035 -0.035] (fully open)   t=60: [0.017 -0.019] (holding the bottle)   t=143: [0.039 -0.039] (open after release)
```

Fingers are symmetric; |value| ≈ 0.04 is fully open. When holding an object
the opening equals the object radius — a usable proxy for "actually grasped".

### obs/joint_states (T,7) — the 7 arm joint angles (rad)

### robot_states (T,8) — compact robot state

`[ee_pos(3), ee_ori axis-angle(3), gripper_qpos(2)]`, i.e. at t=0
`[-0.140 -0.007 0.255 | 3.135 -0.002 -0.106 | 0.035 -0.035]` — the
concatenation of ee_states and gripper_states, saved as one ready-made vector.

### obs/obj_pos (T,7,3), obs/obj_quat (T,7,4) — GT object poses

The 7-object order is given by the `object_names` attr (aligned with seg ids):

| idx | object | seg id | demo_0 position at t=0 | at t=143 |
|---|---|---|---|---|
| 0 | salad dressing 1 (target) | 60 | [0.065 0.087 **0.073**] (on floor) | [-0.016 0.277 **0.090**] (**in basket**) |
| 1 | basket 1 | 70 | [-0.012 0.274 -0.005] (fixed) | same |
| 2..6 | ketchup/soup/cream/milk/tomato | 80..120 | placed, or parked (x≥2) | — |

Quaternions use the **xyzw** convention, world frame. The target's z going
0.073 → (lifted) → 0.090 over time makes this directly usable for 3D position
supervision; `obj_pos[t,0]` landing at the basket's xy is geometric evidence
of task completion.

**The shape is fixed at 7 and does not vary with the number of objects in the
scene**: distractors that are not placed still have entries whose values are
the off-scene parking coordinates (x = 2.0/2.4/2.8/3.2/3.6, y = 2.0). This
lets downstream code batch by fixed index (order = object_names = seg-id
order). Generate the in-scene mask yourself:

```python
in_scene = obj_pos[:, :, 0] < 1.0   # (T,7) bool
```

Always mask parked objects during training (their coordinates are legal
numbers; regressing on them directly will teach the network the parking
position). Alternatively, use `distractor_joints` in scene_log to get the
placed list.

### states (T,110) — full MuJoCo state

`[time(1), qpos(nq), qvel(nv)]` flattened. Uses: exact reproduction of any
frame (`env.set_init_state(states[t])`), offline re-rendering. The object
free-joint slots inside qpos are documented in `object_geometry` / pipeline
code, but **downstream code never needs to parse them** — obj_pos is provided
directly.

### rewards / dones (T,)

Sparse success flags: zero everywhere, with `rewards[-1]=1, dones[-1]=1` on
the **final frame** (successful trajectories).

### phase_id (T,) + attr `phases` — fine-grained stages (action word + per-stage target)

`phase_id[t] ∈ {0,1,2,3}` indexes the demo attr `phases`, each stage carrying
an action word and a target:

```json
[{"phase":0,"action":"move", "target":"salad dressing 1","start":0,  "end":44,
  "instruction":"move the gripper to the salad dressing"},
 {"phase":1,"action":"grasp","target":"salad dressing 1","start":44, "end":64,
  "instruction":"grasp the salad dressing"},
 {"phase":2,"action":"move", "target":"basket 1",        "start":64, "end":110,
  "instruction":"carry the salad dressing to the basket"},
 {"phase":3,"action":"place","target":"basket 1",        "start":110,"end":144,
  "instruction":"place the salad dressing into the basket"}]
```

Conventions: stages 0/1 target the object to pick; stages 2/3 target the
basket. grasp includes the pre-grasp position correction and any regrasp
cycles; place includes the tail hold frames. metainfo carries the same
"phases". Complementary to the coarse subtasks below (paper convention, where
grasp–carry–release is one move): use phases for fine-grained supervision,
subtasks for paper alignment.

### subtask_id (T,) + attr `subtasks` — closed-vocabulary subtasks (primary segmentation)

`subtask_id[t]` indexes the `subtask_vocab` attr (18 words: 15 primitives +
transit/idle/other): in demo_0 `15=transit, 0=move, 16=idle`. The span table
lives in the `subtasks` attr:

```json
[{"action":"transit","object":"salad dressing 1","destination":null,
  "instruction":"move the gripper to the salad dressing","start":0,"end":50},
 {"action":"move","object":"salad dressing 1","destination":"basket 1",
  "instruction":"move the salad dressing into the basket","start":50,"end":124},
 {"action":"transit","object":null,"destination":null,
  "instruction":"retreat the gripper","start":124,"end":141},
 {"action":"idle","object":null,"destination":null,
  "instruction":"keep the arm stationary","start":141,"end":144}]
```

Field meanings: `action` — vocabulary action; `object` — the object the
subtask interacts with / heads toward (the per-stage target object);
`destination` — placement destination (move only); `instruction` — short
language instruction; `[start, end)` — frame span. Boundary convention: move
spans from the **first gripper close** (contact) to the **last open**
(release) — grasp-carry-place is a single move, and regrasps fold inside it.

**Why the two annotations have different boundaries** (same demo_0: grasp in
`phases` starts at 44, move in `subtasks` starts at 50): `phases` counts the
**pre-grasp position-correction frames (settle, 44–50, gripper not yet
closed)** as part of the grasp stage; `subtasks` follows the paper and uses
**gripper–object contact (first close command)** as the boundary, so settle
frames remain transit. Each annotation is internally consistent — just don't
mix their boundaries.

### Images & segmentation

- `*_rgb`: 256×256, **already flipped upright** (unlike raw LIBERO's OpenGL
  bottom-up convention);
- `*_depth`: real depth quantized at **1 cm per level**, uint8 (caps at 255 ≈
  2.55 m);
- `*_depth_mm`: the same depth at **1 mm per level**, uint16 (effectively
  lossless) — use it for point clouds / thin-object geometry;
- `*_seg`: instance segmentation, `0=background, 50=robot end-effector
  (gripper only), 60=target, 70=basket, 80/90/100/110/120=distractors`
  (order = object_names).

### Demo-level attrs

| attr | meaning |
|---|---|
| `num_samples` | frame count T |
| `object_names` / `object_instances` | 7 object display / instance names (defines obj_pos & seg-id order) |
| `phases` | fine-grained stage table (action word + per-stage target + instruction + frame span; pairs with phase_id) |
| `subtasks` / `subtask_vocab` | coarse subtask span table (pairs with subtask_id) / the 18-word vocabulary |

---

## 3. metainfo.json (one entry per demo)

```
keys: phases, subtasks, success, initial_state, task_nouns, task_description,
      target_object, goal_object, object_names, exo_boxes, ego_boxes
```

- `phases` / `subtasks`: the same two segmentation tables as the HDF5 attrs
  (fine/coarse), so metainfo-only consumers get segmentation and per-stage
  targets directly;
- `target_object: "salad dressing 1"`, `goal_object: "basket 1"` — explicit
  targets; no task-name parsing needed downstream;
- `exo_boxes / ego_boxes`: per-frame 2D boxes (agentview / wrist camera),
  format `{object name: [seg_id, [x, y, w, h]]}` with coordinates
  **normalized by image width/height**. Example, demo_0 frame-0 target:
  `[60, [0.5625, 0.4766, 0.0781, 0.1680]]` = box top-left at (0.56W, 0.48H),
  width 0.078W, height 0.168H;
- `initial_state`: the 110-D initial sim state (for scene reproduction in
  evaluation);
- `task_nouns / task_description`:
  `["robot", "salad dressing 1", "basket 1"]` / the task sentence.

## 4. scene_log.json (provenance, one entry per demo)

```
demo_0: scene_id=0, source_demo=demo_28, seed=1009,
        target_old_xy=[0.050,-0.100] → target_new_xy=[0.065, 0.087],
        basket_delta=[-0.013, 0.020],
        distractor_joints=[milk, ketchup, tomato_sauce], n_distractors=3
```

Meaning: this trajectory was synthesized from source **demo_28**; the target
was translated from its source position to the new one (the synthesis offset
`obj_t` = new − old); the basket jitter is `basket_delta` (= `tar_t`); and the
listed 3 distractors were placed. Two demos sharing a `scene_id` have
identical placements and different sources.

## 5. Dataset-level auxiliary files

| file | contents |
|---|---|
| `camera_params.json` | both cameras' intrinsics K; agentview's fixed extrinsics and world→pixel matrix; wrist-camera hand-eye transform T_ee_cam (per-frame extrinsics = T_world_ee @ T_ee_cam); notes that images are flipped (apply row → H-1-row after projection) |
| `object_geometry.json` | each object's `offset_body` (bbox center relative to the free-joint origin in the body frame, e.g. basket z +7.38 cm) and `extents` (full bbox size) — world geometric center = obj_pos + R(quat) @ offset_body |
| `phase_map.json` | phase_id label table |
| `init_states_all.png` | mosaic of every demo's frame 0 (red box = target; title `index sSOURCE·dNDISTRACTORS`) + placement scatter |
| `viz_phases/` | phase plots and annotated videos for sampled demos |
