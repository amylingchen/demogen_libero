# Generating New-Position Trajectories from Source Demos — Method Notes

> This document records the core implementation of the DemoGen→LIBERO pipeline:
> how a single human demonstration (source demo) is transformed into a
> successful trajectory with the object placed at an arbitrary new position.
> Steps follow the pipeline execution order, with the key findings and
> pitfalls of each step. Code lives in `src/demogen_libero/` and `scripts/`.

---

## Overview

```
source demo (HDF5)                 new scene (grid / continuously sampled placement)
   │ ① segmentation                    │ ③ scene sampling (anti-occlusion)
   ▼                                   ▼
(f1,f2,f3) two-stage boundaries  +  obj_t / tar_t (target/basket translation offsets)
   │                                   │
   └───────────► ② trajectory synthesis ◄──────────┘
                 synthesize_uniform
                 (arc-length reparameterization + source speed profile)
                        │
                        ▼
                 ④ closed-loop replay (feedforward + feedback + settle)
                 replay_uniform
                        │
                        ▼
                 ⑤ success filtering + ⑥ OC-format recording (with subtask annotation)
```

Core principle (DemoGen): LIBERO actions are **7-D OSC deltas**
`[dx,dy,dz,dax,day,daz,grip]`; position deltas are **translation-invariant** —
after translating the object and the end-effector together, the same relative
action sequence remains valid. Therefore:

- **skill segments** (grasp / place, contact-rich): actions are **replayed
  verbatim, frame by frame**;
- **motion segments** (free-space moves): **re-planned** for the new
  start/end points.

---

## ① Source-demo segmentation (trajectory.py)

A pick-and-place demo is split into 4 segments:
`motion_1 (reach) → skill_1 (grasp) → motion_2 (transport) → skill_2 (place)`,
with boundaries `(f1, f2, f3)`.

**Automatic segmentation `auto_segment(state, action)`**
(state = absolute EE position sequence):

| Boundary | Criterion |
|---|---|
| `f1` (grasp start) | first frame the gripper command flips from -1 to +1 |
| `f2` (transport start) | first frame after `f1` where EE speed recovers above 9 mm/frame, plus a 2-frame margin |
| `f3` (place start) | 10 frames before the final gripper-open command (keeps the descent into the basket inside skill_2) |

**Regrasp extension `segment_regrasp`**: some human demos contain a
grasp–release–regrasp pattern (multiple gripper open/close cycles). The rules
become: `f1` = **first** close, `f2` = speed recovery after the **last** close,
`f3` = 10 frames before the **last** open — the whole regrasp episode folds
into skill_1, and the function reduces exactly to `auto_segment` for
single-grasp demos (verified frame-identical).

**Pitfall**: automatic segmentation assumes a single grasp/release cycle;
without screening, a regrasp demo gets mis-segmented (once produced a
pathological label with a 1-frame transport). Source demos are therefore
classified first by `screen_sources.py`: `healthy` (single grasp AND open-loop
zero-offset replay succeeds) / `regrasp` (multi-cycle; use regrasp
segmentation) / `bad` (open-loop replay fails — demos with millimeter-scale
contact margins drop the object under any perturbation; even closed-loop
replay cannot save them, so they are discarded).

---

## ② Trajectory synthesis — synthesize_uniform (trajectory.py)

Given offsets `obj_t` (new target position − source position) and `tar_t`
(basket offset), synthesize the **reference path `ref_path`** and the
**base actions `base_actions`**:

1. **Skill segments verbatim**: `s1 = action[f1:f2]`, `s2 = action[f3:]` are
   kept frame-by-frame; their reference path = source EE path + a constant
   offset (`obj_t` for skill_1, `tar_t` for skill_2).
2. **Motion-segment offset ramp, completed early**: motion_1's reference path
   = source path + `w(t)·obj_t`, where `w(t)` rises linearly from 0 to 1 over
   the **first 70%** of the segment (`ramp_frac=0.7`) and then holds — the
   lateral shift happens at cruise height, and the **final descent is purely
   vertical**.
   - Pitfall: the ramp initially spanned the whole segment, so the EE was
     still shifting sideways during the descent → low-altitude sweeps knocked
     over neighboring objects, and the gripper closed while still sliding,
     grasping bottles at a tilt. Both failure modes disappeared after
     completing the ramp early.
3. **Arc-length reparameterization (adaptive duration)**: motion segments are
   resampled by path arc length at the source demo's own average speed —
   **frame count grows with the actual path length** instead of being locked
   to the source frame count.
   - Pitfall 1 (fixed frame count): near and far placements got the same
     number of frames → far ones moved faster per step, producing a
     pathological speed distribution (v_mean varying 5.9–9.4 mm/step).
   - Pitfall 2 (pure constant speed): erased the source's "decelerate when
     approaching the object" profile → approaches were too fast and grasps
     failed.
   - Fix: preserve the **normalized speed profile** during resampling (the
     normalized-time → normalized-arc-length mapping is taken from the source
     segment). Duration ∝ distance, and the tempo follows the human demo.
     After the fix, dataset-wide v_mean converged to 5.8–6.9 mm/step.
4. Motion-segment rotation deltas are resampled with magnitude scaled by the
   frame-count ratio (total rotation preserved); gripper commands are held
   per segment.

---

## ③ Scene sampling (gridscene.py)

Scenes are defined in **absolute coordinates**, so the same scene can be
applied to any source demo's initial state (same placement × multiple source
demos = `demos-per-scene`):

- **Target object**: sampled **continuously and uniformly** inside the
  workspace (not on a grid), with farthest-point scoring (away from already
  used positions) so the whole dataset spreads out. No basket-distance bonus —
  it systematically biased placements away from the basket (learned the hard
  way).
- **Distractors**: 0.11 m grid + 15% jitter, pairwise spacing ≥ `min_spacing`
  (0.10 m), ≥ 0.12 m from the basket; **object identities are fixed in the
  scene** (all sources share the exact same layout).
- **Basket**: reference position ± 1 cm jitter → `tar_t`.
- **Anti-occlusion (two layers)**:
  1. Geometric filter: reject placements nearly collinear with the camera ray
     through any other object (lateral offset < 0.035 m);
  2. Rendered hard gate: after placing, render frame 0 and require each
     object's segmentation blob to exceed a pixel threshold — **size-aware**
     (60 px for flat objects with z-extent < 5 cm, 150 px otherwise).
     Pitfall: a flat 150 px threshold misjudged butter's (7.7×4×1.8 cm)
     **unoccluded** small blob at far positions as "occluded", collapsing the
     task's success rate from 92% to 2%.
- Unselected distractors are parked off-scene (x ≥ 2.0); free-joint
  quaternions are normalized and qvel zeroed.
- `apply_scene` writes the absolute coordinates into the corresponding qpos
  slots of the source demo's 110-D init state, yielding the new initial state
  together with `obj_t / tar_t`.

---

## ④ Closed-loop replay (libero_replay.replay_uniform)

**Why closed-loop is mandatory**: LIBERO's OSC controller targets "current
**actual** EE position + delta" every step and only converges 80–90% within a
step; the shortfall **can never be recovered open-loop**. The human implicitly
compensated for this in the source actions, but our superimposed corrections
have no such compensation — at a 25 cm offset the EE was still 4.4 cm short at
grasp time, guaranteeing a miss. This was the root cause of the early
"all large offsets fail" phase.

Execution policy (motion segments):

```
act[:3] = clip( ff_gain·(ref[t+1]−ref[t])/0.05      # feedforward: reference velocity ×1.25 to offset undershoot
              + clip(gain·(ref[t]−ee)/0.05, ±0.4)   # feedback: pull back to the reference path, capped at 2 cm/step
              , ±1 )
```

Skill segments run strictly open-loop (verbatim actions) so the contact
dynamics match the source demonstration.

**Settle**: before entering skill_1/skill_2, up to 20 "feedback-only, gripper
held" transition frames are inserted until the EE is within 5 mm of the
reference point, and only then does the gripper close / the placement descend —
the residual error of large offsets is absorbed here (this is the key to
succeeding at 25 cm offsets). Phase attribution of settle frames: pre-grasp
correction → skill_1, pre-place → skill_2.

**Recording semantics** (BC-friendly; fixed on request):
- **Pre-step alignment**: `obs[t]` is the observation **before** executing
  `action[t]` (`states[0]`/`ee_pos[0]` = the initial scene), eliminating the
  forward-lunge artifact of "re-executing a motion already completed";
- **5 zero-action warm-up frames** before recording (not recorded) let the
  physics/controller settle;
- **3 zero-action hold frames** appended at the end (recorded, attributed to
  skill_2).

---

## ⑤ Success filtering

`env.check_success()` is checked every frame during replay; success at any
frame counts. Failed attempts **retry with a different source demo** (at most
`demos_per_scene + scene_retries` attempts per scene); scenes that still fall
short are dropped. Typical single-attempt success rates: 85–99% for
single-grasp tasks, ~40% for regrasp sources (contact-sensitive, expected).

---

## ⑥ Recording (oc_obs.py)

Successful trajectories are written in the LIBERO-OC format (field-for-field
compatible with the reference `output/demo`) plus:

- `obs/`: dual-camera RGB, depth (uint8 centimeters + lossless uint16
  millimeters), instance segmentation (ids: robot gripper=50, target=60,
  basket=70, distractors 80–120), proprioception;
- `obs/obj_pos (T,7,3)`, `obs/obj_quat (T,7,4)`: per-frame ground-truth object
  poses (world frame, xyzw), copied straight from robosuite obs so downstream
  code never parses the raw state vector;
- `subtask_id` + `subtasks` attr: closed-vocabulary subtask annotation
  (arXiv:2607.06403) —
  `transit (approach, object=target) → move (first close → last open,
  object=target, destination=basket) → transit (retreat) → idle (tail)`;
  regrasp cycles fold naturally into the single move;
- `phase_id` (internal 4-phase labels, used by tooling) and metainfo
  (per-frame bboxes, `target_object`/`goal_object`, subtasks);
- each dataset directory ships `camera_params.json` (intrinsics + agentview
  extrinsics + hand-eye T_ee_cam) and `object_geometry.json` (mesh-vertex AABB
  center offsets + extents; the basket origin is offset +7.38 cm in z).

---

## Key takeaways at a glance

| Question | Answer |
|---|---|
| Why can skill segments be relocated | OSC delta actions are translation-invariant; shifting EE and object together preserves their relative geometry |
| Why open-loop fails | OSC re-anchors to the actual position each step, undershooting 10–20%/step; superimposed corrections can't catch up open-loop |
| The last centimeter at large offsets | settle to < 5 mm before entering each skill segment |
| When to do the lateral shift | complete it in the first 70% of the motion segment; the descent stays purely vertical, avoiding low sweeps and tilted grasps |
| Speed consistency | arc-length reparameterization (duration ∝ distance) + preserving the source's normalized speed profile |
| Fragile source demos | screen first: discard demos whose open-loop zero-offset replay fails; route regrasp demos through regrasp segmentation |
| Occlusion detection | the geometric ray filter is only a coarse pre-filter; the rendered segmentation pixel count (size-aware threshold) is the hard gate |
| The "overshoot hook" in reach paths | inherited from the human source demos (source overshoot 8.5 cm > generated 4.8 cm), not a synthesis artifact |

## Related scripts

| Script | Purpose |
|---|---|
| `scripts/screen_sources.py` | source-demo health screening (healthy/regrasp/bad) |
| `scripts/run_grid_oc_demo.py` | main generation: scene sampling → synthesis → closed-loop replay → OC recording (`--task/--segment/--primary-mode`) |
| `scripts/run_all_tasks.sh` / `topup_all.sh` | 10-task batch production / top-up to target counts |
| `scripts/build_eval_suite.py` | evaluation init-state scenes disjoint from training (adaptive distance threshold) |
| `scripts/append_regrasp_demos.py` | state-replay append of original regrasp demos |
| `scripts/visualize_init_states.py` / `visualize_phases.py` | initial-state mosaic / phase plots + annotated videos |
| `scripts/dump_camera_params.py` / `dump_object_geometry.py` | camera parameter / object geometry export |
