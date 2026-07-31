# LIBERO_Spatial: relational scenes and paired two-relation data

The `libero_object` pipeline moves a target object anywhere on the table and
retargets the trajectory. That does not transfer to `libero_spatial`, because
there the **instruction is itself a spatial relation** — "pick up the black bowl
**on the stove**", "**between the plate and the ramekin**", "**next to the
ramekin**". Moving the bowl freely makes the instruction false.

This document covers two things built on top of that observation:

1. **Relational rigid groups** — how a spatial scene is augmented while keeping
   its instruction true.
2. **Paired two-relation scenes** — one scene holding BOTH bowls in different
   relations, so language is the only thing that selects the target.

---

## 1. Relational rigid groups

All ten `libero_spatial` tasks share one scene: two identical black bowls, a
plate, a cookie box, a ramekin, and two fixtures (a flat stove, a wooden
cabinet). The target is always `akita_black_bowl_1`, the destination always
`plate_1`, and the second bowl is a visually identical distractor.

Augmentation moves a **rigid group**:

```
group = {target bowl} ∪ {anchor objects} ∪ {anchor fixture}
```

under a single planar transform (translation + yaw about the group centroid).
The internal relation — hence the instruction — is preserved exactly, while the
group travels across the table. Each member keeps its own z, so a bowl stacked
on the ramekin stays stacked and a bowl on the stove stays on the burner.

Free objects move via their free-joint qpos. **Fixtures have no free joint**:
they move via `model.body_pos` / `body_quat`, which is model data that
`env.reset()` rebuilds and wipes — so fixture edits must be re-applied after
every reset, before replay (`spatial_scene.apply_fixture_edits`).

Downstream synthesis is unchanged: `synthesize_uniform` only needs the target's
net displacement `obj_t` and the plate's `tar_t`. The bowl is round, so the
grasp is yaw-invariant and no skill segment changes.

Per-task anchors live in `src/demogen_libero/tasks_config_spatial.json`;
the sampler is `src/demogen_libero/spatial_scene.py`.

---

## 2. Paired two-relation scenes

### Why

With one relation per scene, the distractor bowl is pushed away from every
anchor, so a model can learn a shortcut: *pick whichever bowl is near
something*. Never reading the instruction still scores well.

A paired scene puts **both** bowls in meaningful relations — one on the stove,
one next to the plate — so the shortcut disappears and the instruction is the
only disambiguator.

LIBERO already does this implicitly: in `on_the_stove` source scenes bowl_1
sits on the burner and bowl_2 on the cabinet top; `on_the_wooden_cabinet` is the
same layout with the roles swapped. The same holds for
on_ramekin↔on_cookie_box, next_to_ramekin↔next_to_cookie_box and
from_table_center↔next_to_plate. The work here makes that systematic and
controllable rather than fixed.

### One geometry, two episodes

Every task's BDDL goal is `(And (On akita_black_bowl_1 plate_1))`, and the OC
format keys seg id 60 on the same name. So the bowl the instruction refers to is
written as `akita_black_bowl_1` and the other takes the remaining pose.

The two bowls are the same asset — verified identical: 41 geoms each, same geom
types, sizes, rgba, friction, and total mass 0.005591 kg. **Swapping the poses
is therefore pixel-identical to swapping the names**, and it leaves the BDDL,
the seg-id convention, `target_joint` and the `obj_pos` order untouched.
Changing the goal to `akita_black_bowl_2` instead would silently move seg 60
onto the distractor.

### What validates a scene

Bounding volumes cannot arbitrate collisions for these assets. The stove's
burner is a raised ring whose bounding box swallows the bowl resting inside it,
and a bowl rim legitimately overhangs the flat plate. Discs, oriented boxes and
height-sliced decompositions each either reject scenes LIBERO itself ships or
stop detecting real overlap. So the authorities are **measurements**, each
calibrated on all 500 real source init states:

| what | authority | value on real scenes |
|---|---|---|
| collision | MuJoCo contact penetration depth | max 0.00147 m |
| stability | post-settle **relation drift** | max 0.01919 m |
| visibility | rendered segmentation area | min 491 px |

Relation drift, not raw displacement, decides stability: LIBERO's own
"bowl on the ramekin" scenes re-seat by 0.019 m and are perfectly fine, so a
displacement threshold cannot tell a harmless re-seat from a bowl sliding off.

**A control made only of valid scenes is not a control.** It cannot distinguish
a working checker from one that always says yes. `screen_spatial_pairs.py`
therefore also runs falsification probes — deliberately overlapped
configurations that must be rejected. They caught a real defect: a contact
tolerance applied at slab level silently disabled collision detection for 11 of
21 object pairs, because the plate, cookie box and ramekin are thinner than the
tolerance.

One more calibration result worth knowing: the top-down collinearity
"anti-occlusion" prefilter rejects 44 of 50 real scenes that render at 494+ px.
agentview looks *down*, so xy collinearity is not pixel occlusion. Use the
rendered gate.

### Measured geometry vs the hand-tuned constants

`spatial_scene.OBJECT_RADIUS` / `FIXTURE_RADIUS` are visual estimates and differ
substantially from mesh measurements:

| item | code | measured |
|---|---|---|
| akita_black_bowl | 0.075 | **0.056** |
| plate | 0.100 | **0.069** |
| ramekin | 0.060 | **0.045** |
| cookies | 0.050 | 0.052 |
| flat_stove | 0.120 (disc) | **0.297 × 0.190 box** |
| wooden_cabinet | 0.130 (disc) | **0.313 × 0.283 box** |

The stove matters most: its body **origin sits 0.0965 m outside its footprint
centre**, and "on the stove" holds 0.146 m from that origin (the burner). The
cabinet sits at 155° yaw, so body-frame AABBs are mandatory. Measured table
extent is x ∈ [-0.5, 0.5], y ∈ [-0.6, 0.6], top z = 0.900.

### Which pairs are possible

Of 45 task combinations, 6 fail the planar test because the two relations name
places closer than the exclusion radius (0.24 m):

| pair | naming-point gap | verdict |
|---|---|---|
| between + next_to_ramekin | 0.170 m | ambiguous |
| between + on_ramekin | 0.153 m | ambiguous *(see below)* |
| between + next_to_plate | 0.133 m | ambiguous |
| next_to_ramekin + on_ramekin | 0.126 m | **usable — on vs beside** |
| on_cookie_box + next_to_cookie_box | 0.117 m | **usable — on vs beside** |
| on_cabinet + in_drawer | 0.151 m | ambiguous *(rule limitation)* |

**On vs beside.** Two of these are not ambiguous at all. Planar distance is the
wrong measurement for them, because what separates "the bowl **on** the ramekin"
from "the bowl **next to** the ramekin" is *support*, not position. Measured
naming-point offsets in the anchor's own frame:

| relation | offset from anchor | reading |
|---|---|---|
| on_ramekin | 0.0115 m | concentric — resting on it |
| next_to_ramekin | 0.1218 m | beside it |
| on_cookie_box | 0.000016 m | concentric |
| next_to_cookie_box | 0.1166 m | beside it |

The two bowls still end up 0.117–0.126 m apart, against a measured bowl radius
of 0.056 m — they do not touch. Enable these with
`screen_spatial_pairs.py --shared-anchor include|only`, which admits a pair when
one offset is concentric (< 0.05 m) and the other is not.

Two conditions are structural, not stylistic, and the rule enforces both:

- **The shared anchor must be a free object.** `place_shared_group` places the
  anchor once and hangs both bowls off it, which needs a joint to write the pose
  into. A fixture has none, so two independently drawn groups would each carry
  their own copy of the stove/cabinet and whichever was written last would
  silently win. This is what keeps `on_cabinet + in_drawer` out — even though its
  offsets (0.009 m vs 0.144 m) pass the support test, so it *looks* admissible.
- **Neither relation may name anything else.** An anchor named by only one of the
  two relations is not in the shared group, so it would be scattered as an
  independent object and that relation would simply be false. That is what would
  happen to the plate in `between + on_ramekin`.

So the rule admits exactly two pairs, not four.

The remaining 39 are all sampleable except `from_table_center + on_the_stove`
under the default stove y-band; widening that band to (-0.26, 0.10) yields 93%,
at some cost to the other stove pairs. Sampling cost spans four orders of
magnitude, from 4 to 133,000 draws per scene.

**Reverse plate sampling.** When a relation names the plate, the plate rides
with the group and must still land in the 0.18 × 0.16 m reachable place zone.
Drawing the bowl over the 0.40 × 0.50 m workspace makes that a 1–6% event, and
89–92% of those pairs' rejected draws are exactly that. Sampling the plate
inside the place zone and deriving the bowl from the same rigid offset explores
the identical feasible set from the small side: 4–7× faster on plate-owning
pairs, bit-identical on the rest.

---

## 3. Train / counterfactual / unseen split

Two orthogonal axes: **placement** (seen vs unseen) and **instruction**
(the trained one vs the other task's).

- **train** — target in a seen cell.
- **counterfact** — the *same geometry*, other bowl as target, other task's
  instruction. Pixel-identical image, different correct answer. This set is
  impossible to build from single-task data.
- **unseen** — target in an unseen cell, from geometries not used for training.

Cells come from a **shared 4×4 checkerboard** over the sampler workspace
(0.100 × 0.125 m cells), unseen iff `(cx + cy)` is odd. A shared board already
lands every task between 41% and 59% seen, so no per-task board is needed.

Two things that must be right:

**Role assignment must be balanced per task.** Left to pair enumeration order,
`on_the_wooden_cabinet` and `in_the_top_drawer` get zero training cells (always
the partner) while `next_to_the_ramekin`, `on_the_ramekin` and `between` get
zero counterfactual ones.

**Cells must be re-derived per demo after jitter.** Jitter is ±0.02 m against a
0.100 m cell, and 62% of targets sit within the jitter amplitude of a boundary,
so inheriting the base scene's label would contaminate the split wholesale.

Solvability was verified before committing: 94% of counterfactual and 97% of
unseen scenes yield a working trajectory, against 98% for training scenes — the
unseen cells are not systematically harder.

---

## 4. Scripts

| script | what it does |
|---|---|
| `screen_spatial_sources.py` | per-task source screening (healthy / regrasp / bad) |
| `smoke_spatial_inits.py` | render init-state mosaics per task, no trajectories |
| `run_spatial_oc_demo.py` | single-relation generation (relational rigid group) |
| `run_all_spatial.sh` | all 10 tasks, resumable |
| `screen_spatial_pairs.py` | pair calibration, control, falsification, yields |
| `plot_pair_distribution.py` | per-pair placement scatter |
| `plot_bowl_distribution.py` | source vs generated placement, per task |
| `build_pair_split.py` | the train / counterfact / unseen plan |
| `add_pair_episodes.py` | append episodes for extra pairs to an existing plan |
| `probe_pair_trajectories.py` | trajectory success and scene solvability |
| `run_pair_oc_demo.py` | paired generation from a split plan |
| `run_all_pairs.sh` | one split, several tasks in parallel |
| `patch_fixture_objects.py` | add stove + cabinet to demos generated before that was done inline |
| `patch_regrasp_label.py` | label demos that contain a failed grasp + recovery |

Typical flow:

```bash
PY=.venv/Scripts/python.exe          # .venv/bin/python on Linux

# single-relation data for all 10 tasks
$PY scripts/screen_spatial_sources.py
bash scripts/run_all_spatial.sh

# paired data
$PY scripts/screen_spatial_pairs.py --n 40            # feasibility + calibration
$PY scripts/build_pair_split.py --train-scenes 25 --unseen-scenes 10
$PY scripts/probe_pair_trajectories.py --split train --n 10 --source-retries 6
bash scripts/run_all_pairs.sh train 5
$PY scripts/patch_fixture_objects.py                  # only for pre-inline batches
```

### Adding a pair to a plan that has already been generated

Re-running `build_pair_split.py` renumbers every scene, which invalidates the
`scene_index` in the scene_log of demos already on disk — and both
`patch_fixture_objects.py` and `build_pair_eval_suite.py` dereference it.
`add_pair_episodes.py` is append-only instead: new geometries go on the end of
`plan["scenes"]` so existing indices keep pointing at the same geometry, and the
new episodes carry a `batch` tag.

```bash
$PY scripts/screen_spatial_pairs.py --n 40 --shared-anchor only --merge
$PY scripts/add_pair_episodes.py --batch shared_anchor --train-scenes 10
$PY scripts/run_pair_oc_demo.py --split train --task on_the_ramekin \
    --batch shared_anchor --target-count 52
$PY scripts/build_pair_eval_suite.py --split counterfact
$PY scripts/build_pair_eval_suite.py --split unseen
```

`--batch` also changes how progress is counted: the demos already in the hdf5 came
from other episodes, so charging them against the new ones would skip every one.
`--target-count` stops a task once its hdf5 holds that many demos in total, which
is how a task is topped up rather than doubled.

Per-task parallelism is safe here because each task writes its own HDF5. The
failure recorded for this project (stacked workers, Win32 error 33) needs
*same-file* contention.

---

## 5. Object order with fixtures

The instructions refer to the stove and the cabinet, so those two are recorded as
objects, appended **after** the five free objects so ids 60–100 keep their
meaning:

| idx | instance | seg id |
|---|---|---|
| 0 | akita_black_bowl_1 (target) | 60 |
| 1 | plate_1 (destination) | 70 |
| 2 | akita_black_bowl_2 (distractor) | 80 |
| 3 | cookies_1 | 90 |
| 4 | glazed_rim_porcelain_ramekin_1 | 100 |
| 5 | flat_stove_1 | 110 |
| 6 | wooden_cabinet_1 | 120 |

`run_pair_oc_demo.py` now records them **as the demo is generated** (the extractor
appends the two fixture poses from the sim, since robosuite's per-object
observations exist only for free-jointed bodies, and adds `drawer_pos` /
`drawer_qpos`). Pass `--no-fixtures` for the five-object layout.

The first 485 demos predate that and were fixed up afterwards by
`patch_fixture_objects.py`, which is where the two caveats below come from.
Trajectories are not re-synthesised there: every frame's full sim state is
stored, so frames are restored and re-rendered only.

- `env._get_observations()` returns a **cached** observation; re-rendering needs
  `force_update=True`. Without it the output looks plausible but shows the
  environment's reset state (mask IoU 0.0008 instead of 0.9997).
- Fixture poses are model data and were not logged per demo. They are
  reconstructed from the frame-0 state — jitter moves a group rigidly about its
  bowl, so the bowl's xy gives the translation and its quaternion the yaw — and
  every demo is verified by re-rendering: the five original objects' masks must
  match the stored ones, which a misplaced fixture would break via occlusion.

The drawer is a separate body (`wooden_cabinet_1_cabinet_top`) that tracks the
`wooden_cabinet_1_top_level` slide joint, but instance segmentation treats the
whole cabinet as one instance. Splitting the drawer out needs per-geom
("element") segmentation and is not done here.
