#!/usr/bin/env bash
# Full production: 100 demos per libero_object task (target position repeated
# exactly 2x via demos-per-scene=2, different source demos per repeat), a slice
# of regrasp trajectories where the task has >=3 regrasp sources, plus a
# disjoint eval suite (init states only). Auto camera/geometry dumps + viz run
# at the end of each task's generation.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
SRC_JSON=output/source_screening.json
BASE=output/libero_object_100

# tasks with >=3 regrasp sources get 45+5 scenes; others 50+0
declare -A REGRASP_SCENES=(
  [alphabet_soup]=5 [bbq_sauce]=5 [cream_cheese]=5 [ketchup]=5 [milk]=5
  [orange_juice]=5 [salad_dressing]=5 [tomato_sauce]=5 [butter]=0 [chocolate_pudding]=0
)

i=0
for t in alphabet_soup bbq_sauce butter chocolate_pudding cream_cheese ketchup milk orange_juice salad_dressing tomato_sauce; do
  i=$((i+1))
  rg=${REGRASP_SCENES[$t]}
  healthy=$((50 - rg))
  out=$BASE/$t
  echo "##### TASK $i/10: $t (healthy $healthy scenes + regrasp $rg scenes) #####"

  $PY scripts/run_grid_oc_demo.py --task "$t" --sources-json $SRC_JSON \
      --n-scenes $healthy --demos-per-scene 2 --scene-retries 4 \
      --seed $((1000 + i)) --out-dir "$out" --no-viz --no-dump-aux \
      2>&1 | grep -vE "Gym has been|Please upgrade|Users of this|See the migration" | tail -3

  if [ "$rg" -gt 0 ]; then
    $PY scripts/run_grid_oc_demo.py --task "$t" --sources-json $SRC_JSON \
        --segment regrasp --n-scenes $rg --demos-per-scene 2 --scene-retries 4 \
        --seed $((2000 + i)) --out-dir "$out" \
        2>&1 | grep -vE "Gym has been|Please upgrade|Users of this|See the migration" | tail -3
  else
    # no regrasp batch: run the aux dumps + viz that the second pass would have done
    $PY scripts/dump_camera_params.py --task "$t" --out "$out/camera_params.json" 2>&1 | tail -1
    $PY scripts/dump_object_geometry.py --task "$t" --out "$out/object_geometry.json" 2>&1 | tail -1
    $PY scripts/visualize_init_states.py --dir "$out" 2>&1 | tail -1
    $PY scripts/visualize_phases.py --dir "$out" --demos demo_0 demo_50 demo_99 2>&1 | tail -1
  fi

  $PY scripts/build_eval_suite.py --task "$t" \
      --train-scene-log "$out/scene_log.json" --min-train-dist 0.05 \
      --n-scenes 30 --seed $((9000 + i)) --out-dir "${BASE}/${t}_eval" \
      2>&1 | grep -vE "Gym has been|Please upgrade|Users of this|See the migration" | tail -2

  echo "##### DONE $t #####"
done
echo "##### ALL 10 TASKS COMPLETE #####"
