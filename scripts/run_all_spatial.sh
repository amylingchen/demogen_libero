#!/bin/bash
# Sequentially generate all 10 libero_spatial tasks, 100 demos each
# (20 scenes x 5 demos, per-demo jitter, mixed single-grasp + regrasp). One task
# at a time so there is never more than one process writing an hdf5.
#
# The generator occasionally dies with a silent C-level (MuJoCo/GL) crash after a
# few dozen rollouts, so each task is RE-RUN until its hdf5 holds TARGET demos --
# run_spatial_oc_demo resumes/tops-up an existing file (absolute target).
#
# Usage from repo root:  bash scripts/run_all_spatial.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
TARGET=100
MAX_ATTEMPTS=12

TASKS=(
  pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate
  pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate
  pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
  pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate
  pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate
)

count_demos() {  # echo number of demos in the task hdf5 (0 if none)
  $PY -c "import h5py,glob,sys
p=glob.glob('$1/*_demo.hdf5')
print(len(h5py.File(p[0])['data'].keys()) if p else 0)" 2>/dev/null || echo 0
}

for t in "${TASKS[@]}"; do
  short=$(echo "$t" | sed 's/pick_up_the_black_bowl_//; s/_and_place_it_on_the_plate//')
  outdir="output/libero_spatial_100/$short"
  echo "===== TASK $short  ($(date '+%H:%M:%S')) ====="
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    have=$(count_demos "$outdir")
    if [ "$have" -ge "$TARGET" ]; then break; fi
    echo "--- $short attempt $attempt: have $have/$TARGET ($(date '+%H:%M:%S')) ---"
    $PY scripts/run_spatial_oc_demo.py --task "$t" \
        --n-scenes 20 --demos-per-scene 5 --out-dir "$outdir" --seed "$attempt"
  done
  echo "----- done $short: $(count_demos "$outdir")/$TARGET demos -----"
done
echo "===== ALL 9 TASKS DONE ($(date '+%H:%M:%S')) ====="
