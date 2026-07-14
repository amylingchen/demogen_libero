#!/usr/bin/env bash
# Post-production top-up: append healthy-source scenes (resume mode) to every
# task under output/libero_object_100 until each holds exactly >= TARGET demos,
# then re-run the final viz so the mosaic covers all demos.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
SRC_JSON=output/source_screening.json
BASE=output/libero_object_100
TARGET=${TARGET:-100}

count_demos() {
  $PY - "$1" <<'EOF'
import h5py, glob, os, sys
h5 = glob.glob(os.path.join(sys.argv[1], "*_demo.hdf5"))
if not h5:
    print(0)
else:
    with h5py.File(h5[0], "r") as f:
        print(len(f["data"].keys()) if "data" in f else 0)
EOF
}

i=0
for t in alphabet_soup bbq_sauce butter chocolate_pudding cream_cheese ketchup milk orange_juice salad_dressing tomato_sauce; do
  i=$((i+1))
  out=$BASE/$t
  [ -d "$out" ] || continue
  round=0
  while true; do
    n=$(count_demos "$out")
    if [ "$n" -ge "$TARGET" ]; then echo "== $t: $n/$TARGET OK =="; break; fi
    round=$((round+1))
    if [ "$round" -gt 6 ]; then echo "== $t: stuck at $n/$TARGET after 6 rounds =="; break; fi
    need=$(( (TARGET - n + 1) / 2 ))
    echo "== $t: $n/$TARGET, topping up with $need scenes (round $round) =="
    $PY scripts/run_grid_oc_demo.py --task "$t" --sources-json $SRC_JSON \
        --n-scenes "$need" --demos-per-scene 2 --scene-retries 4 \
        --seed $((3000 + 100*i + round)) --out-dir "$out" --no-viz --no-dump-aux \
        2>&1 | grep -vE "Gym has been|Please upgrade|Users of this|See the migration" | tail -2
  done
  # refresh the mosaic + sample viz to cover the final demo set
  $PY scripts/visualize_init_states.py --dir "$out" 2>&1 | tail -1
done
echo "##### TOP-UP COMPLETE #####"
