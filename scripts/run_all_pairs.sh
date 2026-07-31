#!/usr/bin/env bash
# Generate one split of the paired dataset, running several tasks at once.
#
# Each task writes its own directory and hdf5, so concurrent workers never touch
# the same file -- the failure mode recorded for this project (stacked workers
# appending to one hdf5, Win32 error 33) needs SAME-file contention and cannot
# happen here. Every task is still launched exactly once; re-running the script
# resumes each task from the demo count already in its hdf5.
#
# Usage:
#   bash scripts/run_all_pairs.sh train 5        # split, max concurrent workers
#   bash scripts/run_all_pairs.sh train          # defaults to 5
set -u
SPLIT="${1:-train}"
JOBS="${2:-5}"
PY=".venv/Scripts/python.exe"
OUT_ROOT="output/libero_spatial_pairs"
LOG_DIR="output/logs_pairs_${SPLIT}"
mkdir -p "$LOG_DIR"

TASKS=$(PYTHONIOENCODING=utf-8 $PY - <<'EOF'
import json, sys
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import screen_spatial_pairs as P
plan = json.load(open("output/pair_split.json"))
split = __import__("os").environ.get("SPLIT", "train")
print(" ".join(sorted({P.short(e["task"]) for e in plan["episodes"]})))
EOF
)

echo "split=$SPLIT  concurrency=$JOBS"
echo "tasks: $TASKS"

running=0
for t in $TASKS; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 10; done
  echo "  launching $t"
  $PY scripts/run_pair_oc_demo.py --split "$SPLIT" --task "$t" \
      --out-root "$OUT_ROOT" > "$LOG_DIR/$t.log" 2>&1 &
  sleep 3
done
wait
echo "all tasks finished for split=$SPLIT"

# ------------------------------------------------------------ finishing gate
# A generator's visibility gate can be correct in source and still never have run
# on the bytes it wrote. That is what happened here: before `force_update=True`
# landed, run_pair_oc_demo's gate scored robosuite's cached observation -- the
# env's DEFAULT reset layout, where every object is in plain view -- so it passed
# every candidate, and 76 of 519 train demos ended up carrying an object below the
# generator's own floor, 49 of them at exactly 0 px.
#
# So this check reads the seg that was actually WRITTEN rather than re-rendering
# (a checker that re-simulates can repeat the same mistake), and it runs here
# instead of waiting for someone to remember it. --baseline means only NEW
# violations fail: re-reporting the historical 76 on every top-up would train
# everyone to ignore the gate, which is the failure mode itself.
KP=${KP:-../graphslot-vla-keypoint}
CHECK="$KP/scripts/check_scene_visibility.py"
BASE="$KP/data/spatial_pairs_audit/visibility_baseline_${SPLIT}.json"

if [ ! -f "$CHECK" ]; then
  echo "!! visibility check not found at $CHECK -- set KP=<graphslot repo path>." >&2
  echo "!! generation FINISHED BUT WAS NOT CHECKED for split=$SPLIT." >&2
  exit 3
fi

echo
echo "=== finishing gate: visibility of the frames just written ==="
# --ignore-prefix reproduces the generator's own contract: run_pair_oc_demo's
# gate covers P.FREE_INST (the five free objects) only, so comparing on the same
# object set keeps this apples-to-apples.
$PY "$CHECK" --root "$OUT_ROOT" --splits "$SPLIT" \
    --ignore-prefix flat_stove,wooden_cabinet \
    --baseline "$BASE" \
    --out "$LOG_DIR/visibility_${SPLIT}.json"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo >&2
  echo "!! VISIBILITY GATE FAILED (exit $rc) for split=$SPLIT" >&2
  echo "!! details: $LOG_DIR/visibility_${SPLIT}.json" >&2
  echo "!! do NOT feed this split into training until the new violations are" >&2
  echo "!! resolved, or the baseline is updated deliberately." >&2
fi
exit "$rc"
