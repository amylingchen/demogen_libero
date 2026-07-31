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
