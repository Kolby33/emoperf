#!/usr/bin/env bash
# M2 headline benchmark: sequential vs concurrent, interleaved S C S C with cooldowns,
# so thermal drift on the fanless laptop is spread across both modes.
set -euo pipefail
cd "$(dirname "$0")/.."
N=${N:-100}
COOL=${COOL:-60}
for rep in 1 2; do
  for mode in sequential concurrent; do
    .venv/bin/python bench/run_live.py --mode "$mode" --n "$N" --out "bench/results/m2_${mode}_r${rep}.json"
    sleep "$COOL"
  done
done
.venv/bin/python bench/summarize_live.py bench/results/m2_*_r*.json
