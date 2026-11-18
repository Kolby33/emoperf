#!/usr/bin/env bash
# M4 final: real-time end-to-end check of the best configuration, interleaved A B A B.
#   A = best inference config, hangover 0.3 s (same segmentation as M2, isolates the compute speedup)
#   B = same, hangover 0.2 s (segmenter sweep: -100 ms endpoint delay, 1 more split)
set -euo pipefail
cd "$(dirname "$0")/.."
OPTS='{"asr_device": "mps", "mask_cache": true}'
for rep in 1 2; do
  for h in 0.3 0.2; do
    .venv/bin/python bench/run_live.py --mode concurrent --n 100 --opts "$OPTS" --hangover "$h" \
        --out "bench/results/m4_live_best_h${h}_r${rep}.json"
    sleep 60
  done
done
.venv/bin/python bench/summarize_live.py bench/results/m2_sequential_r1.json bench/results/m2_concurrent_r1.json \
    bench/results/m4_live_best_h0.3_r*.json bench/results/m4_live_best_h0.2_r*.json
