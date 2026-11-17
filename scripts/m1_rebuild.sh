#!/usr/bin/env bash
# Rebuild the whole M1 baseline from the extracted MELD archive:
#   audio extraction -> label index -> embedding cache -> fusion head (3 seeds x 3 modes) -> evaluation.
# Expects data/raw/extracted/ (from MELD.Raw.tar.gz) and data/labels/ (GitHub CSVs, commit e8cedf2).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DEVICE=${DEVICE:-mps}

for pair in train:train_splits dev:dev_splits_complete test:output_repeated_splits_test; do
  s=${pair%%:*}; d=${pair#*:}
  $PY pipeline/data/extract_audio.py --split "$s" --videos-dir "data/raw/extracted/$d" \
      --out-dir "data/audio/$s" --manifest "data/audio/${s}_manifest.csv"
done
$PY pipeline/data/build_index.py --csv-dir data/labels --audio-dir data/audio --out-dir data/index
$PY pipeline/embed/cache_embeddings.py --device "$DEVICE" --check-cpu 32
$PY pipeline/fusion/train.py
$PY pipeline/fusion/evaluate.py --out bench/results/m1_eval.json
