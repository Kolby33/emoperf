"""Sweep the energy segmenter's hangover (and chunk size) on the 100-utterance stream.

No models run: segmentation is independent of inference, so this is fast. For each
setting it reports how many ground-truth utterances get exactly one segment, are split,
merged with a neighbour, or missed, and the endpoint delay (hangover + chunk rounding),
which adds directly to end-to-end latency.

Usage: python bench/segmenter_sweep.py --n 100
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.live.segmenter import EnergySegmenter  # noqa: E402
from pipeline.live.stream import SR, build_stream  # noqa: E402


def evaluate(stream, hangover_s, chunk_ms, threshold_db=-55.0):
    seg = EnergySegmenter(threshold_db=threshold_db, hangover_s=hangover_s)
    chunk = SR * chunk_ms // 1000
    segs = []
    for i in range(0, len(stream.audio), chunk):
        segs += seg.push(stream.audio[i:i + chunk])
    segs += seg.flush()
    s_start = np.array([s.start for s in segs])
    s_end = np.array([s.end for s in segs])
    n_one = n_split = n_miss = 0
    for g in stream.gt.itertuples():
        ov = (np.minimum(s_end, g.end) - np.maximum(s_start, g.start)) > 0
        n = int(ov.sum())
        n_one += n == 1
        n_split += n > 1
        n_miss += n == 0
    merged = sum(int(((np.minimum(s.end, stream.gt.end) - np.maximum(s.start, stream.gt.start)) > 0).sum()) > 1
                 for s in segs)
    # endpoint delay in stream time: decision point (end of the chunk in which the segment closed) - speech end
    delay = [((s.closed_at + chunk - 1) // chunk * chunk - s.end) / SR for s in segs]
    return {"hangover_s": hangover_s, "chunk_ms": chunk_ms, "segments": len(segs), "gt_one": n_one,
            "gt_split": n_split, "gt_missed": n_miss, "segments_merging_gt": merged,
            "endpoint_delay_ms_p50": float(np.median(delay) * 1e3), "endpoint_delay_ms_p95": float(np.percentile(delay, 95) * 1e3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("bench/results/m4_segmenter_sweep.json"))
    args = ap.parse_args()
    stream = build_stream(Path("data/index/test.csv"), Path("bench/subsets/test_fixed_500.txt"), n=args.n)
    rows = [evaluate(stream, h, 20) for h in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7)]
    rows += [evaluate(stream, 0.3, c) for c in (10, 50, 100)]
    print("| hangover (s) | chunk (ms) | segments | GT 1 segment | split | missed | merged | endpoint delay p50 / p95 (ms) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['hangover_s']} | {r['chunk_ms']} | {r['segments']} | {r['gt_one']} | {r['gt_split']} | "
              f"{r['gt_missed']} | {r['segments_merging_gt']} | {r['endpoint_delay_ms_p50']:.0f} / {r['endpoint_delay_ms_p95']:.0f} |")
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
