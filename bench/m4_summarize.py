"""Summarize M4 variant results into one markdown table (mean of rounds, plus round-to-round spread).

Accuracy columns compare each variant with the baseline on the same 100 utterances:
  ΔwF1           weighted F1 change vs baseline
  same preds     fraction of utterances with the same emotion prediction as baseline
  same text      fraction with an identical ASR transcript (from git-ignored data/runs/m4)

Usage: python bench/m4_summarize.py [--tag ""] [--out docs/m4_results.md]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("bench/results/m4"))
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = defaultdict(list)
    for p in sorted(args.dir.glob("*_r*.json")):
        d = json.loads(p.read_text())
        full = Path("data/runs/m4") / p.name
        d["_full"] = json.loads(full.read_text()) if full.exists() else None
        runs[d["variant"]].append(d)

    base = runs[args.baseline][0]
    base_preds = {u["utt_key"]: u["pred"] for u in base["per_utt"]}
    base_text = base["_full"]["transcripts"] if base["_full"] else None
    b_inf = np.mean([r["stages"]["inference"]["p50_ms"] for r in runs[args.baseline]])

    lines = ["| variant | inference p50 (ms) | p95 (ms) | speedup p50 | ASR p50 | audio enc p50 | RTF | "
             "peak RSS (MB) | wF1 | ΔwF1 | WER | same preds | same text | rounds |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, rs in runs.items():
        g = lambda f: np.array([f(r) for r in rs])  # noqa: E731
        p50 = g(lambda r: r["stages"]["inference"]["p50_ms"])
        p95 = g(lambda r: r["stages"]["inference"]["p95_ms"])
        r0 = rs[0]
        same_p = np.mean([base_preds[u["utt_key"]] == u["pred"] for u in r0["per_utt"]])
        same_t = (np.mean([base_text[k] == v for k, v in r0["_full"]["transcripts"].items()])
                  if base_text and r0["_full"] else float("nan"))
        spread = f"±{(p50.max() - p50.min()) / 2:.1f}" if len(rs) > 1 else ""
        therm = " ⚠thermal" if any(r["thermal_warning"] for r in rs) else ""
        lines.append(
            f"| {name}{therm} | {p50.mean():.1f} {spread} | {p95.mean():.1f} | {b_inf / p50.mean():.2f}× | "
            f"{np.mean(g(lambda r: r['stages']['asr']['p50_ms'])):.1f} | "
            f"{np.mean(g(lambda r: r['stages']['audio_enc']['p50_ms'])):.1f} | "
            f"{np.mean(g(lambda r: r['rtf_overall'])):.3f} | {np.mean(g(lambda r: r['peak_rss_mb'])):.0f} | "
            f"{r0['weighted_f1']:.3f} | {r0['weighted_f1'] - base['weighted_f1']:+.3f} | {r0['wer']:.3f} | "
            f"{same_p:.0%} | {same_t:.0%} | {len(rs)} |")
    md = "\n".join(lines)
    print(md)
    if args.out:
        args.out.write_text(md + "\n")


if __name__ == "__main__":
    main()
