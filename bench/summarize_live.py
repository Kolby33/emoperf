"""Print a markdown latency table from one or more bench/run_live.py JSON results.

Usage: python bench/summarize_live.py bench/results/m2_*.json
"""

import json
import sys
from pathlib import Path

ROWS = [("endpoint_delay", "endpoint delay (hangover)"), ("queue_wait", "queue wait"),
        ("audio_enc", "audio encoder"), ("branch_audio", "branch A total"), ("asr", "ASR (Whisper generate)"),
        ("text_enc", "text encoder"), ("branch_text", "branch B total"), ("branches_wall", "branches wall time"),
        ("fusion", "fusion head"), ("inference", "inference total"), ("e2e", "end-to-end")]


def main(paths):
    runs = [(Path(p).stem.replace("m2_", ""), json.loads(Path(p).read_text())) for p in paths]
    print("| stage | " + " | ".join(f"{n} p50 / p95 (ms)" for n, _ in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for key, label in ROWS:
        cells = [f"{r['summary']['stages'][key]['p50_ms']:.1f} / {r['summary']['stages'][key]['p95_ms']:.1f}"
                 for _, r in runs]
        print(f"| {label} | " + " | ".join(cells) + " |")
    extra = [
        ("RTF overall", lambda s: f"{s['rtf_overall']:.3f}"),
        ("RTF per segment p50 / p95", lambda s: f"{s['rtf_per_segment']['p50']:.3f} / {s['rtf_per_segment']['p95']:.3f}"),
        ("text branch is critical path", lambda s: f"{s['critical_path']['text_branch_longer_fraction']:.0%}"),
        ("weighted F1 live / offline gold", lambda s: f"{s['accuracy_live']['weighted_f1']:.3f} / "
                                                      f"{s['accuracy_offline_gold_text_same_utts']['weighted_f1']:.3f}"),
        ("ASR WER", lambda s: f"{s['asr_wer']:.3f}"),
        ("GT utts: 1 segment / split / missed",
         lambda s: f"{s['segmentation']['gt_exactly_one_segment']} / {s['segmentation']['gt_split']} / "
                   f"{s['segmentation']['gt_missed']}"),
        ("queue max depth / ring overrun", lambda s: f"{s['queue_max_depth']} / {s['ring_overrun_samples']}"),
    ]
    for label, f in extra:
        print(f"| {label} | " + " | ".join(f(r["summary"]) for _, r in runs) + " |")
    therm = [r["env"]["thermal_after"].splitlines()[0][:40] for _, r in runs]
    print(f"\nthermal after each run: {therm}")
    print("runs:", [(n, r["timestamp"], r["env"]["git_commit"]) for n, r in runs])


if __name__ == "__main__":
    main(sys.argv[1:])
