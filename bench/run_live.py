"""Benchmark the live pipeline on a simulated real-time stream; write results as JSON.

Usage:
  python bench/run_live.py --mode sequential --n 100 --out bench/results/m2_seq.json
  python bench/run_live.py --mode concurrent --n 100 --out bench/results/m2_conc.json
  python bench/run_live.py --mode concurrent --segmenter oracle --n 100 ...

Latency definitions are in pipeline/live/runner.py. Accuracy: each ground-truth utterance
takes the prediction of the segment that overlaps it most (missed utterance = wrong).
"""

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transformers  # noqa: E402

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

import pandas as pd  # noqa: E402

from pipeline.fusion.model import FusionHead  # noqa: E402
from pipeline.live.runner import RunConfig, run  # noqa: E402
from pipeline.live.stages import Stages  # noqa: E402
from pipeline.live.stream import SR, build_stream  # noqa: E402

STAGE_KEYS = ["endpoint_delay", "queue_wait", "audio_feat", "audio_enc", "branch_audio", "asr_feat", "asr",
              "text_enc", "branch_text", "branches_wall", "fusion", "response", "inference", "e2e"]


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def norm_words(s):
    s = s.lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9' ]+", " ", s).split()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def stats(xs):
    xs = np.asarray(xs, float) * 1e3
    return {"p50_ms": float(np.percentile(xs, 50)), "p95_ms": float(np.percentile(xs, 95)),
            "mean_ms": float(xs.mean()), "max_ms": float(xs.max()), "n": int(len(xs))}


TEXT_FIELDS = {"text", "transcript", "response_text"}


def strip_text(report):
    out = dict(report)
    out["segments"] = [{k: v for k, v in s.items() if k not in TEXT_FIELDS} for s in report["segments"]]
    out["gt"] = [{k: v for k, v in g.items() if k not in TEXT_FIELDS} for g in report["gt"]]
    out["text_fields_removed"] = sorted(TEXT_FIELDS)
    return out


def offline_reference(gt, head_path):
    idx = pd.read_csv("data/index/test.csv", keep_default_na=False)
    pos = idx.reset_index().set_index("utt_key").loc[gt.utt_key, "index"].values
    t = torch.from_numpy(np.load("data/emb/test_text.npy")[pos].astype(np.float32))
    a = torch.from_numpy(np.load("data/emb/test_audio.npy")[pos].astype(np.float32))
    ckpt = torch.load(head_path, weights_only=True)
    head = FusionHead(ckpt["mode"])
    head.load_state_dict(ckpt["state_dict"])
    with torch.inference_mode():
        return head.eval()(t, a).argmax(-1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sequential", "concurrent"], default="sequential")
    ap.add_argument("--segmenter", choices=["energy", "oracle"], default="energy")
    ap.add_argument("--n", type=int, default=100, help="first n utterances of the fixed test subset")
    ap.add_argument("--threads", type=int, default=4, help="torch intra-op threads")
    ap.add_argument("--chunk-ms", type=int, default=20)
    ap.add_argument("--pace", type=float, default=1.0)
    ap.add_argument("--head", type=Path, default=Path("models/fusion_text_audio_seed0.pt"))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    therm_before = sh(["pmset", "-g", "therm"])
    stream = build_stream(Path("data/index/test.csv"), Path("bench/subsets/test_fixed_500.txt"), n=args.n)
    t = time.perf_counter()
    stages = Stages(args.head)
    load_s = time.perf_counter() - t

    # Warm-up on dev clips (never on the measured test utterances). First-call cost is reported.
    dev = pd.read_csv("data/index/dev.csv", keep_default_na=False)
    warm = []
    for w in dev.wav[: args.warmup]:
        x, _ = sf.read(w, dtype="float32")
        t = time.perf_counter()
        a, _ = stages.audio_branch(x)
        tr, te, _ = stages.text_branch(x)
        stages.fuse(te, a)
        warm.append(time.perf_counter() - t)

    cfg = RunConfig(mode=args.mode, segmenter=args.segmenter, chunk_ms=args.chunk_ms, pace=args.pace)
    print(f"stream: {len(stream.gt)} utterances, {stream.seconds:.1f} s; mode={args.mode} "
          f"segmenter={args.segmenter} threads={args.threads}", flush=True)
    res = run(stream, stages, cfg)
    therm_after = sh(["pmset", "-g", "therm"])
    segs = pd.DataFrame(res.segments)

    # ---- match segments to ground truth ---------------------------------------------
    gt = stream.gt.copy()
    best_seg, n_overlap = [], []
    for g in gt.itertuples():
        ov = (np.minimum(segs.end, g.end) - np.maximum(segs.start, g.start)).clip(lower=0)
        n_overlap.append(int((ov > 0).sum()))
        best_seg.append(int(ov.idxmax()) if ov.max() > 0 else -1)
    gt["seg"], gt["n_segments"] = best_seg, n_overlap
    seg_gt_count = [int(((np.minimum(s.end, gt.end) - np.maximum(s.start, gt.start)) > 0).sum())
                    for s in segs.itertuples()]
    live_pred = np.array([segs.pred[s] if s >= 0 else -1 for s in gt.seg])
    offline_pred = offline_reference(gt, args.head)
    y = gt.label.values

    def acc(p):
        return {"weighted_f1": float(f1_score(y, p, labels=range(7), average="weighted", zero_division=0)),
                "macro_f1": float(f1_score(y, p, labels=range(7), average="macro", zero_division=0)),
                "accuracy": float((y == p).mean())}

    edits = refs = 0
    for g in gt.itertuples():
        ref = norm_words(g.text)
        hyp = norm_words(segs.transcript[g.seg]) if g.seg >= 0 else []
        edits += edit_distance(ref, hyp)
        refs += len(ref)

    audio_s = stream.seconds
    busy = segs.inference.sum()
    summary = {
        "stages": {k: stats(segs[k]) for k in STAGE_KEYS},
        "rtf_per_segment": {"p50": float(segs.rtf.median()), "p95": float(segs.rtf.quantile(0.95))},
        "rtf_overall": float(busy / audio_s),
        "critical_path": {"text_branch_longer_fraction": float((segs.branch_text > segs.branch_audio).mean()),
                          "branch_audio_p50_ms": float(segs.branch_audio.median() * 1e3),
                          "branch_text_p50_ms": float(segs.branch_text.median() * 1e3),
                          "branches_wall_p50_ms": float(segs.branches_wall.median() * 1e3)},
        "segmentation": {"n_gt": len(gt), "n_segments": len(segs),
                         "gt_exactly_one_segment": int((gt.n_segments == 1).sum()),
                         "gt_split": int((gt.n_segments > 1).sum()), "gt_missed": int((gt.n_segments == 0).sum()),
                         "segments_spanning_multiple_gt": int(sum(c > 1 for c in seg_gt_count)),
                         "segments_without_gt": int(sum(c == 0 for c in seg_gt_count)),
                         "forced_closes": int(segs.forced.sum())},
        "accuracy_live": acc(live_pred),
        "accuracy_offline_gold_text_same_utts": acc(offline_pred),
        "asr_wer": edits / max(refs, 1),
        "ring_overrun_samples": res.ring_overrun_samples,
        "queue_max_depth": res.queue_max_depth,
        "producer_lateness_ms": {"p50": float(np.median(res.chunk_lateness) * 1e3),
                                 "p99": float(np.percentile(res.chunk_lateness, 99) * 1e3),
                                 "max": float(res.chunk_lateness.max() * 1e3)},
        "wall_seconds": res.wall_seconds, "audio_seconds": audio_s,
    }
    report = {
        "benchmark": "m2_live", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {**vars(cfg), "n": args.n, "threads": args.threads, "interop_threads": torch.get_num_interop_threads(),
                   "head": str(args.head), "asr_model": stages.asr_model_name, "device": "cpu"},
        "env": {"torch": torch.__version__, "transformers": transformers.__version__,
                "python": platform.python_version(), "macos": platform.mac_ver()[0],
                "git_commit": sh(["git", "rev-parse", "--short", "HEAD"]),
                "power": sh(["pmset", "-g", "batt"]).splitlines()[0] if sh(["pmset", "-g", "batt"]) else "",
                "thermal_before": therm_before, "thermal_after": therm_after},
        "model_load_seconds": load_s, "warmup_seconds": warm,
        "summary": summary,
        "segments": [{k: (v if not isinstance(v, np.generic) else v.item()) for k, v in r.items()}
                     for r in res.segments],
        "gt": gt.to_dict("records"),
    }
    # Full detail (with MELD transcripts and ASR output) stays in git-ignored data/runs/.
    # The committed result file has no dataset text, only IDs, labels, timings and metrics.
    full = Path("data/runs") / f"{args.out.stem}_full.json"
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(report, indent=1, default=float))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(strip_text(report), indent=1, default=float))

    s = summary["stages"]
    print(f"wall {res.wall_seconds:.1f}s for {audio_s:.1f}s audio | overall RTF {summary['rtf_overall']:.3f} | "
          f"warm-up first call {warm[0]*1e3:.0f} ms")
    for k in STAGE_KEYS:
        print(f"  {k:>15}: p50 {s[k]['p50_ms']:8.1f} ms   p95 {s[k]['p95_ms']:8.1f} ms   max {s[k]['max_ms']:8.1f} ms")
    print("  segmentation:", summary["segmentation"])
    print(f"  accuracy live {summary['accuracy_live']} | offline gold {summary['accuracy_offline_gold_text_same_utts']}")
    print(f"  ASR WER {summary['asr_wer']:.3f} | queue max depth {res.queue_max_depth} | ring overrun "
          f"{res.ring_overrun_samples} | producer lateness {summary['producer_lateness_ms']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
