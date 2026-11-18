"""M4: offline per-utterance benchmark of one pipeline variant (one change vs baseline).

Runs every stage on the first --n utterances of the fixed test subset, using the
ground-truth clips (no streaming, no segmenter), so each run takes about a minute and
measures inference latency only. End-to-end latency of the final configuration is
measured with bench/run_live.py.

Each variant runs in its own process, because thread counts, env vars (e.g.
VECLIB_MAXIMUM_THREADS) and monkeypatches must not leak between variants.

Usage:
  python bench/m4_bench.py --variant baseline --out bench/results/m4/baseline_r1.json
  python bench/m4_bench.py --variant trim1 --opts '{"asr_trim_pad_s": 1.0}'
"""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transformers  # noqa: E402

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from bench.run_live import edit_distance, norm_words  # noqa: E402
from pipeline.live.stages import StageOptions, Stages  # noqa: E402

KEYS = ["audio_enc", "branch_audio", "asr_enc", "asr_dec", "asr", "text_enc", "branch_text", "inference"]


def pstats(xs):
    xs = np.asarray(xs) * 1e3
    return {"p50_ms": float(np.percentile(xs, 50)), "p95_ms": float(np.percentile(xs, 95)),
            "mean_ms": float(xs.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--opts", default="{}", help="JSON of StageOptions fields")
    ap.add_argument("--mode", choices=["sequential", "concurrent"], default="sequential")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    therm_before = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    opts = StageOptions(**json.loads(args.opts))
    if opts.branch_threads:
        opts.branch_threads = tuple(opts.branch_threads)
    t = time.perf_counter()
    S = Stages(Path("models/fusion_text_audio_seed0.pt"), opts)
    load_s = time.perf_counter() - t

    idx = pd.read_csv("data/index/test.csv", keep_default_na=False).set_index("utt_key")
    keys = Path("bench/subsets/test_fixed_500.txt").read_text().split()[: args.n]
    clips = [sf.read(idx.loc[k, "wav"], dtype="float32")[0] for k in keys]
    pool = ThreadPoolExecutor(2) if args.mode == "concurrent" else None

    def one(x):
        t0 = time.perf_counter()
        if pool is None:
            a, ta = S.audio_branch(x)
            t1 = time.perf_counter()
            tr, te, tt = S.text_branch(x)
            t2 = time.perf_counter()
            ba, bb = t1 - t0, t2 - t1
        else:
            def timed(fn):
                s = time.perf_counter()
                r = fn(x)
                return r, time.perf_counter() - s
            fa, fb = pool.submit(timed, S.audio_branch), pool.submit(timed, S.text_branch)
            (a, ta), ba = fa.result()
            (tr, te, tt), bb = fb.result()
        label, _, _ = S.fuse(te, a)
        S.respond(label, tr)
        return {**ta, **tt, "branch_audio": ba, "branch_text": bb, "inference": time.perf_counter() - t0,
                "pred": label, "transcript": tr}

    dev = pd.read_csv("data/index/dev.csv", keep_default_na=False)
    warm = []
    for w in dev.wav[: args.warmup]:  # dev clips only; first call includes torch.compile time
        s = time.perf_counter()
        one(sf.read(w, dtype="float32")[0])
        warm.append(time.perf_counter() - s)

    rows = []
    for k, x in zip(keys, clips):
        r = one(x)
        r.update(utt_key=k, duration_s=len(x) / 16000, label=int(idx.loc[k, "label"]))
        rows.append(r)
    df = pd.DataFrame(rows)
    therm_after = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout

    edits = refs = 0
    for r in df.itertuples():
        ref = norm_words(idx.loc[r.utt_key, "text"])
        edits += edit_distance(ref, norm_words(r.transcript))
        refs += len(ref)
    y, p = df.label.values, df.pred.values
    report = {
        "variant": args.variant, "mode": args.mode, "threads": args.threads, "opts": json.loads(args.opts),
        "env_vars": {k: v for k, v in os.environ.items() if k.startswith(("VECLIB", "OMP", "KMP", "PYTORCH"))},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "torch": torch.__version__,
        "macos": platform.mac_ver()[0],
        "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
        "n": len(df), "load_seconds": load_s, "warmup_seconds": warm,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,  # macOS: bytes
        # RSS does not include Metal (GPU driver) allocations, so report them separately for MPS variants
        "mps_driver_allocated_mb": (torch.mps.driver_allocated_memory() / 2**20
                                    if torch.backends.mps.is_available() else None),
        "thermal_warning": ("No thermal warning" not in therm_before) or ("No thermal warning" not in therm_after),
        "stages": {k: pstats(df[k]) for k in KEYS if k in df},
        "rtf_overall": float(df.inference.sum() / df.duration_s.sum()),
        "weighted_f1": float(f1_score(y, p, labels=range(7), average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, p, labels=range(7), average="macro", zero_division=0)),
        "wer": edits / refs,
        "asr_frames_mean": float(df.asr_frames.mean()), "asr_tokens_mean": float(df.asr_tokens.mean()),
        "per_utt": df.drop(columns=["transcript"]).to_dict("records"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=float))
    full = Path("data/runs/m4") / args.out.name
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps({"variant": args.variant, "transcripts": dict(zip(df.utt_key, df.transcript)),
                                "preds": dict(zip(df.utt_key, df.pred.astype(int).tolist()))}, indent=1))
    s = report["stages"]
    print(f"{args.variant:>18}: inference p50 {s['inference']['p50_ms']:6.1f} p95 {s['inference']['p95_ms']:6.1f} ms | "
          f"asr {s['asr']['p50_ms']:6.1f} audio_enc {s['audio_enc']['p50_ms']:5.1f} | RTF {report['rtf_overall']:.3f} | "
          f"wF1 {report['weighted_f1']:.3f} WER {report['wer']:.3f} | RSS {report['peak_rss_mb']:.0f} MB | "
          f"warm-up {warm[0]:.1f}s", flush=True)


if __name__ == "__main__":
    main()
