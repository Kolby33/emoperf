"""M3: profile the pipeline stages with torch.profiler and quantify the three hypotheses.

Runs every stage, sequentially, on the first N utterances of the fixed test subset
(the ground-truth clips, so segmentation is not part of this). Whisper is split into
its encoder call and its decode loop (generate with precomputed encoder_outputs; the
token IDs are checked to be identical to a plain generate call).

Outputs:
  traces/m3_profile.json            Chrome trace (open in https://ui.perfetto.dev); git-ignored
  bench/results/m3_profile.json     summary: per-stage time, top ops, shapes, decode-gap analysis

Note: profiling adds overhead, so absolute times here are inflated. Use them for the
breakdown (which stage, which op), and use bench/run_live.py numbers for latency.

Usage: python bench/profile_stages.py --n 30
"""

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transformers  # noqa: E402

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

from pipeline.live.stages import SR, Stages  # noqa: E402

LABELS = ["audio_feat", "audio_enc", "asr_feat", "asr_encoder", "asr_decode", "text_enc", "fusion"]


@torch.inference_mode()
def run_one(S: Stages, x: np.ndarray):
    with record_function("audio_feat"):
        xa = x[: S.max_samples]
        inp = S.fe(xa, sampling_rate=SR, return_tensors="pt").input_values
    with record_function("audio_enc"):
        hs = torch.stack(S.audio_model(inp, output_hidden_states=True).hidden_states, 1)
        a = hs.mean(2)[0]
    with record_function("asr_feat"):
        f = S.asr_proc.feature_extractor(x[: 30 * SR], sampling_rate=SR, return_tensors="pt").input_features
    with record_function("asr_encoder"):
        enc = S.asr.model.encoder(f)
    with record_function("asr_decode"):
        ids = S.asr.generate(input_features=f, encoder_outputs=enc, max_new_tokens=S.max_new_tokens)
        text = S.asr_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    with record_function("text_enc"):
        t = S.embed_text(text)
    with record_function("fusion"):
        S.head(t[None], a[None]).argmax()
    return ids


def merged_length(intervals):
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0.0)


def decode_gap_analysis(trace_path):
    """For every asr_decode region: how much of its wall time is covered by top-level aten ops
    on the same thread? The uncovered rest is Python / HF generate / dispatcher overhead."""
    ev = json.loads(Path(trace_path).read_text())["traceEvents"]
    regions = [e for e in ev if e.get("ph") == "X" and e.get("name") == "asr_decode"]
    ops = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
    by_tid = defaultdict(list)
    for o in ops:
        by_tid[o["tid"]].append((o["ts"], o["ts"] + o["dur"], o["name"]))
    rows = []
    for r in regions:
        s, e = r["ts"], r["ts"] + r["dur"]
        inside = [(a, b, n) for a, b, n in by_tid[r["tid"]] if a >= s and b <= e]
        # top-level ops only: drop ops fully nested in another op inside the region
        inside.sort(key=lambda t: (t[0], -(t[1] - t[0])))
        top, last_end = [], -1
        for a, b, n in inside:
            if a >= last_end:
                top.append((a, b, n))
                last_end = b
        covered = merged_length([(a, b) for a, b, _ in top])
        rows.append({"wall_us": r["dur"], "op_covered_us": covered, "n_top_ops": len(top)})
    wall = sum(r["wall_us"] for r in rows)
    cov = sum(r["op_covered_us"] for r in rows)
    return {"regions": len(rows), "wall_ms_total": wall / 1e3, "op_covered_ms_total": cov / 1e3,
            "uncovered_fraction": 1 - cov / wall if wall else None,
            "top_level_ops_per_region_mean": float(np.mean([r["n_top_ops"] for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--trace", type=Path, default=Path("traces/m3_profile.json"))
    ap.add_argument("--out", type=Path, default=Path("bench/results/m3_profile.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    S = Stages(Path("models/fusion_text_audio_seed0.pt"))
    idx = pd.read_csv("data/index/test.csv", keep_default_na=False).set_index("utt_key")
    keys = Path("bench/subsets/test_fixed_500.txt").read_text().split()[: args.n]
    clips = [sf.read(idx.loc[k, "wav"], dtype="float32")[0] for k in keys]
    dev = pd.read_csv("data/index/dev.csv", keep_default_na=False)

    # Correctness: split Whisper call == plain generate, on every profiled clip
    with torch.inference_mode():
        for x in clips:
            f = S.asr_proc.feature_extractor(x, sampling_rate=SR, return_tensors="pt").input_features
            ref = S.asr.generate(f, max_new_tokens=S.max_new_tokens)
            assert torch.equal(ref, S.asr.generate(input_features=f, encoder_outputs=S.asr.model.encoder(f),
                                                    max_new_tokens=S.max_new_tokens))
    for w in dev.wav[:3]:  # warm-up, dev clips only
        run_one(S, sf.read(w, dtype="float32")[0])

    # Per-utterance wall timing without the profiler (for the duration-vs-time check)
    per_utt = []
    for k, x in zip(keys, clips):
        t = {}
        with torch.inference_mode():
            t0 = time.perf_counter()
            f = S.asr_proc.feature_extractor(x, sampling_rate=SR, return_tensors="pt").input_features
            enc = S.asr.model.encoder(f)
            t1 = time.perf_counter()
            ids = S.asr.generate(input_features=f, encoder_outputs=enc, max_new_tokens=S.max_new_tokens)
            t2 = time.perf_counter()
            S.audio_branch(x)
            t3 = time.perf_counter()
        t = {"utt_key": k, "duration_s": len(x) / SR, "asr_encoder_ms": (t1 - t0) * 1e3,
             "asr_decode_ms": (t2 - t1) * 1e3, "tokens": int(ids.shape[-1]), "audio_branch_ms": (t3 - t2) * 1e3}
        per_utt.append(t)
    pu = pd.DataFrame(per_utt)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        for x in clips:
            run_one(S, x)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.trace))

    ka = prof.key_averages()
    stage_ms = {e.key: e.cpu_time_total / 1e3 for e in ka if e.key in LABELS}
    total_stage = sum(stage_ms.values())
    ops = sorted([e for e in ka if e.key not in LABELS and not e.key.startswith("ProfilerStep")],
                 key=lambda e: e.self_cpu_time_total, reverse=True)
    all_self = sum(e.self_cpu_time_total for e in ops)
    top_ops = [{"op": e.key, "self_ms": e.self_cpu_time_total / 1e3, "share": e.self_cpu_time_total / all_self,
                "calls": e.count} for e in ops[:12]]
    shapes = sorted(prof.key_averages(group_by_input_shape=True), key=lambda e: e.self_cpu_time_total, reverse=True)
    top_shapes = [{"op": e.key, "shapes": str(e.input_shapes)[:160], "self_ms": e.self_cpu_time_total / 1e3,
                   "calls": e.count} for e in shapes if e.key not in LABELS][:10]
    gap = decode_gap_analysis(args.trace)

    dec_per_tok = (pu.asr_decode_ms / pu.tokens).median()
    corr_enc = float(np.corrcoef(pu.duration_s, pu.asr_encoder_ms)[0, 1])
    corr_audio = float(np.corrcoef(pu.duration_s, pu.audio_branch_ms)[0, 1])
    summary = {
        "n": args.n, "threads": args.threads, "torch": torch.__version__, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,  # macOS reports bytes
        "stage_share_under_profiler": {k: {"ms": v, "share": v / total_stage} for k, v in
                                       sorted(stage_ms.items(), key=lambda kv: -kv[1])},
        "top_ops_by_self_cpu": top_ops, "top_op_shapes": top_shapes, "decode_gap": gap,
        "no_profiler": {
            "asr_encoder_ms_p50": float(pu.asr_encoder_ms.median()),
            "asr_encoder_ms_min_max": [float(pu.asr_encoder_ms.min()), float(pu.asr_encoder_ms.max())],
            "asr_decode_ms_p50": float(pu.asr_decode_ms.median()), "decode_ms_per_token_p50": float(dec_per_tok),
            "tokens_p50": float(pu.tokens.median()), "audio_branch_ms_p50": float(pu.audio_branch_ms.median()),
            "corr_duration_vs_asr_encoder": corr_enc, "corr_duration_vs_audio_branch": corr_audio,
            "duration_s_p50": float(pu.duration_s.median()),
            "padding_fraction_p50": float(1 - pu.duration_s.clip(upper=30).median() / 30)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))

    print(f"Stage share of profiled time (n={args.n}, profiler on):")
    for k, v in summary["stage_share_under_profiler"].items():
        print(f"  {k:>12}: {v['ms']:8.1f} ms  {v['share']:6.1%}")
    print("\nTop ops by self CPU time:")
    for o in top_ops:
        print(f"  {o['op']:<36} {o['self_ms']:8.1f} ms {o['share']:6.1%}  calls {o['calls']}")
    print("\nTop (op, input shape):")
    for o in top_shapes:
        print(f"  {o['op']:<22} {o['self_ms']:8.1f} ms  calls {o['calls']:5d}  {o['shapes']}")
    print("\nDecode-loop gap:", gap)
    print("No-profiler timings:", summary["no_profiler"])
    print(f"peak RSS {summary['peak_rss_mb']:.0f} MB; trace -> {args.trace}; summary -> {args.out}")


if __name__ == "__main__":
    main()
