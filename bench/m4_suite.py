"""Run every M4 variant in its own process, two rounds (round 2 in reverse order), with cooldowns.

Each variant changes ONE thing relative to `baseline` (sequential, 4 threads, CPU fp32).
Usage: python bench/m4_suite.py [--only name1,name2] [--rounds 2] [--cool 15]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VARIANTS = {
    # name: (mode, threads, env, opts)
    "baseline": ("sequential", 4, {}, {}),
    # threads (intra-op) on 4 P + 6 E cores
    "threads_1": ("sequential", 1, {}, {}),
    "threads_2": ("sequential", 2, {}, {}),
    "threads_3": ("sequential", 3, {}, {}),
    "threads_6": ("sequential", 6, {}, {}),
    "threads_8": ("sequential", 8, {}, {}),
    "threads_10": ("sequential", 10, {}, {}),
    "veclib_1": ("sequential", 4, {"VECLIB_MAXIMUM_THREADS": "1"}, {}),  # Accelerate's own threading
    # concurrency and thread budget split (audio, text)
    "concurrent": ("concurrent", 4, {}, {}),
    "conc_split_2_2": ("concurrent", 4, {}, {"branch_threads": [2, 2]}),
    "conc_split_1_3": ("concurrent", 4, {}, {"branch_threads": [1, 3]}),
    # M3 hotspot fixes
    "mask_cache": ("sequential", 4, {}, {"mask_cache": True}),
    "trim_pad1_b1": ("sequential", 4, {}, {"asr_trim_pad_s": 1.0, "asr_trim_bucket_s": 1.0}),
    "trim_pad2_b5": ("sequential", 4, {}, {"asr_trim_pad_s": 2.0, "asr_trim_bucket_s": 5.0}),
    "trim_win10": ("sequential", 4, {}, {"asr_trim_pad_s": 0.0, "asr_trim_bucket_s": 10.0}),
    # precision
    "int8_dynamic": ("sequential", 4, {}, {"int8_dynamic": True}),
    "bf16": ("sequential", 4, {}, {"dtype": "bf16"}),
    "fp16": ("sequential", 4, {}, {"dtype": "fp16"}),
    # compiler and devices
    "compile": ("sequential", 4, {}, {"compile": True}),
    "mps_all": ("sequential", 4, {}, {"device": "mps"}),
    "mps_asr_only": ("sequential", 4, {}, {"asr_device": "mps"}),
    # combinations of the changes that helped (not single-change variants)
    "combo_cpu_conc_mask": ("concurrent", 4, {}, {"mask_cache": True}),
    "combo_mps_mask": ("sequential", 4, {}, {"device": "mps", "mask_cache": True}),
    "combo_mps_mask_conc": ("concurrent", 4, {}, {"device": "mps", "mask_cache": True}),
    "combo_mpsasr_cpuaudio_conc_mask": ("concurrent", 4, {}, {"asr_device": "mps", "mask_cache": True}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--cool", type=int, default=15)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tag", default="", help="suffix for result files")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(VARIANTS)
    out_dir = Path("bench/results/m4")
    for r in range(1, args.rounds + 1):
        order = names if r % 2 else names[::-1]
        for name in order:
            mode, threads, env, opts = VARIANTS[name]
            out = out_dir / f"{name}{args.tag}_r{r}.json"
            cmd = [sys.executable, "bench/m4_bench.py", "--variant", name, "--mode", mode, "--threads", str(threads),
                   "--n", str(args.n), "--opts", json.dumps(opts), "--out", str(out)]
            res = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True)
            line = [l for l in res.stdout.splitlines() if l.strip()]
            print(f"r{r} " + (line[-1] if res.returncode == 0 and line else
                              f"{name}: FAILED rc={res.returncode}: {res.stderr.strip().splitlines()[-1:]}"), flush=True)
            time.sleep(args.cool)


if __name__ == "__main__":
    main()
