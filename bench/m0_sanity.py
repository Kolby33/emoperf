"""M0 sanity benchmark: how does PyTorch CPU speed change with thread count?

Two workloads, chosen to behave differently:
  - matmul (fp32, NxN): compute-bound, goes through PyTorch's BLAS backend.
  - gelu on a large vector: memory-bound, uses ATen's own parallel loops.

Thread counts are interleaved across rounds (1,2,4,...,1,2,4,...) so slow drift
from thermal throttling is spread over all settings instead of hitting the last one.

Usage:  python bench/m0_sanity.py [--out bench/results/m0_sanity.json]
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

THREADS = [1, 2, 4, 6, 8, 10]
ROUNDS = 5
ITERS_PER_ROUND = 10
WARMUP = 5
MATMUL_N = 2048
GELU_ELEMS = 32 * 1024 * 1024  # 128 MB of fp32 in, 128 MB out


def pmset_therm() -> str:
    try:
        return subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unavailable"


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def time_it(fn, iters):
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/m0_sanity.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    a = torch.randn(MATMUL_N, MATMUL_N)
    b = torch.randn(MATMUL_N, MATMUL_N)
    x = torch.randn(GELU_ELEMS)
    y = torch.empty_like(x)

    workloads = {
        "matmul_fp32_2048": (lambda: torch.mm(a, b), 2 * MATMUL_N**3, None),
        # writing into a preallocated y avoids timing a 128 MB allocation on every call
        "gelu_fp32_32M": (lambda: torch.ops.aten.gelu.out(x, out=y), None, 2 * GELU_ELEMS * 4),  # read x + write y
    }

    samples = {w: {t: [] for t in THREADS} for w in workloads}
    therm_before = pmset_therm()
    t_start = time.time()

    for r in range(ROUNDS):
        for t in THREADS:
            torch.set_num_threads(t)
            for name, (fn, _, _) in workloads.items():
                time_it(fn, WARMUP if r == 0 else 1)
                samples[name][t] += time_it(fn, ITERS_PER_ROUND)

    therm_after = pmset_therm()

    results = {}
    for name, (_, flops, nbytes) in workloads.items():
        results[name] = {}
        for t in THREADS:
            s = samples[name][t]
            p50 = statistics.median(s)
            row = {"n": len(s), "p50_ms": p50 * 1e3, "p95_ms": pct(s, 0.95) * 1e3, "min_ms": min(s) * 1e3}
            if flops:
                row["gflops_at_p50"] = flops / p50 / 1e9
            if nbytes:
                row["gb_per_s_at_p50"] = nbytes / p50 / 1e9
            results[name][t] = row

    mps = {"available": torch.backends.mps.is_available()}
    if mps["available"]:
        dev = torch.device("mps")
        am, bm = a.to(dev), b.to(dev)
        for _ in range(WARMUP):
            torch.mm(am, bm)
        torch.mps.synchronize()

        def mm_sync():
            torch.mm(am, bm)
            torch.mps.synchronize()

        s = time_it(mm_sync, ROUNDS * ITERS_PER_ROUND)
        p50 = statistics.median(s)
        mps["matmul_fp32_2048"] = {"n": len(s), "p50_ms": p50 * 1e3, "p95_ms": pct(s, 0.95) * 1e3,
                                   "gflops_at_p50": 2 * MATMUL_N**3 / p50 / 1e9}
        diff = (torch.mm(am, bm).cpu() - torch.mm(a, b)).abs().max().item()
        mps["max_abs_diff_vs_cpu"] = diff

    report = {
        "benchmark": "m0_sanity",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": round(time.time() - t_start, 1),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "config": {"threads": THREADS, "rounds": ROUNDS, "iters_per_round": ITERS_PER_ROUND,
                   "warmup": WARMUP, "matmul_n": MATMUL_N, "gelu_elems": GELU_ELEMS,
                   "interop_threads": torch.get_num_interop_threads()},
        "thermal_before": therm_before,
        "thermal_after": therm_after,
        "cpu": results,
        "mps": mps,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"torch {torch.__version__}, wrote {out}")
    for name in results:
        print(f"\n{name}")
        print(f"{'threads':>7} {'p50 ms':>9} {'p95 ms':>9}  rate")
        for t, row in results[name].items():
            rate = f"{row['gflops_at_p50']:.0f} GFLOP/s" if "gflops_at_p50" in row else f"{row['gb_per_s_at_p50']:.1f} GB/s"
            print(f"{t:>7} {row['p50_ms']:>9.2f} {row['p95_ms']:>9.2f}  {rate}")
    if mps.get("matmul_fp32_2048"):
        m = mps["matmul_fp32_2048"]
        print(f"\nMPS matmul: p50 {m['p50_ms']:.2f} ms, {m['gflops_at_p50']:.0f} GFLOP/s, "
              f"max |diff| vs CPU {mps['max_abs_diff_vs_cpu']:.2e}")
    print("\nthermal before:", therm_before.splitlines()[0] if therm_before else "")
    print("thermal after: ", therm_after.splitlines()[0] if therm_after else "")


if __name__ == "__main__":
    main()
