"""Summarize an Instruments Time Profiler trace: which cores and how many threads do the compute.

Time Profiler takes ~1 sample per ms per running thread; each sample records the core
it ran on (P "S Core" or E "E Core") and the call stack. A sample counts as "compute" if
its stack contains PyTorch, OpenMP or Accelerate code.

Reports, for the last --window-s seconds of the trace (the streamed part of a
bench/run_live.py run, after model loading and warm-up):
  - compute samples on P-cores vs E-cores
  - number of distinct OS threads doing compute
  - how many compute threads were running at once (per 1 ms bin): the parallelism

Usage:
  xcrun xctrace record --template "Time Profiler" --output traces/x.trace --launch -- \
      .venv/bin/python bench/run_live.py --mode concurrent --n 10 --out /tmp/x.json
  python bench/analyze_xctrace.py traces/x.trace --window-s 30
"""

import argparse
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

COMPUTE_LIBS = ("libtorch_cpu", "libomp", "libBLAS", "libvDSP", "libLAPACK", "libBNNS")


def export(trace: Path) -> ET.Element:
    with tempfile.NamedTemporaryFile(suffix=".xml") as f:
        subprocess.run(["xcrun", "xctrace", "export", "--input", str(trace), "--output", f.name, "--xpath",
                        '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]'], check=True,
                       capture_output=True)
        return ET.parse(f.name).getroot()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--process", default="python")
    args = ap.parse_args()

    root = export(args.trace)
    ids = {e.get("id"): e for e in root.iter() if e.get("id")}
    R = lambda e: ids[e.get("ref")] if e is not None and e.get("ref") else e  # noqa: E731

    rows = []
    for row in root.iter("row"):
        proc = R(row.find("process"))
        if proc is None or not proc.get("fmt", "").startswith(args.process):
            continue
        bt = R(row.find("tagged-backtrace"))
        frames = [R(f) for f in bt.findall("frame")] if bt is not None else []
        libs = set()
        for f in frames:
            b = R(f.find("binary"))
            if b is not None and b.get("name"):
                libs.add(b.get("name"))
        core = R(row.find("core")).get("fmt", "")
        leaf_bin = R(frames[0].find("binary")) if frames else None
        rows.append({"t_ns": int(R(row.find("sample-time")).text),
                     "tid": R(row.find("thread")).get("fmt", ""),
                     "core": "P" if "S Core" in core or "P Core" in core else "E",
                     "leaf_lib": leaf_bin.get("name") if leaf_bin is not None else "?",
                     "compute": any(any(c in lib for c in COMPUTE_LIBS) for lib in libs)})
    t_end = max(r["t_ns"] for r in rows)
    win = [r for r in rows if r["t_ns"] >= t_end - args.window_s * 1e9]
    comp = [r for r in win if r["compute"]]
    cores = Counter(r["core"] for r in comp)
    bins = defaultdict(set)
    for r in comp:
        bins[r["t_ns"] // 1_000_000].add(r["tid"])
    par = np.array([len(v) for v in bins.values()])
    n = len(comp)
    print(f"window: last {args.window_s:.0f} s, {len(win)} samples from '{args.process}', {n} compute samples")
    print(f"compute samples on P-cores: {cores['P']} ({cores['P'] / n:.1%}), E-cores: {cores['E']} ({cores['E'] / n:.1%})")
    print(f"distinct threads doing compute: {len({r['tid'] for r in comp})}")
    print(f"compute threads running at once (in busy 1 ms bins): mean {par.mean():.2f}, p50 {np.median(par):.0f}, "
          f"p95 {np.percentile(par, 95):.0f}, max {par.max()}")
    print("  histogram:", dict(sorted(Counter(par.tolist()).items())))
    print(f"busy 1 ms bins: {len(par)} ({len(par) / (args.window_s * 1000):.1%} of window)")
    leaf = Counter(r["leaf_lib"] for r in comp)
    print("innermost library of compute samples (libomp leaf = OpenMP runtime itself, mostly spin-waiting):")
    for lib, c in leaf.most_common(6):
        on_e = sum(1 for r in comp if r["leaf_lib"] == lib and r["core"] == "E")
        print(f"  {lib:<28} {c:6d} ({c / n:5.1%})  of which on E-cores {on_e}")


if __name__ == "__main__":
    main()
