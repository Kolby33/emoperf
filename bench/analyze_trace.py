"""Break a torch.profiler Chrome trace down by labeled region.

For each record_function region name (e.g. asr_decode), sums the self time of every
cpu_op that runs inside it on the same thread, and lists the top ops, with the call
stacks reduced to "which op inside which op". Also lists the input shapes of a chosen op.

Usage:
  python bench/analyze_trace.py traces/m3_profile.json --region asr_decode --shapes-of aten::_slow_conv2d_forward
"""

import argparse
import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--region", default="asr_decode")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--shapes-of", default=None)
    args = ap.parse_args()

    ev = [e for e in json.loads(args.trace.read_text())["traceEvents"] if e.get("ph") == "X"]
    regions = [e for e in ev if e.get("cat") == "user_annotation"]
    ops = [e for e in ev if e.get("cat") == "cpu_op"]

    # self time = op duration minus time of ops directly nested in it (same thread)
    by_tid = defaultdict(list)
    for o in ops:
        by_tid[o["tid"]].append(o)
    self_us = {}
    for tid, lst in by_tid.items():
        lst.sort(key=lambda o: (o["ts"], -o["dur"]))
        stack = []
        for o in lst:
            while stack and stack[-1]["ts"] + stack[-1]["dur"] <= o["ts"]:
                stack.pop()
            self_us[id(o)] = o["dur"]
            if stack:
                self_us[id(stack[-1])] -= o["dur"]
            stack.append(o)

    # which region each op belongs to (region on same tid that contains it)
    reg_by_tid = defaultdict(list)
    for r in regions:
        reg_by_tid[r["tid"]].append((r["ts"], r["ts"] + r["dur"], r["name"]))
    for v in reg_by_tid.values():
        v.sort()
    totals, counts = defaultdict(Counter), defaultdict(Counter)
    for o in ops:
        rs = reg_by_tid.get(o["tid"], [])
        i = bisect.bisect_right(rs, (o["ts"], float("inf"), "")) - 1
        name = rs[i][2] if i >= 0 and rs[i][1] >= o["ts"] + o["dur"] else "(no region)"
        totals[name][o["name"]] += self_us[id(o)]
        counts[name][o["name"]] += 1

    reg_wall = Counter()
    for r in regions:
        reg_wall[r["name"]] += r["dur"]
    t = totals[args.region]
    tot = sum(t.values())
    print(f"region {args.region}: wall {reg_wall[args.region] / 1e3:.1f} ms, op self time {tot / 1e3:.1f} ms")
    for name, us in t.most_common(args.top):
        print(f"  {name:<52} {us / 1e3:8.1f} ms {us / tot:6.1%}  calls {counts[args.region][name]}")

    if args.shapes_of:
        sh = Counter()
        for o in ops:
            if o["name"] == args.shapes_of:
                sh[str(o.get("args", {}).get("Input Dims"))[:120]] += self_us[id(o)]
        print(f"\nshapes of {args.shapes_of} by self time:")
        for s, us in sh.most_common(10):
            print(f"  {us / 1e3:8.1f} ms  {s}")


if __name__ == "__main__":
    main()
