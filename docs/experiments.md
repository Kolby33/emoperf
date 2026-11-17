# Experiments log

Each entry: hypothesis, change, result, kept or reverted, and why. Failed and mixed results stay in the log.

## E0 (M0): Do more threads help, and do E-cores hurt?

**Hypothesis (from the plan):** speed improves up to 4 threads (P-cores only), then adding E-cores makes it slower, because a parallel op waits for its slowest thread.

**Change:** none. Baseline sweep of `torch.set_num_threads` over 1, 2, 4, 6, 8, 10 on two workloads. Numbers are in `docs/methodology.md`; raw data is in `bench/results/m0_sanity.json`.

**Result: mixed. It depends on the op.**

- **Matmul (Accelerate):** threads barely matter. 1 thread already gets 85% of the best rate (1791 vs 2102 GFLOP/s, at 4 threads). Past 4 threads it gets 13% slower (1824 GFLOP/s at 10). The hypothesis holds weakly here.
- **GELU (ATen OpenMP):** it keeps improving all the way to 10 threads: 23.4 ms at 1 thread, 6.6 ms at 4, 5.0 ms at 10. Adding the 6 E-cores cut the time by another 24% compared with 4 threads. The hypothesis is **wrong** for this op.

**Possible explanations (not yet verified):**

- *Matmul:* Accelerate may run large matmuls on the chip's matrix hardware (SME), not the NEON units of each core. If that matrix unit is shared per cluster, extra CPU threads wouldn't add throughput. To check: profile the call in M3, and compare with my own SME2 kernel in M5.
- *GELU:* one thread reaches only 11.5 GB/s. That's probably limited by the math for erf, not memory, so more cores (even slow ones) keep helping. 53 GB/s at 10 threads may or may not be the memory limit. To check: STREAM-style bandwidth test in M5.
- The 4-thread GELU p95 (9.97 ms against a 6.63 ms p50) has outliers that the other thread counts don't. Unexplained; watch for it in later runs.

**Follow-up trace (Time Profiler, 100 fp32 2048×2048 matmuls, `traces/m0_Time_Profiler.trace`, not committed):**

- 1626 samples landed inside Accelerate. In 99.9% of them, the innermost function was in `libBLAS.dylib`. The hottest single address accounted for 82.8% of those samples.
- Those samples were on **only 2 threads**, even though PyTorch was set to 4 OpenMP threads. Accelerate seems to handle its own threading (it uses `libdispatch`), so `torch.set_num_threads` probably doesn't control it. That would explain why the matmul thread sweep is flat.
- The samples are split about evenly between P-cores and E-cores (828 vs 798). So the matmul doesn't always run on P-cores.
- libBLAS has no symbol names, so the trace can't tell whether that hot address uses SME instructions. Disassembling it with lldb failed because attaching wasn't permitted. Still open; next tries are CPU Counters or Processor Trace in M3.

**Takeaway for M4:** the best thread count is per op, not global. The default (4) is a reasonable start, but the pipeline should be swept end to end, not tuned from this microbenchmark.

## E0b (M0): MPS and CPU matmul give bit-identical results. Open question.

**Observation:** `torch.equal(cpu_result, mps_result)` is True for fp32 2048×2048 and 1000×1000 matmuls. Both have exactly the same max error against a float64 reference (5.40e-4 and 1.80e-4). Timings differ (CPU 4 threads 8.17 ms, MPS 6.04 ms), so they aren't trivially the same code path.

**Why it's surprising:** different hardware usually adds numbers in a different order, and fp32 rounding then makes the results differ slightly.

**Not concluded.** Next step: check in M3 which device and kernels actually run (Metal System Trace needs Xcode). Until then, I make no claim that MPS speedups are GPU speedups.
