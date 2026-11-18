# Methodology

How every benchmark in this repo is run, and the machine it runs on. If a number in the README doesn't match the setup described here, the number is wrong.

## Machine (recorded 2026-11-17)

| Item | Value | Source |
|---|---|---|
| Model | MacBook Air, Mac17,3 (fanless) | `system_profiler SPHardwareDataType` |
| Chip | Apple M5 | `sysctl machdep.cpu.brand_string` |
| CPU cores | 10: 4 performance ("Super", perflevel0) + 6 efficiency (perflevel1). No SMT, 10 hardware threads | `sysctl hw.perflevel*` |
| P-core caches | L1i 192 KB, L1d 128 KB per core; L2 16 MB shared by the 4 P-cores | `sysctl hw.perflevel0` |
| E-core caches | L1i 128 KB, L1d 64 KB per core; L2 6 MB shared by the 6 E-cores | `sysctl hw.perflevel1` |
| Cache line / page | 128 B / 16 KB | `hw.cachelinesize`, `hw.pagesize` |
| Memory | 16 GB unified (shared by CPU, GPU, Neural Engine) | `hw.memsize` |
| GPU | Apple M5, 8 cores, Metal 4 | `system_profiler SPDisplaysDataType` |
| SIMD / matrix ISA | NEON, FP16, BF16, EBF16, DotProd, I8MM, SME, SME2, SME2p1, SME_F16F16, SME_B16B16 (full list in `sysctl hw.optional.arm`) | `sysctl hw.optional.arm.FEAT_*` |
| OS | macOS 26.6.2 (25G83) | `sw_vers` |
| Free disk | 291 GB | `df -h` |

## Software

| Item | Version |
|---|---|
| Apple clang | 17.0.0 (clang-1700.6.4.2), Command Line Tools 26.3 |
| cmake | 4.5.0 (Homebrew) |
| ffmpeg | 9.0.2 (Homebrew) |
| uv | 0.13.9 (Homebrew) |
| Python | 3.12.16 in `.venv`, created with `uv venv --python 3.12`. Separate from Anaconda. 3.12 was picked for the widest ML/audio wheel coverage. |
| PyTorch | 2.15.0 (all pins in `requirements.txt`) |

### PyTorch build facts that matter for performance

From `torch.__config__.show()` and `torch.__config__.parallel_info()`:

- **BLAS and LAPACK: Apple Accelerate.** Matmuls are not PyTorch's own code; they run in Apple's library.
- **No MKL, no oneDNN (MKLDNN).** The x86 fast paths don't exist here.
- **Parallel backend: OpenMP, bundled with the wheel.** Default intra-op threads = **4** (the P-core count), inter-op threads = 10.
- **Quantized engines: `qnnpack` only.** This matters for M4 int8 experiments.
- **MPS (Metal GPU backend): built and available.**
- **`torch.profiler` activities: CPU only.** It can't see MPS GPU work; that needs Metal System Trace in Instruments.

## Profiling tools available

| Tool | Status |
|---|---|
| `torch.profiler` + Perfetto | Works, CPU only |
| `sample`, `spindump` (stack sampling) | Present in `/usr/bin` |
| `powermetrics` (clocks, power, thermal) | Present, needs `sudo`, so I run it myself |
| Instruments / `xctrace` 27.2 (Xcode 27.2, build 27B52) | Works. Recorded test traces with Time Profiler and Metal System Trace. Templates available include CPU Profiler, CPU Counters, Processor Trace, Metal System Trace, Core ML and Power Profiler. Each Time Profiler sample records whether it ran on a P-core ("S Core") or an E-core. |
| `lldb` attach | Fails ("could not pause execution"). Probably needs debugging permission (Developer Mode / `DevToolsSecurity`), a security setting I haven't changed. |

Traces go in `traces/` (git-ignored). To read one from the command line, export it with `xcrun xctrace export --input X.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]'`. Sample stacks are `tagged-backtrace` elements; elements are written once with an `id` and then referenced by `ref`.

## Run conditions (every benchmark)

- Plugged in (AC power), Low Power Mode off (`pmset -g`, `lowpowermode 0`).
- Thermal state recorded before and after (`pmset -g therm`) and stored in the result JSON.
- Variants interleaved across rounds (A B C A B C ...), so slow thermal drift is spread across all of them.
- Warm-up iterations first; report p50 and p95; report the sample count.
- Fixed seeds (`torch.manual_seed(0)`).
- Burst runs (seconds) and sustained runs (minutes) are reported separately when they differ. The laptop has no fan, so sustained runs may be slower.
- Not yet controlled: which other apps are open. Record this per run from M2 onward.

## M0 sanity benchmark

`python bench/m0_sanity.py` writes `bench/results/m0_sanity.json`. Run 2026-11-17 16:17, 7.6 s wall time (a burst run), AC power, no thermal warnings before or after. 5 rounds × 10 iterations = 50 samples per cell, threads interleaved.

**fp32 matmul 2048×2048** (compute-bound, runs in Accelerate)

| Threads | p50 ms | p95 ms | GFLOP/s at p50 |
|---|---|---|---|
| 1 | 9.59 | 10.69 | 1791 |
| 2 | 8.57 | 9.16 | 2005 |
| 4 | 8.17 | 8.91 | 2102 |
| 6 | 9.19 | 10.06 | 1870 |
| 8 | 9.16 | 9.72 | 1875 |
| 10 | 9.42 | 9.99 | 1824 |

**fp32 GELU on 32M elements** (elementwise, runs in ATen's OpenMP loops; GB/s counts 4 bytes read + 4 bytes written per element)

| Threads | p50 ms | p95 ms | GB/s at p50 |
|---|---|---|---|
| 1 | 23.39 | 24.38 | 11.5 |
| 2 | 11.25 | 11.32 | 23.9 |
| 4 | 6.63 | 9.97 | 40.5 |
| 6 | 6.48 | 6.62 | 41.4 |
| 8 | 5.15 | 5.68 | 52.1 |
| 10 | 5.03 | 5.38 | 53.4 |

**MPS (GPU) fp32 matmul 2048×2048**, timed with `torch.mps.synchronize()` after each call: p50 6.04 ms, 2844 GFLOP/s, 50 samples.

What this run does and doesn't show is logged in `docs/experiments.md` (E0).

## Dataset (M1)

- MELD raw archive from https://huggingface.co/datasets/declare-lab/MELD (mirror of the official University of Michigan file). Size and sha256 are in `docs/experiments.md` (E1). Labels come from GitHub declare-lab/MELD commit `e8cedf2`.
- License: the repo and the dataset card say GPL-3.0. The clips come from the TV series *Friends*, so clips, audio and embeddings are **never committed or published**. Only code, utterance IDs and summary metrics are.
- Cite: Poria et al., "MELD", ACL 2019; Chen et al., "EmotionLines", arXiv:1802.08379.
- Audio: 16 kHz mono 16-bit PCM WAV via ffmpeg 9.0.2 (`pipeline/data/extract_audio.py`). 13,715 clips in 70 s with 8 workers.
- Rebuild everything: `scripts/m1_rebuild.sh`. Evaluation only: `python pipeline/fusion/evaluate.py`.
- Encoders: `distilbert/distilroberta-base` (Apache-2.0, pretrained on OpenWebText) and `facebook/wav2vec2-base` (Apache-2.0, pretrained on LibriSpeech). Neither was fine-tuned on emotion data, so the MELD test set can't leak through the encoders. (`j-hartmann/emotion-english-distilroberta-base` was rejected for this reason: it was trained on several emotion datasets, including MELD.)

## Live benchmark (M2)

- Command: `bench/m2_runs.sh` (sequential and concurrent, interleaved S C S C, 60 s cooldown), or `python bench/run_live.py --mode {sequential,concurrent} --n 100 --out ...`.
- Stream: the first 100 utterances of the fixed test subset, in key order, joined with 0.8 s of −75 dBFS seeded white noise between them; 358.2 s total. Fed in 20 ms chunks at real-time pace by a producer thread.
- Warm-up: 3 dev-set clips through both branches before the timed stream (never test clips).
- Latency definitions (all from one `time.perf_counter` clock):
  - end-to-end = time the response is ready − time the chunk holding the segment's last speech sample was written
  - endpoint delay = when the segmenter closed the segment − that same write time
  - RTF per segment = inference time / segment duration; overall RTF = total inference time / stream duration
- Accuracy: each ground-truth utterance takes the prediction of the segment that overlaps it most; a missed utterance counts as wrong. The offline reference uses the cached gold-transcript embeddings for the same utterances.
- Committed result JSONs contain no MELD text. Full runs, including transcripts, are written to `data/runs/` (git-ignored).
- Unit tests: `python -m pytest` (ring buffer, bounded queue, segmenter: empty, full, wraparound, overrun, close, threaded and chunk-size invariance cases).

## Optimization experiments (M4)

- Offline per-variant harness: `bench/m4_bench.py` (one process per variant), driven by `bench/m4_suite.py`; summary table `bench/m4_summarize.py` → `docs/m4_variants_table.md`.
- Baseline p50 drifted 192–203 ms across sessions; effects under about 5% are compared within a session only.
- Memory: peak process RSS (`ru_maxrss`) plus `torch.mps.driver_allocated_memory()` for MPS variants, because RSS doesn't count Metal allocations.
- Extra packages for M4 (in `.venv` only): onnxruntime 1.32.0, onnx 1.23.2, onnxscript 0.7.2. ONNX files in `data/onnx/` (git-ignored).
- Final live configuration: `bench/m4_live_runs.sh` (`--opts '{"asr_device": "mps", "mask_cache": true}' --mode concurrent --hangover 0.2`).
