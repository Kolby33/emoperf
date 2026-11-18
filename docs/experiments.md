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

## E1 (M1): Data problems found while building the dataset

Source: `MELD.Raw.tar.gz` from the Hugging Face mirror of declare-lab/MELD, 10,878,146,150 bytes, sha256 `a56b4407d574195cbce470d86f9c9d72fcfea59b0e34502ecd4babee4a5c613e` (matches the published hash). Labels: GitHub declare-lab/MELD at commit `e8cedf2`.

| Problem | Evidence | Decision |
|---|---|---|
| `train_sent_emo.csv` isn't in the top level of the archive | archive listing; it's inside `train.tar.gz` | Use the GitHub CSVs for all splits |
| The archive CSVs have broken apostrophes (`\x92`, a Windows-1252 leftover) | 2695 / 309 / 743 train/dev/test utterance texts differ from GitHub; keys and emotion labels are identical | Use the GitHub CSVs (clean text for the tokenizer) |
| 132 test clips exist twice (`diaX_uttY.mp4` and `final_videos_testdiaX_uttY.mp4`) with different bytes | For all 131 labeled pairs, the prefixed file's length matches the label's StartTime/EndTime (median error 0.00 s); the regular file is off by a median of 1.49 s | Prefer the prefixed file (`extract_audio.py`) |
| `dia125_utt3.mp4` (train) won't decode | ffmpeg: "Invalid data found when processing input" | Dropped and logged in the manifest |
| 12 train and 2 dev clips are shorter than 0.1 s | `build_index.py` | Dropped (too short to encode). **No test utterances dropped** |
| 2 test labels look wrong (`dia155_utt3` has start = end; `dia242_utt2` is off by 1.7 s) | duration check | Kept; test set stays complete |
| Very long clips: max 305 s (test), 41 s (train) | duration stats; label timestamps agree | Audio encoder truncates to 20 s (7 train, 2 dev, 2 test clips affected) |

Final sizes: train 9976, dev 1106, **test 2610 (all labeled test utterances)**. Audio: 8.69 / 0.96 / 2.40 h.

## E2 (M1): Offline baseline, frozen encoders + fusion head

**Setup:** text `distilbert/distilroberta-base` (all 7 hidden states), audio `facebook/wav2vec2-base` (all 13 hidden states), each mean-pooled. Head: learned softmax layer mix per modality, then LayerNorm, concatenation and an MLP (256 hidden, dropout 0.3). AdamW lr 1e-3, batch 128, class weights ∝ 1/sqrt(freq), early stopping on dev weighted F1 (patience 8). 3 seeds per mode. Raw data: `bench/results/m1_baseline.json`.

**Hypothesis:** audio adds signal beyond the text, so text+audio > text > audio > majority.

**Result (test, n = 2610, mean ± std over seeds 0, 1, 2):**

| Model | Weighted F1 | Macro F1 | Accuracy |
|---|---|---|---|
| Majority class (neutral) | 0.3127 | 0.0928 | 0.4812 |
| Audio only | 0.4638 ± 0.0037 | 0.2758 ± 0.0041 | 0.4871 ± 0.0061 |
| Text only | 0.6046 ± 0.0021 | 0.4225 ± 0.0088 | 0.6132 ± 0.0029 |
| Text + audio | 0.6133 ± 0.0082 | 0.4284 ± 0.0129 | 0.6227 ± 0.0127 |

- Audio alone is well above majority (+0.15 weighted F1), so the audio encoder carries emotion signal.
- **Text+audio vs text is +0.009 weighted F1, about the size of the seed spread (std up to 0.008). With 3 seeds, I don't claim a fusion gain.** Possible reasons: concatenation lets the head lean on the stronger text features, and mean-pooling the audio throws away timing. Later experiment ideas: modality dropout, a gate, attention pooling over audio frames.
- Learned audio layer weights peak at wav2vec2 layers 6–8 (0.13–0.15 vs uniform 0.077) in the audio-only head. That's consistent with middle layers carrying the most paralinguistic information.
- The weak classes are disgust (F1 0.09) and fear (0.12): 68 and 50 test examples, often predicted as anger or neutral.
- Training takes 1–4 s per run on the CPU, because the encoders run once (caching took 450 s on MPS).

**Reference model for M2 onward:** `fusion_text_audio_seed0.pt`, chosen as seed 0 in advance and not picked by test score. Test weighted F1 0.6032, macro F1 0.4107. Fixed 500-utterance subset (`bench/subsets/test_fixed_500.txt`, seed 0): weighted F1 0.6233, macro F1 0.4011. These are the accuracy numbers later optimizations get compared against.

**Embedding cache check:** MPS and CPU wav2vec2 outputs on 32 dev clips: max relative difference 3.9e-4, min cosine 0.99999976. Most of the difference comes from storing as fp16.

## E3 (M2): Live pipeline, sequential vs concurrent branches

**Setup:** 100 utterances (first 100 keys of `bench/subsets/test_fixed_500.txt`) joined into one 358.2 s stream with 0.8 s gaps of −75 dBFS noise, fed in 20 ms chunks at real-time pace. Energy segmenter: −55 dBFS threshold, 0.3 s hangover. Everything on CPU, `torch.set_num_threads(4)`, fp32. ASR `openai/whisper-base.en` with greedy decoding. Head `fusion_text_audio_seed0.pt`. Runs interleaved S C S C with 60 s cooldowns (`bench/m2_runs.sh`), git `8f68f42`, AC power, no thermal warnings before or after any run. Raw data: `bench/results/m2_{sequential,concurrent}_r{1,2}.json`.

**Hypothesis:** running branch A (audio) and branch B (ASR + text) at the same time cuts inference latency from A + B to max(A, B), saving about the audio branch's time (~60 ms).

**Result (ms, p50 / p95, per detected segment, n = 107 segments per run):**

| Stage | Sequential r1 | Sequential r2 | Concurrent r1 | Concurrent r2 |
|---|---|---|---|---|
| Endpoint delay (hangover) | 300.4 / 304.2 | 300.5 / 303.9 | 300.4 / 307.3 | 300.2 / 306.1 |
| Branch A: audio encoder total | 62.0 / 121.3 | 63.2 / 126.1 | 86.6 / 196.2 | 89.5 / 210.7 |
| Branch B: ASR (Whisper generate) | 150.7 / 226.8 | 148.8 / 236.3 | 180.6 / 300.9 | 198.1 / 310.0 |
| Branch B total (ASR + text encoder) | 161.8 / 239.5 | 160.1 / 248.4 | 200.4 / 331.0 | 221.0 / 331.1 |
| Inference total | 223.6 / 342.1 | 223.4 / 356.7 | 201.2 / 331.5 | 221.6 / 331.6 |
| **End-to-end (speech end to output)** | **525.3 / 641.1** | **526.6 / 660.4** | **501.9 / 634.4** | **518.0 / 631.4** |
| RTF overall (busy time / audio time) | 0.072 | 0.073 | 0.065 | 0.068 |

Fusion head 0.1 ms and response template < 0.1 ms in every run. Queue max depth 1, ring overrun 0, and the producer was a median 3.9 ms late per chunk (p99 4.9 ms, max 7.6–17.9 ms) without drifting.

**Findings:**

1. **Concurrency helps much less than the theory says.** The ideal concurrent time is max(A, B) ≈ 162 ms (sequential branch B). The measured result is 201–222 ms. Inference p50 improved by only 2–22 ms (1–10%), and the two concurrent runs differ from each other by 20 ms, about as much as the effect itself. p95 improved by 11–25 ms.
2. **Why: the branches slow each other down.** Running together, the audio encoder went from 62 to 87–90 ms and ASR from 149–151 to 181–198 ms. Each branch starts its own 4-thread OpenMP team, so 8 compute threads compete for 4 P-cores (the rest land on E-cores or wait). That contention adds 39–60 ms to the critical path and cancels most of the ~62 ms we hoped to save. This is a hypothesis for M3/M4: thread sweeps (e.g. 2 + 2 threads) should test it.
3. **The text branch is the critical path in 99–100% of segments,** and ASR is 90–93% of it. Whisper is the thing to optimize.
4. **The largest part of end-to-end latency is the 300 ms hangover,** waiting for silence to decide the utterance ended (57–60% of p50 end-to-end). Compute is about 200–225 ms of it. Cutting the hangover trades latency for more split utterances; that's an M4 experiment.
5. **Plenty of headroom:** RTF 0.065–0.073 means the pipeline is busy only about 7% of real time, which is why the queue never backs up.

**Accuracy (the same in all 4 runs, because the pipeline is deterministic):**

| On the same 100 utterances | Weighted F1 | Macro F1 | Accuracy |
|---|---|---|---|
| Offline (gold transcript, oracle segments) | 0.599 | 0.424 | 0.62 |
| Live (Whisper transcript, energy segmenter) | 0.503 | 0.332 | 0.53 |

- Live loses 0.096 weighted F1. Two causes, not yet separated: ASR word error rate of 31.5% on this TV audio (laughter, crosstalk, music), and segmentation (92 utterances got exactly one segment, 7 were split, 1 was missed). An oracle-segmenter run would separate them.
- n = 100 is small: one utterance is 1% accuracy. These accuracy numbers are a reference for later speed changes, not a model-quality claim.

**Not explained yet:** the first warm-up call (after the models load) took 290, 506, 997 and 2032 ms in the four runs.

**Hypotheses for M3 (write down before profiling):**
- H1: Whisper's encoder always processes a 30 s padded input (3000 mel frames), even for a 2.5 s utterance, so most of its encoder compute goes to padding.
- H2: In concurrent mode, OpenMP oversubscription (2 × 4 threads on 4 P-cores) is what slows both branches.
- H3: Whisper's token-by-token decode loop has significant Python overhead per token, which also holds the GIL and slows the other branch.

## E4 (M3): Profiling — the top three hotspots

**Setup:** `python bench/profile_stages.py --n 30` (torch.profiler, CPU activities, `record_shapes`, `profile_memory`) over the first 30 utterances of the fixed subset, using ground-truth clips with every stage run sequentially and 4 threads. Whisper is split into its encoder call and its decode loop (`generate` with precomputed `encoder_outputs`); the token IDs are asserted identical to a plain `generate` on all 30 clips. Chrome trace in `traces/m3_profile.json` (350 MB, git-ignored; open in Perfetto). Summary in `bench/results/m3_profile.json`; per-region breakdown in `bench/results/m3_trace_breakdown.txt` (`bench/analyze_trace.py`). Instruments Time Profiler traces of 10-utterance live runs: `bench/results/m3_xctrace_{sequential,concurrent}.txt` (`bench/analyze_xctrace.py`).

Profiler overhead: profiled Whisper encoder time per clip (91.6 ms) is close to the unprofiled 92.7 ms, but the decode loop is inflated (75.6 vs 63.6 ms, about 19%) because it's made of many small ops. Shares below are under the profiler; the ms values marked "no profiler" are not.

**Where the time goes (share of profiled stage time):** Whisper encoder 41.1%, Whisper decode 33.9%, wav2vec2 audio encoder 19.7%, text encoder 4.3%, Whisper feature extraction 0.8%, fusion head < 0.1%.

### Hotspot 1: Whisper encoder runs on 30 s of input, about 92% of it padding (H1: confirmed)

- Every clip goes in as `(1, 80, 3000)` mel frames and comes out as `(1, 1500, 512)`, whatever its length. The median clip is 2.52 s, so the median input is 91.6% padding.
- No profiler: encoder 92.7 ms p50 (range 91.7–97.9 ms) on clips from 0.59 to 7.38 s. **Correlation between clip length and encoder time: r = 0.06.** For comparison, the wav2vec2 branch, which only sees the real audio, has r = 0.99.
- Inside the encoder: `addmm` (linear layers on `[1500, 512]` activations) 39.9%, and attention `_scaled_dot_product_flash_attention_for_cpu` on `[1, 8, 1500, 64]` 38.5%. That one (op, shape) pair, 180 calls, is the single most expensive in the whole profile: 1049 ms.
- **Hypothesis for M4:** feeding the encoder only the real frames (rounded up to some bucket) would make its time scale with clip length. Attention cost grows with length², so it would shrink even faster. Risks: Whisper was trained on 30 s windows and has a fixed 1500-position embedding, and the HF implementation checks the input length. This needs a code change plus a WER and F1 check, not just a flag.

### Hotspot 2: Whisper decode loop, about 7 ms per token (H3 as stated: refuted; a different cause found)

- No profiler: decode 63.6 ms p50 for a median of 9 tokens, **7.1 ms per token.**
- H3 said the gaps between ops (Python / `generate` overhead) dominate. **They don't:** top-level aten ops cover 89.3% of decode wall time; only 10.7% is gaps. On average each clip's decode runs 3417 top-level ops.
- What the time actually is (decode op self time, 2025 ms):
  - `aten::mm` 31.1%, mostly the logits projection `[1, 512] × [512, 51864]` (399 ms for 309 calls), done once per token over the whole vocabulary.
  - `aten::addmm` 25.3% from 14,418 tiny calls (`[1, 512] × [512, 512]` and similar).
  - **`aten::sort` 13.4% + `isin` 2.1% + `_unique` 1.1%** come from Hugging Face's Whisper logits processors. On every step, they call `torch.isin(vocab_tensor, ...)` over the full 51,864-token vocabulary, once against the 90 suppressed tokens (`SuppressTokensLogitsProcessor`, `transformers/generation/logits_process.py:1945`) and once against a 2-token set. `isin` sorts the whole vocabulary through `_unique`. The mask is the same every step, but it's rebuilt 339 times for 30 clips: about 1 ms per token.
- The decode loop mostly doesn't use the 4 threads: in the sequential Time Profiler trace, 1199 of 2241 busy 1 ms bins had only one compute thread running.
- **Hypotheses for M4:** compute the suppress mask once (expected about −1 ms per token, same tokens out); fewer intra-op threads for the decode, which is too small to split well.

### Hotspot 3: In concurrent mode the two branches fight for P-cores (H2: supported)

From Instruments Time Profiler (about 1 sample per ms per running thread; last 30 s of a 10-utterance live run, i.e. the streamed part):

| | Sequential | Concurrent |
|---|---|---|
| Distinct threads doing compute | 6 | 13 |
| Compute threads running at once, p95 / max | 4 / 4 | 5 / 8 |
| Compute samples on E-cores | 13.5% | 28.2% |
| Total compute samples (same 10 clips) | 4630 | 5734 (+24%) |
| Accelerate BLAS samples on E-cores | 408 | 787 |

- Each branch thread starts its own 4-thread OpenMP team, so up to 8 compute threads run on a 4-P-core chip. The extra threads land on E-cores, and the same matmuls cost more CPU time. That explains why each branch got slower in E3 (audio encoder 62 → 87–90 ms, ASR 149–151 → 181–198 ms) and why concurrency saved so little.
- **OpenMP spin-waiting isn't the cause:** the OpenMP runtime is the innermost library in only 0.3% of compute samples; 55–56% are inside Accelerate BLAS and 41% inside libtorch.
- **Hypothesis for M4:** split the budget (for example 2 + 2 threads, or ASR 3 + audio 1) so the two branches together fit on the 4 P-cores.

### Smaller finding: wav2vec2's convolutions use PyTorch's generic fallback

`aten::_slow_conv2d_forward` is 28.1% of the audio encoder's op time. The shapes are wav2vec2's 1-D feature-encoder convolutions (e.g. `[1, 512, 1, 23618] * [512, 512, 1, 3]`, size growing with the clip) plus Whisper's two input convolutions. This PyTorch build has no oneDNN (`USE_MKLDNN=OFF`, see M0), so convolutions run on the generic path. It's a candidate for ONNX Runtime or Core ML in M4.

**Memory:** peak RSS during the profiled run was 4.6 GB, but `profile_memory` and `record_shapes` inflate this. It isn't a pipeline number; M4 measures memory without the profiler.

## E5 (M4): Optimization experiments, one change at a time

**Method:** `bench/m4_suite.py` runs each variant in a fresh process (so thread settings, environment variables and patches can't leak between variants), on the first 100 utterances of the fixed subset, ground-truth clips, oracle segmentation. 5 dev-clip warm-ups first (they also absorb `torch.compile` compile time). Two rounds per variant, round 2 in reverse order. 15 s cooldowns, AC power, no thermal warnings in any run. Each variant changes **one thing** from `baseline` (sequential, `torch.set_num_threads(4)`, CPU, fp32, M2 code with Whisper split into encoder + decode, which is bit-identical). The `combo_*` rows combine changes that helped. Full table: `docs/m4_variants_table.md` (from `bench/m4_summarize.py`). Raw data: `bench/results/m4/*.json`.

**Baseline drift:** the baseline was measured in three sessions: 195.9, 192.7 and 200.5 ms p50 (range of single runs 192.0–203.2 ms). **Differences under about 5% are inside that drift.** Small effects below are compared against the baseline from the same session.

**Accuracy check:** every variant reports weighted F1 on the 100 utterances, Whisper WER against MELD transcripts, and the share of utterances where the emotion prediction and the transcript are identical to the baseline's.

### Results (inference latency per utterance, mean of 2 rounds)

| Variant (one change) | p50 ms | p95 ms | vs baseline | wF1 (Δ) | WER | same preds / text | Kept? |
|---|---|---|---|---|---|---|---|
| baseline (CPU fp32, 4 threads) | 195.9 (session 1) | 376.1 | 1.00× | 0.497 | 0.320 | — | — |
| threads = 1 | 275.6 | 493.1 | 0.71× | same | same | 100% / 100% | no |
| threads = 2 / 3 | 211.4 / 199.6 | 396.9 / 365.9 | 0.93× / 0.98× | same | same | 100% / 100% | no |
| threads = 6 / 8 / 10 | 196.0 / 192.6 / 195.9 | 361.4 / 356.8 / 359.5 | 1.00× / 1.02× / 1.00× | same | same | 100% / 100% | no (no gain) |
| `VECLIB_MAXIMUM_THREADS=1` | 201.1 | 394.2 | 0.97× | same | same | 100% / 100% | no |
| concurrent branches (4 + 4 threads) | 180.3 | 314.4 | 1.09× | same | same | 100% / 100% | yes (CPU) |
| concurrent, threads split 2 + 2 | 197.1 | 320.8 | 0.99× | same | same | 100% / 100% | no |
| concurrent, threads split 1 + 3 | 191.4 | 326.8 | 1.02× | same | same | 100% / 100% | no |
| cached suppress masks | 193.6 (ASR −5.5 ms) | 356.3 | 1.01× | same | same | 100% / 100% | **yes** (free, identical output) |
| Whisper input trimmed: audio + 1 s, 1 s buckets | 148.1 | 547.4 | 1.32× | 0.495 (−0.003) | **2.389** | 78% / 29% | **no** |
| trimmed: audio + 2 s, 5 s buckets | 121.9 | 532.2 | 1.61× | 0.472 (−0.025) | **1.017** | 90% / 49% | **no** |
| trimmed: 10 s windows | 141.6 (±23.7) | 345.7 | 1.38× | 0.460 (−0.037) | 0.333 | 94% / 61% | **no** |
| int8 dynamic quantization (qnnpack) | 404.2 | 810.2 | 0.48× | 0.493 (−0.005) | 0.380 | 87% / 69% | **no** |
| bf16 weights + activations (CPU) | 3224.9 | 8252.3 | 0.06× | 0.499 (+0.002) | 0.318 | 99% / 95% | **no** (RTF 1.43: not real-time) |
| fp16 weights + activations (CPU) | 2843.2 | 7304.0 | 0.07× | 0.497 | 0.319 | 100% / 98% | **no** (RTF 1.28) |
| `torch.compile` (3 encoders, dynamic shapes) | 199.3 | 386.5 | 0.97× (same session: 192.7) | same | same | 100% / 100% | no |
| all 3 models on MPS (GPU) | 101.4 (session 3) | 180.8 | 1.98× (vs session-3 baseline 200.5) | same | same | 100% / 100% | yes |
| only Whisper on MPS | 120.1 | 252.3 | 1.63× | same | same | 100% / 100% | yes |

ONNX Runtime 1.32 (`bench/m4_ort.py`, `bench/results/m4_ort.json`), encoders only, 100 clips:

| Encoder | PyTorch CPU | ORT CPU EP | ORT Core ML EP |
|---|---|---|---|
| wav2vec2 + pooling, p50 | 34.9 ms | 48.8 ms (0.72×), max rel. diff 1.3e-5 | 110.8 ms (0.32×), max rel. diff 5.4e-2 |
| Whisper encoder (3000 frames), p50 | 96.0 ms | 117.1 ms (0.82×), max rel. diff 4.2e-4 | 202.5 ms (0.47×), max rel. diff 1.6e-1 |

ORT's log: Core ML supports only 386/716 wav2vec2 nodes (graph split into 109 partitions) and 192/231 Whisper-encoder nodes (33 partitions), so execution keeps switching between Core ML and the CPU. The large differences suggest Core ML used reduced precision. Not kept.

### Combinations (same session as the last baseline, 197.9 / 203.2 ms)

| Combination | p50 ms | p95 ms | vs baseline (200.5) | Accuracy |
|---|---|---|---|---|
| CPU: concurrent + mask cache | 178.0 | 292.9 | 1.13× | identical |
| all MPS + mask cache | 99.4 | 181.9 | 2.02× | identical |
| all MPS + mask cache + concurrent | 96.1 | 175.9 | 2.09× | identical |
| **Whisper on MPS ‖ wav2vec2 on CPU, concurrent + mask cache** | **82.8** | **137.4** | **2.42× (p95 2.71×)** | identical (100% same preds and text) |

Memory: process RSS doesn't include Metal (GPU driver) allocations. For the best combination, RSS was 2094 MB plus 1177 MB reported by `torch.mps.driver_allocated_memory()`, about 3.3 GB in total, against 2.1 GB for the CPU baseline. Variants with all models on MPS: RSS 727–733 MB plus about 2.2 GB in Metal. The speedup costs roughly +1.2 GB of memory.

### What each result means

1. **More threads stop helping after 4.** 1 → 2 → 4 threads: 275.6 → 211.4 → about 196 ms. 6, 8 and 10 threads (adding E-cores) are within noise of 4. And `VECLIB_MAXIMUM_THREADS=1`, which limits Accelerate's own threads, costs only 3%. Consistent with E0: most of the time is in Accelerate matmuls, which barely scale with CPU threads on this chip.
2. **Splitting the thread budget between concurrent branches didn't fix contention** (H2 fix attempt: failed). With fewer threads, ASR got slower (180–186 ms) more than the audio branch got faster. Concurrency only paid off once the branches ran on **different hardware**: Whisper on the GPU and wav2vec2 on the CPU. Then the audio encoder ran at 38 ms, the same as alone, instead of 65 ms when sharing the CPU or 49 ms when sharing the GPU.
3. **Cached suppress masks: a real but small win** (M3 hotspot 2). Same-session ASR p50 went from 151.5 to 146.0 ms (−3.6%), with identical tokens on all 100 utterances. Smaller than the ~9 ms I predicted from the profile (1 ms × 9 tokens). The profiler overstated the cost, because it inflates small ops (E4: decode had about 19% profiler overhead).
4. **Trimming Whisper's padded input is fast but breaks ASR** (M3 hotspot 1 fix: failed). ASR dropped to 74–90 ms, but Whisper hallucinated on short windows. WER went to 2.39 (outputs much longer than the reference, i.e. repetition loops), and p95 got *worse* (547 ms) because hallucinated outputs take many decode steps. Even 10 s windows cost −0.037 wF1. The hotspot is real, but fixing it needs a model trained for shorter windows, not trimming.
5. **Lower precision is a large slowdown on this CPU.** int8 through qnnpack is 2× slower, with worse WER (0.380). bf16 and fp16 are 14–17× slower and no longer real-time. The fp32 path runs on Accelerate's matrix hardware; these paths don't (M4 probe: a Whisper-sized feed-forward layer took 5.1 ms in fp32, 17.8 int8, 28.3 fp16, 45.6 bf16).
6. **`torch.compile` didn't help:** the encoders are matmul-bound, Inductor calls the same Accelerate matmuls, and the decode loop, where small-op overhead lives, stayed eager.
7. **The GPU is the big win.** Whisper's cost is many medium-sized matmuls over 1500 positions; on MPS ASR went from about 150 to about 73 ms.

## E6 (M4): Real-time end-to-end with the best configuration

**Setup:** `bench/m4_live_runs.sh`, the same 100-utterance real-time stream as M2. Configuration: Whisper on MPS, wav2vec2 + text encoder on CPU, branches concurrent, mask cache on. Hangover 0.3 s (same segmentation as M2) and 0.2 s (from the segmenter sweep), interleaved A B A B with 60 s cooldowns, git `8922d19`, no thermal warnings. Raw data: `bench/results/m4_live_best_h{0.3,0.2}_r{1,2}.json`.

**Segmenter sweep** (`bench/segmenter_sweep.py`, no models; `bench/results/m4_segmenter_sweep.json`):

| Hangover | GT utterances with 1 segment / split / missed | Endpoint delay p50 |
|---|---|---|
| 0.1 s | 86 / 13 / 1 | 100 ms |
| 0.2 s | 91 / 8 / 1 | 200 ms |
| 0.3 s (M2) | 92 / 7 / 1 | 300 ms |
| 0.5 s | 94 / 5 / 1 | 500 ms |
| 0.7 s | 96 / 3 / 1 | 700 ms |

No setting merged two utterances (the 0.8 s gaps are longer than any hangover). Chunk size: 10 and 20 ms identical; 50 / 100 ms only add 20–40 ms p50 rounding delay.

**Result (ms, p50 / p95):**

| | M2 sequential (r1) | M2 concurrent (r1) | Best, hangover 0.3 (r1, r2) | Best, hangover 0.2 (r1, r2) |
|---|---|---|---|---|
| ASR | 150.7 / 226.8 | 180.6 / 300.9 | 107.0 / 141.7, 114.2 / 184.6 | 104.8 / 147.1, 104.8 / 142.8 |
| Inference total | 223.6 / 342.1 | 201.2 / 331.5 | 126.4 / 161.9, 137.6 / 241.4 | 126.8 / 169.0, 126.8 / 165.9 |
| **End-to-end** | **525.3 / 641.1** | 501.9 / 634.4 | 427.1 / 462.5, 436.8 / 542.5 | **327.1 / 372.0, 328.6 / 366.2** |
| RTF overall | 0.072 | 0.065 | 0.039, 0.045 | 0.041, 0.041 |
| Live wF1 / WER | 0.503 / 0.315 | 0.503 / 0.315 | 0.503 / 0.315 | 0.503 / 0.318 |

- **End-to-end p50: 525 → 328 ms (−37%); p95: 641 → 366–372 ms (−42 to −43%).** 86–97 ms comes from compute (inference p50 224 → 126–138 ms) and 100 ms from the shorter hangover.
- Accuracy is unchanged. The 0.2 s hangover split one more utterance (91 / 8 / 1), but the weighted F1 on these 100 is the same (0.503); WER went 0.315 → 0.318.
- **Open question:** in the live stream, MPS ASR p50 is 105–114 ms, but offline (back-to-back clips) it's 72 ms. Not measured yet. One hypothesis: the GPU slows its clock during the ~3.5 s idle gaps between utterances, so each utterance starts on a "cold" GPU. Checking that needs `powermetrics` (sudo), so I'd run it with you. Because of this, the live speedup (1.6–1.8× on inference) is smaller than the offline one (2.4×). The README should quote the live number for latency.
