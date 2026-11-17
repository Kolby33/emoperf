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
