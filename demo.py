"""Live demo: speak into the Mac's microphone, get an emotion tag and a response per utterance.

    python demo.py                      # microphone (macOS asks once for microphone access)
    python demo.py --source meld --n 8  # no microphone: replay MELD test clips in real time
    python demo.py --source file --file some.wav

Pipeline (the M4 best configuration): 20 ms chunks -> ring buffer -> energy segmenter with
adaptive noise floor -> [Whisper on MPS -> text encoder] in parallel with [wav2vec2 on CPU]
-> fusion head (trained on gold + Whisper text) -> response template.

Everything runs locally and offline: no audio or text leaves the machine. Stop with Ctrl+C.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models are cached; never call the network
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import bisect  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

warnings.filterwarnings("ignore")
import transformers  # noqa: E402

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

from pipeline.fusion.model import EMOTIONS  # noqa: E402
from pipeline.live.bounded_queue import CLOSED, BoundedQueue  # noqa: E402
from pipeline.live.ring_buffer import RingBuffer  # noqa: E402
from pipeline.live.segmenter import EnergySegmenter  # noqa: E402
from pipeline.live.stages import StageOptions, Stages  # noqa: E402

SR = 16000
CHUNK = 320  # 20 ms


class FeedClock:
    """Remembers when each block of samples arrived, to timestamp 'speech ended'."""

    def __init__(self):
        self.ends, self.times = [], []  # appended by one producer thread only

    def add(self, end_pos):
        self.ends.append(end_pos)
        self.times.append(time.perf_counter())

    def time_of(self, pos):
        i = bisect.bisect_left(self.ends, pos)
        return self.times[min(i, len(self.times) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["mic", "meld", "file"], default="mic")
    ap.add_argument("--file", type=Path, help="16 kHz mono WAV for --source file")
    ap.add_argument("--n", type=int, default=8, help="MELD test clips for --source meld")
    ap.add_argument("--device", type=int, default=None, help="input device index (see sounddevice.query_devices())")
    ap.add_argument("--head", type=Path, default=Path("models/fusion_text_audio_trainboth_seed0.pt"))
    ap.add_argument("--hangover", type=float, default=0.2)
    ap.add_argument("--margin-db", type=float, default=12.0, help="speech must be this far above the room noise")
    ap.add_argument("--calibrate-s", type=float, default=1.0)
    ap.add_argument("--save", action="store_true", help="save the session (incl. transcripts) to data/runs/")
    args = ap.parse_args()

    print("Loading models ...", flush=True)
    torch.set_num_threads(4)
    stages = Stages(args.head, StageOptions(asr_device="mps", mask_cache=True))
    dev_wav = Path("data/audio/dev/dia0_utt0.wav")
    if dev_wav.exists():  # warm-up so the first real utterance isn't slow
        x, _ = sf.read(dev_wav, dtype="float32")
        for _ in range(2):
            stages.audio_branch(x)
            stages.text_branch(x)

    ring = RingBuffer(30 * SR)
    q = BoundedQueue(8)
    clock = FeedClock()
    stop = threading.Event()
    seg = EnergySegmenter(threshold_db=-60.0, hangover_s=args.hangover, adaptive=True, margin_db=args.margin_db)
    pool = ThreadPoolExecutor(2)
    results, truth = [], []
    status = {"mic_overflows": 0}

    # ---- audio source ----------------------------------------------------------
    mic = None
    if args.source == "mic":
        import sounddevice as sd
        print(f"Calibrating: please stay quiet for {args.calibrate_s:.0f} s ...", flush=True)
        quiet = sd.rec(int(args.calibrate_s * SR), samplerate=SR, channels=1, dtype="float32", device=args.device)
        sd.wait()
        floor = seg.calibrate(quiet[:, 0])
        print(f"Room noise {floor:.0f} dBFS -> speech threshold {seg.current_threshold_db():.0f} dBFS", flush=True)
        pos = [0]

        def callback(indata, frames, t, st):  # runs on the audio thread: keep it short
            if st.input_overflow:
                status["mic_overflows"] += 1
            ring.write(indata[:, 0])
            pos[0] += frames
            clock.add(pos[0])

        mic = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=CHUNK, callback=callback,
                             device=args.device)
        mic.start()
        print("Listening. Speak, pause briefly, and watch the output. Ctrl+C to stop.\n", flush=True)
    else:
        if args.source == "file":
            audio, sr = sf.read(args.file, dtype="float32")
            assert sr == SR and audio.ndim == 1, "file must be 16 kHz mono"
        else:
            import pandas as pd
            idx = pd.read_csv("data/index/test.csv", keep_default_na=False).set_index("utt_key")
            keys = Path("bench/subsets/test_fixed_500.txt").read_text().split()[: args.n]
            gap = np.zeros(int(0.8 * SR), np.float32) + np.random.default_rng(0).standard_normal(int(0.8 * SR)).astype(
                np.float32) * 10 ** (-75 / 20)
            parts, p = [gap], len(gap)
            for k in keys:
                x, _ = sf.read(idx.loc[k, "wav"], dtype="float32")
                truth.append((p, p + len(x), idx.loc[k, "emotion"]))
                parts += [x, gap]
                p += len(x) + len(gap)
            audio = np.concatenate(parts)
        seg.calibrate(audio[: int(0.5 * SR)])
        print(f"Replaying {len(audio) / SR:.1f} s of audio in real time. Ctrl+C to stop.\n", flush=True)

        def player():
            t0 = time.perf_counter()
            for k in range(0, len(audio), CHUNK):
                if stop.is_set():
                    break
                delay = t0 + (k + CHUNK) / SR - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                ring.write(audio[k:k + CHUNK])
                clock.add(min(k + CHUNK, len(audio)))
            time.sleep(args.hangover + 0.1)
            ring.write(np.zeros(int((args.hangover + 0.1) * SR), np.float32) + 1e-6)  # let the last segment close
            stop.set()

        threading.Thread(target=player, daemon=True).start()

    # ---- segmenter and worker threads ---------------------------------------------
    def segmenter():
        while True:
            x = ring.read(CHUNK, timeout=0.5)
            if x is None:
                if stop.is_set():
                    for s in seg.flush():
                        q.put((s, time.perf_counter()))
                    q.close()
                    return
                continue
            for s in seg.push(x):
                q.put((s, time.perf_counter()))

    def worker():
        while True:
            item = q.get()
            if item is CLOSED:
                return
            s, t_closed = item
            t0 = time.perf_counter()
            fa, fb = pool.submit(stages.audio_branch, s.audio), pool.submit(stages.text_branch, s.audio)
            a_emb, _ = fa.result()
            transcript, t_emb, tt = fb.result()
            label, probs, _ = stages.fuse(t_emb, a_emb)
            response, _ = stages.respond(label, transcript)
            t_done = time.perf_counter()
            e2e = t_done - clock.time_of(s.end)
            order = np.argsort(probs)[::-1]
            r = {"start_s": s.start / SR, "duration_s": len(s.audio) / SR, "transcript": transcript,
                 "emotion": EMOTIONS[label], "prob": float(probs[label]), "second": EMOTIONS[order[1]],
                 "second_prob": float(probs[order[1]]), "response": response, "e2e_ms": e2e * 1e3,
                 "inference_ms": (t_done - t0) * 1e3, "endpoint_ms": (t_closed - clock.time_of(s.end)) * 1e3}
            mark = ""
            if truth:
                ov = [(min(s.end, e) - max(s.start, b), emo) for b, e, emo in truth]
                best = max(ov)
                if best[0] > 0:
                    r["true"] = best[1]
                    mark = f"   true: {best[1]} {'(match)' if best[1] == r['emotion'] else '(miss)'}"
            results.append(r)
            print(f"[{r['start_s']:6.1f}s, {r['duration_s']:.1f}s] \"{transcript}\"\n"
                  f"    -> {r['emotion'].upper()} {r['prob']:.0%} (next: {r['second']} {r['second_prob']:.0%}){mark}\n"
                  f"    {response}\n"
                  f"    latency {r['e2e_ms']:.0f} ms after you stopped (waiting for silence {r['endpoint_ms']:.0f} ms, "
                  f"models {r['inference_ms']:.0f} ms)\n", flush=True)

    threads = [threading.Thread(target=segmenter, daemon=True), threading.Thread(target=worker, daemon=True)]
    for t in threads:
        t.start()

    def finish(*_):
        stop.set()

    signal.signal(signal.SIGINT, finish)
    while not stop.is_set():
        time.sleep(0.1)
    if mic is not None:
        mic.stop()
        mic.close()
    for t in threads:
        t.join(timeout=10)
    pool.shutdown()

    if results:
        e2e = np.array([r["e2e_ms"] for r in results])
        inf = np.array([r["inference_ms"] for r in results])
        print(f"--- {len(results)} utterances | latency p50 {np.median(e2e):.0f} ms, p95 {np.percentile(e2e, 95):.0f} ms "
              f"| models p50 {np.median(inf):.0f} ms")
        if truth:
            scored = [r for r in results if "true" in r]
            print(f"--- emotion matched the MELD label in {sum(r['true'] == r['emotion'] for r in scored)}"
                  f"/{len(scored)} utterances")
    if status["mic_overflows"]:
        print(f"--- warning: the microphone buffer overflowed {status['mic_overflows']} times (audio was lost)")
    if args.save and results:
        out = Path("data/runs") / f"demo_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "results": results}, indent=1))
        print(f"--- saved to {out} (git-ignored)")


if __name__ == "__main__":
    sys.exit(main())
