"""A live emotion session: audio source -> ring buffer -> segmenter -> models -> events.

Used by demo.py (terminal output) and the local web page (pipeline/live/web_server.py).
Listeners receive plain dict events:

  {"type": "status", "state": "loading|calibrating|listening|replaying|stopped", "message": str}
  {"type": "calibrated", "floor_db": float, "threshold_db": float}
  {"type": "level", "db": float, "floor_db": float, "threshold_db": float, "speaking": bool}   (10 per second)
  {"type": "utterance", "start_s", "duration_s", "transcript", "emotion", "prob", "probs": {emotion: p},
   "response", "e2e_ms", "endpoint_ms", "inference_ms", "asr_ms", "audio_ms", "true"?: str}
  {"type": "summary", "n", "e2e_p50_ms", "e2e_p95_ms", "inference_p50_ms", "matched"?: int, "scored"?: int}

Pipeline = the M4 best configuration (Whisper on MPS in parallel with wav2vec2 on CPU,
mask cache), the mixed-text fusion head (E7), and the adaptive-noise-floor segmenter.
"""

import bisect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from pipeline.fusion.model import EMOTIONS
from pipeline.live.bounded_queue import CLOSED, BoundedQueue
from pipeline.live.ring_buffer import RingBuffer
from pipeline.live.segmenter import EnergySegmenter
from pipeline.live.stages import StageOptions, Stages

SR = 16000
CHUNK = 320  # 20 ms

SILENT_MIC_MESSAGE = (
    "The microphone is sending pure silence (every sample is exactly 0). On macOS this means the app this "
    "terminal runs in has no microphone permission: macOS returns zeros instead of an error. Fix: run the demo "
    "from the Terminal app (it asks for permission the first time), or turn the app on in System Settings > "
    "Privacy & Security > Microphone, then restart the demo.")


@dataclass
class SessionConfig:
    source: str = "mic"            # mic | meld | file
    file: Path | None = None
    n: int = 8                     # MELD clips for source=meld
    device: int | None = None      # input device index
    head: Path = Path("models/fusion_text_audio_trainboth_seed0.pt")
    hangover: float = 0.2
    margin_db: float = 12.0
    calibrate_s: float = 1.0


class FeedClock:
    """When each block of samples arrived (for 'time since speech ended')."""

    def __init__(self):
        self.ends, self.times = [], []

    def add(self, end_pos):
        self.ends.append(end_pos)
        self.times.append(time.perf_counter())

    def time_of(self, pos):
        i = bisect.bisect_left(self.ends, pos)
        return self.times[min(i, len(self.times) - 1)]


def block_db(x):
    return float(10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-12)) if len(x) else -120.0


class LiveSession:
    def __init__(self, cfg: SessionConfig):
        self.cfg = cfg
        self.listeners = []
        self.results = []
        self.truth = []                     # (start, end, emotion) for source=meld
        self.stop_event = threading.Event()
        self.mic_overflows = 0
        self._level = -120.0
        self._threads = []
        self._mic = None

    # ---- events -----------------------------------------------------------------
    def on(self, fn):
        self.listeners.append(fn)

    def emit(self, ev):
        for fn in list(self.listeners):
            try:
                fn(ev)
            except Exception:  # noqa: BLE001 - a broken listener must not stop the pipeline
                pass

    def status(self, state, message=""):
        self.emit({"type": "status", "state": state, "message": message})

    # ---- setup ------------------------------------------------------------------
    def load(self):
        self.status("loading", "Loading models ...")
        torch.set_num_threads(4)
        self.stages = Stages(self.cfg.head, StageOptions(asr_device="mps", mask_cache=True))
        warm = Path("data/audio/dev/dia0_utt0.wav")
        if warm.exists():  # so the first real utterance isn't slow
            x, _ = sf.read(warm, dtype="float32")
            for _ in range(2):
                self.stages.audio_branch(x)
                self.stages.text_branch(x)
        self.ring = RingBuffer(30 * SR)
        self.q = BoundedQueue(8)
        self.clock = FeedClock()
        self.seg = EnergySegmenter(threshold_db=-60.0, hangover_s=self.cfg.hangover, adaptive=True,
                                   margin_db=self.cfg.margin_db)
        self.pool = ThreadPoolExecutor(2)

    def start(self):
        c = self.cfg
        if c.source == "mic":
            import sounddevice as sd
            self.status("calibrating", f"Stay quiet for {c.calibrate_s:.0f} s while the room noise is measured ...")
            quiet = sd.rec(int(c.calibrate_s * SR), samplerate=SR, channels=1, dtype="float32", device=c.device)
            sd.wait()
            if not np.any(quiet):
                self.status("error", SILENT_MIC_MESSAGE)
            self.seg.calibrate(quiet[:, 0])
            self._emit_calibrated()
            pos = [0]

            def callback(indata, frames, t, st):  # audio thread: keep it short
                if st.input_overflow:
                    self.mic_overflows += 1
                x = indata[:, 0]
                self.ring.write(x)
                pos[0] += frames
                self.clock.add(pos[0])
                self._level = max(self._level, block_db(x))

            self._mic = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=CHUNK,
                                       callback=callback, device=c.device)
            self._mic.start()
            if np.any(quiet):
                self.status("listening", "Listening. Speak, then pause briefly.")
        else:
            audio = self._load_replay_audio()
            self.seg.calibrate(audio[: int(0.5 * SR)])
            self._emit_calibrated()
            self.status("replaying", f"Replaying {len(audio) / SR:.0f} s of {'MELD clips' if c.source == 'meld' else c.file}")
            self._spawn(self._player, audio)
        self._spawn(self._segmenter)
        self._spawn(self._worker)
        self._spawn(self._level_ticker)

    def _emit_calibrated(self):
        self.emit({"type": "calibrated", "floor_db": self.seg.noise_floor_db,
                   "threshold_db": self.seg.current_threshold_db()})

    def _load_replay_audio(self):
        c = self.cfg
        if c.source == "file":
            audio, sr = sf.read(c.file, dtype="float32")
            assert sr == SR and audio.ndim == 1, "file must be 16 kHz mono"
            return audio
        import pandas as pd
        idx = pd.read_csv("data/index/test.csv", keep_default_na=False).set_index("utt_key")
        keys = Path("bench/subsets/test_fixed_500.txt").read_text().split()[: c.n]
        rng = np.random.default_rng(0)
        gap = lambda: (rng.standard_normal(int(0.8 * SR)) * 10 ** (-75 / 20)).astype(np.float32)  # noqa: E731
        parts, p = [gap()], int(0.8 * SR)
        for k in keys:
            x, _ = sf.read(idx.loc[k, "wav"], dtype="float32")
            self.truth.append((p, p + len(x), idx.loc[k, "emotion"]))
            parts += [x, gap()]
            p += len(x) + int(0.8 * SR)
        return np.concatenate(parts)

    def _spawn(self, fn, *a):
        t = threading.Thread(target=fn, args=a, daemon=True)
        t.start()
        self._threads.append(t)

    # ---- threads ----------------------------------------------------------------
    def _player(self, audio):
        t0 = time.perf_counter()
        for k in range(0, len(audio), CHUNK):
            if self.stop_event.is_set():
                break
            delay = t0 + (k + CHUNK) / SR - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            x = audio[k:k + CHUNK]
            self.ring.write(x)
            self.clock.add(min(k + CHUNK, len(audio)))
            self._level = max(self._level, block_db(x))
        if not self.stop_event.is_set():  # trailing quiet so the last utterance closes, then finish
            tail = (np.random.default_rng(1).standard_normal(int((self.cfg.hangover + 0.2) * SR)) * 1e-4).astype(np.float32)
            self.ring.write(tail)
            time.sleep(self.cfg.hangover + 1.0)
            self.stop()

    def _level_ticker(self):
        silent_ticks, warned = 0, False
        while not self.stop_event.is_set():
            time.sleep(0.1)
            lvl, self._level = self._level, -120.0
            if self.cfg.source == "mic":  # all-zero input for 3 s: permission problem, say so once
                silent_ticks = silent_ticks + 1 if lvl <= -119.9 else 0
                if silent_ticks >= 30 and not warned:
                    self.status("error", SILENT_MIC_MESSAGE)
                    warned = True
            self.emit({"type": "level", "db": lvl, "floor_db": self.seg.noise_floor_db,
                       "threshold_db": self.seg.current_threshold_db(), "speaking": self.seg._active})

    def _segmenter(self):
        while True:
            x = self.ring.read(CHUNK, timeout=0.5)
            if x is None:
                if self.stop_event.is_set():
                    for s in self.seg.flush():
                        self.q.put((s, time.perf_counter()))
                    self.q.close()
                    return
                continue
            for s in self.seg.push(x):
                self.q.put((s, time.perf_counter()))

    def _worker(self):
        while True:
            item = self.q.get()
            if item is CLOSED:
                return
            s, t_closed = item
            t0 = time.perf_counter()

            def timed(fn):
                a = time.perf_counter()
                r = fn(s.audio)
                return r, time.perf_counter() - a

            fa, fb = self.pool.submit(timed, self.stages.audio_branch), self.pool.submit(timed, self.stages.text_branch)
            (a_emb, _), t_audio = fa.result()
            (transcript, t_emb, _), t_text = fb.result()
            label, probs, _ = self.stages.fuse(t_emb, a_emb)
            response, _ = self.stages.respond(label, transcript)
            t_done = time.perf_counter()
            t_end = self.clock.time_of(s.end)
            r = {"type": "utterance", "start_s": s.start / SR, "duration_s": len(s.audio) / SR,
                 "transcript": transcript, "emotion": EMOTIONS[label], "prob": float(probs[label]),
                 "probs": {e: float(p) for e, p in zip(EMOTIONS, probs)}, "response": response,
                 "e2e_ms": (t_done - t_end) * 1e3, "endpoint_ms": (t_closed - t_end) * 1e3,
                 "inference_ms": (t_done - t0) * 1e3, "asr_ms": t_text * 1e3, "audio_ms": t_audio * 1e3}
            if self.truth:
                best = max((min(s.end, e) - max(s.start, b), emo) for b, e, emo in self.truth)
                if best[0] > 0:
                    r["true"] = best[1]
            self.results.append(r)
            self.emit(r)

    # ---- teardown ---------------------------------------------------------------
    def stop(self):
        self.stop_event.set()

    def wait(self):
        while not self.stop_event.is_set():
            time.sleep(0.1)
        if self._mic is not None:
            self._mic.stop()
            self._mic.close()
        for t in self._threads:
            t.join(timeout=10)
        self.pool.shutdown()
        self.emit(self.summary())
        self.status("stopped", "Stopped.")

    def summary(self):
        ev = {"type": "summary", "n": len(self.results), "mic_overflows": self.mic_overflows}
        if self.results:
            e2e = np.array([r["e2e_ms"] for r in self.results])
            ev.update(e2e_p50_ms=float(np.median(e2e)), e2e_p95_ms=float(np.percentile(e2e, 95)),
                      inference_p50_ms=float(np.median([r["inference_ms"] for r in self.results])))
            scored = [r for r in self.results if "true" in r]
            if scored:
                ev.update(scored=len(scored), matched=sum(r["true"] == r["emotion"] for r in scored))
        return ev
