"""Threaded live pipeline.

  producer thread ── writes 20 ms chunks at real-time pace ──> RingBuffer
  segmenter thread ── reads chunks, energy VAD ──> BoundedQueue(segments)
  worker thread ── per segment:
        sequential:  audio branch, then text branch
        concurrent:  audio branch ‖ text branch (2-thread pool), join
      ── fusion ── response ──> results

Timestamps (time.perf_counter, one clock for all threads) recorded per segment:
  t_fed      when the chunk holding the segment's last speech sample was written
  t_closed   when the segmenter decided the utterance had ended (after hangover)
  t_start    when the worker dequeued it
  t_done     when the response text was ready
so that
  endpoint delay = t_closed - t_fed      (waiting for silence: the hangover)
  queue wait     = t_start - t_closed    (worker busy with an earlier segment)
  inference      = t_done - t_start
  end-to-end     = t_done - t_fed        (utterance end to output, the headline latency)
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from pipeline.fusion.model import EMOTIONS
from pipeline.live.bounded_queue import CLOSED, BoundedQueue
from pipeline.live.ring_buffer import RingBuffer
from pipeline.live.segmenter import EnergySegmenter, Segment
from pipeline.live.stream import SR, Stream


class OracleSegmenter:
    """Emits the ground-truth utterance boundaries, as soon as the stream passes each end.
    Used to separate inference latency from segmentation errors and hangover delay."""

    def __init__(self, stream: Stream):
        self.audio, self.bounds = stream.audio, list(zip(stream.gt.start, stream.gt.end))
        self.pos, self.next = 0, 0

    def push(self, chunk):
        self.pos += len(chunk)
        out = []
        while self.next < len(self.bounds) and self.bounds[self.next][1] <= self.pos:
            s, e = self.bounds[self.next]
            out.append(Segment(start=s, end=e, audio=self.audio[s:e].copy(), closed_at=self.pos))
            self.next += 1
        return out

    def flush(self):
        return []


@dataclass
class RunConfig:
    mode: str = "sequential"          # or "concurrent"
    segmenter: str = "energy"         # or "oracle"
    chunk_ms: int = 20
    pace: float = 1.0                 # 1.0 = real time; 0 = as fast as possible (no latency meaning)
    queue_size: int = 8
    ring_seconds: float = 30.0
    threshold_db: float = -55.0
    hangover_s: float = 0.3


@dataclass
class RunResult:
    segments: list = field(default_factory=list)
    chunk_times: np.ndarray | None = None
    chunk_lateness: np.ndarray | None = None
    ring_overrun_samples: int = 0
    queue_max_depth: int = 0
    wall_seconds: float = 0.0


def run(stream: Stream, stages, cfg: RunConfig) -> RunResult:
    chunk = SR * cfg.chunk_ms // 1000
    n_chunks = -(-len(stream.audio) // chunk)
    chunk_times = np.zeros(n_chunks)
    lateness = np.zeros(n_chunks)
    ring = RingBuffer(int(cfg.ring_seconds * SR))
    q = BoundedQueue(cfg.queue_size)
    seg = (OracleSegmenter(stream) if cfg.segmenter == "oracle"
           else EnergySegmenter(threshold_db=cfg.threshold_db, hangover_s=cfg.hangover_s))
    results, errors = [], []
    pool = ThreadPoolExecutor(2, thread_name_prefix="branch") if cfg.mode == "concurrent" else None

    def producer():
        t0 = time.perf_counter()
        for k in range(n_chunks):
            if cfg.pace > 0:
                target = t0 + (k + 1) * cfg.chunk_ms / 1000 / cfg.pace  # chunk k is "captured" at its end
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                lateness[k] = max(0.0, time.perf_counter() - target)
            ring.write(stream.audio[k * chunk:(k + 1) * chunk])
            chunk_times[k] = time.perf_counter()
        ring.close()

    def segmenter():
        while True:
            x = ring.read(chunk, timeout=5.0)
            if x is None:
                tail = ring.read_remaining()
                segs = seg.push(tail) + seg.flush() if len(tail) else seg.flush()
                for s in segs:
                    q.put((s, time.perf_counter()))
                q.close()
                return
            for s in seg.push(x):
                q.put((s, time.perf_counter()))

    def worker():
        try:
            while True:
                item = q.get()
                if item is CLOSED:
                    return
                s, t_closed = item
                t_start = time.perf_counter()
                if pool is None:
                    ta0 = time.perf_counter()
                    a_emb, a_t = stages.audio_branch(s.audio)
                    ta1 = time.perf_counter()
                    transcript, t_emb, t_t = stages.text_branch(s.audio)
                    tb1 = time.perf_counter()
                    branch_a, branch_b = ta1 - ta0, tb1 - ta1
                else:
                    def timed(fn):
                        t = time.perf_counter()
                        r = fn(s.audio)
                        return r, time.perf_counter() - t
                    fa, fb = pool.submit(timed, stages.audio_branch), pool.submit(timed, stages.text_branch)
                    (a_emb, a_t), branch_a = fa.result()
                    (transcript, t_emb, t_t), branch_b = fb.result()
                t_branches = time.perf_counter()
                label, probs, f_t = stages.fuse(t_emb, a_emb)
                response, r_t = stages.respond(label, transcript)
                t_done = time.perf_counter()
                results.append({"start": s.start, "end": s.end, "closed_at": s.closed_at, "forced": s.forced,
                                "t_closed": t_closed, "t_start": t_start, "t_done": t_done,
                                "branches_wall": t_branches - t_start, "branch_audio": branch_a,
                                "branch_text": branch_b, **a_t, **t_t, **f_t, **r_t,
                                "pred": label, "pred_emotion": EMOTIONS[label], "transcript": transcript,
                                "response_text": response})
        except Exception as e:  # surface worker failures instead of hanging the run
            errors.append(e)
            q.close()

    threads = [threading.Thread(target=f, name=f.__name__) for f in (producer, segmenter, worker)]
    t_run = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_run
    if pool:
        pool.shutdown()
    if errors:
        raise errors[0]

    for r in results:
        last_chunk = (r["end"] - 1) // chunk
        r["t_fed"] = float(chunk_times[last_chunk])
        r["endpoint_delay"] = r["t_closed"] - r["t_fed"]
        r["queue_wait"] = r["t_start"] - r["t_closed"]
        r["inference"] = r["t_done"] - r["t_start"]
        r["e2e"] = r["t_done"] - r["t_fed"]
        r["duration_s"] = (r["end"] - r["start"]) / SR
        r["rtf"] = r["inference"] / r["duration_s"]
    return RunResult(results, chunk_times, lateness, ring.overrun_samples, q.max_depth, wall)
