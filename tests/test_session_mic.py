"""Silent-microphone detection, with a fake `sounddevice` module (no real microphone, no models)."""

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pipeline.live.bounded_queue import BoundedQueue
from pipeline.live.ring_buffer import RingBuffer
from pipeline.live.segmenter import EnergySegmenter
from pipeline.live.session import SR, FeedClock, LiveSession, SessionConfig


def fake_sounddevice(level_db):
    """level_db=None: the device returns exact zeros (what macOS does without permission)."""
    rng = np.random.default_rng(0)

    def make(n):
        if level_db is None:
            return np.zeros((n, 1), np.float32)
        return (rng.standard_normal((n, 1)) * 10 ** (level_db / 20)).astype(np.float32)

    class InputStream:
        def __init__(self, callback, blocksize, **_):
            self.cb, self.bs, self.run = callback, blocksize, False

        def start(self):
            self.run = True

            def loop():
                st = types.SimpleNamespace(input_overflow=False)
                while self.run:
                    self.cb(make(self.bs), self.bs, None, st)
                    time.sleep(self.bs / SR)
            threading.Thread(target=loop, daemon=True).start()

        def stop(self):
            self.run = False

        def close(self):
            pass

    return types.SimpleNamespace(rec=lambda n, **_: make(n), wait=lambda: None, InputStream=InputStream)


def run_session(level_db, seconds):
    sys.modules["sounddevice"] = fake_sounddevice(level_db)
    try:
        s = LiveSession(SessionConfig(source="mic", calibrate_s=0.5))
        s.ring, s.q, s.clock = RingBuffer(30 * SR), BoundedQueue(8), FeedClock()
        s.seg = EnergySegmenter(threshold_db=-60.0, adaptive=True)
        s.pool = ThreadPoolExecutor(1)
        events = []
        s.on(events.append)
        s.start()
        time.sleep(seconds)
        s.stop()
        s.wait()
        return [e for e in events if e["type"] == "status"]
    finally:
        del sys.modules["sounddevice"]


def test_all_zero_microphone_is_reported():
    statuses = run_session(None, 3.5)
    states = [e["state"] for e in statuses]
    assert "error" in states and "listening" not in states
    assert "permission" in next(e for e in statuses if e["state"] == "error")["message"]


def test_real_microphone_noise_is_not_reported():
    states = [e["state"] for e in run_session(-55.0, 3.5)]
    assert "listening" in states and "error" not in states
