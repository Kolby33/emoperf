"""Energy-based utterance segmentation (a minimal voice-activity detector).

Per 20 ms frame: energy in dBFS = 10*log10(mean(x^2)). A frame is "active" if it is
above `threshold_db`. An utterance starts at the first active frame and ends once
`hangover_s` of consecutive inactive frames have passed. That waiting time is the
endpointing delay: the price of not cutting someone off at a short pause.

Segments shorter than `min_speech_s` are dropped as clicks or noise. Very long
segments are force-closed at `max_segment_s` so latency stays bounded.

The segmenter is fed arbitrary-sized chunks and keeps absolute sample positions,
so every segment can be matched back to the ground-truth utterances afterwards.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    start: int          # absolute sample index (inclusive)
    end: int            # absolute sample index (exclusive), last active frame end
    audio: np.ndarray   # samples [start, end)
    closed_at: int      # absolute sample index at which the segmenter decided the end
    forced: bool = False


class EnergySegmenter:
    def __init__(self, sr=16000, frame_s=0.02, threshold_db=-55.0, hangover_s=0.3,
                 min_speech_s=0.2, max_segment_s=20.0, adaptive=False, margin_db=12.0,
                 floor_window_s=3.0, floor_percentile=10.0, max_floor_db=-30.0):
        """adaptive=False: fixed threshold_db (used by every benchmark).
        adaptive=True (live microphone): estimate the room's noise floor by minimum statistics,
        i.e. the floor_percentile-th percentile of the last floor_window_s seconds of frame
        energies (all frames, speech included: speech always has pauses, so a low percentile
        stays at the noise level, while a lasting change in room noise is followed within about
        one window). A frame is active if it is margin_db above the floor. threshold_db is the
        lower bound of the threshold; the floor is capped at max_floor_db. calibrate() sets the
        starting floor from a moment of silence.

        (A first version learned the floor only from inactive frames. A unit test showed the
        flaw: if the room gets louder by more than margin_db, every frame looks like speech and
        the floor can never update.)"""
        self.sr = sr
        self.frame = int(sr * frame_s)
        self.threshold_db = threshold_db
        self.adaptive = adaptive
        self.margin_db = margin_db
        self.max_floor_db = max_floor_db
        self.floor_percentile = floor_percentile
        self._recent_db = deque(maxlen=max(1, round(floor_window_s / frame_s)))
        self.noise_floor_db = threshold_db - margin_db
        self.hangover_frames = max(1, round(hangover_s / frame_s))
        self.min_frames = max(1, round(min_speech_s / frame_s))
        self.max_frames = round(max_segment_s / frame_s)
        self._pending = np.zeros(0, dtype=np.float32)  # samples not yet forming a full frame
        self._pos = 0              # absolute index of _pending[0]
        self._active = False
        self._seg_frames: list[np.ndarray] = []
        self._seg_start = 0
        self._silent_run = 0
        self._last_active_end = 0

    def push(self, chunk: np.ndarray) -> list[Segment]:
        """Feed samples; return any segments that closed during this chunk."""
        out = []
        buf = np.concatenate([self._pending, chunk.astype(np.float32, copy=False)])
        n_frames = len(buf) // self.frame
        if n_frames:
            frames = buf[: n_frames * self.frame].reshape(n_frames, self.frame)
            db = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
            for i in range(n_frames):
                f_start = self._pos + i * self.frame
                if self.adaptive:
                    self._recent_db.append(db[i])
                    if len(self._recent_db) == self._recent_db.maxlen:
                        self.noise_floor_db = min(self.max_floor_db,
                                                  float(np.percentile(self._recent_db, self.floor_percentile)))
                active = db[i] > self.current_threshold_db()
                seg = self._step(frames[i], active, f_start)
                if seg is not None:
                    out.append(seg)
        self._pending = buf[n_frames * self.frame:].copy()
        self._pos += n_frames * self.frame
        return out

    def current_threshold_db(self) -> float:
        if not self.adaptive:
            return self.threshold_db
        return max(self.threshold_db, self.noise_floor_db + self.margin_db)

    def calibrate(self, silence: np.ndarray) -> float:
        """Set the noise floor from a recording of the quiet room (median frame energy)."""
        n = len(silence) // self.frame
        if n:
            f = silence[: n * self.frame].reshape(n, self.frame).astype(np.float64)
            db = 10 * np.log10(np.mean(f ** 2, 1) + 1e-12)
            self.noise_floor_db = min(self.max_floor_db, float(np.median(db)))
            self._recent_db.extend(db.tolist())
        return self.noise_floor_db

    def flush(self) -> list[Segment]:
        """End of stream: close an open segment, if any."""
        if self._active and self._seg_frames:
            return [s for s in [self._close(self._pos, forced=False)] if s is not None]
        return []

    def _step(self, frame, active, f_start):
        if not self._active:
            if active:
                self._active = True
                self._seg_start = f_start
                self._seg_frames = [frame]
                self._silent_run = 0
                self._last_active_end = f_start + self.frame
            return None
        self._seg_frames.append(frame)
        if active:
            self._silent_run = 0
            self._last_active_end = f_start + self.frame
        else:
            self._silent_run += 1
        if self._silent_run >= self.hangover_frames:
            return self._close(f_start + self.frame, forced=False)
        if len(self._seg_frames) >= self.max_frames:
            self._last_active_end = f_start + self.frame
            return self._close(f_start + self.frame, forced=True)
        return None

    def _close(self, closed_at, forced):
        audio = np.concatenate(self._seg_frames)[: self._last_active_end - self._seg_start]
        n_active_frames = len(audio) // self.frame
        start = self._seg_start
        self._active, self._seg_frames, self._silent_run = False, [], 0
        if n_active_frames < self.min_frames:
            return None
        return Segment(start=start, end=start + len(audio), audio=audio, closed_at=closed_at, forced=forced)
