"""Single-producer / single-consumer ring buffer for float32 audio samples.

Fixed capacity, preallocated once, no per-write allocation. Positions are absolute
sample counters (they never wrap); the storage index is position % capacity. That
makes "how many samples are readable" just write_pos - read_pos, and lets callers
refer to any sample by its absolute position in the stream.

Real-time rule: the producer (the audio source) must never block. If the consumer
falls behind and the buffer is full, the oldest unread samples are overwritten and
counted in `overrun_samples`, the same thing a sound card driver does.
"""

import threading

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._write_pos = 0  # absolute index of the next sample to write
        self._read_pos = 0   # absolute index of the next sample to read
        self._closed = False
        self.overrun_samples = 0
        self._cond = threading.Condition()

    def available(self) -> int:
        with self._cond:
            return self._write_pos - self._read_pos

    @property
    def read_pos(self) -> int:
        return self._read_pos

    def write(self, x: np.ndarray) -> None:
        """Append samples. Never blocks; overwrites the oldest unread data if full."""
        x = np.asarray(x, dtype=np.float32)
        n = len(x)
        if n == 0:
            return
        with self._cond:
            if self._closed:
                raise RuntimeError("write to closed RingBuffer")
            if n > self.capacity:  # only the newest `capacity` samples can survive
                skipped = n - self.capacity
                self._write_pos += skipped
                x = x[skipped:]
                n = self.capacity
            start = self._write_pos % self.capacity
            first = min(n, self.capacity - start)
            self._buf[start:start + first] = x[:first]
            self._buf[:n - first] = x[first:]  # wraparound part (empty slice if none)
            self._write_pos += n
            unread = self._write_pos - self._read_pos
            if unread > self.capacity:
                lost = unread - self.capacity
                self.overrun_samples += lost
                self._read_pos += lost
            self._cond.notify_all()

    def read(self, n: int, timeout: float | None = None) -> np.ndarray | None:
        """Block until n samples are readable, then return a copy of them and advance.

        Returns None on timeout, or when the buffer is closed with fewer than n left
        (call read_remaining() to drain the tail).
        """
        if n > self.capacity:
            raise ValueError("cannot read more than capacity at once")
        with self._cond:
            ok = self._cond.wait_for(lambda: self._write_pos - self._read_pos >= n or self._closed, timeout)
            if not ok or self._write_pos - self._read_pos < n:
                return None
            return self._take(n)

    def read_remaining(self) -> np.ndarray:
        with self._cond:
            return self._take(self._write_pos - self._read_pos)

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _take(self, n: int) -> np.ndarray:
        start = self._read_pos % self.capacity
        first = min(n, self.capacity - start)
        out = np.empty(n, dtype=np.float32)
        out[:first] = self._buf[start:start + first]
        out[first:] = self._buf[:n - first]
        self._read_pos += n
        return out
