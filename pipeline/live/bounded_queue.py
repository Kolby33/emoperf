"""Bounded blocking FIFO queue between pipeline threads.

Python's queue.Queue would work; this is written out (deque + two condition
variables, the textbook monitor) so that its behavior is explicit and testable:

- put() blocks while full: backpressure, so a slow consumer slows the producer
  instead of memory growing without bound.
- get() blocks while empty.
- close() wakes everyone: get() drains what is left, then returns the CLOSED sentinel.

It also records the maximum depth it reached, which shows whether a stage fell behind.
"""

import threading
from collections import deque

CLOSED = object()


class QueueClosed(Exception):
    pass


class BoundedQueue:
    def __init__(self, maxsize: int):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._items = deque()
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)
        self._closed = False
        self.max_depth = 0

    def __len__(self):
        with self._lock:
            return len(self._items)

    def put(self, item, timeout: float | None = None) -> bool:
        """Returns False on timeout. Raises QueueClosed if the queue was closed."""
        with self._not_full:
            ok = self._not_full.wait_for(lambda: len(self._items) < self.maxsize or self._closed, timeout)
            if self._closed:
                raise QueueClosed
            if not ok:
                return False
            self._items.append(item)
            self.max_depth = max(self.max_depth, len(self._items))
            self._not_empty.notify()
            return True

    def get(self, timeout: float | None = None):
        """Returns the oldest item, None on timeout, or CLOSED once closed and drained."""
        with self._not_empty:
            ok = self._not_empty.wait_for(lambda: self._items or self._closed, timeout)
            if self._items:
                item = self._items.popleft()
                self._not_full.notify()
                return item
            return CLOSED if self._closed else None  # None = timed out

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_full.notify_all()
            self._not_empty.notify_all()
