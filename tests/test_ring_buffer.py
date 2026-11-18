import threading
import time

import numpy as np
import pytest

from pipeline.live.ring_buffer import RingBuffer


def arange(a, b):
    return np.arange(a, b, dtype=np.float32)


def test_empty_read_times_out():
    rb = RingBuffer(8)
    t0 = time.perf_counter()
    assert rb.read(1, timeout=0.05) is None
    assert time.perf_counter() - t0 >= 0.04
    assert rb.available() == 0


def test_fifo_order():
    rb = RingBuffer(8)
    rb.write(arange(0, 5))
    np.testing.assert_array_equal(rb.read(3), arange(0, 3))
    np.testing.assert_array_equal(rb.read(2), arange(3, 5))
    assert rb.available() == 0


def test_exactly_full_no_overrun():
    rb = RingBuffer(8)
    rb.write(arange(0, 8))
    assert rb.available() == 8 and rb.overrun_samples == 0
    np.testing.assert_array_equal(rb.read(8), arange(0, 8))


def test_wraparound_write_and_read():
    rb = RingBuffer(8)
    rb.write(arange(0, 6))
    rb.read(5)                       # read_pos=5, storage index 5
    rb.write(arange(6, 13))          # 7 samples: indices 6,7 then wraps to 0..4
    assert rb.available() == 8
    np.testing.assert_array_equal(rb.read(8), arange(5, 13))  # read also wraps
    assert rb.overrun_samples == 0


def test_overrun_drops_oldest():
    rb = RingBuffer(8)
    rb.write(arange(0, 6))
    rb.write(arange(6, 12))          # 12 written, only 8 fit: 0..3 lost
    assert rb.overrun_samples == 4
    assert rb.read_pos == 4
    np.testing.assert_array_equal(rb.read(8), arange(4, 12))


def test_write_larger_than_capacity():
    rb = RingBuffer(4)
    rb.write(arange(0, 10))
    assert rb.overrun_samples == 6
    np.testing.assert_array_equal(rb.read(4), arange(6, 10))


def test_read_more_than_capacity_rejected():
    with pytest.raises(ValueError):
        RingBuffer(4).read(5)


def test_close_unblocks_reader_and_drains_tail():
    rb = RingBuffer(8)
    rb.write(arange(0, 3))
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("r", rb.read(5)))
    t.start()
    time.sleep(0.05)
    rb.close()
    t.join(1)
    assert not t.is_alive() and result["r"] is None
    np.testing.assert_array_equal(rb.read_remaining(), arange(0, 3))
    with pytest.raises(RuntimeError):
        rb.write(arange(0, 1))


def _threaded(capacity, total, write_size, pace_s):
    rb = RingBuffer(capacity)
    got = []

    def consumer():
        while True:
            x = rb.read(160, timeout=1.0)
            if x is None:
                got.append(rb.read_remaining())
                return
            got.append(x)

    t = threading.Thread(target=consumer)
    t.start()
    for i in range(0, total, write_size):
        rb.write(arange(i, min(i + write_size, total)))
        if pace_s:
            time.sleep(pace_s)
    rb.close()
    t.join(5)
    return rb, np.concatenate(got)


def test_threaded_unpaced_producer_overruns_but_never_corrupts():
    # A real-time producer never blocks, so an unpaced one outruns the consumer.
    # Invariant: every sample is either delivered (in order, no duplicates) or counted as lost.
    total = 50_000
    rb, out = _threaded(capacity=1024, total=total, write_size=100, pace_s=0)
    assert len(out) + rb.overrun_samples == total
    assert np.all(np.diff(out) > 0)                # strictly increasing: no duplicates/reordering
    assert out[-1] == total - 1                    # newest data always survives


def test_threaded_paced_producer_loses_nothing():
    total = 16_000                                  # 1 s of audio in 20 ms chunks, at ~10x real time
    rb, out = _threaded(capacity=8_000, total=total, write_size=320, pace_s=0.002)
    assert rb.overrun_samples == 0
    np.testing.assert_array_equal(out, arange(0, total))
