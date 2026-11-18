import threading
import time

import pytest

from pipeline.live.bounded_queue import CLOSED, BoundedQueue, QueueClosed


def test_empty_get_times_out():
    q = BoundedQueue(2)
    assert q.get(timeout=0.05) is None


def test_full_put_times_out_and_keeps_items():
    q = BoundedQueue(2)
    assert q.put(1) and q.put(2)
    assert q.put(3, timeout=0.05) is False
    assert len(q) == 2 and q.max_depth == 2


def test_fifo_and_wraparound_of_deque():
    q = BoundedQueue(3)
    out = []
    for i in range(10):          # repeatedly fill/drain past capacity
        q.put(i)
        if len(q) == 3:
            out.append(q.get())
    while len(q):
        out.append(q.get())
    assert out == list(range(10))


def test_full_put_unblocks_when_consumer_gets():
    q = BoundedQueue(1)
    q.put("a")
    done = threading.Event()

    def producer():
        q.put("b")
        done.set()

    t = threading.Thread(target=producer)
    t.start()
    time.sleep(0.05)
    assert not done.is_set()     # blocked: backpressure
    assert q.get() == "a"
    t.join(1)
    assert done.is_set() and q.get() == "b"


def test_close_drains_then_returns_sentinel():
    q = BoundedQueue(4)
    q.put(1)
    q.put(2)
    q.close()
    assert q.get() == 1 and q.get() == 2
    assert q.get() is CLOSED
    with pytest.raises(QueueClosed):
        q.put(3)


def test_close_wakes_blocked_getter_and_putter():
    q = BoundedQueue(1)
    res = {}
    g = threading.Thread(target=lambda: res.setdefault("get", q.get()))
    g.start()
    time.sleep(0.02)
    q.close()
    g.join(1)
    assert res["get"] is CLOSED

    q2 = BoundedQueue(1)
    q2.put(0)

    def putter():
        try:
            q2.put(1)
        except QueueClosed:
            res["put"] = "closed"

    p = threading.Thread(target=putter)
    p.start()
    time.sleep(0.02)
    q2.close()
    p.join(1)
    assert res["put"] == "closed"


def test_many_producers_one_consumer_no_loss():
    q = BoundedQueue(8)
    n_prod, per = 4, 500

    def producer(k):
        for i in range(per):
            q.put((k, i))

    ts = [threading.Thread(target=producer, args=(k,)) for k in range(n_prod)]
    for t in ts:
        t.start()
    got = [q.get(timeout=2) for _ in range(n_prod * per)]
    for t in ts:
        t.join()
    assert sorted(got) == sorted((k, i) for k in range(n_prod) for i in range(per))
    for k in range(n_prod):      # per-producer order preserved
        assert [i for kk, i in got if kk == k] == list(range(per))
    assert q.max_depth <= 8
