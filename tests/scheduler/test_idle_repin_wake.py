"""空闲收包在换血窗口到点时醒来，多卡用同一条唤醒包对齐。"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from freetoken.scheduler.io import _REPIN_WAKE, SchedulerIOMixin
from freetoken.utils.mp import ZmqPullQueue, ZmqPushQueue


def _scheduler_with_pending_hotness():
    from freetoken.moe.hotness import ExpertHotness
    from freetoken.moe.hot_pin import HotExpertRepinManager
    from freetoken.scheduler.scheduler import Scheduler
    from tests.moe.test_hot_pin import (
        _init_pins,
        _load_pinned_slot_contents,
        _make_pinned_cache,
    )

    cache = _make_pinned_cache(num_layers=1, num_experts=8, pins=(1,))
    _init_pins(cache, pins=(1,))
    _load_pinned_slot_contents(cache)
    cache.hotness = ExpertHotness(
        1, 8, torch.device("cpu"), None, flush_interval_s=60, window_interval_s=10,
    )
    cache.repin_manager = HotExpertRepinManager(
        cache, cache.hotness, interval_s=10, gain=1.5, max_swaps=1,
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(moe_offload_cache=cache)
    scheduler.cache_manager = SimpleNamespace(check_integrity=lambda: None)
    scheduler.prefill_manager = SimpleNamespace(runnable=False)
    scheduler.decode_manager = SimpleNamespace(runnable=False)
    cache.hotness.record(0, torch.full((20, 1), 5, dtype=torch.int32))
    return scheduler


@pytest.mark.parametrize("num_ranks", [1, 2])
def test_idle_receive_flushes_short_request_hotness_before_next_request(monkeypatch, num_ranks):
    """A short request's routing counts must produce a new pin set while receive blocks."""
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    ranks = [_scheduler_with_pending_hotness() for _ in range(num_ranks)]
    caches = [rank.engine.moe_offload_cache for rank in ranks]
    clock[0] += 1
    for rank, cache in zip(ranks, caches):
        rank._process_last_data(None)
        assert not cache.hotness.has_window
        assert int(cache.hotness.counts.sum()) == 20

    barrier = threading.Barrier(num_ranks)
    messages: queue.Queue[bytes] = queue.Queue()
    pins_before_request = []

    class _Tokenizer:
        calls = 0

        def get_raw(self, timeout_ms=None):
            barrier.wait(timeout=5)
            self.calls += 1
            if self.calls <= 2:
                assert timeout_ms is not None and timeout_ms > 0
                clock[0] += timeout_ms / 1000
                return None
            pins_before_request.extend(cache.pinned_id_lists() for cache in caches)
            return b"NEXT_REQUEST"

        def get(self, timeout_ms=None):
            return self.get_raw(timeout_ms)

        def decode(self, raw):
            return raw

        def empty(self):
            return True

    ranks[0]._recv_from_tokenizer = _Tokenizer()
    if num_ranks == 1:
        results = [ranks[0]._recv_msg_single_rank(blocking=True)]
    else:
        class _Sub:
            def get_raw(self):
                barrier.wait(timeout=5)
                return messages.get(timeout=5)

            def decode(self, raw):
                return raw

        class _Group:
            def broadcast(self, tensor, root=0):
                # The tokenizer has no messages after the first blocking receive.
                tensor.fill_(0)
                return SimpleNamespace(wait=lambda: barrier.wait(timeout=5))

        ranks[0]._send_into_ranks = SimpleNamespace(put_raw=messages.put)
        ranks[1]._recv_from_rank0 = _Sub()
        for rank in ranks:
            rank.tp_cpu_group = _Group()
            rank.sync_all_ranks = lambda: barrier.wait(timeout=5)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [
                pool.submit(ranks[0]._recv_msg_multi_rank0, blocking=True),
                pool.submit(ranks[1]._recv_msg_multi_rank1, blocking=True),
            ]
            results = [result.result(timeout=5) for result in pending]

    assert results == [[b"NEXT_REQUEST"]] * num_ranks
    assert pins_before_request == [[[5]]] * num_ranks
    for cache in caches:
        assert cache.hotness.has_window
        assert int(cache.hotness.counts.sum()) == 0
        assert cache.hotness.cumulative_counts()[0, 5] == 20


def test_single_rank_recv_repins_again_when_the_wait_times_out():
    """超时返回空时再进一次 idle，直到真正收到请求。"""
    addr = f"ipc:///tmp/ft-idle-{uuid.uuid4().hex}.sock"
    pull = ZmqPullQueue(addr, create=True, decoder=lambda payload: payload["x"])
    push = ZmqPushQueue(addr, create=False, encoder=lambda item: {"x": item})
    io = SchedulerIOMixin.__new__(SchedulerIOMixin)
    io._recv_from_tokenizer = pull
    calls = {"n": 0}
    started = threading.Event()

    def _idle():
        calls["n"] += 1
        if calls["n"] == 1:
            started.set()

    io.run_when_idle = _idle
    io._repin_wait_ms = lambda: 20

    def _send():
        assert started.wait(2)
        time.sleep(0.05)
        push.put("hello")

    sender = threading.Thread(target=_send)
    sender.start()
    try:
        got = io._recv_msg_single_rank(blocking=True)
    finally:
        sender.join(timeout=2)
        pull.stop()
        push.stop()
    assert got == ["hello"]
    assert calls["n"] >= 2


def test_multi_rank_wake_repins_both_ranks_before_the_request():
    """rank0 超时后发唤醒包，两边各换一次血，再一起收下一条请求。"""
    messages: queue.Queue[bytes] = queue.Queue()
    barrier = threading.Barrier(2)
    counts = {"r0": 0, "r1": 0}

    class _Tokenizer:
        def __init__(self):
            self.calls = 0

        def get_raw(self, timeout_ms=None):
            self.calls += 1
            if self.calls < 3:
                return None
            return b"MSG"

        def decode(self, raw):
            return raw

        def empty(self):
            return True

        def get(self, timeout_ms=None):
            raw = self.get_raw(timeout_ms)
            return None if raw is None else self.decode(raw)

    class _Pub:
        def put_raw(self, raw):
            messages.put(raw)

    class _Sub:
        def get_raw(self, timeout_ms=None):
            return messages.get(timeout=5)

        def decode(self, raw):
            return raw

    class _Group:
        def broadcast(self, tensor, root=0):
            class _Wait:
                def wait(self):
                    return None

            return _Wait()

    rank0 = SchedulerIOMixin.__new__(SchedulerIOMixin)
    rank1 = SchedulerIOMixin.__new__(SchedulerIOMixin)
    rank0._recv_from_tokenizer = _Tokenizer()
    rank0._send_into_ranks = _Pub()
    rank0.tp_cpu_group = _Group()
    rank1.tp_cpu_group = rank0.tp_cpu_group
    rank0._repin_wait_ms = lambda: 1
    rank0.run_when_idle = lambda: counts.__setitem__("r0", counts["r0"] + 1)
    rank0.sync_all_ranks = lambda: barrier.wait(timeout=5)
    rank1._recv_from_rank0 = _Sub()
    rank1.run_when_idle = lambda: counts.__setitem__("r1", counts["r1"] + 1)
    rank1.sync_all_ranks = lambda: barrier.wait(timeout=5)

    out: dict[str, list] = {}

    def _run0():
        out["r0"] = rank0._recv_msg_multi_rank0(blocking=True)

    def _run1():
        out["r1"] = rank1._recv_msg_multi_rank1(blocking=True)

    thread1 = threading.Thread(target=_run1)
    thread0 = threading.Thread(target=_run0)
    thread1.start()
    thread0.start()
    thread0.join(timeout=5)
    thread1.join(timeout=5)
    assert not thread0.is_alive() and not thread1.is_alive()
    assert out["r0"] == [b"MSG"]
    assert out["r1"] == [b"MSG"]
    # 进空闲一次，两次超时/唤醒再各一次。
    assert counts == {"r0": 3, "r1": 3}
    assert _REPIN_WAKE.startswith(b"\x00")
