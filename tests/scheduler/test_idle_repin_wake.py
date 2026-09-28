"""空闲收包在换血窗口到点时醒来，多卡用同一条唤醒包对齐。"""

from __future__ import annotations

import queue
import threading
import time
import uuid

from freetoken.scheduler.io import _REPIN_WAKE, SchedulerIOMixin
from freetoken.utils.mp import ZmqPullQueue, ZmqPushQueue


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
