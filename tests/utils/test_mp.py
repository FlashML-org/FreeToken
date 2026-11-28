"""The scheduler's request fan-out (``ZmqPubQueue`` -> ``ZmqSubQueue``): a PUB socket drops what it sends before a
subscription reaches it, so the first send waits for every named subscriber, and names the ones that never came."""

from __future__ import annotations

import threading
import time

import pytest
from freetoken.utils.mp import ZmqPubQueue, ZmqSubQueue


def _late_subscribers(addr, names, delay=0.3):
    """Connect one subscriber per name after ``delay``, as ranks that finish starting after rank 0 does."""
    subs = []

    def connect():
        time.sleep(delay)
        subs.extend(ZmqSubQueue(addr, create=False, decoder=dict, name=name) for name in names)

    thread = threading.Thread(target=connect)
    thread.start()
    return thread, subs


@pytest.mark.parametrize("wait", [True, False])
def test_first_send_reaches_subscribers_that_connect_late(tmp_path, wait):
    addr = f"ipc://{tmp_path}/fanout"
    pub = ZmqPubQueue(addr, create=True, encoder=dict, subscribers=["rank 1", "rank 2"] if wait else ())
    thread, subs = _late_subscribers(addr, ["rank 1", "rank 2"])
    try:
        pub.put({"request": 1})
        thread.join()
        got = [sub.socket.poll(1000) and sub.get() for sub in subs]
        if wait:
            assert got == [{"request": 1}] * 2
        else:  # a plain PUB sent it before anyone listened
            assert got == [0, 0]
    finally:
        thread.join()
        for queue in (*subs, pub):
            queue.stop()


def test_wait_times_out_naming_the_subscriber_that_never_came(tmp_path):
    addr = f"ipc://{tmp_path}/fanout"
    pub = ZmqPubQueue(addr, create=True, encoder=dict, subscribers=["rank 1", "rank 2"], subscribe_timeout=1.0)
    # rank 1 connects twice (a reconnect): it still stands for one rank, not for rank 2
    thread, subs = _late_subscribers(addr, ["rank 1", "rank 1"], delay=0.1)
    try:
        with pytest.raises(RuntimeError, match=r"^rank 2 never subscribed"):
            pub.put({"request": 1})
    finally:
        thread.join()
        for queue in (*subs, pub):
            queue.stop()
