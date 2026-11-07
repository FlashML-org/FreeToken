from __future__ import annotations

import msgpack
import pytest
import time

from freetoken.utils import ZmqPubQueue, ZmqSubQueue

ADDR = "tcp://127.0.0.1:5590"


def encode(obj: dict) -> dict:
    return obj


def decode(raw: dict) -> dict:
    return raw


def _close(*queues) -> None:
    for q in queues:
        try:
            q.stop()
        except Exception:
            pass


def test_pub_wait_for_subscribers_then_deliver():
    pub = sub = None
    try:
        pub = ZmqPubQueue(ADDR, create=True, encoder=encode)
        sub = ZmqSubQueue(ADDR, create=False, decoder=decode)
        pub.wait_for_subscribers(1, timeout_s=10.0)
        msg = {"role": "broadcast", "seq": 1}
        pub.put(msg)
        deadline = time.monotonic() + 2.0
        while sub.empty() and time.monotonic() < deadline:
            pass
        assert sub.empty() is False
        assert sub.get() == msg
    finally:
        _close(sub, pub)


def test_pub_wait_zero_returns_immediately():
    pub = None
    try:
        pub = ZmqPubQueue(ADDR, create=True, encoder=encode)
        pub.wait_for_subscribers(0, timeout_s=1.0)
    finally:
        _close(pub)


def test_pub_wait_timeout_without_subscribers():
    pub = None
    try:
        pub = ZmqPubQueue(ADDR, create=True, encoder=encode)
        with pytest.raises(TimeoutError):
            pub.wait_for_subscribers(1, timeout_s=1.0)
    finally:
        _close(pub)


def test_payload_roundtrip_bytes_identical():
    """XPUB must deliver the same msgpack bytes a PUB would."""
    pub = sub = None
    try:
        pub = ZmqPubQueue(ADDR, create=True, encoder=encode)
        sub = ZmqSubQueue(ADDR, create=False, decoder=decode)
        pub.wait_for_subscribers(1, timeout_s=10.0)
        obj = {"tokens": [1, 2, 3], "nested": {"a": b"\x00\x01"}}
        packed = msgpack.packb(obj, use_bin_type=True)
        pub.put_raw(packed)
        assert sub.get() == obj
    finally:
        _close(sub, pub)
