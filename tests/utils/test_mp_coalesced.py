# SPDX-License-Identifier: Apache-2.0
"""Regression tests for coalesced msgpack frames on ZMQ queues (issue #452).

A single ZMQ frame should carry exactly one msgpack object, but a read can surface
a frame containing two packed objects back to back. ``msgpack.unpackb`` raises
``ExtraData`` there and the worker process dies. The buffered unpacker decodes the
first object and holds the remainder for the next get().
"""

import asyncio

import msgpack
import pytest

from freetoken.utils.mp import (
    ZmqAsyncPullQueue,
    ZmqAsyncPushQueue,
    ZmqPubQueue,
    ZmqPullQueue,
    ZmqPushQueue,
    ZmqSubQueue,
)


def _identity_decoder(obj):
    return obj


def _addr(port):
    return f"tcp://127.0.0.1:{port}"


class TestCoalescedFrames:
    """A frame carrying two packed objects must yield two get() calls, not a crash."""

    def test_sync_pull_coalesced_frame(self):
        push = ZmqPushQueue(_addr(5601), create=True, encoder=lambda o: o)
        pull = ZmqPullQueue(_addr(5601), create=False, decoder=_identity_decoder)
        try:
            # Simulate the coalesced frame: two packb outputs concatenated.
            frame = msgpack.packb({"i": 1}, use_bin_type=True) + msgpack.packb({"i": 2}, use_bin_type=True)
            pull._unpacker.feed(frame)

            assert pull.get() == {"i": 1}
            assert pull.get() == {"i": 2}
            assert len(pull._unpacker) == 0
        finally:
            push.stop()
            pull.stop()

    def test_decode_coalesced_raw(self):
        from freetoken.utils.mp import _CoalescedUnpacker

        unpacker = _CoalescedUnpacker()
        frame = msgpack.packb("first", use_bin_type=True) + msgpack.packb("second", use_bin_type=True)
        unpacker.feed(frame)
        assert unpacker.take() == "first"
        assert unpacker.take() == "second"

    def test_empty_still_true_with_buffered_only(self):
        push = ZmqPushQueue(_addr(5602), create=True, encoder=lambda o: o)
        pull = ZmqPullQueue(_addr(5602), create=False, decoder=_identity_decoder)
        try:
            # nothing on socket and no buffer -> empty
            assert pull.empty()
            # a coalesced frame arrives and is fully buffered -> socket empty but buffer has 1 left
            push.put({"x": 1})
            pull._unpacker.feed(pull.get_raw())
            assert pull.get() == {"x": 1}
            assert pull.empty()
        finally:
            push.stop()
            pull.stop()

    def test_sub_queue_coalesced(self):
        import msgpack

        pub = ZmqPubQueue(_addr(5603), create=True, encoder=lambda o: o)
        sub = ZmqSubQueue(_addr(5603), create=False, decoder=_identity_decoder)
        try:
            import time

            time.sleep(0.1)  # SUB subscription propagation
            pub.put({"a": 1})
            pub.put({"a": 2})
            time.sleep(0.1)
            # Deliver both objects even if they arrive in one frame.
            assert sub.get() == {"a": 1}
            assert sub.get() == {"a": 2}
        finally:
            pub.stop()
            sub.stop()


async def test_async_pull_coalesced():
    import sys

    from freetoken.utils.mp import ZmqAsyncPullQueue as _AsyncPull
    from freetoken.utils.mp import ZmqAsyncPushQueue as _AsyncPush

    # zmq.asyncio needs a Selector loop on Windows; the project runs on Linux CI.
    if sys.platform == "win32":
        pytest.skip("zmq.asyncio requires SelectorEventLoop on Windows (prod runs on Linux)")
    push = ZmqAsyncPushQueue(_addr(5604), create=True, encoder=lambda o: o)
    pull = ZmqAsyncPullQueue(_addr(5604), create=False, decoder=_identity_decoder)
    try:
        await push.put({"n": 1})
        await push.put({"n": 2})
        await asyncio.sleep(0.1)

        # Even if both messages arrive in a single frame, two gets must succeed.
        assert await pull.get() == {"n": 1}
        assert await pull.get() == {"n": 2}
    finally:
        push.stop()
        pull.stop()