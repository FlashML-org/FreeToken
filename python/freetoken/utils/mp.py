from __future__ import annotations

from typing import Any, Callable, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio

T = TypeVar("T")


class _CoalescedUnpacker:
    """
    Buffered msgpack decoder tolerant of coalesced frames.

    A single ZMQ frame should carry exactly one msgpack object, but on some platforms
    (issue #452: Windows, offload MoE) a read can surface a frame containing two packed
    objects back to back. ``msgpack.unpackb`` raises ``ExtraData`` in that case and kills
    the worker process. Feeding every received frame through one ``Unpacker`` instead
    decodes the first object and buffers the remainder, so the next ``get()`` returns it
    in order instead of crashing.
    """

    def __init__(self) -> None:
        self._unpacker = msgpack.Unpacker(raw=False)
        self._pending: list[Any] = []

    def feed(self, frame: bytes) -> None:
        """Queue up every msgpack object contained in ``frame`` (usually exactly one)."""
        self._unpacker.feed(frame)
        for obj in self._unpacker:
            self._pending.append(obj)

    def take(self) -> Any:
        """Pop the oldest undelivered object; raises StopIteration when the buffer is empty."""
        return self._pending.pop(0)

    def __len__(self) -> int:
        return len(self._pending)


class ZmqPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder
        self._unpacker = _CoalescedUnpacker()

    def get(self) -> T:
        if not len(self._unpacker):
            event = self.socket.recv()
            self._unpacker.feed(event)
        obj = self._unpacker.take()
        return self.decoder(obj)

    def get_raw(self) -> bytes:
        # Raw path (multi-rank broadcast) must stay unbuffered: callers pair empty()/get_raw()
        # with a rank-wide count, so a buffered remainder would desynchronize the loop.
        if len(self._unpacker):
            raise RuntimeError(
                "get_raw() cannot be mixed with buffered get() on the same queue: "
                "the unpacker holds undelivered objects that would be skipped."
            )
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        return len(self._unpacker) == 0 and self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder
        self._unpacker = _CoalescedUnpacker()

    async def get(self) -> T:
        if not len(self._unpacker):
            event = await self.socket.recv()
            self._unpacker.feed(event)
        obj = self._unpacker.take()
        return self.decoder(obj)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder
        self._unpacker = _CoalescedUnpacker()

    def get(self) -> T:
        if not len(self._unpacker):
            event = self.socket.recv()
            self._unpacker.feed(event)
        obj = self._unpacker.take()
        return self.decoder(obj)

    def empty(self) -> bool:
        return len(self._unpacker) == 0 and self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()