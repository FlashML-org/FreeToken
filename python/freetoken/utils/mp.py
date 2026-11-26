from __future__ import annotations

import time
from typing import Callable, Collection, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio

T = TypeVar("T")


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

    def get(self) -> T:
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def get_raw(self) -> bytes:
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

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

    async def get(self) -> T:
        event = await self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
        subscribers: Collection[str] = (),
        subscribe_timeout: float | None = None,
    ):
        """``subscribers``: the names of the ``ZmqSubQueue``s that must have subscribed before the first send, since PUB
        silently drops what it sends before a subscription reaches it. ``subscribe_timeout`` (seconds) bounds that wait,
        which ``wait_for_subscribers`` runs up front and the first send runs if no one did."""
        self.context = zmq.Context()
        # an XPUB reports every subscription, including the topic each named subscriber adds (ZmqSubQueue)
        self.socket = self.context.socket(zmq.XPUB if subscribers else zmq.PUB)
        if subscribers:
            self.socket.setsockopt(zmq.XPUB_VERBOSE, 1)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder
        self._unjoined = set(subscribers)
        self._subscribe_timeout = subscribe_timeout

    def wait_for_subscribers(self) -> None:
        """Block until every named subscriber has subscribed; raise naming the missing ones after ``subscribe_timeout``."""
        deadline = None if self._subscribe_timeout is None else time.monotonic() + self._subscribe_timeout
        while self._unjoined:
            wait = None if deadline is None else max(0, int((deadline - time.monotonic()) * 1000))
            if not self.socket.poll(wait):
                missing = ", ".join(sorted(self._unjoined))
                raise RuntimeError(f"{missing} never subscribed to this publisher within {self._subscribe_timeout:g} s")
            msg = self.socket.recv()
            # 0x01 + topic is a subscribe (0x00 an unsubscribe); a subscriber that reconnects subscribes again, which the
            # set absorbs, and the unnamed "" subscriptions name no one
            if msg[:1] == b"\x01":
                self._unjoined.discard(msg[1:].decode(errors="replace"))

    def put_raw(self, raw: bytes):
        if self._unjoined:
            self.wait_for_subscribers()
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        self.put_raw(msgpack.packb(self.encoder(obj), use_bin_type=True))

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
        name: str | None = None,
    ):
        """``name``: also subscribe to that topic, which nothing is published on, so a ``ZmqPubQueue`` waiting for its
        ``subscribers`` can tell this subscriber's arrival from the others'."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        if name is not None:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, name)
        self.decoder = decoder

    def get(self) -> T:
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()
