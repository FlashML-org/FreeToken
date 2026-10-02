from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

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

    def get(self, timeout_ms: int | None = None) -> T | None:
        raw = self.get_raw(timeout_ms=timeout_ms)
        if raw is None:
            return None
        return self.decode(raw)

    def get_raw(self, timeout_ms: int | None = None) -> bytes | None:
        if timeout_ms is not None and self.socket.poll(timeout=timeout_ms) == 0:
            return None
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

    def get(self, timeout_ms: int | None = None) -> T | None:
        raw = self.get_raw(timeout_ms=timeout_ms)
        if raw is None:
            return None
        return self.decode(raw)

    def get_raw(self, timeout_ms: int | None = None) -> bytes | None:
        if timeout_ms is not None and self.socket.poll(timeout=timeout_ms) == 0:
            return None
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()
