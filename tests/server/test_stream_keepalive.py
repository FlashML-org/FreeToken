"""A stream that stays silent -- the prefill of a long prompt -- keeps its client, and notices when
the client leaves instead of prefilling for nobody."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from freetoken.server import api_server
from freetoken.server.api_server import FrontendManager

# Converts a stream that never ends into a failure; the streams below finish in milliseconds.
HANG_GUARD_SECONDS = 10.0


class _Request:
    def __init__(self) -> None:
        self.gone = False

    async def is_disconnected(self) -> bool:
        return self.gone


def _manager(aborted: list[int]) -> SimpleNamespace:
    async def abort_user(uid: int) -> None:
        aborted.append(uid)

    return SimpleNamespace(abort_user=abort_user)


def _is_sse_comment(chunk: bytes) -> bool:
    return chunk.startswith(b":") and chunk.endswith(b"\n\n")


def test_a_silent_stream_sends_comments_until_its_next_chunk(monkeypatch):
    monkeypatch.setattr(api_server, "STREAM_KEEPALIVE_SECONDS", 0.01, raising=False)
    released = asyncio.Event()

    async def chunks():
        yield b"data: first\n\n"
        await released.wait()
        yield b"data: last\n\n"

    async def consume() -> list[bytes]:
        out = []
        request = _Request()
        async for chunk in FrontendManager.stream_with_cancellation(_manager([]), chunks(), request, 7):
            out.append(chunk)
            if _is_sse_comment(chunk):
                released.set()
        return out

    out = asyncio.run(asyncio.wait_for(consume(), HANG_GUARD_SECONDS))
    assert out[0] == b"data: first\n\n" and out[-1] == b"data: last\n\n"
    assert len(out) > 2 and all(_is_sse_comment(chunk) for chunk in out[1:-1])


def test_a_client_that_leaves_during_the_silence_aborts_its_request(monkeypatch):
    monkeypatch.setattr(api_server, "STREAM_KEEPALIVE_SECONDS", 0.01, raising=False)
    prefill_done = asyncio.Event()

    async def chunks():
        yield b"data: first\n\n"
        await prefill_done.wait()
        yield b"data: never read\n\n"

    async def scenario() -> tuple[list[bytes], list[int]]:
        aborted = asyncio.Event()
        seen: list[int] = []

        async def abort_user(uid: int) -> None:
            seen.append(uid)
            aborted.set()

        request = _Request()
        out: list[bytes] = []

        async def consume() -> None:
            manager = SimpleNamespace(abort_user=abort_user)
            async for chunk in FrontendManager.stream_with_cancellation(manager, chunks(), request, 9):
                out.append(chunk)
                request.gone = True

        consumer = asyncio.ensure_future(consume())
        # Only the wrapper noticing the departure can end this wait; the guard below fails it.
        await aborted.wait()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        return out, seen

    out, seen = asyncio.run(asyncio.wait_for(scenario(), HANG_GUARD_SECONDS))
    assert out == [b"data: first\n\n"]
    assert seen == [9]
    assert not prefill_done.is_set()


def test_a_stream_that_never_goes_quiet_carries_no_comments(monkeypatch):
    monkeypatch.setattr(api_server, "STREAM_KEEPALIVE_SECONDS", HANG_GUARD_SECONDS, raising=False)

    async def chunks():
        for index in range(5):
            yield f"data: {index}\n\n".encode()

    async def consume() -> list[bytes]:
        return [
            chunk
            async for chunk in FrontendManager.stream_with_cancellation(
                _manager([]), chunks(), _Request(), 3
            )
        ]

    out = asyncio.run(asyncio.wait_for(consume(), HANG_GUARD_SECONDS))
    assert out == [f"data: {index}\n\n".encode() for index in range(5)]
