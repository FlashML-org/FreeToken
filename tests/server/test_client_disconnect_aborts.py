"""A client that hangs up must stop the generation.

Streaming responses already notice: yielding a chunk into a dead socket is observable,
and stream_with_cancellation acts on it. A NON-streaming request awaits the whole
completion and never looks -- so a caller that disconnects leaves the GPUs generating to
max_tokens for an answer nobody will read, with the queue behind it waiting on that work.

On an offload MoE that is minutes of the whole machine per abandoned request.
"""

from __future__ import annotations

import asyncio
import pathlib

import pytest


class _Request:
    """Stand-in for a Starlette Request: disconnects after `after` polls."""

    def __init__(self, after: int | None = None):
        self.after = after
        self.polls = 0

    async def is_disconnected(self) -> bool:
        self.polls += 1
        return self.after is not None and self.polls > self.after


class _Ack:
    def __init__(self, finished=False):
        self.finished = finished
        self.incremental_output = ""
        self.prompt_tokens_delta = 0
        self.completion_tokens_delta = 1
        self.cached_tokens = 0
        self.error = None


class _State:
    """Only the pieces wait_for_ack_watching_client touches."""

    def __init__(self, n_acks=100):
        from freetoken.server.api_server import FrontendManager

        self.n_acks = n_acks
        self.yielded = 0
        self.aborted: list[int] = []
        self._fn = FrontendManager.wait_for_ack_watching_client

    async def wait_for_ack(self, uid):
        for i in range(self.n_acks):
            self.yielded += 1
            yield _Ack(finished=(i == self.n_acks - 1))
            await asyncio.sleep(0)

    async def abort_user(self, uid):
        self.aborted.append(uid)

    def watch(self, uid, request):
        return self._fn(self, uid, request)


async def _drain(state, request):
    got = 0
    try:
        async for _ack in state.watch(7, request):
            got += 1
    except asyncio.CancelledError:
        pass
    # abort_user is scheduled, not awaited, so let the loop run it.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    return got


class TestADisconnectStopsTheGeneration:
    def test_it_stops_early_instead_of_running_to_completion(self):
        state = _State(n_acks=100)
        got = asyncio.run(_drain(state, _Request(after=3)))
        assert got < 100, "the loop ran to completion for a client that had hung up"
        assert got <= 5

    def test_it_aborts_the_request(self):
        state = _State(n_acks=100)
        asyncio.run(_drain(state, _Request(after=2)))
        assert state.aborted == [7], "the engine must be told to drop the request"

    def test_a_connected_client_receives_everything(self):
        state = _State(n_acks=10)
        got = asyncio.run(_drain(state, _Request(after=None)))
        assert got == 10
        assert state.aborted == []

    def test_no_request_means_no_watching(self):
        # Offline / internal callers have no HTTP request; they must still work.
        state = _State(n_acks=10)
        got = asyncio.run(_drain(state, None))
        assert got == 10
        assert state.aborted == []


class TestTheNonStreamingPathsUseIt:
    """The wrapper is worthless if the handlers keep calling the bare loop."""

    @staticmethod
    def _src(name: str) -> str:
        return (
            pathlib.Path(__file__).resolve().parents[2]
            / "python" / "freetoken" / "server" / name
        ).read_text()

    def test_generate_full_watches_the_client(self):
        src = self._src("generation.py")
        body = src[src.index("async def _generate_full_impl") :]
        assert "_acks(state, uid, request)" in body, (
            "a non-streaming ack loop that does not watch the client will generate to "
            "max_tokens for a caller that has gone"
        )

    def test_the_helper_tolerates_a_state_without_the_watcher(self):
        # Not every state implementation has it (the test fakes do not). A missing
        # disconnect check must cost the abort, not the request.
        src = self._src("generation.py")
        assert 'getattr(state, "wait_for_ack_watching_client", None)' in src

    def test_chat_completions_passes_its_request_down(self):
        src = self._src("openai_api.py")
        assert "request=request" in src, (
            "generate_full cannot watch a client it was never given"
        )

    def test_completions_watches_too(self):
        src = self._src("openai_api.py")
        assert "_acks(state, uid, request)" in src
