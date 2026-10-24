"""Cancellation-path tests for FrontendManager.stream_with_cancellation / abort_user."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from freetoken.message import AbortMsg
from freetoken.server.api_server import FrontendManager


class _Stats:
    def __init__(self):
        self.aborts = []

    def on_abort(self, uid):
        self.aborts.append(uid)


def _state(send_impl=None):
    st = SimpleNamespace(
        ack_map={7: [object()]},
        event_map={7: asyncio.Event()},
        aborted_uids={},
        stats=_Stats(),
    )
    sent = []

    async def default_send(msg):
        sent.append(msg)

    st.send_one = send_impl or default_send
    st.sent = sent
    return st


class _Request:
    def __init__(self, disconnected=False):
        self._disconnected = disconnected

    async def is_disconnected(self):
        return self._disconnected


async def _consume(state, request, uid=7):
    state.abort_user = lambda request_uid: FrontendManager.abort_user(state, request_uid)
    async for _ in FrontendManager.stream_with_cancellation(state, _never(), request, uid):
        pass


async def _never():
    await asyncio.sleep(3600)
    yield b""  # pragma: no cover


def test_cancellation_sends_one_abort_and_cleans_maps_inline():
    async def run():
        state = _state()
        task = asyncio.create_task(_consume(state, _Request()))
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert asyncio.all_tasks() - {asyncio.current_task()} == set()
        assert len(state.sent) == 1
        assert isinstance(state.sent[0], AbortMsg)
        assert state.sent[0].uid == 7
        assert state.ack_map == {}
        assert state.event_map == {}
        assert state.stats.aborts == [7]

    asyncio.run(run())


def test_abort_delivery_failure_preserves_cancellation():
    async def boom(msg):
        raise RuntimeError("zmq down")

    async def run():
        state = _state(send_impl=boom)
        task = asyncio.create_task(_consume(state, _Request()))
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_abort_user_is_idempotent():
    async def run():
        state = _state()

        await FrontendManager.abort_user(state, 7)
        assert len(state.sent) == 1

        await FrontendManager.abort_user(state, 7)
        assert len(state.sent) == 1
        assert state.stats.aborts == [7]

    asyncio.run(run())


def test_cancellation_after_ack_loop_cleanup_still_aborts():
    """The production disconnect path: the CancelledError lands on wait_for_ack's
    innermost await, and its finally empties both maps BEFORE the stream wrapper
    gets to run. A claim keyed on the maps no-ops exactly then; the AbortMsg must
    still reach the scheduler (which acks aborts for uids it no longer has)."""

    async def run():
        state = _state()

        async def drain_like_wait_for_ack():
            try:
                await asyncio.sleep(3600)
                yield b""  # pragma: no cover
            finally:
                state.ack_map.pop(7, None)
                state.event_map.pop(7, None)

        async def consume():
            state.abort_user = lambda uid: FrontendManager.abort_user(state, uid)
            async for _ in FrontendManager.stream_with_cancellation(
                state, drain_like_wait_for_ack(), _Request(), 7
            ):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert state.ack_map == {} and state.event_map == {}
        assert [type(m) for m in state.sent] == [AbortMsg]
        assert state.sent[0].uid == 7
        assert state.stats.aborts == [7]

    asyncio.run(run())


def test_late_close_delivers_abort():
    """A body closed after the response started surfaces as aclose() on the stream
    wrapper: GeneratorExit, which `except CancelledError` never sees. The finally
    must deliver the abort for any stream that did not run to completion."""

    async def run():
        state = _state()
        state.abort_user = lambda uid: FrontendManager.abort_user(state, uid)

        async def chunks():
            while True:
                yield b"data: x\n\n"

        agen = FrontendManager.stream_with_cancellation(state, chunks(), _Request(), 7)
        assert await agen.__anext__() == b"data: x\n\n"
        await agen.aclose()

        assert [type(m) for m in state.sent] == [AbortMsg]
        assert state.sent[0].uid == 7
        assert state.stats.aborts == [7]

    asyncio.run(run())


def test_concurrent_aborts_send_one_message():
    """The accounting drain gathers abort_user over every in-flight uid and can race
    a disconnect abort for the same request; the claim set must collapse them."""

    async def run():
        state = _state()
        await asyncio.gather(
            FrontendManager.abort_user(state, 7),
            FrontendManager.abort_user(state, 7),
        )
        assert len(state.sent) == 1
        assert state.stats.aborts == [7]

    asyncio.run(run())


def test_normal_completion_sends_no_abort():
    async def run():
        state = _state()

        async def one():
            yield b"data: x\n\n"

        async for _ in FrontendManager.stream_with_cancellation(state, one(), _Request(), 7):
            pass
        assert state.sent == []
        assert 7 in state.ack_map  # wait_for_ack owns normal-path cleanup, not the stream

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Non-streaming handlers: a disconnected client must abort the engine request
# (same shielded abort_user as the streaming path) instead of letting it run to
# max_tokens while later requests queue behind it.
# --------------------------------------------------------------------------- #

from freetoken.server import openai_api
from freetoken.server.api_models import ChatCompletionRequest, CompletionRequest


def _ack(text, finished=False):
    return SimpleNamespace(
        error=None,
        incremental_output=text,
        finished=finished,
        prompt_tokens_delta=0,
        completion_tokens_delta=1,
        cached_tokens=0,
        finish_reason="stop" if finished else None,
        matched_stop=None,
        logprobs=None,
    )


class _ApiState:
    """State fake for the openai_api handlers: hangs forever when given no acks."""

    def __init__(self, acks=None):
        self.config = SimpleNamespace(
            model_path="/models/unit-model",
            served_model_name="unit-model",
            tool_call_parser="llama3",
            reasoning_parser=None,
        )
        self.acks = acks
        self.sent = []
        self.aborted = []

    def new_user(self):
        return 7

    async def send_one(self, msg):
        self.sent.append(msg)

    async def wait_for_ack(self, uid):
        if self.acks is None:
            await asyncio.sleep(3600)
        for ack in self.acks or []:
            yield ack

    async def abort_user(self, uid):
        self.aborted.append(uid)


def _chat_req(**kwargs):
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    payload.update(kwargs)
    return ChatCompletionRequest(**payload)


def test_chat_non_stream_disconnect_delivers_abort(monkeypatch):
    monkeypatch.setattr(openai_api, "_DISCONNECT_POLL_SECONDS", 0.01)
    state = _ApiState(acks=None)  # generation never finishes on its own

    resp = asyncio.run(
        openai_api.handle_chat_completion(_chat_req(), _Request(disconnected=True), state, {})
    )

    assert resp.status_code == 499
    assert state.aborted == [7]


def test_completion_non_stream_disconnect_delivers_abort(monkeypatch):
    monkeypatch.setattr(openai_api, "_DISCONNECT_POLL_SECONDS", 0.01)
    state = _ApiState(acks=None)
    req = CompletionRequest(model="m", prompt="hello", max_tokens=8)

    resp = asyncio.run(
        openai_api.handle_completion(req, _Request(disconnected=True), state, {})
    )

    assert resp.status_code == 499
    assert state.aborted == [7]


def test_non_stream_connected_client_gets_result_without_abort(monkeypatch):
    monkeypatch.setattr(openai_api, "_DISCONNECT_POLL_SECONDS", 0.01)
    state = _ApiState(acks=[_ack("Hi", finished=True)])

    result = asyncio.run(
        openai_api.handle_chat_completion(_chat_req(), _Request(disconnected=False), state, {})
    )

    assert result["choices"][0]["message"]["content"] == "Hi"
    assert state.aborted == []
