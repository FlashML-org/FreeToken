"""The shared generation layer logs each request with its real token totals, so accounting is
independent of which endpoint served it — covering what the HTTP middleware can't record for a
stream (it fires before the totals are known)."""
from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

# Same shim as the sibling server tests: the venv may hold a non-editable install, and without
# this the file only tests the source tree when a test that does insert it collects first.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from freetoken.message import UserReply  # noqa: E402
from freetoken.server import api_server, request_ring  # noqa: E402
from freetoken.server.generation import (  # noqa: E402
    GenDone,
    GenSpec,
    generate_events,
    generate_full,
)


@pytest.fixture(autouse=True)
def served_model_name():
    """_record_generation stamps the row from api_server._served_model_name(), which reads the
    module-global app state -- not the state handed to generate_*. Pin it so the recorded model
    name is deterministic instead of a leftover from whichever test ran before."""
    prev = api_server._GLOBAL_STATE
    api_server._GLOBAL_STATE = SimpleNamespace(
        config=SimpleNamespace(served_model_name="unit-model")
    )
    yield
    api_server._GLOBAL_STATE = prev


class FakeState:
    """Yields canned acks in place of the scheduler; carries only what the generation helpers
    read (`config.reasoning_parser`, `config.served_model_name`, `wait_for_ack`)."""

    def __init__(self, replies: list[UserReply]) -> None:
        self.config = SimpleNamespace(
            mm=SimpleNamespace(max_pixels=None, text_model_only=False, disabled_encoders=frozenset()),
            model_path="/m",
            served_model_name="unit-model",
            tool_call_parser="llama3",
            reasoning_parser=None,
        )
        self._replies = replies

    def new_user(self) -> int:
        return 42

    async def wait_for_ack(self, uid: int):
        assert uid == 42
        for reply in self._replies:
            yield reply


def _ack(prompt: int = 0, completion: int = 0, out: str = "", finished: bool = False) -> UserReply:
    return UserReply(
        uid=42,
        incremental_output=out,
        finished=finished,
        prompt_tokens_delta=prompt,
        completion_tokens_delta=completion,
        finish_reason="stop" if finished else None,
    )


def _spec() -> GenSpec:
    return GenSpec(messages=[{"role": "user", "content": "hi"}], sampling_params=SimpleNamespace())


def _row(*, ttft_ms: int | None) -> request_ring.RequestRecord:
    return request_ring.RequestRecord(
        ts="2026-01-01T00:00:00Z", method="POST", path="/v1/messages", status=200,
        model="unit-model", duration_ms=1000, ttft_ms=ttft_ms, prompt_tokens=1,
        completion_tokens=1, stream=True, error=None,
    )


def _last_row() -> dict:
    rows, _ = request_ring.requests_since(0, 1000)
    return rows[-1]


def test_non_stream_records_the_request_with_real_token_totals():
    request_ring.reset()
    st = FakeState([_ack(prompt=5, completion=1, out="a"), _ack(completion=2, out="bc", finished=True)])
    result = asyncio.run(generate_full(42, _spec(), st, source="/v1/chat/completions"))
    assert (result.prompt_tokens, result.completion_tokens) == (5, 3)
    row = _last_row()
    assert row["path"] == "/v1/chat/completions"
    assert row["stream"] is False
    assert (row["prompt_tokens"], row["completion_tokens"]) == (5, 3)
    assert row["status"] == 200 and row["error"] is None
    assert row["model"] == "unit-model"


def test_stream_records_the_totals_from_gendone():
    request_ring.reset()
    st = FakeState([_ack(prompt=7, completion=1, out="x"), _ack(completion=4, out="yz", finished=True)])

    async def drain():
        done = None
        async for ev in generate_events(42, _spec(), st, source="/v1/messages"):
            if isinstance(ev, GenDone):
                done = ev
        return done

    done = asyncio.run(drain())
    assert (done.prompt_tokens, done.completion_tokens) == (7, 5)
    row = _last_row()
    assert row["path"] == "/v1/messages" and row["stream"] is True
    assert (row["prompt_tokens"], row["completion_tokens"]) == (7, 5)


def test_stream_still_records_the_row_when_the_client_disconnects_mid_stream():
    request_ring.reset()
    st = FakeState([_ack(prompt=9, completion=2, out="p"), _ack(completion=2, out="q", finished=True)])

    async def abort_after_first():
        gen = generate_events(42, _spec(), st, source="/v1/responses")
        async for _ev in gen:
            break  # the consumer stops early
        await gen.aclose()  # Starlette closes the generator on disconnect -> runs the finally

    asyncio.run(abort_after_first())
    row = _last_row()
    # Still logged on disconnect (the point); tokens are 0 since GenDone never arrived.
    assert row["path"] == "/v1/responses" and row["stream"] is True
    assert row["prompt_tokens"] == 0 and row["completion_tokens"] == 0


def test_a_generation_error_records_the_row_as_failed():
    request_ring.reset()
    st = FakeState([UserReply(uid=42, incremental_output="", finished=True, error="boom")])
    try:
        asyncio.run(generate_full(42, _spec(), st, source="/v1/chat/completions"))
    except Exception:
        pass
    row = _last_row()
    assert row["status"] == 500 and row["error"] == "boom"


def test_no_source_opts_out_of_recording():
    request_ring.reset()
    st = FakeState([_ack(prompt=1, completion=1, out="z", finished=True)])
    asyncio.run(generate_full(42, _spec(), st))  # no source
    rows, _ = request_ring.requests_since(0, 1000)
    assert rows == []


def test_stream_records_a_ttft_within_the_request_duration():
    request_ring.reset()
    st = FakeState([_ack(prompt=3, completion=1, out="a"), _ack(completion=1, out="b", finished=True)])

    async def drain():
        async for _ev in generate_events(42, _spec(), st, source="/v1/messages"):
            pass

    asyncio.run(drain())
    row = _last_row()
    assert row["ttft_ms"] is not None and 0 <= row["ttft_ms"] <= row["duration_ms"]


def test_non_stream_records_no_ttft():
    """generate_full hands the client one response: there is no first-token instant to observe."""
    request_ring.reset()
    st = FakeState([_ack(prompt=3, completion=2, out="ab", finished=True)])
    asyncio.run(generate_full(42, _spec(), st, source="/v1/chat/completions"))
    assert _last_row()["ttft_ms"] is None


def test_ttft_mean_covers_only_the_rows_that_have_one():
    """Non-streaming generations and middleware-logged rows carry no TTFT; averaging them in
    as zeros would drag the mean toward 0 as soon as anything hits /health."""
    request_ring.reset()
    for ms in (100, 200, 300):
        request_ring.record_request(_row(ttft_ms=ms))
    request_ring.record_request(_row(ttft_ms=None))
    assert request_ring.requests_ttft_mean_ms() == 200


def test_ttft_mean_is_zero_without_samples():
    request_ring.reset()
    request_ring.record_request(_row(ttft_ms=None))
    assert request_ring.requests_ttft_mean_ms() == 0
