"""Admission bounds at the real manager, plus protocol-specific HTTP overload errors."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from freetoken.message import UserReply
from freetoken.server import api_server, request_ring
from freetoken.server.accounting import AdmissionClosedError
from freetoken.server.admission import AdmissionThrottledError, admission_limit
from freetoken.server.anthropic_api import register_anthropic_routes
from freetoken.server.args import ServerArgs, parse_args
from freetoken.server.openai_api import register_openai_routes
from freetoken.server.responses_api import register_responses_routes


def manager(limit=0, scheduler_limit=4):
    return api_server.FrontendManager(
        config=SimpleNamespace(
            max_concurrent_requests=limit, max_running_req=scheduler_limit,
            model_path="/test", served_model_name="test", reasoning_parser=None,
            tool_call_parser="llama3",
        ),
        send_tokenizer=None, recv_tokenizer=None, maintenance_state="serving",
    )


def test_default_is_unlimited_and_automatic_capacity_is_explicit():
    assert ServerArgs.max_concurrent_requests == 0
    obj = manager()
    for _ in range(50):
        obj.new_user()
    assert len(obj.ack_map) == len(obj.event_map) == 50
    assert admission_limit(obj.config) == 0
    assert admission_limit(manager(-1, 3).config) == 3


def test_parallel_admission_never_exceeds_the_cap_or_mutates_on_rejection():
    obj = manager(3)

    async def admit():
        try:
            return obj.new_user()
        except AdmissionThrottledError:
            return None

    async def run():
        return await asyncio.gather(*(admit() for _ in range(20)))

    results = asyncio.run(run())
    assert [uid for uid in results if uid is not None] == [0, 1, 2]
    assert obj.uid_counter == 3
    assert len(obj.ack_map) == len(obj.event_map) == obj.stats.active == 3


def test_maintenance_gate_wins_over_capacity():
    obj = manager(1)
    obj.new_user()
    obj.maintenance_state = "stopping"
    with pytest.raises(AdmissionClosedError):
        obj.new_user()


def test_terminal_reply_frees_capacity_before_consumer_closes_the_generator():
    async def run():
        obj = manager(1)
        uid = obj.new_user()
        obj.ack_map[uid] = [UserReply(uid, "done", True)]
        obj.event_map[uid].set()
        replies = obj.wait_for_ack(uid)
        assert (await anext(replies)).finished
        assert obj.ack_map == obj.event_map == {}
        # Keep the previous generator alive: cleanup must not depend on its GC.
        assert obj.new_user() == uid + 1
        await replies.aclose()

    asyncio.run(run())


def test_abort_releases_maps_without_faking_terminal_accounting():
    async def run():
        obj = manager(1)
        sent = []

        async def send(msg):
            sent.append(msg)

        obj.send_one = send
        uid = obj.new_user()
        await obj.abort_user(uid)
        assert obj.ack_map == obj.event_map == {}
        assert obj.stats.active == 1  # still waiting for the scheduler's terminal ack
        assert sent[0].uid == uid
        assert obj.new_user() == uid + 1

    asyncio.run(run())


@pytest.mark.parametrize("latency, expected", [(0, 1), (1, 1), (1000, 1), (1001, 2), (3250, 4)])
def test_retry_after_uses_the_existing_p95_with_a_fixed_empty_fallback(latency, expected):
    request_ring.reset()
    if latency:
        request_ring.record_request(request_ring.RequestRecord(
            ts="", method="POST", path="/v1/messages", status=200, model="test",
            duration_ms=latency, ttft_ms=None, prompt_tokens=None, completion_tokens=None,
            stream=False, error=None,
        ))
    assert AdmissionThrottledError().retry_after == expected


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("path", [
    "/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/completions", "/generate",
])
def test_all_adapters_return_429_before_streaming_and_do_not_submit(monkeypatch, path, stream):
    request_ring.reset()
    obj = manager(1)
    obj.new_user()

    async def forbidden_send(msg):
        raise AssertionError("saturated requests must never enter IPC")

    state = SimpleNamespace(config=obj.config, new_user=obj.new_user, send_one=forbidden_send,
                            maintenance_state="serving")
    app = FastAPI()
    register_openai_routes(app, lambda: state, lambda: {})
    register_anthropic_routes(app, lambda: state, lambda: {})
    register_responses_routes(app, lambda: state, lambda: {})
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
    app.post("/generate")(api_server.generate)
    payload = {"model": "test", "stream": stream, "max_tokens": 4}
    if path in ("/v1/chat/completions", "/v1/messages"):
        payload["messages"] = [{"role": "user", "content": "hi"}]
    elif path == "/v1/responses":
        payload["input"] = "hi"
    else:
        payload["prompt"] = "hi"
    with TestClient(app) as http:
        response = http.post(path, json=payload)
    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"
    assert response.headers["content-type"].startswith("application/json")
    if path != "/generate":
        assert response.json()["error"]["type"] == "rate_limit_error"
    assert obj.uid_counter == 1
    assert obj.stats.active == 1


def test_cli_rejects_an_invalid_limit_before_loading_a_model():
    with pytest.raises(SystemExit) as exc:
        parse_args(["--model", "unused", "--max-concurrent-requests", "-2"])
    assert exc.value.code == 2
