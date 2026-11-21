"""CPU-only tests for the bounded prepare-stop accounting barrier."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from freetoken.message import UserReply
from freetoken.server.accounting import (
    AccountingDrainError,
    AdmissionClosedError,
    _is_loopback,
    prepare_stop_accounting,
    register_accounting_routes,
)
from freetoken.server.api_server import FrontendManager
from freetoken.server.stats import StatsTracker


def _state(*, maintenance: str = "serving", ready_at: float | None = None):
    stats = StatsTracker()

    async def abort_user(uid: int) -> None:
        stats.on_abort(uid)
        stats.observe(UserReply(uid=uid, incremental_output="", finished=True, error="aborted"))

    return SimpleNamespace(
        maintenance_state=maintenance,
        instance_id="generation-1",
        config=SimpleNamespace(served_model_name="model-a"),
        stats=stats,
        ready_at=ready_at,
        abort_user=abort_user,
    )


def test_idle_prepare_stop_seals_totals_and_is_idempotent(monkeypatch):
    state = _state(ready_at=90.0)
    state.stats.prompt_tokens_total = 12
    state.stats.completion_tokens_total = 7
    monkeypatch.setattr("freetoken.server.accounting.time.monotonic", lambda: 100.8)

    first = asyncio.run(prepare_stop_accounting(state))
    assert first == {
        "instance_id": "generation-1",
        "model_id": "model-a",
        "prompt_tokens_total": 12,
        "completion_tokens_total": 7,
        "uptime_s": 10,
        "drain_complete": True,
    }
    assert state.maintenance_state == "stopping"

    # A retry after the daemon lost the HTTP response gets the original sealed document.
    state.stats.completion_tokens_total = 999
    assert asyncio.run(prepare_stop_accounting(state)) == first


def test_prepare_stop_waits_for_a_natural_terminal_reply():
    async def run():
        state = _state()
        state.stats.on_new_user(4)

        async def finish() -> None:
            await asyncio.sleep(0.02)
            state.stats.observe(
                UserReply(
                    uid=4,
                    incremental_output="x",
                    finished=True,
                    prompt_tokens_delta=8,
                    completion_tokens_delta=1,
                )
            )

        finisher = asyncio.create_task(finish())
        result = await prepare_stop_accounting(
            state, drain_timeout_s=0.2, abort_timeout_s=0.1
        )
        await finisher
        return state, result

    state, result = asyncio.run(run())
    assert state.stats.active == 0
    assert result["prompt_tokens_total"] == 8
    assert result["completion_tokens_total"] == 1


def test_prepare_stop_aborts_after_bounded_drain_and_waits_for_terminal_ack():
    state = _state()
    state.stats.on_new_user(5)
    result = asyncio.run(
        prepare_stop_accounting(state, drain_timeout_s=0.0, abort_timeout_s=0.1)
    )
    assert result["drain_complete"] is True
    assert state.stats.active == 0
    assert state.stats.completed == 0  # an aborted request is terminal, not completed


def test_missing_abort_terminal_fails_closed_and_keeps_admission_shut():
    state = _state()
    state.stats.on_new_user(6)

    async def abort_without_ack(uid: int) -> None:
        state.stats.on_abort(uid)

    state.abort_user = abort_without_ack
    with pytest.raises(AccountingDrainError, match="abort barrier timed out"):
        asyncio.run(
            prepare_stop_accounting(state, drain_timeout_s=0.0, abort_timeout_s=0.01)
        )
    assert state.maintenance_state == "stopping"
    assert state.stats.active == 1
    assert not hasattr(state, "_sealed_accounting")


def test_loading_engine_can_seal_zero_without_being_reopened():
    state = _state(maintenance="loading")
    result = asyncio.run(prepare_stop_accounting(state))
    assert result["prompt_tokens_total"] == result["completion_tokens_total"] == 0
    assert state.maintenance_state == "stopping"


def test_frontend_new_user_refuses_work_after_stop_gate_closes():
    manager = FrontendManager(
        config=SimpleNamespace(served_model_name="model-a"),
        send_tokenizer=None,
        recv_tokenizer=None,
        maintenance_state="stopping",
    )
    with pytest.raises(AdmissionClosedError, match="stopping"):
        manager.new_user()
    assert manager.stats.active == 0


def test_prepare_stop_route_reports_fail_closed_timeout():
    state = _state()
    state.stats.on_new_user(7)

    async def abort_without_ack(uid: int) -> None:
        state.stats.on_abort(uid)

    state.abort_user = abort_without_ack
    app = FastAPI()
    register_accounting_routes(app, lambda: state)
    response = TestClient(app, client=("127.0.0.1", 50000)).post(
        "/v1/admin/prepare-stop",
        json={"drain_timeout_s": 0, "abort_timeout_s": 0},
    )
    assert response.status_code == 503
    assert response.json()["engine_preserved"] is True
    assert response.json()["drain_complete"] is False


def test_prepare_stop_route_rejects_non_loopback_without_closing_admission():
    state = _state()
    app = FastAPI()
    register_accounting_routes(app, lambda: state)
    response = TestClient(app, client=("192.0.2.10", 50000)).post(
        "/v1/admin/prepare-stop",
        json={},
    )
    assert response.status_code == 403
    assert state.maintenance_state == "serving"
    assert not hasattr(state, "_sealed_accounting")


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "::ffff:127.0.0.1"])
def test_loopback_recognizes_ipv4_ipv6_and_mapped_ipv4(host):
    assert _is_loopback(host)


@pytest.mark.parametrize(
    "path, body",
    [
        ("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": {"x": 1 << 64}}),
        ("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": {"x": -(1 << 63) - 1}}),
        ("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}], "top_k": 1 << 64}),
        ("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1 << 64}),
        ("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stop": "\ud800"}),
        ("/v1/completions", {"model": "m", "prompt": "hi", "max_tokens": 1 << 64}),
        ("/v1/completions", {"model": "m", "prompt": "hi", "top_k": 1 << 64, "stream": True}),
        ("/v1/messages", {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}], "top_k": 1 << 64}),
        ("/v1/messages", {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}], "stop_sequences": ["\ud800"]}),
        ("/v1/responses", {"model": "m", "input": "hi", "top_k": 1 << 64}),
    ],
)
def test_unpackable_request_fields_return_400_without_leaking_active_uids(path, body):
    import msgpack
    from freetoken.message import BaseTokenizerMsg
    from freetoken.server.anthropic_api import register_anthropic_routes
    from freetoken.server.openai_api import register_openai_routes
    from freetoken.server.responses_api import register_responses_routes

    class _PackingPushQueue:
        async def put(self, obj) -> None:
            msgpack.packb(BaseTokenizerMsg.encoder(obj), use_bin_type=True)

    class _IdlePullQueue:
        async def get(self):
            await asyncio.Future()

    manager = FrontendManager(
        config=SimpleNamespace(
            served_model_name="m",
            max_output_tokens=128,
            mm=SimpleNamespace(text_model_only=True, disabled_encoders=frozenset()),
        ),
        send_tokenizer=_PackingPushQueue(),
        recv_tokenizer=_IdlePullQueue(),
        maintenance_state="serving",
    )
    app = FastAPI()
    register_openai_routes(app, lambda: manager, lambda: {})
    register_anthropic_routes(app, lambda: manager, lambda: {})
    register_responses_routes(app, lambda: manager, lambda: {})

    import json

    resp = TestClient(app, raise_server_exceptions=False).post(
        path,
        content=json.dumps(body).encode("ascii"),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert manager.stats.active == 0
    assert manager.ack_map == {}
    assert manager.event_map == {}

    sealed = asyncio.run(
        prepare_stop_accounting(manager, drain_timeout_s=0.0, abort_timeout_s=0.0)
    )
    assert sealed["drain_complete"] is True

