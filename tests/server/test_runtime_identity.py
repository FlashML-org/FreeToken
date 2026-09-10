"""CPU-only tests for the loopback runtime identity control plane."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from freetoken.server.accounting import prepare_stop_accounting, register_accounting_routes
from freetoken.server.api_server import (
    FrontendManager,
    _TRUSTED_PROXY_HOSTS,
    _refresh_shell_runtime_identity,
    _uvicorn_config,
    dispatch_rebuild,
)
from freetoken.server.args import parse_args
from freetoken.server.control_api import (
    RUNTIME_IDENTITY_SCHEMA,
    build_runtime_identity,
    build_runtime_identity_snapshot,
    register_control_routes,
)


class _Config:
    def to_dict(self) -> dict:
        return {"torch_dtype": "bfloat16"}


def _parse(*extra: str):
    with patch("freetoken.utils.cached_load_hf_config", return_value=_Config()):
        parsed, run_shell = parse_args(["--model", "/private/models/model-a", *extra])
    assert run_shell is False
    return parsed


def _state(*, nonce: str | None = "runtime_nonce_123456") -> SimpleNamespace:
    return SimpleNamespace(
        instance_id="instance-1",
        maintenance_state="loading",
        fatal_error=None,
        config=SimpleNamespace(
            launch_nonce=nonce,
            served_model_name="model-a",
            model_path="/private/models/model-a",
        ),
        runtime_identity=None,
    )


def test_launch_nonce_is_optional_and_accepts_only_bounded_url_safe_text():
    assert _parse().launch_nonce is None
    assert _parse("--launch-nonce", "runtime_nonce-1234").launch_nonce == "runtime_nonce-1234"

    for invalid in ("short", "contains space 123", "slash/value/123456", "x" * 129):
        with pytest.raises(SystemExit) as exc_info:
            _parse("--launch-nonce", invalid)
        assert exc_info.value.code == 2


def test_launch_nonce_help_is_available_without_a_model_or_toolchain_resolution(capsys):
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--help"])
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--launch-nonce" in help_text
    assert "non-secret supervisor correlation token" in " ".join(help_text.split())


def test_snapshot_has_runtime_owned_process_facts_and_structured_worker_roles():
    state = _state()
    processes = [
        SimpleNamespace(name="freetoken-TP0-scheduler", pid=os.getpid()),
        SimpleNamespace(name="freetoken-detokenizer-0", pid=os.getpid()),
        SimpleNamespace(name="freetoken-tokenizer-2", pid=os.getpid()),
    ]

    snapshot = build_runtime_identity_snapshot(state, processes)
    state.runtime_identity = snapshot
    doc = build_runtime_identity(state)

    assert doc is not None
    assert doc["schema"] == RUNTIME_IDENTITY_SCHEMA
    assert doc["schema_version"] == 1
    assert doc["correlation"] == {"launch_nonce": "runtime_nonce_123456"}
    assert doc["runtime"]["instance_id"] == "instance-1"
    frontend = doc["runtime"]["frontend"]
    assert frontend["role"] == "frontend"
    assert frontend["pid"] == os.getpid()
    assert frontend["pgid"] == os.getpgid(os.getpid())
    assert isinstance(frontend["starttime"], int)
    assert frontend["starttime"] > 0
    assert [(worker["role"], worker["index"]) for worker in doc["runtime"]["workers"]] == [
        ("scheduler", 0),
        ("detokenizer", 0),
        ("tokenizer", 2),
    ]
    assert all(worker["pid"] == os.getpid() for worker in doc["runtime"]["workers"])
    assert doc["lifecycle"] == {"status": "loading", "maintenance_state": "loading"}


def test_snapshot_is_privacy_narrow_and_missing_nonce_is_explicit():
    state = _state(nonce=None)
    state.runtime_identity = build_runtime_identity_snapshot(
        state, [SimpleNamespace(name="unknown", pid=None)]
    )
    doc = build_runtime_identity(state)
    assert doc is not None
    assert doc["correlation"]["launch_nonce"] is None
    assert doc["runtime"]["workers"][0] == {
        "worker_id": "unknown",
        "role": "worker",
        "index": None,
        "pid": None,
        "pgid": None,
        "starttime": None,
    }

    encoded = json.dumps(doc, sort_keys=True)
    assert "/private/" not in encoded
    for forbidden in ("argv", "environment", "executable", "model_path", "source_hash"):
        assert forbidden not in encoded


@pytest.mark.parametrize(
    ("maintenance", "fatal", "expected"),
    [
        ("loading", None, "loading"),
        ("serving", None, "ready"),
        ("rebuilding", None, "ready"),
        ("failed", None, "failed"),
        ("stopping", None, "stopping"),
        ("serving", "worker died", "failed"),
    ],
)
def test_lifecycle_changes_without_mutating_bound_process_identity(
    maintenance: str, fatal: str | None, expected: str
):
    state = _state()
    state.runtime_identity = build_runtime_identity_snapshot(state, [])
    bound_runtime = json.dumps(state.runtime_identity["runtime"], sort_keys=True)
    state.maintenance_state = maintenance
    state.fatal_error = fatal

    doc = build_runtime_identity(state)
    assert doc is not None
    assert doc["lifecycle"] == {
        "status": expected,
        "maintenance_state": maintenance,
    }
    assert json.dumps(doc["runtime"], sort_keys=True) == bound_runtime


def test_frontend_stop_is_terminal_against_late_ready_callback():
    manager = FrontendManager(
        config=SimpleNamespace(served_model_name="model-a", launch_nonce=None),
        send_tokenizer=None,
        recv_tokenizer=None,
    )
    manager.mark_stopping()
    assert manager.maintenance_state == "stopping"
    assert manager.mark_ready() is False
    assert manager.maintenance_state == "stopping"
    assert manager.ready_at is None

    manager.maintenance_state = "failed"
    manager.mark_stopping()
    assert manager.maintenance_state == "failed"


def test_prepare_stop_cannot_overwrite_a_terminal_frontend_failure():
    manager = FrontendManager(
        config=SimpleNamespace(served_model_name="model-a", launch_nonce=None),
        send_tokenizer=None,
        recv_tokenizer=None,
        maintenance_state="serving",
    )
    manager.mark_failed("scheduler exited")

    result = asyncio.run(prepare_stop_accounting(manager))

    assert result["drain_complete"] is True
    assert manager.maintenance_state == "failed"
    assert manager.fatal_error == "scheduler exited"


def _rebuild_reply(request_id: str, status: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        status=status,
        moe_cache_size=0,
        num_pages=0,
        mamba_slots=0,
        num_swa_pages=0,
        error=None,
    )


def test_stop_is_terminal_against_late_rebuild_reply_and_dispatch_rollback():
    stopped = SimpleNamespace(
        maintenance_state="stopping",
        fatal_error=None,
        rebuild_futures={},
        last_rebuild=None,
    )
    FrontendManager._resolve_rebuild(stopped, _rebuild_reply("late", "ok"))
    assert stopped.maintenance_state == "stopping"

    async def should_not_send(_msg) -> None:
        raise AssertionError("stopping engine must reject before dispatch")

    stopped.send_one = should_not_send
    rejected = asyncio.run(dispatch_rebuild(stopped, moe_cache_size=8, num_pages=None))
    assert rejected["status"] == "rejected"
    assert stopped.maintenance_state == "stopping"
    assert stopped.rebuild_futures == {}

    async def stop_then_fail(_msg) -> None:
        stopped.maintenance_state = "stopping"
        raise RuntimeError("transport closed during stop")

    stopped.maintenance_state = "serving"
    stopped.send_one = stop_then_fail
    result = asyncio.run(dispatch_rebuild(stopped, moe_cache_size=8, num_pages=None))
    assert result["status"] == "failed"
    assert stopped.maintenance_state == "stopping"


def test_uvicorn_pins_proxy_trust_to_loopback_peers(monkeypatch):
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")
    config = _uvicorn_config("0.0.0.0", 1919)
    assert config.proxy_headers is True
    assert config.forwarded_allow_ips == list(_TRUSTED_PROXY_HOSTS)


def test_runtime_identity_route_is_loopback_only_and_fails_closed_when_unbound():
    state = _state()
    state.runtime_identity = build_runtime_identity_snapshot(state, [])
    app = FastAPI(version="test")
    register_control_routes(app, lambda: state)

    loopback = TestClient(app, client=("127.0.0.1", 50000)).get("/v1/runtime/identity")
    assert loopback.status_code == 200
    assert loopback.json()["runtime"]["instance_id"] == "instance-1"

    remote = TestClient(app, client=("192.0.2.10", 50000)).get(
        "/v1/runtime/identity",
        headers={"X-Forwarded-For": "127.0.0.1"},
    )
    assert remote.status_code == 403
    assert remote.json() == {"error": "loopback access required"}

    proxy_app = ProxyHeadersMiddleware(app, trusted_hosts=list(_TRUSTED_PROXY_HOSTS))
    proxied_remote = TestClient(proxy_app, client=("127.0.0.1", 50000)).get(
        "/v1/runtime/identity",
        headers={"X-Forwarded-For": "192.0.2.10"},
    )
    assert proxied_remote.status_code == 403
    assert proxied_remote.json() == {"error": "loopback access required"}

    state.runtime_identity = None
    missing = TestClient(app, client=("::1", 50000)).get("/v1/runtime/identity")
    assert missing.status_code == 503
    assert missing.json() == {"error": "runtime identity is not bound"}


def test_loopback_proxy_preserves_remote_client_for_admin_authorization():
    state = FrontendManager(
        config=SimpleNamespace(served_model_name="model-a", launch_nonce=None),
        send_tokenizer=None,
        recv_tokenizer=None,
        maintenance_state="serving",
    )
    app = FastAPI()
    register_accounting_routes(app, lambda: state)
    proxy_app = ProxyHeadersMiddleware(app, trusted_hosts=list(_TRUSTED_PROXY_HOSTS))

    response = TestClient(proxy_app, client=("127.0.0.1", 50000)).post(
        "/v1/admin/prepare-stop",
        headers={"X-Forwarded-For": "192.0.2.10"},
        json={},
    )

    assert response.status_code == 403
    assert state.maintenance_state == "serving"


def test_shell_identity_refreshes_worker_pgid_at_ready_boundary():
    state = _state()
    state.config.shell_mode = True
    state.backend_processes = [SimpleNamespace(name="freetoken-TP0-scheduler", pid=123)]
    state.runtime_identity = {
        "runtime": {"workers": [{"pid": 123, "pgid": 100, "starttime": 456}]}
    }

    with (
        patch("freetoken.server.control_api.os.getpid", return_value=999),
        patch("freetoken.server.control_api.read_starttime", side_effect=[456, 111]),
        patch("freetoken.server.control_api.proc_pgid", side_effect=[123, 999]),
    ):
        _refresh_shell_runtime_identity(state)

    assert state.runtime_identity["runtime"]["frontend"] == {
        "role": "frontend",
        "pid": 999,
        "pgid": 999,
        "starttime": 111,
    }
    assert state.runtime_identity["runtime"]["workers"][0]["pgid"] == 123


def test_ready_publication_cannot_pair_with_stale_shell_identity():
    state = FrontendManager(
        config=SimpleNamespace(
            served_model_name="model-a",
            launch_nonce="runtime_nonce_123456",
            shell_mode=True,
        ),
        send_tokenizer=None,
        recv_tokenizer=None,
    )
    state.backend_processes = [
        SimpleNamespace(name="freetoken-TP0-scheduler", pid=os.getpid())
    ]
    state.runtime_identity = build_runtime_identity_snapshot(state, state.backend_processes)
    state.runtime_identity["runtime"]["workers"][0]["pgid"] = -1

    copy_started = threading.Event()
    allow_copy = threading.Event()
    original_deepcopy = copy.deepcopy

    def blocking_deepcopy(value):
        copy_started.set()
        assert allow_copy.wait(timeout=2)
        return original_deepcopy(value)

    result = {}
    with patch("freetoken.server.control_api.copy.deepcopy", side_effect=blocking_deepcopy):
        reader = threading.Thread(
            target=lambda: result.setdefault("doc", build_runtime_identity(state))
        )
        reader.start()
        assert copy_started.wait(timeout=2)

        writer = threading.Thread(target=state.mark_ready)
        writer.start()
        allow_copy.set()
        reader.join(timeout=2)
        writer.join(timeout=2)

    assert not reader.is_alive()
    assert not writer.is_alive()
    assert result["doc"]["lifecycle"]["status"] == "loading"
    assert result["doc"]["runtime"]["workers"][0]["pgid"] == -1

    ready = build_runtime_identity(state)
    assert ready is not None
    assert ready["lifecycle"]["status"] == "ready"
    assert ready["runtime"]["workers"][0]["pgid"] == os.getpgid(os.getpid())


def test_run_api_server_binds_identity_after_worker_spawn(monkeypatch):
    from freetoken.server import api_server

    class Queue:
        def __init__(self, *_args, **_kwargs):
            pass

        def stop(self) -> None:
            pass

    class Thread:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def start(self) -> None:
            pass

    observed = {}

    def run(_app, **kwargs) -> None:
        observed["uvicorn_kwargs"] = kwargs
        observed["identity"] = api_server._GLOBAL_STATE.runtime_identity

    config = SimpleNamespace(
        sampling_defaults="none",
        use_dummy_weight=True,
        model_path="/private/models/model-a",
        served_model_name="model-a",
        launch_nonce="runtime_nonce_123456",
        server_host="127.0.0.1",
        server_port=1919,
        cors_origins="",
        zmq_frontend_addr="inproc://frontend",
        zmq_tokenizer_addr="inproc://tokenizer",
        frontend_create_tokenizer_link=True,
    )
    worker = SimpleNamespace(name="freetoken-TP0-scheduler", pid=os.getpid())
    handle = SimpleNamespace(processes=[worker])

    monkeypatch.setattr(api_server, "ZmqAsyncPullQueue", Queue)
    monkeypatch.setattr(api_server, "ZmqAsyncPushQueue", Queue)
    monkeypatch.setattr(api_server, "install_cors", lambda *_args: None)
    monkeypatch.setattr(api_server, "init_request_logging", lambda: None)
    monkeypatch.setattr(api_server, "install_polling_access_log_filter", lambda: None)
    monkeypatch.setattr(api_server.threading, "Thread", Thread)
    monkeypatch.setattr(api_server.uvicorn, "run", run)

    prior = api_server._GLOBAL_STATE
    api_server._GLOBAL_STATE = None
    try:
        api_server.run_api_server(config, lambda: handle, run_shell=False)
        identity = observed["identity"]
        assert identity["correlation"]["launch_nonce"] == "runtime_nonce_123456"
        assert identity["runtime"]["workers"][0]["pid"] == os.getpid()
        assert observed["uvicorn_kwargs"]["proxy_headers"] is True
        assert observed["uvicorn_kwargs"]["forwarded_allow_ips"] == list(
            api_server._TRUSTED_PROXY_HOSTS
        )
    finally:
        api_server._GLOBAL_STATE = prior
