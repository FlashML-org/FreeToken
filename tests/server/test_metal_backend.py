"""Tests for the Apple Silicon Metal backend (server/metal.py).

Pure-process tests: no GPU, no Metal engine, no torch. They cover the
backend resolver, the upstream port allocator, and the proxy routes
mounted on a FastAPI app against a real loopback upstream (uvicorn in a
thread).

Run:  PYTHONPATH=python <venv>/bin/python -m pytest tests/server/test_metal_backend.py -v
"""

from __future__ import annotations

import os
import sys
import threading
from types import SimpleNamespace

import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from freetoken.server import metal  # noqa: E402


# ---------------------------------------------------------------- resolver --

def test_resolve_backend_explicit_passthrough():
    assert metal.resolve_backend("mlx") == "mlx"
    assert metal.resolve_backend("llama") == "llama"


def test_resolve_backend_rejects_unknown():
    try:
        metal.resolve_backend("tpu")
    except RuntimeError as e:
        assert "unknown --backend" in str(e)
    else:
        raise AssertionError("expected RuntimeError for unknown backend")


def test_pick_upstream_port_defaults_into_reserved_range():
    for preferred in (None, 0):
        port = metal._pick_upstream_port(preferred)
        assert metal._UPSTREAM_PORT_MIN <= port <= metal._UPSTREAM_PORT_MAX, port


def test_pick_upstream_port_prefers_free_preferred_port():
    # 19099 is inside the range and effectively never occupied in CI.
    assert metal._pick_upstream_port(19099) == 19099


# ------------------------------------------------------------- proxy routes --

def _start_upstream(port: int):
    """Serve a tiny OpenAI-ish upstream on a loopback port; return a stop()."""
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "test-model"}]}

    @app.post("/v1/chat/completions")
    async def chat():
        return {
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hello from upstream"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
        }

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):  # wait for the listener
        if server.started:
            break
        import time

        time.sleep(0.05)
    assert server.started, "upstream test server failed to start"

    def stop():
        server.should_exit = True
        thread.join(timeout=5)

    return stop


def test_proxy_roundtrip_to_real_upstream():
    port = metal._pick_upstream_port(0)
    stop = _start_upstream(port)
    try:
        handle = metal.MetalBackendHandle(
            processes=[SimpleNamespace(poll=lambda: None)],
            upstream_base_url=f"http://127.0.0.1:{port}",
            backend="mlx",
        )
        assert handle.is_alive()

        proxy = FastAPI()
        metal.register_metal_proxy_routes(proxy, lambda: handle)
        client = TestClient(proxy, raise_server_exceptions=False)

        r = client.get("/v1/models")
        assert r.status_code == 200
        assert [m["id"] for m in r.json()["data"]] == ["test-model"]

        r = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "hello from upstream"

        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
    finally:
        stop()


def test_proxy_503_when_backend_not_alive():
    handle = metal.MetalBackendHandle(
        processes=[SimpleNamespace(poll=lambda: 1)],  # exited
        upstream_base_url="http://127.0.0.1:1",
        backend="mlx",
    )
    assert not handle.is_alive()

    proxy = FastAPI()
    metal.register_metal_proxy_routes(proxy, lambda: handle)
    client = TestClient(proxy, raise_server_exceptions=False)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503


def test_handle_terminate_is_safe_on_empty():
    handle = metal.MetalBackendHandle()
    handle.terminate()  # no processes -> must not raise
    assert not handle.is_alive()


# ----------------------------------------------------- wire-format translation --

def test_reasoning_field_renamed_streaming_and_body():
    """mlx_lm's thinking channel is `reasoning`; FreeToken's clients (shell,
    bench) read `reasoning_content` (the vLLM/SGLang name)."""
    chunk = b'data: {"choices": [{"delta": {"reasoning": "think"}}]}'
    out = metal._rewrite_reasoning_field(chunk)
    assert b'"reasoning_content":' in out and b'"reasoning":' not in out
    # value text that merely contains the word stays untouched
    chunk2 = b'data: {"choices": [{"delta": {"content": "the reasoning: here"}}]}'
    assert metal._rewrite_reasoning_field(chunk2) == chunk2


def test_health_reports_maintenance_serving():
    """Tools gate on maintenance == "serving" (bench wait_ready, daemon)."""
    app = FastAPI()
    handle = metal.MetalBackendHandle(
        processes=[SimpleNamespace(poll=lambda: None)],
        upstream_base_url="http://127.0.0.1:1",
        backend="mlx",
    )
    metal.register_metal_proxy_routes(app, lambda: handle)
    client = TestClient(app, raise_server_exceptions=False)
    doc = client.get("/health").json()
    assert doc["status"] == "ok"
    assert doc["maintenance"] == "serving"
    # ctl parity endpoints exist
    assert client.get("/v1/requests").status_code == 200
    assert client.get("/v1/stats").status_code == 200


# --------------------------------------------------------- model switching --

def test_model_load_requires_model_field():
    app = FastAPI()
    handle = metal.MetalBackendHandle(
        processes=[SimpleNamespace(poll=lambda: None)],
        upstream_base_url="http://127.0.0.1:1",
        backend="mlx",
    )
    metal.register_metal_proxy_routes(app, lambda: handle)
    client = TestClient(app, raise_server_exceptions=False)

    # missing field
    r = client.post("/v1/model/load", json={"nomodel": "x"})
    assert r.status_code == 400
    # bad JSON body
    r = client.post("/v1/model/load", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_model_load_same_model_is_a_noop():
    app = FastAPI()
    handle = metal.MetalBackendHandle(
        processes=[SimpleNamespace(poll=lambda: None)],
        upstream_base_url="http://127.0.0.1:1",
        backend="mlx",
        model_path="old-model",
    )
    metal.register_metal_proxy_routes(app, lambda: handle)
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/v1/model/load", json={"model": "old-model"})
    assert r.status_code == 200
    assert r.json()["detail"] == "already serving this model"


def test_model_load_switches_to_new_handle(monkeypatch):
    """A successful switch swaps processes/upstream/model atomically."""
    app = FastAPI()
    old = metal.MetalBackendHandle(
        processes=[SimpleNamespace(poll=lambda: None)],
        upstream_base_url="http://127.0.0.1:1",
        backend="mlx",
        model_path="old-model",
    )
    new_proc = SimpleNamespace(poll=lambda: None, terminate=lambda: None)

    def fake_launch(backend, model, upstream_port=None):
        assert (backend, model) == ("mlx", "new-model")
        return metal.MetalBackendHandle(
            processes=[new_proc],
            upstream_base_url="http://127.0.0.1:2",
            backend="mlx",
            model_path="new-model",
        )

    monkeypatch.setattr(metal, "launch_metal_backend", fake_launch)
    metal.register_metal_proxy_routes(app, lambda: old)
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/v1/model/load", json={"model": "new-model"})
    assert r.status_code == 200
    assert old.model_path == "new-model"
    assert old.upstream_base_url == "http://127.0.0.1:2"
    assert len(old.processes) == 1 and old.processes[0] is new_proc


def test_model_load_failure_keeps_old_model(monkeypatch):
    """A failed launch must leave the old engine serving."""
    app = FastAPI()
    old = metal.MetalBackendHandle(
        processes=[SimpleNamespace(poll=lambda: None)],
        upstream_base_url="http://127.0.0.1:1",
        backend="mlx",
        model_path="old-model",
    )

    def fake_launch(backend, model, upstream_port=None):
        raise RuntimeError("download failed")

    monkeypatch.setattr(metal, "launch_metal_backend", fake_launch)
    metal.register_metal_proxy_routes(app, lambda: old)
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/v1/model/load", json={"model": "new-model"})
    assert r.status_code == 500
    assert "download failed" in r.json()["detail"]
    assert old.model_path == "old-model"  # unchanged
    assert old.upstream_base_url == "http://127.0.0.1:1"
