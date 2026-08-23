"""Tests for the Apple Silicon Metal backend (server/metal.py).

Pure-process tests: no GPU, no Metal engine, no torch. They cover the
backend resolver, the upstream port allocator, and the proxy routes
mounted on a FastAPI app against a real loopback upstream (uvicorn in a
thread).

Run:  PYTHONPATH=python <venv>/bin/python -m pytest tests/server/test_metal_backend.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from types import SimpleNamespace

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient


def _free_port() -> int:
    """A random free port (the 19000-19 range is littered with TIME_WAIT
    residue from real Metal runs)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

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
        handle.load_state = "ready"
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
        model_path="test-model",
    )
    handle.load_state = "ready"
    handle.load_ended_at = time.monotonic()
    metal.register_metal_proxy_routes(app, lambda: handle)
    client = TestClient(app, raise_server_exceptions=False)
    doc = client.get("/health").json()
    assert doc["status"] == "ok"
    assert doc["maintenance"] == "serving"
    # ctl parity endpoints exist
    assert client.get("/v1/requests").status_code == 200
    assert client.get("/v1/stats").status_code == 200


# ------------------------------------------------------- load supervision --

def test_health_reports_loading_progress():
    """While the engine loads, /health answers loading + byte progress in the
    CUDA contract the shell renders (phase + done_bytes/total_bytes)."""
    app = FastAPI()
    handle = metal.MetalBackendHandle(
        processes=[SimpleNamespace(poll=lambda: None)],
        upstream_base_url="http://127.0.0.1:1",
        backend="mlx",
        model_path="big-model",
    )
    # Default state is "starting" -> loading without weights total (unknown).
    doc = handle.health_doc()
    assert doc["status"] == "loading"
    assert doc["phase"] == "starting"
    assert "progress" not in doc or doc["progress"]["total_bytes"] == 0

    handle.load_state = "loading"
    handle.load_phase = "weights"
    handle.weights_bytes = 10 << 30
    metal.register_metal_proxy_routes(app, lambda: handle)
    client = TestClient(app, raise_server_exceptions=False)
    doc = client.get("/health").json()
    assert doc["status"] == "loading"
    assert doc["phase"] == "weights"
    assert doc["progress"]["total_bytes"] == 10 << 30
    assert 0 <= doc["progress"]["done_bytes"] <= 10 << 30


def test_health_reports_error_with_reason():
    app = FastAPI()
    handle = metal.MetalBackendHandle(
        processes=[SimpleNamespace(poll=lambda: None)],
        upstream_base_url="http://127.0.0.1:1",
        backend="mlx",
        model_path="m",
    )
    handle._set_state("error", error="model load failed: HTTP timeout")
    metal.register_metal_proxy_routes(app, lambda: handle)
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "error"
    assert "model load failed" in r.json()["message"]


def test_generation_503_with_phase_while_loading():
    """A generation request during the load must answer immediately with the
    loading state, not queue behind the weight load forever."""
    app = FastAPI()
    handle = metal.MetalBackendHandle(
        processes=[SimpleNamespace(poll=lambda: None)],
        upstream_base_url="http://127.0.0.1:1",
        backend="mlx",
        model_path="m",
    )
    handle.load_state = "loading"
    handle.load_phase = "weights"
    metal.register_metal_proxy_routes(app, lambda: handle)
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503
    assert "still loading" in r.json()["detail"]
    assert r.json()["phase"] == "weights"
    # Reads are not gated: /v1/models still proxies (a 502 here only because
    # the fixture's upstream address has nothing listening).
    assert client.get("/v1/models").status_code == 502


def test_upstream_resident_bytes_of_exited_process():
    """Progress probing must tolerate a dead/missing pid (best-effort)."""
    class Dead:
        pid = 999999999  # no such process

    assert metal._upstream_resident_bytes([Dead()]) == 0


def test_warm_up_generation_hits_completions():
    port = metal._pick_upstream_port(0)
    seen = []

    def stop():
        pass

    app = FastAPI()

    @app.post("/v1/completions")
    async def completions(payload: dict):
        seen.append(payload)
        return {"choices": [{"text": "x"}]}

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        metal._warm_up_generation(f"http://127.0.0.1:{port}", model_id="test-model", timeout=10)
        assert seen and seen[0]["max_tokens"] == 1
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_stream_signalled_in_body_not_accept_header():
    """The OpenAI SDK sends Accept: application/json even for streaming
    requests; streaming is signalled in the body. Detecting from the header
    alone routed streamed generations through the buffered path -- the client
    got the whole answer in one burst and live tok/s read 0."""
    import httpx as _httpx

    port = metal._pick_upstream_port(0)
    first_chunk_at: list[float] = []
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(payload: dict):
        async def gen():
            # The upstream writes + flushes each SSE event (mlx_lm does this
            # per token); a buffered proxy delivers these as one burst.
            for i in range(3):
                yield (
                    f'data: {{"choices":[{{"delta":{{"content":"{i}"}}}}]}}\n\n'
                ).encode()
                await asyncio.sleep(0.15)

        return StreamingResponse(gen(), media_type="text/event-stream")

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        handle = metal.MetalBackendHandle(
            processes=[SimpleNamespace(poll=lambda: None)],
            upstream_base_url=f"http://127.0.0.1:{port}",
            backend="mlx",
            model_path="m",
        )
        handle.load_state = "ready"
        proxy_app = FastAPI()
        metal.register_metal_proxy_routes(proxy_app, lambda: handle)

        # Real uvicorn + raw socket: TestClient buffers the whole response
        # before iteration, which would make any proxy look burst-shaped.
        proxy_port = _free_port()
        proxy_server = uvicorn.Server(
            uvicorn.Config(proxy_app, host="127.0.0.1", port=proxy_port, log_level="error")
        )
        threading.Thread(target=proxy_server.run, daemon=True).start()
        for _ in range(100):
            if proxy_server.started:
                break
            time.sleep(0.05)

        # No Accept: text/event-stream -- exactly what the OpenAI SDK sends.
        body = json.dumps(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }
        ).encode()
        with socket.create_connection(("127.0.0.1", proxy_port), timeout=5) as s:
            s.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                b"Accept: application/json\r\nContent-Type: application/json\r\n"
                b"Content-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            t0 = time.time()
            stamps: list[float] = []
            buf = b""
            while len(stamps) < 3 and time.time() - t0 < 5:
                data = s.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n\n" in buf:
                    chunk, buf = buf.split(b"\n\n", 1)
                    if b"data:" in chunk:
                        stamps.append(time.time() - t0)
        assert len(stamps) >= 3, f"expected 3 SSE events, got {len(stamps)}"
        # A buffered proxy delivers them as one burst (spread ~0); progressive
        # delivery spreads >= ~0.2s for three events 150ms apart.
        spread = stamps[-1] - stamps[0]
        assert spread >= 0.2, f"chunks delivered as one burst (spread {spread:.3f}s)"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        proxy_server.should_exit = True


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
    """A switch stops the old engine BEFORE starting the new one, and adopts
    the new engine's identity on the shared handle.

    Sequential (stop, then start) is the point: two concurrent engines
    over-commit the Metal working set and deadlock on this hardware."""
    app = FastAPI()
    old_proc = SimpleNamespace(poll=lambda: None, terminate=lambda: None)
    events: list[str] = []
    old = metal.MetalBackendHandle(
        processes=[old_proc],
        upstream_base_url="http://127.0.0.1:1",
        backend="mlx",
        model_path="old-model",
    )
    old.load_state = "ready"
    new_proc = SimpleNamespace(poll=lambda: None, terminate=lambda: None)

    real_stop = metal._stop_processes

    def seen_stop(processes):
        events.append(f"stopped:{processes is old.processes}")
        real_stop(processes)

    def fake_launch(backend, model, upstream_port=None, state_handle=None):
        events.append("launch")
        h = metal.MetalBackendHandle(
            processes=[new_proc],
            upstream_base_url="http://127.0.0.1:2",
            backend="mlx",
            model_path="new-model",
        )
        h.load_state = "ready"
        # A switch passes state_handle: the watcher publishes onto the SHARED
        # handle, so simulate that publication.
        if state_handle is not None:
            state_handle.load_state = "ready"
        return h

    monkeypatch.setattr(metal, "launch_metal_backend", fake_launch)
    monkeypatch.setattr(metal, "_stop_processes", seen_stop)
    metal.register_metal_proxy_routes(app, lambda: old)
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/v1/model/load", json={"model": "new-model"})
    assert r.status_code == 200
    assert r.json()["model"] == "new-model"
    # The old engine died before the new load began -- never both resident.
    assert events == ["stopped:True", "launch"]
    assert old.model_path == "new-model"
    assert old.upstream_base_url == "http://127.0.0.1:2"
    assert len(old.processes) == 1 and old.processes[0] is new_proc
    assert old.load_state == "ready"


def test_model_load_failure_keeps_old_model(monkeypatch):
    """A failed launch reports the error; the old engine is already gone
    (sequential switch), so the honest state is the error itself."""
    app = FastAPI()
    old_proc = SimpleNamespace(poll=lambda: None, terminate=lambda: None)
    old = metal.MetalBackendHandle(
        processes=[old_proc],
        upstream_base_url="http://127.0.0.1:1",
        backend="mlx",
        model_path="old-model",
    )

    def fake_launch(backend, model, upstream_port=None, state_handle=None):
        raise RuntimeError("download failed")

    monkeypatch.setattr(metal, "launch_metal_backend", fake_launch)
    metal.register_metal_proxy_routes(app, lambda: old)
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/v1/model/load", json={"model": "new-model"})
    assert r.status_code == 500
    assert "download failed" in r.json()["detail"]
