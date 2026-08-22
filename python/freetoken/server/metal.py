"""Apple Silicon (Metal) backends for ``ft serve``.

This module wires Apple's own, already-built Metal runtimes as the inference
engine behind FreeToken's OpenAI/Anthropic/Responses API. It does NOT port any
of the CUDA/Triton kernels (there is no macOS build of triton/flashinfer/
sglang-kernel, and FreeToken's native fast path is irreducibly CUDA). Instead it
reuses two Apple-proven upstreams:

  * ``mlx``  (``mlx_lm.server``)  -- Apple's MLX framework running on the MPS
    (Metal) GPU. OpenAI-compatible ``/v1/*`` HTTP server.
  * ``llama`` (``llama.cpp``'s ``llama-server``) -- Metal-backed GGUF server.
    OpenAI- and Anthropic-Compatible ``/v1/*`` and ``/v1/messages`` HTTP server.

FreeToken keeps serving its OpenAI/Anthropic/Responses surface on the configured
host/port; this module launches the chosen upstream as a child process and
proxies the generation routes to it. Running the CUDA scheduler path is entirely
untouched (see ``server/launch.py``), so ``ft serve`` on a CUDA box behaves
exactly as before and ``ft serve --backend mlx|llama`` re-targets to Metal.

Backend resolution rules:
  * ``cuda``  -> native FreeToken scheduler (unchanged default behaviour).
  * ``mlx``   -> mlx_lm.server (requires the ``mlx-lm`` package).
  * ``llama`` -> llama.cpp llama-server (requires the ``llama-server`` binary).
  * ``auto``  -> CUDA when available and usable; otherwise the first Metal
    runtime that is installed/importable.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from freetoken.utils import init_logger

logger = init_logger(__name__)

#: Upstream serves on a loopback port inside this range (FreeToken's own API keeps
#: the user-facing ``server_port``).
_UPSTREAM_PORT_MIN = 19000
_UPSTREAM_PORT_MAX = 19999
_ADDRESS_HEALTH_TIMEOUT_S = float(
    os.environ.get("FREETOKEN_METAL_READY_TIMEOUT", "180")
)
#: mlx httpd advertises readiness with this line on stderr; llama-server is probed
#: over HTTP. Both are also verified by a live ``/v1/models`` round-trip.
_STARTED_ONCE_TIMEOUT_S = float(os.environ.get("FREETOKEN_METAL_START_TIMEOUT", "60"))


def _pick_upstream_port(preferred: int | None) -> int:
    """Pick an upstream port for the Metal engine.

    Uses ``preferred`` when given and free; otherwise scans the reserved range
    for a free loopback port. FreeToken's own API never occupies this range (it
    defaults to 1919), so collisions are effectively limited to another Metal
    backend instance."""
    if preferred is not None and preferred > 0:
        return _claim_port(preferred) or _scan_free_port()
    return _scan_free_port()


def _claim_port(port: int) -> int | None:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return None
    return port


def _scan_free_port() -> int:
    import socket

    for port in range(_UPSTREAM_PORT_MIN, _UPSTREAM_PORT_MAX):
        if _claim_port(port) is not None:
            return port
    raise RuntimeError("no free loopback port for the Metal backend")


def _any_cuda_usable() -> bool:
    """True when the native CUDA scheduler path is usable on this host."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 -- not available at all is fine
        return False


def mlx_importable() -> bool:
    import importlib.util

    return importlib.util.find_spec("mlx_lm") is not None


def llama_binary() -> str | None:
    for exe in ("llama-server",):
        path = shutil.which(exe)
        if path:
            return path
    return None


def resolve_backend(requested: str) -> str:
    """Resolve a ``--backend`` value to a concrete choice (``cuda/mlx/llama``).

    ``cuda`` is accepted as-is. ``mlx``/``llama`` require their upstream to be
    present. ``auto`` prefers CUDA when usable, then mlx, then llama. Raises a
    clear error when the requested backend cannot run here."""
    if requested == "cuda":
        if not _any_cuda_usable():
            raise RuntimeError(
                "--backend cuda requested but no usable CUDA GPU was found "
                "on this host."
            )
        return "cuda"
    if requested == "mlx":
        if not mlx_importable():
            raise RuntimeError(
                "--backend mlx requested but mlx_lm is not importable. "
                "Install it with: uv pip install 'mlx-lm'"
            )
        return "mlx"
    if requested == "llama":
        if llama_binary() is None:
            raise RuntimeError(
                "--backend llama requested but 'llama-server' was not found "
                "on PATH. Install llama.cpp, or use --backend mlx."
            )
        return "llama"
    if requested != "auto":
        raise RuntimeError(
            f"unknown --backend {requested!r} (expected auto, cuda, mlx, or llama)"
        )
    if _any_cuda_usable():
        logger.info("backend=auto resolved to cuda (native CUDA scheduler)")
        return "cuda"
    if mlx_importable():
        logger.info("backend=auto resolved to mlx (Apple Silicon MLX)")
        return "mlx"
    if llama_binary() is not None:
        logger.info("backend=auto resolved to llama (llama.cpp Metal)")
        return "llama"
    raise RuntimeError(
        "FREETOKEN: no usable inference backend. No CUDA GPU, mlx_lm, or "
        "llama-server was found. Install mlx-lm (Apple Silicon) or llama.cpp."
    )


@dataclass
class MetalBackendHandle:
    """Handle to a launched Metal inference engine (blunt stand-in for the CUDA
    scheduler's ``BackendHandle``; the API layer only needs processes + readiness)."""
    processes: list[subprocess.Popen] = field(default_factory=list)
    upstream_base_url: str = ""
    backend: str = ""

    def terminate(self) -> None:
        for p in self.processes:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:  # noqa: BLE001 -- best-effort teardown
                continue

    def is_alive(self) -> bool:
        return any(p.poll() is None for p in self.processes)


def _wait_for_readiness(url: str, process: subprocess.Popen, *, timeout: float) -> None:
    """Poll ``<url>/v1/models`` until the upstream answers or ``timeout`` elapses.

    Also surfaces any early stdout/stderr lines so a launch failure is visible
    instead of a silent timeout."""
    deadline = time.monotonic() + timeout
    last_err = ""
    drained: list[str] = []
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Metal backend exited during startup "
                f"(code {process.returncode}): {''.join(drained[-8:])}"
            )
        try:
            proc = subprocess.run(
                ["curl", "-fsS", f"{url}/v1/models"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0:
                return
            last_err = (proc.stderr or proc.stdout or "").strip()
        except Exception as exc:  # noqa: BLE001 -- not ready yet, keep polling
            last_err = str(exc)
        if process.stdout is not None:
            line = process.stdout.readline()
            if line:
                drained.append(line)
        if process.stderr is not None:
            line = process.stderr.readline()
            if line:
                drained.append(line)
        time.sleep(0.5)
    raise RuntimeError(
        f"Metal backend at {url} did not become ready within {timeout:.0f}s. "
        f"Last probe: {last_err}. Output:\n{''.join(drained[-12:])}"
    )


def _launch_mlx(model_path: str, port: int) -> MetalBackendHandle:
    import sys

    py = sys.executable
    cmd = [
        py,
        "-m",
        "mlx_lm.server",
        "--model",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    logger.info("launching Metal backend (mlx): %s", shlex.join(cmd))
    proc = subprocess.Popen(
        cmd,
        # Own the child's stdout/stderr so we can detect startup failures and
        # drain logs; the child inherits env so HF/mlx settings pass through.
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    url = f"http://127.0.0.1:{port}"
    _wait_for_readiness(url, proc, timeout=_ADDRESS_HEALTH_TIMEOUT_S)
    return MetalBackendHandle(
        processes=[proc], upstream_base_url=url, backend="mlx"
    )


def _launch_llama(model_path: str, port: int, **kwargs: Any) -> MetalBackendHandle:
    binary = llama_binary()
    assert binary is not None
    cmd = [
        binary,
        "-m",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        # Metal backend (Apple Silicon)
        "-ngl",
        "999",
    ]
    logger.info("launching Metal backend (llama.cpp): %s", shlex.join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    url = f"http://127.0.0.1:{port}"
    _wait_for_readiness(url, proc, timeout=_ADDRESS_HEALTH_TIMEOUT_S)
    return MetalBackendHandle(
        processes=[proc], upstream_base_url=url, backend="llama"
    )


def launch_metal_backend(
    backend: str, model_path: str, upstream_port: int | None = None
) -> MetalBackendHandle:
    port = _pick_upstream_port(upstream_port)
    if backend == "mlx":
        return _launch_mlx(model_path, port)
    if backend == "llama":
        return _launch_llama(model_path, port)
    raise RuntimeError(f"unsupported Metal backend: {backend!r}")


# --- HTTP proxy over the FreeToken API -------------------------------------


async def _proxy_stream(
    upstream_base_url: str,
    path: str,
    body: bytes,
    headers: dict[str, str],
) -> AsyncIterator[bytes]:
    """Forward a request to the upstream and stream the (possibly SSE) bytes back."""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", f"{upstream_base_url}{path}", content=body, headers=headers
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk


def register_metal_proxy_routes(
    app: FastAPI, get_backend: Any
) -> None:
    """Proxy generation routes to the Metal upstream.

    The user-facing surface (``/v1/chat/completions``, ``/v1/completions``,
    ``/v1/models``, and Anthropic's ``/v1/messages``) is forwarded verbatim to
    the upstream, which already implements the OpenAI/Anthropic-compatible
    protocol. Streaming responses pass through as SSE."""

    @app.post("/v1/chat/completions")
    async def proxy_chat(request: Request):
        return await _forward(request, get_backend)

    @app.post("/v1/completions")
    async def proxy_completions(request: Request):
        return await _forward(request, get_backend)

    @app.post("/v1/messages")
    async def proxy_messages(request: Request):
        return await _forward(request, get_backend)

    @app.post("/v1/responses")
    async def proxy_responses(request: Request):
        return await _forward(request, get_backend)

    @app.post("/v1/embeddings")
    async def proxy_embeddings(request: Request):
        return await _forward(request, get_backend)

    @app.get("/v1/models")
    async def proxy_models(request: Request):
        return await _forward(request, get_backend, method="GET")

    @app.get("/health")
    async def metal_health(request: Request):
        """Lightweight health: checks the upstream Flask/uvicorn is alive. Keeps the
        desktop/shell poll working with the same semantics as the CUDA path."""
        handle = get_backend()
        if handle is None or not handle.is_alive():
            return JSONResponse({"status": "down"}, status_code=503)
        return {"status": "ok"}


def _strip_host_header(headers: dict[str, str]) -> dict[str, str]:
    out = {k: v for k, v in headers.items() if k.lower() not in {"host", "content-length"}}
    out["accept"] = headers.get("accept", "application/json")
    return out


async def _forward(
    request: Request,
    get_backend: Any,
    method: str = "POST",
) -> Response:
    handle = get_backend()
    if handle is None or not handle.is_alive():
        return JSONResponse(
            {"detail": "Metal backend is not running"}, status_code=503
        )
    upstream = handle.upstream_base_url
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"

    body = await request.body()
    headers = _strip_host_header(dict(request.headers))
    stream = "text/event-stream" in (headers.get("accept") or "")

    if stream:
        return StreamingResponse(
            _proxy_stream(upstream, path, body, headers),
            media_type="text/event-stream",
        )

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            r = await client.request(
                method, f"{upstream}{path}", content=body, headers=headers
            )
    except httpx.HTTPError as exc:
        return JSONResponse({"detail": f"Metal upstream error: {exc}"}, status_code=502)
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type"),
    )
