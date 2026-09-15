"""Runtime cache rebuild against a real server.

Boots ``ft serve`` on a small checkpoint and drives POST /v1/cache/rebuild for real: the whole
chain of HTTP route -> control message -> scheduler idle gate -> engine teardown -> pool resize
-> page-table refresh -> CUDA-graph re-capture, then checks the server still generates.

This is the primary gate for the rebuild path. The destructive orchestration has no in-process
seam worth stubbing -- what matters is that a rebuild lands on a live engine and the server keeps
serving afterwards, which only a real boot can show. The narrower pieces that a rebuild cannot
reach from outside (maintenance-state latching, pool identity on resize) stay in
server/test_rebuild_maintenance.py and kvcache/test_kv_cache_rebuild.py.

Gated behind ``needs_weights``:

  FREETOKEN_REBUILD_TEST_MODEL  small local model dir (falls back to FREETOKEN_TEST_MODEL)
  FREETOKEN_REBUILD_MIN_FREE_GIB  free-GPU-memory gate (default 20)
  FREETOKEN_REBUILD_BOOT_TIMEOUT  seconds to wait for "serving" (default 300)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.needs_weights

# One boot serves every check below, so the pages a rebuild grows into have to fit on top of the
# model. Small and round: the engine sizes the rest from what is left.
BOOT_PAGES = 4000
GROWN_PAGES = 6000


def _model_dir() -> Path | None:
    value = os.environ.get("FREETOKEN_REBUILD_TEST_MODEL") or os.environ.get("FREETOKEN_TEST_MODEL")
    return Path(value).expanduser() if value else None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _free_gpu_gib() -> float:
    free, _total = torch.cuda.mem_get_info()
    return free / (1 << 30)


def _post(base: str, path: str, payload: dict, timeout: float = 180.0) -> tuple[int, dict]:
    """(status code, decoded body). A refused rebuild answers 503 with a JSON body, so the
    non-2xx path carries signal and must not raise."""
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(base: str, path: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(base + path, timeout=timeout) as response:
        return json.loads(response.read())


def _wait_until_serving(base: str, proc: subprocess.Popen, deadline: float) -> None:
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            if _get(base, "/v1/cache/status")["state"] == "serving":
                return
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
            pass  # not listening yet, or still loading
        time.sleep(2.0)
    raise TimeoutError("server never reached the serving state")


def _generate(base: str) -> int:
    """Tokens the engine decoded -- the point is that it still runs, not what it says.

    Counts rather than reads the reply. Which field the text lands in depends on the
    checkpoint: ``enable_thinking`` is read by the qwen/glm/gemma/dsv4 templates and is
    inert for gpt-oss, whose Harmony channels are not gated by it, so its trace goes to
    ``reasoning_content`` and ``content`` can be empty on a perfectly healthy engine.
    ``usage.completion_tokens`` is accumulated from the scheduler's own acks, upstream of
    every reasoning parser, so it says decode ran without depending on any of that.
    """
    code, body = _post(
        base,
        "/v1/chat/completions",
        {
            "model": "rebuild-test",
            "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
            "max_tokens": 32,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert code == 200, body
    return int(body["usage"]["completion_tokens"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cache rebuild e2e needs CUDA")
def test_cache_rebuild_resizes_a_live_engine_and_keeps_serving(tmp_path):
    model_dir = _model_dir()
    if model_dir is None:
        pytest.skip("set FREETOKEN_REBUILD_TEST_MODEL to a small local model directory")
    if not model_dir.is_dir():
        pytest.skip(f"model is not downloaded: {model_dir}")

    min_free = float(os.environ.get("FREETOKEN_REBUILD_MIN_FREE_GIB", "20"))
    free_gib = _free_gpu_gib()
    if free_gib < min_free:
        pytest.skip(f"needs ~{min_free:.0f} GiB free; only {free_gib:.2f} GiB")

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    log = (tmp_path / "serve.log").open("w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "freetoken",
            "--model-path", str(model_dir),
            "--served-model-name", "rebuild-test",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--num-pages", str(BOOT_PAGES),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": "python"},
    )
    try:
        boot_timeout = float(os.environ.get("FREETOKEN_REBUILD_BOOT_TIMEOUT", "300"))
        _wait_until_serving(base, proc, time.monotonic() + boot_timeout)

        geometry = _get(base, "/v1/cache/status")["geometry"]
        assert geometry["num_pages"] == BOOT_PAGES
        assert _generate(base) > 0  # a baseline generation, before anything is torn down

        # 1. Grow the KV pool. The reply carries the geometry the engine actually landed on.
        code, reply = _post(base, "/v1/cache/rebuild", {"num_pages": GROWN_PAGES})
        assert (code, reply["status"]) == (200, "ok"), reply
        assert reply["num_pages"] == GROWN_PAGES
        status = _get(base, "/v1/cache/status")
        assert status["state"] == "serving"
        assert status["geometry"]["num_pages"] == GROWN_PAGES
        assert status["last_rebuild"]["status"] == "ok"
        # Graphs were re-captured against the new pool: decoding still works.
        assert _generate(base) > 0

        # 2. The second axis, if this checkpoint has a GDN state pool. Resizing it moves a
        #    different pool through the same teardown/re-capture path.
        mamba_slots = status["geometry"]["num_mamba_slots"]
        if mamba_slots:
            grown_slots = mamba_slots + 8
            code, reply = _post(base, "/v1/cache/rebuild", {"num_mamba_slots": grown_slots})
            assert (code, reply["status"]) == (200, "ok"), reply
            geometry = _get(base, "/v1/cache/status")["geometry"]
            assert geometry["num_mamba_slots"] == grown_slots
            # A pool-only resize must not disturb the pool the previous rebuild grew.
            assert geometry["num_pages"] == GROWN_PAGES
            assert _generate(base) > 0

        # 3. An unfittable target is rejected BEFORE anything is freed, and the engine that was
        #    serving a moment ago keeps serving on its old cache.
        code, reply = _post(base, "/v1/cache/rebuild", {"num_pages": 99_999_999})
        # A refusal is a 503 carrying the reason, not a silent 200 the client would read as done.
        assert (code, reply["status"]) == (503, "rejected"), reply
        assert reply["error"]
        status = _get(base, "/v1/cache/status")
        assert status["state"] == "serving"
        assert status["geometry"]["num_pages"] == GROWN_PAGES  # untouched by the rejection
        assert _generate(base) > 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        log.close()
