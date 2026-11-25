"""--tp start-up against a real server: the first request sent the moment the server reports ready is answered.

Rank 0 fans each request out to the other ranks over ZMQ PUB/SUB. A request published before a rank's subscription
arrived was dropped there, leaving rank 0 in the scheduler's broadcast and the other rank waiting for the request until
the gloo timeout. --tp 2 --moe-strategy hybrid with numa host pools hit it on about every other start, so this starts
the server a few times and sends one request at the first /health "ok" of each.

Gated behind ``needs_weights`` and two GPUs:

  FREETOKEN_TP_TEST_MODEL     local split-expert checkpoint (DeepSeek-V4-Flash, GLM-5.3-Flash); falls back to FREETOKEN_TEST_MODEL
  FREETOKEN_TP_STARTS         starts to try (default 3)
  FREETOKEN_TP_BOOT_TIMEOUT   seconds to wait for "ok" (default 1200)
  FREETOKEN_TP_EXTRA_ARGS     additional ft serve flags (e.g. --text-model-only --max-seq-len 8192)
"""

from __future__ import annotations

import json
import os
import shlex
import signal
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


def _model_dir() -> Path | None:
    value = os.environ.get("FREETOKEN_TP_TEST_MODEL") or os.environ.get("FREETOKEN_TEST_MODEL")
    return Path(value).expanduser() if value else None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _health(base: str) -> str | None:
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as response:
            return json.loads(response.read()).get("status")
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None  # not listening yet


def _complete(base: str) -> str:
    request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps({"model": "tp-test", "prompt": "def fibonacci(n):", "max_tokens": 8, "temperature": 0.0}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # a few tokens take well under a second here
        return json.loads(response.read())["choices"][0]["text"]


def _stop(proc: subprocess.Popen) -> None:
    """Kill every rank: a rank that outlives the test holds its GPU and fails the next start's memory check."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(1.0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_first_request_after_ready_is_answered_under_tp(tmp_path):
    model_dir = _model_dir()
    if model_dir is None:
        pytest.skip("set FREETOKEN_TP_TEST_MODEL to a local split-expert checkpoint (DeepSeek-V4-Flash, GLM-5.3-Flash)")
    if not model_dir.exists():
        pytest.skip(f"model is not downloaded: {model_dir}")

    boot_timeout = float(os.environ.get("FREETOKEN_TP_BOOT_TIMEOUT", "1200"))
    for start in range(int(os.environ.get("FREETOKEN_TP_STARTS", "3"))):
        port = _free_port()
        base = f"http://127.0.0.1:{port}"
        log = (tmp_path / f"serve_{start}.log").open("w")
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "freetoken",
                "--model-path", str(model_dir),
                "--served-model-name", "tp-test",
                "--host", "127.0.0.1",
                "--port", str(port),
                "--tp", "2",
                "--moe-strategy", "hybrid",
                "--moe-host-pool", "numa",
                *shlex.split(os.environ.get("FREETOKEN_TP_EXTRA_ARGS", "")),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": "python"},
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + boot_timeout
            while _health(base) != "ok":
                assert proc.poll() is None, f"start {start}: server exited with {proc.returncode}, see {log.name}"
                assert time.monotonic() < deadline, f"start {start}: server never reported ok, see {log.name}"
                time.sleep(0.2)
            try:
                assert _complete(base)
            except (urllib.error.URLError, TimeoutError) as e:
                pytest.fail(f"start {start}: the first request after ready was not answered ({e}), see {log.name}")
        finally:
            _stop(proc)
            log.close()
