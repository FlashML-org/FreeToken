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
import glob
import json
import os
import queue
import shlex
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
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
#: How long the eager warm-up generation may take. This is the real model-load
#: budget (a multi-shard download or a 50 GiB weight load happens inside it):
#: mlx_lm queues the request in its generation thread and answers only once the
#: weights are resident, so the warm-up doubles as a supervised load. A hang
#: here (e.g. a dead CDN socket) surfaces as an error after this timeout
#: instead of blocking the user's first request forever.
_LOAD_TIMEOUT_S = float(os.environ.get("FREETOKEN_METAL_LOAD_TIMEOUT", "3600"))
#: mlx httpd advertises readiness with this line on stderr; llama-server is probed
#: over HTTP. Both are also verified by a live ``/v1/models`` round-trip.
_STARTED_ONCE_TIMEOUT_S = float(os.environ.get("FREETOKEN_METAL_START_TIMEOUT", "60"))
_POLL_INTERVAL_S = 0.5


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
    scheduler's ``BackendHandle``; the API layer only needs processes + readiness).

    ``load_state`` is the lifecycle the /health route reports: ``starting``
    (process spawned, port not yet listening) -> ``loading`` (upstream answers
    /v1/models but weights are still coming down / into memory) -> ``ready``
    (a warm-up generation succeeded: the engine can actually generate) ->
    ``error`` (process died, load timed out, or warm-up failed). Shared with
    the proxy threads via a lock so /health never tears while reading it."""
    processes: list[subprocess.Popen] = field(default_factory=list)
    upstream_base_url: str = ""
    backend: str = ""
    model_path: str = ""
    load_state: str = "starting"
    load_phase: str = ""
    load_error: str = ""
    load_started_at: float = 0.0
    load_ended_at: float = 0.0
    weights_bytes: int = 0
    # The CUDA supervisor contract (supervisor.drain_ready): progress tuples
    # flow through ``ack_queue`` and one final ack completes readiness. The
    # load watcher speaks it, so ``ft serve --backend mlx`` gets live
    # /health progress and the maintenance flip for free, unmodified.
    ack_queue: Any = field(default_factory=queue.Queue)
    expected_acks: int = 1
    _switch_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _state_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------ load state --
    def _set_state(
        self, state: str, *, phase: str = "", error: str = ""
    ) -> None:
        with self._state_lock:
            self.load_state = state
            if phase:
                self.load_phase = phase
            if error:
                self.load_error = error
            if state in ("ready", "error"):
                self.load_ended_at = time.monotonic()

    def health_doc(self) -> dict[str, Any]:
        """The /health document for this engine, in the CUDA path's contract:
        ``status: loading`` carries ``phase`` + byte progress the shell renders
        (``loading (weights): 4.2/15.0 GiB``), ``error`` carries the reason."""
        with self._state_lock:
            state = self.load_state
            phase = self.load_phase
            error = self.load_error
            weights = self.weights_bytes
        doc: dict[str, Any] = {"model": self.model_path, "backend": self.backend}
        if state == "error":
            doc["status"] = "error"
            doc["message"] = error or "Metal backend failed to load"
            return doc
        if state == "ready":
            doc["status"] = "ok"
            doc["maintenance"] = "serving"
            doc["uptime_s"] = max(0, int(time.monotonic() - self.load_ended_at))
            return doc
        done = _upstream_resident_bytes(self.processes)
        doc["status"] = "loading"
        doc["phase"] = phase or "starting"
        if weights > 0:
            doc["progress"] = {
                "done_bytes": min(done, weights),
                "total_bytes": weights,
            }
        return doc

    # ------------------------------------------------------------ lifecycle --
    def terminate(self) -> None:
        _stop_processes(self.processes)

    def is_alive(self) -> bool:
        return any(p.poll() is None for p in self.processes)

    def is_ready(self) -> bool:
        """True once a warm-up generation has proven the engine generates.

        The proxy's generation routes gate on this (503 while loading) so a
        user request cannot queue behind the weight load forever."""
        return self.load_state == "ready"

    def switch_model(self, model_path: str) -> None:
        """Serve a different model: stop the old engine *first*, then start the new.

        Sequential, not concurrent. Concurrent loads (old engine resident while
        the new one streams in) over-commit the Metal working set -- two ~50 GiB
        engines against a ~107 GiB wired limit is the deadlock this machine
        kept hitting during switches. Stopping first frees the old model's
        memory before the new load starts, at the cost of a load-window gap in
        serving (the shell's /health progress bar covers it).

        Failure semantics: if the new engine fails to come up, the old model is
        gone -- the server reports the error rather than pretending. Rolling
        back would mean reloading the old weights (the same cost as retrying),
        so the honest state is an error the user can act on.
        """
        with self._switch_lock:
            old_processes = self.processes
            self.processes = []
            self.upstream_base_url = ""
            self.model_path = model_path
            with self._state_lock:
                self.load_state = "loading"
                self.load_phase = "stopping"
                self.load_error = ""
                self.weights_bytes = 0
            # A switch drains the queue the supervisor may still be reading;
            # its terminal acks for the old engine must not leak into the new
            # one's readiness handshake. _drain_acks takes _state_lock itself,
            # so it MUST run outside the block above (non-reentrant Lock --
            # calling it inside self-deadlocked every model switch).
            self._drain_acks()
            # Full escalate-to-kill teardown BEFORE the new load: the child's
            # output pipe is drained (see _drain_process_output), and a partial
            # teardown here would leave it holding its port -- and its memory.
            _stop_processes(old_processes)
            # Sequential from here: the watcher publishes load state onto THIS
            # handle (state_handle=self) while the load runs, and the launch
            # returns a handle carrying the new engine's identity.
            new = launch_metal_backend(
                self.backend,
                model_path,
                upstream_port=None,
                state_handle=self,
            )
            # Block until the watcher reaches a terminal state. /v1/model/load
            # must not answer "ok" while the new engine is still loading -- the
            # shell budgets this call for the full download + load. Raises on
            # failure so the route reports the reason instead of a false ok.
            self._wait_load_terminal()
            self.processes = new.processes
            self.upstream_base_url = new.upstream_base_url
            self.model_path = new.model_path

    def _wait_load_terminal(self, timeout: float | None = None) -> None:
        """Block until this handle's load watcher reports ready or error.

        Polls ``load_state`` (the watcher's publication point) -- not the
        ack_queue, which the CUDA-side supervisor may be draining concurrently
        in ``ft serve --backend mlx`` mode."""
        deadline = time.monotonic() + (timeout if timeout else _LOAD_TIMEOUT_S)
        while time.monotonic() < deadline:
            with self._state_lock:
                state = self.load_state
            if state == "ready":
                return
            if state == "error":
                with self._state_lock:
                    raise RuntimeError(self.load_error or "model load failed")
            time.sleep(0.25)
        raise RuntimeError("model load timed out")

    def _drain_acks(self) -> None:
        """Drop stale acks so a prior engine's terminal events cannot be read
        as the next engine's readiness."""
        with self._state_lock:
            while True:
                try:
                    self.ack_queue.get_nowait()
                except queue.Empty:
                    return


def _drain_process_output(proc: subprocess.Popen, name: str) -> None:
    """Read a child's stdout until EOF on a daemon thread.

    The children are launched with ``stdout=PIPE`` so launch failures are
    visible, but an unread pipe fills (64 KiB) and then blocks the child inside
    ``write()`` forever -- including ignoring SIGTERM. Draining on a thread
    keeps the child healthy and makes ``terminate()`` actually work.
    """
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            logger.debug("%s: %s", name, line.rstrip())
    except Exception:  # noqa: BLE001 -- draining must never raise
        pass


def _stop_processes(processes: list[subprocess.Popen]) -> None:
    for p in processes:
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:  # noqa: BLE001 -- best-effort teardown
            continue
    for p in processes:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
                p.wait(timeout=5)
            except Exception:  # noqa: BLE001 -- best-effort teardown
                continue
        except Exception:  # noqa: BLE001 -- best-effort teardown
            continue


def _start_drain_thread(proc: subprocess.Popen, name: str) -> None:
    t = threading.Thread(target=_drain_process_output, args=(proc, name), daemon=True)
    t.start()


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


def _upstream_resident_bytes(processes: list[subprocess.Popen]) -> int:
    """How many bytes of the model the upstream has resident, from RSS.

    A crude but truthful progress signal for a lazy loader: mlx_lm reads
    weights into unified memory on the first generation, so RSS climbing
    toward ``weights_bytes`` IS the load."""
    total = 0
    for p in processes:
        try:
            rss_kb = int(
                subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(p.pid)],
                    capture_output=True,
                    text=True,
                    timeout=2,
                ).stdout.strip()
                or 0
            )
            total += rss_kb * 1024
        except Exception:  # noqa: BLE001 -- progress is best-effort
            continue
    return total


def _weights_bytes_on_disk(model_path: str) -> int:
    """Total weight-file bytes for a local model dir, else 0 (unknown)."""
    path = os.path.expanduser(model_path)
    if not os.path.isdir(path):
        return 0
    total = 0
    for name in ("*.safetensors", "*.gguf", "*.bin"):
        for f in glob.glob(os.path.join(path, name)):
            try:
                total += os.path.getsize(f)
            except OSError:
                continue
    return total


def _hf_cache_dir(repo_id: str) -> str | None:
    """Snapshot dir in the local HF cache for ``repo_id``, when fully present.

    Used to (a) size the load before it starts and (b) decide whether the
    child can run with HF_HUB_OFFLINE=1 -- which stops a stalled CDN retry
    from hanging the load forever (the failure mode seen on this host)."""
    base = os.environ.get("HF_HUB_CACHE") or os.path.expanduser(
        "~/.cache/huggingface/hub"
    )
    repo_dir = os.path.join(base, "models--" + repo_id.replace("/", "--"))
    snaps = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snaps):
        return None
    snapshots = [
        d
        for d in (os.path.join(snaps, x) for x in os.listdir(snaps))
        if os.path.isdir(d)
    ]
    if not snapshots:
        return None
    snap = max(snapshots, key=os.path.getmtime)
    if _weights_bytes_on_disk(snap) == 0:
        return None  # no weights resolved (download incomplete or empty)
    return snap


def _warm_up_generation(url: str, model_id: str, timeout: float) -> None:
    """Drive one tiny generation so weights load before we call the engine ready.

    mlx_lm serves /v1/models and accepts requests while weights are still on
    disk, then blocks the first generation until the load finishes -- so "the
    port answers" is NOT ready. This 1-token request runs the actual load
    under supervision: it either proves the engine can generate (and warms
    kernels/caches along the way) or it fails with a reason we can report.

    ``model_id`` must be the id the upstream actually serves: mlx_lm resolves
    the model field against the HF cache, and an id that is not cached (plus
    our HF_HUB_OFFLINE=1) makes it 404 before any weights move."""
    body = json.dumps(
        {
            "model": model_id,
            "prompt": "1",
            "max_tokens": 1,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"warm-up generation failed: HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001 -- body text is best-effort
            pass
        raise RuntimeError(
            f"warm-up generation failed: HTTP {exc.code}"
            + (f": {detail}" if detail else "")
        ) from exc


def _watch_mlx_load(
    handle: MetalBackendHandle,
    url: str,
    proc: subprocess.Popen,
    *,
    timeout: float,
) -> None:
    """Supervise the upstream's load: port up -> weights loading -> ready/error.

    Runs on a daemon thread so the API server binds immediately and /health
    reports live progress while the engine loads. Progress is ALSO pushed as
    ("progress", desc, done, total) tuples on ``handle.ack_queue`` -- the CUDA
    supervisor's protocol (supervisor.drain_ready) -- so ``ft serve --backend
    mlx`` renders the same live /health phases through the stock supervisor,
    and a terminal ("error", reason) / plain ack drives its maintenance gate."""
    handle._set_state("loading", phase="starting")
    try:
        _wait_for_readiness(url, proc, timeout=_ADDRESS_HEALTH_TIMEOUT_S)
    except RuntimeError as exc:
        handle._set_state("error", error=str(exc))
        handle.ack_queue.put(("error", str(exc)))
        return
    handle._set_state("loading", phase="weights")
    handle.ack_queue.put(("progress", "Loading weights (Metal)", 0, handle.weights_bytes))

    # Sample RSS toward the weights total while the warm-up generation runs:
    # that request IS the load (mlx_lm generates only after weights arrive),
    # so its duration is exactly when progress moves.
    done_holder = {"done": 0, "stop": False}

    def _progress_sampler() -> None:
        while not done_holder["stop"]:
            done_holder["done"] = _upstream_resident_bytes(handle.processes)
            if handle.weights_bytes > 0:
                handle.ack_queue.put(
                    (
                        "progress",
                        "Loading weights (Metal)",
                        min(done_holder["done"], handle.weights_bytes),
                        handle.weights_bytes,
                    )
                )
            time.sleep(_POLL_INTERVAL_S)

    sampler = threading.Thread(
        target=_progress_sampler, name="freetoken-metal-progress", daemon=True
    )
    sampler.start()
    try:
        _warm_up_generation(url, model_id=handle.model_path, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 -- any warm-up failure is a load failure
        done_holder["stop"] = True
        sampler.join(timeout=2)
        if proc.poll() is not None:
            reason = f"Metal backend exited during load (code {proc.returncode})"
        else:
            reason = f"model load failed: {exc}"
        handle._set_state("error", error=reason)
        handle.ack_queue.put(("error", reason))
        return
    done_holder["stop"] = True
    sampler.join(timeout=2)
    if proc.poll() is not None:
        reason = f"Metal backend exited during load (code {proc.returncode})"
        handle._set_state("error", error=reason)
        handle.ack_queue.put(("error", reason))
        return
    handle._set_state("ready")
    handle.ack_queue.put("ready")


def _gemma4_chat_template() -> str | None:
    """Google's canonical Gemma 4 chat template, for gemma-4 snapshots whose
    repo ships none (the 26b-a4b base repo uploads tokenizer files without a
    chat_template; the -it repos carry chat_template.jinja).

    Loaded from the bundled gemma4_chat_template.jinja (Google Gemma
    Engineering, 2026-07-09 -- fixes tool-calling loops, turn closures, and
    thinking content-ordering). Do NOT hand-edit: the exact turn grammar
    (<|turn>/<turn|>, <|channel>thought, <|think|>, tool blocks) is what the
    model was trained on, and near-misses make it degenerate into raw
    text completion (endless repetition, API-doc regurgitation).

    Returns None when the asset is missing -- other models either ship their
    own template (correct as-is) or are not chat models."""
    global _GEMMA4_TEMPLATE
    if _GEMMA4_TEMPLATE is None:
        path = os.path.join(os.path.dirname(__file__), "gemma4_chat_template.jinja")
        try:
            with open(path, "r", encoding="utf-8") as f:
                _GEMMA4_TEMPLATE = f.read()
        except OSError:
            return None
    return _GEMMA4_TEMPLATE


_GEMMA4_TEMPLATE: str | None = None


def _needs_turn_stop_token(body: dict[str, Any]) -> bool:
    """True when the request targets a gemma-4 model served through mlx_lm.

    Those snapshots ship no chat template and define eos as <eos> only, so the
    model's turn-end token (<turn|>) is not in the engine's stop set: the model
    answers and then keeps generating a fake USER:/ASSISTANT: transcript. The
    proxy injects <turn|> into the request's stop list for exactly those
    models; everything else keeps its engine-default behavior."""
    return _is_gemma4_model(body.get("model") or "")


def _is_gemma4_model(model: str) -> bool:
    """Match a served model id / path against the gemma-4 family (the id may be
    a HF repo id like google/gemma-4-26b-a4b or a local snapshot path)."""
    m = model.lower()
    return "gemma-4" in m or "gemma4" in m


def _launch_mlx(
    model_path: str, port: int, *, state_handle: MetalBackendHandle | None = None
) -> MetalBackendHandle:
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
    # gemma-4's snapshot ships NO chat template (neither in tokenizer_config.json
    # nor as chat_template.jinja), so without help mlx_lm serves raw-text prompts
    # to a chat model. Pass the Gemma 4 turn format explicitly (the same wire
    # format transformers' own serving utils document: turns open with
    # "<|turn>user\n"/"<|turn>model\n" and close with "<turn|>"). Stopping is
    # handled at the proxy (stop-word injection), since the model's
    # generation_config eos_token_id is only <eos> (1) and mlx_lm builds its
    # stop set from that alone.
    if _is_gemma4_model(model_path):
        cmd += ["--chat-template", _gemma4_chat_template()]
    # When the weights are already fully in the local HF cache, pin the child
    # to them: no revalidation round-trip, and no stalled-CDN retry can hang
    # the lazy load. A repo id that is not cached (or only partially) keeps
    # online mode so the download happens as before.
    env = os.environ.copy()
    cache_dir = _hf_cache_dir(model_path)
    if cache_dir is not None:
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        logger.info(
            "mlx: using cached snapshot %s (HF_HUB_OFFLINE=1)", cache_dir
        )
    logger.info("launching Metal backend (mlx): %s", shlex.join(cmd))
    proc = subprocess.Popen(
        cmd,
        # Own the child's stdout/stderr so we can detect startup failures and
        # drain logs; the child inherits env so HF/mlx settings pass through.
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    _start_drain_thread(proc, "mlx")
    url = f"http://127.0.0.1:{port}"
    # In a switch, the watcher publishes state onto the caller's shared handle
    # (the one the proxy routes already read); the returned handle only
    # carries the engine's process/upstream/url for the caller to adopt.
    return_handle = MetalBackendHandle(
        processes=[proc],
        upstream_base_url=url,
        backend="mlx",
        model_path=model_path,
    )
    watcher_handle = state_handle or return_handle
    if cache_dir is not None:
        weights = _weights_bytes_on_disk(cache_dir)
        return_handle.weights_bytes = weights
        watcher_handle.weights_bytes = weights
    watcher_handle.load_started_at = time.monotonic()
    threading.Thread(
        target=_watch_mlx_load,
        args=(watcher_handle, url, proc),
        kwargs={"timeout": _LOAD_TIMEOUT_S},
        name=f"freetoken-metal-load-{port}",
        daemon=True,
    ).start()
    return return_handle


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
    _start_drain_thread(proc, "llama")
    url = f"http://127.0.0.1:{port}"
    _wait_for_readiness(url, proc, timeout=_ADDRESS_HEALTH_TIMEOUT_S)
    return MetalBackendHandle(
        processes=[proc], upstream_base_url=url, backend="llama", model_path=model_path
    )


def launch_metal_backend(
    backend: str,
    model_path: str,
    upstream_port: int | None = None,
    *,
    state_handle: MetalBackendHandle | None = None,
) -> MetalBackendHandle:
    """Launch an upstream engine and return a handle to it.

    ``state_handle``: when given, the load watcher publishes its state onto
    THAT handle instead of the returned one. Used by ``switch_model``, which
    owns a shared handle the proxy routes already read from -- without this,
    the watcher would update a detached object and /health would report the
    old model's state forever."""
    port = _pick_upstream_port(upstream_port)
    if backend == "mlx":
        return _launch_mlx(model_path, port, state_handle=state_handle)
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
    """Forward a request to the upstream and stream the (possibly SSE) bytes back.

    SSE chunks are rewritten on the fly so the Metal upstream speaks FreeToken's
    wire format: mlx_lm splits the thinking channel as ``delta.reasoning`` while
    FreeToken (and vLLM/SGLang, which the shell/bench clients read) uses
    ``delta.reasoning_content``.

    Chunks are yielded as they arrive -- no buffering. The upstream flushes per
    token (mlx_lm writes + flushes each SSE event), so with body-based stream
    detection this delivers tokens as they decode."""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", f"{upstream_base_url}{path}", content=body, headers=headers
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes(chunk_size=None):
                if b'"reasoning"' in chunk:
                    chunk = _rewrite_reasoning_field(chunk)
                yield chunk


def _rewrite_reasoning_field(chunk: bytes) -> bytes:
    """Rename ``"reasoning"`` to ``"reasoning_content"`` inside SSE data lines.

    Byte-level and conservative: only touches the exact ``"reasoning":`` JSON
    key (mlx_lm's name for the thinking channel), never the value text, and
    leaves non-data bytes (keepalives, separators) untouched."""
    return chunk.replace(b'"reasoning":', b'"reasoning_content":')


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
        """List models. Overridden when the upstream reports more than the one it
        actually serves (mlx_lm lists the whole local HF cache): the proxy reports
        the served model only, so clients (ft shell) label the right one."""
        response = await _forward(request, get_backend, method="GET")
        handle = get_backend()
        if (
            isinstance(response, Response)
            and response.status_code == 200
            and handle is not None
            and handle.model_path
        ):
            try:
                doc = json.loads(response.body)
            except Exception:  # noqa: BLE001 -- keep upstream's answer as-is
                return response
            data = doc.get("data") if isinstance(doc, dict) else None
            if not isinstance(data, list):
                return response
            ids = [
                item.get("id")
                for item in data
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            ]
            if ids == [handle.model_path]:
                return response  # already truthful
            import time as _time

            doc["data"] = [
                {"id": handle.model_path, "object": "model", "created": doc.get("created", int(_time.time()))}
            ]
            return JSONResponse(doc)
        return response

    @app.get("/v1/model/list")
    async def proxy_model_list(request: Request):
        return await _forward(request, get_backend, method="GET")

    @app.post("/v1/model/load")
    async def model_load(request: Request):
        """Switch the Metal upstream to a different model.

        Accepts ``{"model": "<path or HF id>"}`` and relaunches the upstream
        engine (mlx/llama.cpp) on a fresh port while the old one keeps serving,
        then swaps the proxy target. On failure the server keeps the old model
        (the new upstream never launched) and returns the reason.
        """
        handle = get_backend()
        if handle is None or not handle.is_alive():
            return JSONResponse(
                {"detail": "Metal backend is not running"}, status_code=503
            )
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 -- bad JSON is a client error
            return JSONResponse(
                {"detail": "request body must be JSON with a 'model' field"},
                status_code=400,
            )
        model = body.get("model") if isinstance(body, dict) else None
        if not model or not isinstance(model, str):
            return JSONResponse(
                {"detail": "request body must be JSON with a 'model' field"},
                status_code=400,
            )
        if model == handle.model_path:
            return {
                "status": "ok",
                "model": model,
                "detail": "already serving this model",
            }
        try:
            # The switch stops the old engine and waits out the new one's load
            # (download + weights) -- minutes for a big model. Run it off the
            # event loop so /health (which reports the load's live progress)
            # keeps answering while it runs.
            await asyncio.to_thread(handle.switch_model, model)
        except Exception as exc:  # noqa: BLE001 -- a failed switch must not kill the server
            return JSONResponse(
                {"detail": f"model switch failed: {exc}"}, status_code=500
            )
        return {"status": "ok", "model": handle.model_path}

    @app.get("/v1/stats")
    async def metal_stats(request: Request):
        """Stable shape for the shell's status-bar poller. The Metal upstream has
        no CUDA-style stats; every pool reports empty so the bar shows just the
        model + token counters, which the shell counts client-side."""
        return {
            "kv": None,
            "mamba": None,
            "swa": None,
            "vram_bytes": 0,
        }

    @app.get("/v1/cache/status")
    async def metal_cache_status(request: Request):
        """Minimal geometry doc so the shell's startup read resolves instead of
        404-ing: no MoE cache on the Metal path, no reasoning gears advertised
        (the upstream applies its own chat template)."""
        return {
            "geometry": {
                "moe_cache_size": 0,
                "moe_cache_policy": "none",
                "reasoning": {"gears": [], "kwargs": {}, "default": None},
            },
            "pools": {},
        }

    @app.get("/v1/requests")
    async def metal_requests(request: Request):
        """``ft ctl requests`` parity. The Metal path has no engine-side request
        ring (the CUDA scheduler owns that); an empty list is the truthful
        answer -- per-request accounting lives in the upstream's own logs."""
        return {"requests": []}

    @app.get("/health")
    async def metal_health(request: Request):
        """Same contract the CUDA path answers: loading -> ok -> error, with
        byte progress while loading. Tools gate on ``maintenance == "serving"``
        (bench_decode_moe.wait_ready, the daemon, the desktop); the shell's
        attach path renders ``phase`` + ``progress`` as
        ``loading (weights): 4.2/15.0 GiB``. Reported from the handle's own
        supervised load state, not just process liveness: mlx_lm answers
        /v1/models before weights load, so "port up" would lie."""
        handle = get_backend()
        if handle is None or not handle.is_alive():
            return JSONResponse({"status": "down"}, status_code=503)
        doc = handle.health_doc()
        if doc.get("status") == "error":
            return JSONResponse(doc, status_code=503)
        return doc


def _strip_host_header(headers: dict[str, str]) -> dict[str, str]:
    out = {k: v for k, v in headers.items() if k.lower() not in {"host", "content-length"}}
    out["accept"] = headers.get("accept", "application/json")
    return out


def _inject_turn_stop(path: str, body: bytes, stream: bool) -> tuple[bytes, bool]:
    """Add "<turn|>" to a gemma-4 generation request's stop list.

    Returns the (possibly rewritten) body and the streaming decision. The body
    is rewritten only for chat/completions-style generation routes on a
    gemma-4 model that does not already stop on the turn token; parse failures
    and non-generation routes pass through untouched."""
    if not path.endswith(("/chat/completions", "/completions", "/messages", "/responses")):
        return body, stream
    try:
        payload = json.loads(body or b"{}")
    except (ValueError, UnicodeDecodeError):
        return body, stream
    if not isinstance(payload, dict) or not _needs_turn_stop_token(payload):
        return body, stream
    stop = payload.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        stop = []
    if "<turn|>" in stop:
        return body, stream
    stop.append("<turn|>")
    payload["stop"] = stop
    return json.dumps(payload).encode(), stream


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
    if method == "POST" and not getattr(handle, "is_ready", lambda: True)():
        # The engine is still loading (download or weights). Answer immediately
        # with the state instead of letting the request queue behind the load
        # forever -- the failure mode this proxy used to hang on.
        doc = handle.health_doc()
        detail = f"model is still loading ({doc.get('phase', 'starting')})"
        return JSONResponse({"detail": detail, **doc}, status_code=503)
    upstream = handle.upstream_base_url
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"

    body = await request.body()
    headers = _strip_host_header(dict(request.headers))
    # Streaming is signalled in the BODY ("stream": true), not the Accept
    # header -- the OpenAI SDK sends Accept: application/json even for
    # streaming requests. Deciding from the header routed streamed
    # generations through the buffered path: the client got the whole answer
    # in one burst at the end and live token counts read 0/burst.
    stream = "text/event-stream" in (headers.get("accept") or "")
    if not stream:
        try:
            payload = json.loads(body or b"{}")
        except (ValueError, UnicodeDecodeError):
            payload = {}
        stream = bool(payload.get("stream")) if isinstance(payload, dict) else False

    # gemma-4's turn-end token (<turn|>) is not in mlx_lm's stop set (the
    # model's generation_config lists only <eos>), so without this the model
    # answers and then keeps generating a fake USER:/ASSISTANT: transcript
    # until max_tokens. Inject it as a stop word for exactly those models.
    body, stream = _inject_turn_stop(request.url.path, body, stream)

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
    content = r.content
    # Same reasoning-channel rename as the streaming path, for non-streaming
    # completions (mlx_lm's "reasoning" -> FreeToken's "reasoning_content").
    if b'"reasoning":' in content:
        content = content.replace(b'"reasoning":', b'"reasoning_content":')
    return Response(
        content=content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type"),
    )
