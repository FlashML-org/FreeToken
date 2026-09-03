from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from freetoken import __version__
from freetoken.core import SamplingParams
from freetoken.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    CacheRebuildMsg,
    CacheRebuildReply,
    TokenizeMsg,
    UserReply,
)
from freetoken.utils import (
    ZmqAsyncPullQueue,
    ZmqAsyncPushQueue,
    init_logger,
    load_generation_sampling,
)
from pydantic import BaseModel

from .args import ServerArgs
from .anthropic_api import register_anthropic_routes
from .accounting import AdmissionClosedError, register_accounting_routes
from .control_api import register_control_routes
from .openai_api import register_openai_routes
from . import request_ring
from .access_log_filter import install_polling_access_log_filter
from .request_logger import init as init_request_logging, log_request
from .responses_api import register_responses_routes
from .stats import StatsTracker

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None
# Recommended sampling defaults from the checkpoint's generation_config.json, applied to
# request fields the caller left unspecified (sglang's sampling_defaults='model').
_MODEL_SAMPLING: Dict[str, Any] = {}
# Set on an orderly stop (uvicorn lifespan shutdown fired by SIGTERM/SIGINT, or the shell
# signal handler). The backend supervisor reads this so a worker exiting AS PART of that
# shutdown is treated as expected — no ERROR log, no "failed" latch. See run_backend_supervisor.
_SHUTTING_DOWN = threading.Event()
BACKEND_DEATH_EXIT_GRACE_S = 10.0


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _terminate_backend_workers(processes: List[Any]) -> None:
    """Best-effort, non-blocking teardown of the backend worker processes on an orderly stop.

    Called from the shutdown path AFTER ``_SHUTTING_DOWN`` is set, so the supervisor's liveness
    watch is guaranteed to observe the flag before it sees these deaths — the exits are then
    attributed to the stop, not misreported as a crash (the flag/death race, otherwise decided
    by OS signal-delivery order, is settled in our favor). Also ensures the non-daemon workers
    are actually torn down when an external stop signals only the main process.

    Never blocks (no join) and never raises: a worker already gone / an unqueryable handle is
    fine — this only nudges live ones toward exit."""
    for p in processes or []:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:  # noqa: BLE001 -- already-gone / unqueryable handle: nothing to do
            continue


def _exit_after_backend_death(grace_s: float) -> threading.Timer:
    def _stop() -> None:
        if _SHUTTING_DOWN.is_set():
            return  # an external stop got here first
        logger.error("Backend worker is gone and cannot be restarted; stopping the API server")
        os.kill(os.getpid(), signal.SIGTERM)

    timer = threading.Timer(grace_s, _stop)
    timer.daemon = True
    timer.start()
    return timer


def _reap_backend_workers(processes: List[Any], timeout: float = 5.0) -> None:
    """Wait out a preceding ``_terminate_backend_workers`` and SIGKILL whatever is still
    standing. Only the shell path needs this: it owns the process lifetime end to end (no
    outer signal takes the process down for it), and a worker that ignored SIGTERM would keep
    the GPU and the IPC sockets after the shell has already returned to the user's terminal."""
    for p in processes or []:
        try:
            p.join(timeout=timeout)
            if p.is_alive():
                p.kill()
        except Exception:  # noqa: BLE001 -- already-gone / unqueryable handle: nothing to do
            continue


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    # Stable identity for this serve process. Generated before the backend is ready so every
    # /health state (loading/ok/error) and /v1/stats can identify the same engine generation.
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Runtime cache-rebuild control plane (correlated by uuid request_id, separate from
    # the int-uid generation ack machinery).
    rebuild_futures: Dict[str, asyncio.Future] = field(default_factory=dict)
    # Lifecycle gate. Starts "loading" (uvicorn binds before weights finish; the three
    # API adapters 503 until this flips) -> "serving" once all workers ack ready ->
    # "rebuilding"/"failed" for runtime cache rebuilds.
    maintenance_state: str = "loading"
    last_rebuild: Dict[str, Any] | None = None
    load_progress: Any = None
    # Monotonic timestamp the server became ready; drives /health + /v1/stats uptime without
    # being affected by wall-clock adjustments.
    ready_at: float | None = None
    # Set when a backend worker dies (startup OR post-ready). /health reports status "error"
    # with this message; the API gate already blocks new work via maintenance_state="failed".
    fatal_error: str | None = None
    # Runtime metrics for /v1/stats: throughput sliding window + last-known kv/mamba/vram
    # snapshot, fed from every UserReply in listen().
    stats: Any = None
    # Optional backend metadata delivered once on the ack path at ready: per-unit cache VRAM
    # costs {"kv_bytes_per_token", "moe_bytes_per_expert", "mamba_bytes_per_slot"}. None until
    # the ("meta", …) ack arrives (or forever, on an engine build that doesn't emit one).
    unit_bytes: Dict[str, int] | None = None
    # Pool-budget baseline from the same ("meta", …) ack: free VRAM captured after the weights
    # loaded but BEFORE any cache pool (KV/MoE/GDN) was allocated (Engine._post_weights_free).
    # Constant for the engine's lifetime — the slider upper bounds derive from it directly,
    # so they don't drift with allocator caching or runtime allocations. 0 until meta arrives.
    free_vram_bytes: int = 0
    # Per-pool floor unit counts {"kv_tokens", "moe_experts", "mamba_slots"} the rebuild path
    # enforces, from the ("meta", …) ack. None until it arrives (older engine); limits fall back.
    cache_floors: Dict[str, int] | None = None
    # Actual pool sizes allocated at load {"num_pages", "page_size", "moe_cache_size",
    # "num_mamba_slots"}, from the same ack. Seeds geometry before the first generation reply
    # (the running snapshot channel) has anything. None until meta arrives.
    cache_pools: Dict[str, int] | None = None
    # one {index, name, uuid, total_bytes} per TP rank, from the same ack; /v1/stats gpus
    gpus: List[Dict[str, Any]] = field(default_factory=list)
    # Backend worker Process handles (TP schedulers + tokenizer/detokenizer), captured from the
    # BackendHandle after start_backend(). The orderly-shutdown path (lifespan / shell signal
    # handler) tears these down itself, AFTER setting _SHUTTING_DOWN, so the supervisor observes
    # the shutdown flag before the ensuing deaths. See _terminate_backend_workers.
    backend_processes: List[Any] = field(default_factory=list)
    # Event loop the listener runs on, captured when the listener starts (_create_listener_once).
    # Lets a cross-thread caller — the supervisor thread's failure callback — marshal rebuild
    # future resolution back onto the loop (asyncio Futures are not thread-safe). None until the
    # first send_one starts the listener.
    _loop: Any = None
    # Frontend-side tokenizer for /v1/messages/count_tokens, built lazily on the first count
    # and cached for the process lifetime. It is the SAME TokenizeManager the engine's
    # tokenizer worker runs (chat template + generation prompt, DSV4/GGUF handling), so a
    # counted prompt tokenizes identically to a generated one. The engine's tokenizer lives in
    # another process — this is a deliberate second, read-only instance that keeps counting off
    # the generation path. The lock dedupes concurrent first builds.
    _frontend_tokenizer: Any = None
    _frontend_tokenizer_lock: Any = field(default_factory=threading.Lock)
    # One-shot guard for warm_frontend_tokenizer(); benign if two polls race it.
    _frontend_warm_started: bool = False

    def __post_init__(self) -> None:
        if self.stats is None:
            self.stats = StatsTracker()

    def frontend_tokenizer(self) -> Any:
        """Lazily build and cache the frontend-side tokenizer used by count_tokens (see the
        ``_frontend_tokenizer`` field). Called from a worker thread — never on the event loop."""
        with self._frontend_tokenizer_lock:
            if self._frontend_tokenizer is None:
                from freetoken.tokenizer.tokenize import TokenizeManager
                from freetoken.utils import load_tokenizer

                self._frontend_tokenizer = TokenizeManager(load_tokenizer(self.config.model_path))
            return self._frontend_tokenizer

    def warm_frontend_tokenizer(self) -> None:
        """Build the frontend tokenizer and probe its thinking profile off-thread,
        once — /v1/cache/status polls call this so the gear picker self-populates
        without ever blocking the event loop on a tokenizer load."""
        if self._frontend_warm_started:
            return
        self._frontend_warm_started = True

        def _warm() -> None:
            try:
                self.frontend_tokenizer().thinking_profile()
            except Exception:  # noqa: BLE001 -- warmup only; real faults surface on use
                pass

        threading.Thread(target=_warm, daemon=True, name="frontend-tokenizer-warm").start()

    def new_user(self) -> int:
        if self.maintenance_state != "serving":
            raise AdmissionClosedError(
                f"server unavailable: engine is {self.maintenance_state}"
            )
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        self.stats.on_new_user(uid)
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            if isinstance(msg, CacheRebuildReply):
                self._resolve_rebuild(msg)
                continue
            for msg in _unwrap_msg(msg):
                # Global accounting follows actual admitted/sampled work even after the HTTP
                # client disconnects and abort_user removes its ack queue. Delivery to a live
                # request remains gated below, but observation must happen first.
                self.stats.observe(msg)
                if msg.uid not in self.ack_map:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _resolve_rebuild(self, msg: CacheRebuildReply) -> None:
        """Terminal transition for a rebuild: rebuilding -> serving | failed. This is the ONLY
        path that reopens the gate dispatch_rebuild latches to "rebuilding", so it must always
        land on a definite state — including for a reply that arrives after the HTTP wait timed
        out (its future is already gone) — so a rebuild can never wedge the server in
        "rebuilding" forever.

        Ordering matters. Wake any waiter and record the result first, then decide the gate:
          - A fatal worker death latched "failed" (the watchdog) OUTRANKS this reply. A reply
            that raced a crash (e.g. a buffered "ok") must not resurrect a dead engine to
            "serving" — a crashed backend cannot serve. Leave it latched.
          - Otherwise only a genuine destructive "failed" latches maintenance; "ok"/"busy"/
            "rejected"/"unsupported" all leave the prior cache intact, so the engine keeps
            serving."""
        self.last_rebuild = {
            "request_id": msg.request_id,
            "status": msg.status,
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "mamba_slots": msg.mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
            "error": msg.error,
        }
        fut = self.rebuild_futures.pop(msg.request_id, None)
        if fut is not None and not fut.done():
            fut.set_result(self.last_rebuild)
        if self.fatal_error is not None:
            # A dead backend stays failed regardless of any (possibly stale/buffered) reply.
            self.maintenance_state = "failed"
            return
        self.maintenance_state = "failed" if msg.status == "failed" else "serving"

    def fail_pending_rebuilds(self, message: str) -> None:
        """Resolve every in-flight rebuild waiter as failed. Called from the supervisor thread
        when a worker death latches a fatal error: no CacheRebuildReply will ever arrive, so a
        caller blocked in dispatch_rebuild's ``await`` would otherwise hang until its full
        timeout (up to 300s). Runs the resolution ON the event loop via call_soon_threadsafe —
        asyncio Futures are not thread-safe, and _resolve_rebuild only ever calls set_result
        from the loop thread (the listen() task), so we mirror that here from off-thread."""
        loop = self._loop
        if loop is None:
            return  # listener never started -> no futures could be pending
        result = {"status": "failed", "error": message}

        def _resolve_all() -> None:
            for request_id in list(self.rebuild_futures):
                fut = self.rebuild_futures.pop(request_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(dict(result))

        try:
            loop.call_soon_threadsafe(_resolve_all)
        except RuntimeError:
            # Loop already closed (shutdown racing the crash): nothing left to wake.
            pass

    def _create_listener_once(self):
        if not self.initialized:
            self._loop = asyncio.get_running_loop()
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]
        # finally, not a trailing statement: every consumer breaks out of its `async for` at
        # the terminal ack, leaving this generator suspended at the yield. Cleanup written
        # after the loop would then only run on paths nobody takes, leaking both maps once
        # per completed request. GeneratorExit runs the finally.
        try:
            while True:
                await event.wait()
                event.clear()

                pending = self.ack_map[uid]
                self.ack_map[uid] = []
                ack = None
                for ack in pending:
                    yield ack
                if ack and ack.finished:
                    break
        finally:
            self.ack_map.pop(uid, None)
            self.event_map.pop(uid, None)

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            if ack.incremental_output:
                # SSE: JSON-encode (so a newline inside a token can't break the frame)
                # and terminate every event with a blank line.
                yield f"data: {json.dumps({'text': ack.incremental_output}, ensure_ascii=False)}\n\n".encode()
            if ack.finished:
                break
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        self.stats.on_abort(uid)
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()
        # Tear the workers down ourselves (best-effort). _SHUTTING_DOWN is already set by the
        # time shutdown() runs, so the supervisor attributes the ensuing deaths to the stop.
        _terminate_backend_workers(self.backend_processes)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # Orderly shutdown (uvicorn traps SIGINT/SIGTERM and runs this on the way out). Flag it
    # BEFORE tearing anything down so the backend supervisor treats the workers' ensuing
    # exit as expected rather than a crash — no spurious ERROR / "failed" latch during stop.
    _SHUTTING_DOWN.set()
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


def install_cors(app: FastAPI, origins_csv: str) -> None:
    """Attach CORS headers for browser/webview clients (e.g. the desktop app).

    No-op when the allow-list is empty; must run before the app starts serving."""
    origins = [o.strip() for o in origins_csv.split(",") if o.strip()]
    if not origins:
        return
    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if "*" in origins else origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


app = FastAPI(title="FreeToken API Server", version=__version__, lifespan=lifespan)
register_openai_routes(app, get_global_state, lambda: _MODEL_SAMPLING)
register_anthropic_routes(app, get_global_state, lambda: _MODEL_SAMPLING)
register_responses_routes(app, get_global_state, lambda: _MODEL_SAMPLING)
register_control_routes(app, get_global_state, lambda: _MODEL_SAMPLING)
register_accounting_routes(app, get_global_state)


# Paths the HTTP middleware logs into the request ring. The three chat protocols funnel through
# the shared generation layer and are recorded there instead (with real token totals) — kept out
# here to avoid a duplicate token-less row. These two run their own ack loop, so they stay logged
# here as before (without per-request tokens). See generation.py `_record_generation`.
_TRACKED_REQUEST_PREFIXES = (
    "/v1/completions",
    "/generate",
)

# Subpaths that share a tracked prefix but are NOT generation requests. count_tokens never
# enters generation accounting, and its first-touch tokenizer load would otherwise dominate the
# /v1/stats p95 and pollute /v1/requests — exclude it before the prefix check below.
_UNTRACKED_REQUEST_PREFIXES = ("/v1/messages/count_tokens",)


def _served_model_name() -> str | None:
    st = _GLOBAL_STATE
    cfg = getattr(st, "config", None) if st is not None else None
    return getattr(cfg, "served_model_name", None)


@app.middleware("http")
async def _record_request_middleware(request: Request, call_next):
    """Time every generation request into the ring for /v1/requests + /v1/stats p95. Single-
    model server, so model = served_model_name; stream is inferred from the response media
    type. Token counts are P3 (SSE usage arrives after the handler returns) — kept as None."""
    path = request.url.path
    if path.startswith(_UNTRACKED_REQUEST_PREFIXES) or not path.startswith(
        _TRACKED_REQUEST_PREFIXES
    ):
        return await call_next(request)
    import time as _time

    start = _time.monotonic()
    response = await call_next(request)
    duration_ms = int((_time.monotonic() - start) * 1000)
    ctype = response.headers.get("content-type", "")
    request_ring.record_request(
        request_ring.RequestRecord(
            ts=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
            method=request.method,
            path=path,
            status=response.status_code,
            model=_served_model_name(),
            duration_ms=duration_ms,
            ttft_ms=None,
            prompt_tokens=None,
            completion_tokens=None,
            stream=ctype.startswith("text/event-stream"),
            error=None,
        )
    )
    return response


class CacheRebuildRequest(BaseModel):
    moe_cache_size: int | None = None
    num_pages: int | None = None
    # Usable GDN (mamba) state-pool slots (matches the status-bar total); the engine adds the
    # reserved padding sink internally.
    num_mamba_slots: int | None = None
    # Window pool sizing (DSV4 or a sliding-window model in --cache-type radix). Two forms, at
    # most one: num_swa_pages pins the window to an absolute page count (usable, in the pool's own
    # page unit -- P for DSV4, 1 token for radix-SWA); swa_full_tokens_ratio in (0, 1] is a
    # convenience the server converts to num_swa_pages = ceil(ratio x full window) at the current
    # (or requested) anchor. Internally only num_swa_pages flows to the engine.
    num_swa_pages: int | None = None
    swa_full_tokens_ratio: float | None = None
    # Only "if_idle" (reject unless the scheduler is idle) is supported today. "drain" mode
    # is deferred (needs the drain-gate machinery); constraining the Literal makes an
    # unsupported value fail fast with a 422 at the API layer instead of a generic 503.
    mode: Literal["if_idle"] = "if_idle"
    timeout: float = 300.0


async def dispatch_rebuild(
    state: FrontendManager,
    *,
    moe_cache_size: int | None,
    num_pages: int | None,
    num_mamba_slots: int | None = None,
    num_swa_pages: int | None = None,
    mode: str = "if_idle",
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """Send a cache-rebuild request to the scheduler and await its result, managing the
    maintenance gate. Returns the scheduler's result dict, or a synthesized
    ``{"status": "failed"|"timeout"}`` on dispatch error / timeout. Every caller reaches it
    through ``POST /v1/cache/rebuild`` (``ft ctl cache``, the desktop panel, the shell's
    ``/cache``), which does the pre-flight maintenance_state checks (409/503 short-circuits)."""
    request_id = str(uuid.uuid4())
    fut = asyncio.get_running_loop().create_future()
    state.rebuild_futures[request_id] = fut
    state.maintenance_state = "rebuilding"
    try:
        await state.send_one(
            CacheRebuildMsg(
                request_id=request_id,
                moe_cache_size=moe_cache_size,
                num_pages=num_pages,
                num_mamba_slots=num_mamba_slots,
                num_swa_pages=num_swa_pages,
                mode=mode,
            )
        )
    except Exception as e:  # noqa: BLE001
        # The enqueue failed, so the scheduler never received the request and the engine is
        # untouched. Roll the gate back to serving (else a transient ZMQ error would latch
        # maintenance forever with no reply ever arriving to clear it) and surface the error.
        state.rebuild_futures.pop(request_id, None)
        state.maintenance_state = "serving"
        return {"status": "failed", "error": f"failed to dispatch rebuild: {e!r}"}
    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    except asyncio.TimeoutError:
        # Do NOT reopen the maintenance gate here: the scheduler may still be mid-rebuild
        # (e.g. a slow CUDA-graph recapture) and the backend request was not cancelled.
        # Leave maintenance_state == "rebuilding" so new generation and new rebuilds stay
        # blocked; the eventual CacheRebuildReply flips it to serving/failed via
        # _resolve_rebuild. Drop the now-cancelled future so it does not linger.
        state.rebuild_futures.pop(request_id, None)
        return {"status": "timeout", "request_id": request_id}


def _resolve_num_swa_pages(state: FrontendManager, req: CacheRebuildRequest) -> int | None:
    """External accepts num_swa_pages OR swa_full_tokens_ratio; internally only num_swa_pages
    flows. Convert a ratio to an absolute window at the requested (or current) full anchor, in the
    window pool's own page unit (P for DSV4, 1 token for radix-SWA)."""
    if req.swa_full_tokens_ratio is None:
        return req.num_swa_pages
    config = state.config
    pools = getattr(state, "cache_pools", None) or {}
    last = getattr(state, "last_rebuild", None) or {}
    num_pages = req.num_pages or int(
        last.get("num_pages") or getattr(state.stats, "kv_total_pages", 0)
        or pools.get("num_pages", 0) or 0
    )
    page_size = int(pools.get("page_size", 0) or getattr(config, "page_size", 1) or 1)
    is_dsv4 = getattr(getattr(config, "model_config", None), "dsv4_args", None) is not None
    swa_page_size = page_size if is_dsv4 else 1
    window_tokens = int(round(req.swa_full_tokens_ratio * num_pages * page_size))
    return max(1, -(-window_tokens // swa_page_size))  # ceil-div to the pool's page unit


@app.post("/v1/cache/rebuild")
async def cache_rebuild(req: CacheRebuildRequest):
    """Trigger a runtime KV/MoE cache resize. Blocks until the scheduler reports a result
    (or timeout). New generation is gated (503) while a rebuild is in flight."""
    state = get_global_state()
    if state.maintenance_state == "loading":
        return JSONResponse(
            {"status": "loading", "error": "model is still loading; cannot rebuild cache yet"},
            status_code=503,
        )
    if state.maintenance_state == "failed":
        return JSONResponse(
            {"status": "failed", "error": "server latched in maintenance; restart required"},
            status_code=503,
        )
    if state.maintenance_state == "rebuilding":
        return JSONResponse(
            {"status": "busy", "error": "a cache rebuild is already in progress"},
            status_code=409,
        )
    if state.maintenance_state == "stopping":
        return JSONResponse(
            {"status": "busy", "error": "engine stop is in progress"},
            status_code=409,
        )
    if req.num_swa_pages is not None and req.swa_full_tokens_ratio is not None:
        return JSONResponse(
            {"status": "failed", "error": "pass num_swa_pages OR swa_full_tokens_ratio, not both"},
            status_code=422,
        )
    if req.swa_full_tokens_ratio is not None and not 0.0 < req.swa_full_tokens_ratio <= 1.0:
        return JSONResponse(
            {"status": "failed", "error": "swa_full_tokens_ratio must be in (0, 1]"},
            status_code=422,
        )
    result = await dispatch_rebuild(
        state,
        moe_cache_size=req.moe_cache_size,
        num_pages=req.num_pages,
        num_mamba_slots=req.num_mamba_slots,
        num_swa_pages=_resolve_num_swa_pages(state, req),
        mode=req.mode,
        timeout=req.timeout,
    )
    if result["status"] == "timeout":
        return JSONResponse(result, status_code=504)
    return JSONResponse(result, status_code=200 if result["status"] == "ok" else 503)


def _cache_limits(geo: dict, unit_bytes: dict, pool_budget: int, floors: dict) -> dict:
    """Per-pool adjustable {min, max} bounds for the desktop cache sliders, so their ranges stay
    sensible instead of spanning the whole theoretical space:

      min -- the engine's rebuild floor for that pool (from the ("meta", …) floors; falls back to
              page_size for KV and num_experts for MoE, both known here, and 0 for mamba).
      max -- the ideal-case unit count if the whole cache budget went to this one pool:
                  max = pool_budget // unit_cost,
              where pool_budget is ``cache_budget_bytes`` -- the SAME ceiling the rebuild
              fit-check enforces (memory_ratio x pre-model baseline − weights), so a slider can
              never offer a size the rebuild would reject. It excludes the (1-memory_ratio)
              CUDA-graph/activation headroom, is constant for the engine's lifetime, and is
              measured before any pool existed, so each pool's ideal max is simply budget/cost
              -- no current-occupancy correction -- and the bounds are stable across rebuilds.

              Still optimistic in one way: it is the ``fixed=0`` whole-cache ceiling, while a
              real rebuild also subtracts ``fixed_cache_size`` (GDN state pool, the radix-SWA
              single-request floor / pinned window). Pools sized near their max on a model that
              carries those may still be rejected; the desktop's whole-config budget line is
              what catches that.

              MoE is additionally capped at the model's own routed-expert count: past that the
              budget could buy slots the model has nothing to put in.

    A pool whose unit cost or the budget signal is unknown (0) reports max 0 -- the desktop
    reads that as "unknown" and keeps its own bounds. Non-MoE / non-hybrid models therefore
    report 0 max (and a 0 floor) for the pools they lack. Defensive like the unit_bytes/num_experts
    blocks: any bad read degrades to all-zero bounds and never raises."""
    try:
        page_size = int(geo["page_size"])
        kv_per_token = int(unit_bytes["kv_per_token"])
        moe_per_expert = int(unit_bytes["moe_per_expert"])
        mamba_per_slot = int(unit_bytes["mamba_per_slot"])
        budget = int(pool_budget) if pool_budget and int(pool_budget) > 0 else 0

        def ideal(unit_cost: int) -> int:
            if unit_cost <= 0 or budget <= 0:
                return 0  # unit cost or budget signal unknown -> "unknown", desktop falls back
            return budget // unit_cost

        swa_per_token = int(unit_bytes.get("swa_per_token", 0) or 0)
        kv_min = int(floors.get("kv_tokens", page_size) or 0)
        moe_min = int(floors.get("moe_experts", geo["num_experts"]) or 0)
        mamba_min = int(floors.get("mamba_slots", 0) or 0)
        # SWA pool floor (single-request working set); the desktop uses it to lower-bound the
        # SWA-token estimate at a given reuse ratio. max is the ideal per-token ceiling.
        swa_min = int(floors.get("swa_tokens", 0) or 0)
        # The expert cache is the one pool with a ceiling of its own: caching more slots than
        # the model HAS routed experts buys nothing (every expert is already resident), so on a
        # small MoE model with a big card budget//cost overshoots the whole model. The other
        # pools have no such bound -- KV, window and GDN capacity all keep paying off with more
        # concurrent requests and longer prefix reuse.
        total_experts = int(geo["num_experts"]) * int(geo["num_moe_layers"])
        moe_max = ideal(moe_per_expert)
        if total_experts > 0:
            moe_max = min(moe_max, total_experts)
        return {
            "kv_tokens": {"min": kv_min, "max": ideal(kv_per_token)},
            "moe_experts": {"min": moe_min, "max": moe_max},
            "mamba_slots": {"min": mamba_min, "max": ideal(mamba_per_slot)},
            "swa_tokens": {"min": swa_min, "max": ideal(swa_per_token)},
        }
    except Exception:  # noqa: BLE001 -- limits are a nicety; a bad read must not 500 the poll
        return {
            "kv_tokens": {"min": 0, "max": 0},
            "moe_experts": {"min": 0, "max": 0},
            "mamba_slots": {"min": 0, "max": 0},
            "swa_tokens": {"min": 0, "max": 0},
        }


def _reasoning_geometry(state: Any) -> dict | None:
    """The ``geometry.reasoning`` block, from the frontend tokenizer's probed
    thinking profile. Peeks rather than builds: a cold tokenizer only kicks the
    warmup thread (the profile itself is a handful of microsecond renders once
    the tokenizer exists, safe to run inline)."""
    manager = getattr(state, "_frontend_tokenizer", None)
    if manager is None:
        state.warm_frontend_tokenizer()
        return None
    from .model_meta import derive_think_gears

    derived = derive_think_gears(
        manager.thinking_profile(),
        parser_configured=bool(getattr(state.config, "reasoning_parser", None)),
    )
    if derived is None:
        return None
    gears, default, kwargs = derived
    return {"gears": list(gears), "default": default, "kwargs": kwargs}


def cache_geometry(state: Any) -> dict:
    """Current cache geometry for the desktop cache panel. Each pool size resolves
    most-recent-truth first: the last rebuild's result, else the running UserReply snapshot
    (updated per generation), else the load-time allocation from the ("meta", …) ack — so the
    panel shows the real allocation from the moment the server is ready, not zeros until the
    first chat. moe_cache_size falls back further to the configured size (rate resolved to a
    slot count, matching the shell status bar). num_experts / num_moe_layers are the per-layer
    expert count and MoE layer count from model_config, so callers can derive the moe pool
    bounds; both are 0 when config.model_config is absent/raises (dummy configs) or non-MoE."""
    from .model_meta import moe_cache_size as configured_moe_cache_size

    tr = state.stats
    config = state.config
    last = getattr(state, "last_rebuild", None) or {}
    pools = getattr(state, "cache_pools", None) or {}
    num_pages = int(last.get("num_pages") or tr.kv_total_pages or pools.get("num_pages", 0) or 0)
    num_mamba_slots = int(
        last.get("mamba_slots") or tr.mamba_total_slots or pools.get("num_mamba_slots", 0) or 0
    )
    page_size = int(pools.get("page_size", 0) or getattr(config, "page_size", 1) or 1)
    moe_cache_size = last.get("moe_cache_size")
    if moe_cache_size is None:
        moe_cache_size = int(pools.get("moe_cache_size", 0) or 0) or configured_moe_cache_size(
            config
        )
    # Effective window/full ratio: derived from the last rebuild's pinned window (num_swa_pages,
    # the internal currency), else the load-time meta value. 0.0 for models without a window pool.
    swa_full_tokens_ratio = float(getattr(state, "swa_full_tokens_ratio", 0.0) or 0.0)
    last_swa_pages = last.get("num_swa_pages")
    if last_swa_pages:
        is_dsv4 = getattr(getattr(config, "model_config", None), "dsv4_args", None) is not None
        full = num_pages if is_dsv4 else num_pages * page_size
        if full > 0:
            swa_full_tokens_ratio = min(1.0, last_swa_pages / full)
    try:
        model_config = config.model_config
        num_experts = int(getattr(model_config, "num_experts", 0) or 0)
        num_moe_layers = int(getattr(model_config, "num_moe_layers", 0) or 0)
    except Exception:
        num_experts = 0
        num_moe_layers = 0
    # Per-unit VRAM costs from the backend's ("meta", …) ack (compute_cache_unit_bytes). All 0
    # when the meta never arrived (older engine build / still loading) or a dummy config is in
    # play; same defensive guard as num_experts so a missing/odd unit_bytes can't 500 the poll.
    try:
        ub = getattr(state, "unit_bytes", None) or {}
        unit_bytes = {
            "kv_per_token": int(ub.get("kv_bytes_per_token", 0) or 0),
            "moe_per_expert": int(ub.get("moe_bytes_per_expert", 0) or 0),
            "mamba_per_slot": int(ub.get("mamba_bytes_per_slot", 0) or 0),
            "swa_per_token": int(ub.get("swa_bytes_per_token", 0) or 0),
        }
    except Exception:
        unit_bytes = {"kv_per_token": 0, "moe_per_expert": 0, "mamba_per_slot": 0, "swa_per_token": 0}
    # Thinking control: the gears a client can offer and the chat_template_kwargs
    # each selects, derived from the checkpoint's own probed template (no
    # per-family registry). None until the frontend tokenizer is warm — the
    # first status poll kicks the warmup thread and later polls see the gears,
    # so the picker self-populates without ever blocking this route.
    try:
        reasoning = _reasoning_geometry(state)
    except Exception:
        reasoning = None
    geo = {
        "num_pages": num_pages,
        "page_size": page_size,
        "moe_cache_size": moe_cache_size,
        "num_mamba_slots": num_mamba_slots,
        "num_experts": num_experts,
        "num_moe_layers": num_moe_layers,
        # Eviction policy of the MoE slot cache ("lru"). Reported so a client can label the
        # pool without having to know how the server was started.
        "moe_cache_policy": getattr(config, "moe_cache_policy", None),
        "unit_bytes": unit_bytes,
        "swa_full_tokens_ratio": swa_full_tokens_ratio,
        # The window pool's page unit for num_swa_pages: DSV4 = P (== page_size), radix-SWA = 1
        # token, 0 for models without a window pool. Lets a client denominate the swa control.
        "swa_page_size": int(pools.get("swa_page_size", 0) or 0),
        # Concrete current window size in that unit (usable pages): the last rebuild's value if it
        # pinned/derived one, else the load-time pool size. 0 for models without a window pool.
        "num_swa_pages": int(last.get("num_swa_pages") or pools.get("num_swa_pages", 0) or 0),
        # Engine's exact total cache VRAM budget (all pools), from the ("meta", …) ack. 0 when
        # unknown (pre-budget engine) — the desktop then reverse-derives it from the limits.
        "cache_budget_bytes": int(getattr(state, "cache_budget_bytes", 0) or 0),
        "reasoning": reasoning,
    }
    # Per-pool slider bounds, sized against the cache budget the rebuild fit-check actually
    # enforces — NOT the raw post-weights free VRAM, which is larger by the (1-memory_ratio)
    # graph/activation headroom and would offer slider room every rebuild rejects. Both come
    # from the ("meta", …) ack and are constant for the engine's lifetime; the free-VRAM value
    # stays the fallback for an ack without a budget. 0 -> limits report 0 max and the desktop
    # keeps its own bounds.
    pool_budget = int(getattr(state, "cache_budget_bytes", 0) or 0) or int(
        getattr(state, "free_vram_bytes", 0) or 0
    )
    floors = getattr(state, "cache_floors", None) or {}
    geo["limits"] = _cache_limits(geo, unit_bytes, pool_budget, floors)
    return geo


@app.get("/v1/cache/status")
async def cache_status():
    state = get_global_state()
    return {
        "state": state.maintenance_state,
        "last_rebuild": state.last_rebuild,
        "geometry": cache_geometry(state),
    }


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    log_request("/generate", req, request)
    state = get_global_state()
    if state.maintenance_state != "serving":
        detail = "model is still loading" if state.maintenance_state == "loading" else "cache rebuild in progress"
        return JSONResponse({"error": f"server unavailable: {detail}"}, status_code=503)
    if req.max_tokens < 1:
        return JSONResponse({"error": f"max_tokens must be at least 1, got {req.max_tokens}"}, status_code=400)
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


def _install_shell_stop_handlers() -> None:
    """Shell mode runs uvicorn off the main thread, where its own signal capture is a no-op, so
    an external stop would kill this process and orphan the non-daemon backend workers. Flag the
    orderly stop FIRST -- so the supervisor attributes the ensuing worker exits to the stop
    instead of reporting a crash -- terminate them, then chain to whatever handler was there
    before (SIG_DFL restored and re-raised when it isn't callable, so the signal still stops us).

    SIGHUP is handled for the same reason as SIGTERM, and it matters more here: the workers run
    outside our process group in shell mode (see launch.py:_detach_process_group), so a closed
    terminal no longer reaches them on its own and this handler is what tears them down.

    SIGINT is deliberately left alone: the shell binds it, per turn, to "cancel this turn"."""
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}

    def _flag_shutdown(signum, frame) -> None:
        _SHUTTING_DOWN.set()
        _terminate_backend_workers(_GLOBAL_STATE.backend_processes)
        prev = previous.get(signum)
        if callable(prev):
            prev(signum, frame)
        else:
            signal.signal(signum, prev if prev is not None else signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    for sig in previous:
        signal.signal(sig, _flag_shutdown)


def _serve_and_run_shell(host: str, port: int) -> None:
    """Shell mode: serve the API here, and attach the terminal client to it over the loopback.

    The shell is an ordinary API client (see ``freetoken.shell``), so shell mode is just
    ``ft serve`` plus that client -- one generation path for every caller, none of it
    shell-private. uvicorn owns the HTTP surface on a worker thread and, through its lifespan,
    the orderly shutdown that flags ``_SHUTTING_DOWN`` before the workers exit; the main thread
    belongs to the TUI. Access logging is off because those lines would land in the middle of
    the chat as it streams -- ``/v1/requests`` still records every request for ``ft ctl``.

    Signals: uvicorn's capture is a no-op off the main thread, so ^C stays with the shell, which
    binds it to "cancel this turn". That only works because the engine workers leave our process
    group in shell mode (launch.py:_detach_process_group) -- otherwise the terminal would deliver
    that same ^C to them and the shell would go on chatting with a dead engine. In exchange, the
    stop signals they no longer receive are relayed by _install_shell_stop_handlers below."""
    from freetoken.launch import resolve_server_url
    from freetoken.shell.tui import run_shell

    # Resolved before anything is started, so a bad address fails while there is still nothing
    # to tear down. 0.0.0.0/:: are bind addresses, not destinations; resolve_server_url maps
    # them to loopback. An IPv6 bind host has to be bracketed before it can go into a URL.
    netloc = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    origin = resolve_server_url(f"http://{netloc}").origin

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, access_log=False))
    thread = threading.Thread(target=server.run, name="freetoken-uvicorn", daemon=True)
    thread.start()
    _install_shell_stop_handlers()
    try:
        # The engine is still loading here (uvicorn binds first, by design): the client waits on
        # /health and echoes the same load progress the desktop app polls for. A ^C during that
        # wait is a stop, not a crash -- exit through the teardown below, not a traceback.
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(run_shell(origin, connect_grace=30.0))
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        # Belt and braces: if uvicorn's lifespan shutdown did not run (thread wedged), flag the
        # stop and tear the workers down here so nothing outlives the shell.
        _SHUTTING_DOWN.set()
        _terminate_backend_workers(_GLOBAL_STATE.backend_processes)
        _reap_backend_workers(_GLOBAL_STATE.backend_processes)


def run_api_server(config: ServerArgs, start_backend: Callable[[], "Any"], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, also attach the interactive terminal shell to the served API.
    """

    global _GLOBAL_STATE, _MODEL_SAMPLING

    if config.sampling_defaults == "model" and not config.use_dummy_weight:
        _MODEL_SAMPLING = load_generation_sampling(config.model_path)
    # Always surface the effective default sampling (model-recommended where available,
    # else framework defaults), since unspecified request fields resolve to these.
    logger.info(
        "Default sampling config (source=%s): temperature=%s, top_k=%s, top_p=%s",
        "model" if _MODEL_SAMPLING else "framework",
        _MODEL_SAMPLING.get("temperature", 0.0),
        _MODEL_SAMPLING.get("top_k", -1),
        _MODEL_SAMPLING.get("top_p", 1.0),
    )

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    # Create/validate FREETOKEN_API_LOG_DIR and start the writer thread up front, so a
    # bad path is reported at boot rather than silently on the first request.
    install_cors(app, config.cors_origins)
    init_request_logging()
    # Hide the frequent health/stats/requests/cache-status polling of the desktop app (and of
    # the shell's status bar) from uvicorn's access log; non-polling access lines are
    # unaffected. See access_log_filter.py; toggle back on with LOG_LEVEL=DEBUG.
    install_polling_access_log_filter()

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    from .supervisor import LoadProgress, run_backend_supervisor

    _GLOBAL_STATE.load_progress = LoadProgress()
    handle = start_backend()
    # Hold the worker handles so the orderly-shutdown path can tear them down itself (after
    # setting _SHUTTING_DOWN) rather than relying on OS signal-delivery order.
    _GLOBAL_STATE.backend_processes = list(getattr(handle, "processes", None) or [])

    def _on_ready() -> None:
        # A stop requested while weights were loading has already sealed admission.  The backend
        # may finish its ready handshake before SIGTERM arrives; never reopen that gate after the
        # daemon has received a final accounting snapshot.
        if _GLOBAL_STATE.maintenance_state == "loading":
            _GLOBAL_STATE.maintenance_state = "serving"
            _GLOBAL_STATE.ready_at = time.monotonic()
            logger.info(f"API server is ready to serve on {host}:{port}")

    def _on_failure(message: str) -> None:
        _GLOBAL_STATE.fatal_error = message
        _GLOBAL_STATE.maintenance_state = "failed"
        logger.error("Backend supervisor: %s", message)
        # No CacheRebuildReply will ever arrive from a dead backend, so wake any caller blocked
        # in dispatch_rebuild's await now — otherwise it strands until the full rebuild timeout.
        _GLOBAL_STATE.fail_pending_rebuilds(message)
        # Then take the whole serve down (see _exit_after_backend_death). Shell mode is excluded:
        # a person is sitting at that TUI, the API is theirs alone, and its stop path is ^C.
        if not run_shell:
            _exit_after_backend_death(BACKEND_DEATH_EXIT_GRACE_S)

    def _on_meta(meta: dict) -> None:
        # Per-unit cache VRAM costs + the free-VRAM seed + per-pool floors + the actual pool
        # sizes allocated at load, delivered once on the ack path; surfaced by cache_geometry
        # (unit_bytes + the limits block + the pre-first-chat pool seed). Unpack the extras
        # aside so unit_bytes keeps its original three-key shape; unknown keys, if any, are inert.
        meta = dict(meta or {})
        _GLOBAL_STATE.free_vram_bytes = int(meta.pop("free_vram_bytes", 0) or 0)
        _GLOBAL_STATE.cache_floors = meta.pop("floors", None)
        _GLOBAL_STATE.cache_pools = meta.pop("pools", None)
        _GLOBAL_STATE.swa_full_tokens_ratio = float(meta.pop("swa_full_tokens_ratio", 0.0) or 0.0)
        _GLOBAL_STATE.cache_budget_bytes = int(meta.pop("cache_budget_bytes", 0) or 0)
        _GLOBAL_STATE.gpus = list(meta.pop("gpus", None) or [])
        _GLOBAL_STATE.unit_bytes = meta

    # Early-bind: supervise the backend on a daemon thread so uvicorn can bind
    # immediately and /health can report loading progress. Shell mode wants exactly the same
    # thing -- its client waits on /health and renders that progress -- so both paths share
    # this supervisor; only who runs uvicorn differs.
    threading.Thread(
        target=run_backend_supervisor,
        args=(handle, _GLOBAL_STATE.load_progress, _on_ready),
        kwargs={
            "on_failure": _on_failure,
            "on_meta": _on_meta,
            # uvicorn's lifespan shutdown sets this on SIGINT/SIGTERM, so the workers'
            # expected exit during stop is not reported as a crash.
            "is_shutting_down": _SHUTTING_DOWN.is_set,
        },
        name="freetoken-backend-supervisor",
        daemon=True,
    ).start()

    if run_shell:
        _serve_and_run_shell(host, port)
        return
    # uvicorn stays on the main thread (signal handling unchanged); ^C reaches the worker group.
    uvicorn.run(app, host=host, port=port)
