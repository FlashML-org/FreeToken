"""Read-only control-plane endpoints consumed by the desktop app: /health (lifecycle),
/v1/stats (runtime metrics, Task 6), /v1/requests (request log ring, Task 5).

All handlers read a shared FrontendManager snapshot via ``get_state``; nothing here touches
the scheduler or blocks. Registered on the app alongside the OpenAI/Anthropic/Responses routes.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from fastapi import FastAPI


def build_health(state: Any, version: str) -> dict:
    """Full-lifecycle health doc: loading -> ok -> error."""
    instance_id = getattr(state, "instance_id", None)
    fatal = getattr(state, "fatal_error", None)
    if fatal:
        return {"status": "error", "message": fatal, "instance_id": instance_id}

    mstate = getattr(state, "maintenance_state", "serving")
    config = getattr(state, "config", None)
    model = getattr(config, "served_model_name", None)

    if mstate == "loading":
        lp = getattr(state, "load_progress", None)
        return {
            "status": "loading",
            "phase": lp.phase if lp is not None else "other",
            "progress": {
                "done_bytes": lp.done_bytes if lp is not None else 0,
                "total_bytes": lp.total_bytes if lp is not None else 0,
            },
            "model": model,
            "instance_id": instance_id,
        }

    ready_at = getattr(state, "ready_at", None)
    uptime_s = max(0, int(time.monotonic() - ready_at)) if ready_at is not None else 0
    doc = {
        "status": "ok",
        "model": model,
        "instance_id": instance_id,
        "uptime_s": uptime_s,
        "maintenance": mstate,
        "version": version,
    }
    # KV quant / context info so dashboards (desktop console, config panel) can show
    # what the engine is actually running without parsing the serve command line.
    if config is not None:
        kv_dtype = getattr(config, "kv_cache_dtype", None)
        if kv_dtype and kv_dtype != "auto":
            doc["kv_cache_dtype"] = kv_dtype
        reserve = getattr(config, "kv_reserve_tokens", None)
        if reserve:
            doc["kv_reserve_tokens"] = reserve
        moe_backend = getattr(config, "moe_backend", None)
        if moe_backend:
            doc["moe_backend"] = moe_backend
    return doc


def register_control_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict] | None = None,
) -> None:
    @app.get("/health")
    async def health():
        return build_health(get_state(), app.version)

    from . import request_ring

    @app.get("/v1/requests")
    async def list_requests(since: int = 0, limit: int = 100):
        limit = max(1, min(limit, 512))
        entries, next_cursor = request_ring.requests_since(since, limit)
        return {"entries": entries, "next_cursor": next_cursor}

    from .stats import build_stats

    @app.get("/v1/stats")
    async def stats():
        doc = build_stats(
            get_state(), request_ring.requests_p95_ms(), request_ring.requests_ttft_mean_ms()
        )
        # Surface the model's recommended sampling (from its generation_config.json / GGUF
        # metadata) so clients can seed their sampling controls per-model instead of guessing.
        if get_model_sampling is not None:
            doc["model"]["sampling"] = get_model_sampling() or {}
        return doc
