"""Read-only control-plane endpoints consumed by the desktop app: /health (lifecycle),
/v1/runtime/identity (local process identity), /v1/stats (runtime metrics, Task 6), and
/v1/requests (request log ring, Task 5).

All handlers read a shared FrontendManager snapshot via ``get_state``; nothing here touches
the scheduler or blocks. Registered on the app alongside the OpenAI/Anthropic/Responses routes.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from freetoken.daemon.osproc import proc_pgid, read_starttime

from .accounting import _is_loopback


RUNTIME_IDENTITY_SCHEMA = "freetoken.runtime_identity"
RUNTIME_IDENTITY_SCHEMA_VERSION = 1


def _process_runtime(pid: int | None) -> dict[str, int | None]:
    if not isinstance(pid, int) or pid <= 0:
        return {"pid": None, "pgid": None, "starttime": None}
    return {
        "pid": pid,
        "pgid": proc_pgid(pid),
        "starttime": read_starttime(pid),
    }


def _worker_role(name: str) -> tuple[str, int | None]:
    if name.startswith("freetoken-TP") and name.endswith("-scheduler"):
        try:
            return "scheduler", int(name[len("freetoken-TP") : -len("-scheduler")])
        except ValueError:
            return "scheduler", None
    if name.startswith("freetoken-detokenizer-"):
        try:
            return "detokenizer", int(name.rsplit("-", 1)[1])
        except ValueError:
            return "detokenizer", None
    if name.startswith("freetoken-tokenizer-"):
        try:
            return "tokenizer", int(name.rsplit("-", 1)[1])
        except ValueError:
            return "tokenizer", None
    return "worker", None


def build_runtime_identity_snapshot(state: Any, processes: list[Any]) -> dict[str, Any]:
    """Bind caller correlation separately from process facts observed by this frontend.

    The snapshot is intentionally narrow: no argv, environment, model paths, executable paths,
    or caller-supplied hash claims. A supervisor can independently bind those external facts to
    this process using the nonce plus PID/PGID/start-time tuple.
    """
    workers = []
    for process in processes:
        name = str(getattr(process, "name", "worker"))
        role, index = _worker_role(name)
        worker = {
            "worker_id": name,
            "role": role,
            "index": index,
            **_process_runtime(getattr(process, "pid", None)),
        }
        workers.append(worker)

    config = getattr(state, "config", None)
    return {
        "schema": RUNTIME_IDENTITY_SCHEMA,
        "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
        "correlation": {
            "launch_nonce": getattr(config, "launch_nonce", None),
        },
        "runtime": {
            "instance_id": getattr(state, "instance_id", None),
            "frontend": {"role": "frontend", **_process_runtime(os.getpid())},
            "workers": workers,
        },
    }


def _build_runtime_identity_unlocked(state: Any) -> dict[str, Any] | None:
    snapshot = getattr(state, "runtime_identity", None)
    if not isinstance(snapshot, dict):
        return None
    doc = copy.deepcopy(snapshot)
    maintenance = str(getattr(state, "maintenance_state", "failed"))
    if getattr(state, "fatal_error", None) is not None:
        status = "failed"
    elif maintenance in {"serving", "rebuilding"}:
        status = "ready"
    elif maintenance in {"loading", "failed", "stopping"}:
        status = maintenance
    else:
        status = "failed"
    doc["lifecycle"] = {
        "status": status,
        "maintenance_state": maintenance,
    }
    return doc


def build_runtime_identity(state: Any) -> dict[str, Any] | None:
    # Production FrontendManager publishes shell-mode PGID refreshes and lifecycle transitions
    # under this lock. Read both under the same lock so a response cannot combine a stale loading
    # snapshot with a newly-ready lifecycle. Compatible external/fake states need not provide it.
    lock = getattr(state, "_maintenance_lock", None)
    if lock is None:
        return _build_runtime_identity_unlocked(state)
    with lock:
        return _build_runtime_identity_unlocked(state)


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
    return {
        "status": "ok",
        "model": model,
        "instance_id": instance_id,
        "uptime_s": uptime_s,
        "maintenance": mstate,
        "version": version,
    }


def register_control_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict] | None = None,
) -> None:
    @app.get("/health")
    async def health():
        return build_health(get_state(), app.version)

    @app.get("/v1/runtime/identity")
    async def runtime_identity(request: Request):
        # This is a process-supervisor surface, not a remotely consumable model API. Keep it
        # loopback-only even when the generation server is deliberately bound to 0.0.0.0.
        if not _is_loopback(request.client.host if request.client else None):
            return JSONResponse(status_code=403, content={"error": "loopback access required"})
        doc = build_runtime_identity(get_state())
        if doc is None:
            return JSONResponse(
                status_code=503,
                content={"error": "runtime identity is not bound"},
            )
        return doc

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
