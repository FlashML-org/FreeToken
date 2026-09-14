"""Fail-closed final-accounting barrier used by the daemon before stopping a serve.

The endpoint closes frontend admission first, lets already-admitted work drain for a bounded
period, then aborts anything still active and waits for the scheduler's terminal abort
acknowledgements.  Only after ``StatsTracker.active`` reaches zero is the cumulative snapshot
sealed.  The daemon persists that snapshot before it sends SIGTERM, so process shutdown can no
longer race the last sampled-token reply.
"""

from __future__ import annotations

import asyncio
import time
from ipaddress import IPv6Address, ip_address
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class AdmissionClosedError(RuntimeError):
    """A generation tried to enter after the prepare-stop gate was closed."""


class AccountingDrainError(RuntimeError):
    """The engine could not reach a sealed accounting state within the bounded stop barrier."""


async def _wait_for_idle(stats: Any, timeout_s: float) -> bool:
    """Wait until all admitted requests have emitted a terminal reply, without an unbounded wait."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, timeout_s)
    while int(getattr(stats, "active", 0)) > 0:
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(0.02, remaining))
    return True


async def prepare_stop_accounting(
    state: Any,
    *,
    drain_timeout_s: float = 5.0,
    abort_timeout_s: float = 3.0,
) -> dict[str, Any]:
    """Close admission, drain/abort accepted work, and return one idempotent sealed snapshot.

    A timeout never reopens admission: the daemon must preserve the process and retry, because
    killing an engine that has not crossed the terminal accounting barrier could lose a late
    sampled token.  A successful result is cached for this process generation so a daemon retry
    after a response loss receives exactly the same totals.
    """

    lock = getattr(state, "_accounting_prepare_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        state._accounting_prepare_lock = lock

    async with lock:
        sealed = getattr(state, "_sealed_accounting", None)
        if sealed is not None:
            return dict(sealed)

        maintenance = getattr(state, "maintenance_state", "serving")
        if maintenance == "rebuilding":
            raise AccountingDrainError("cache rebuild is in progress; retry stop after it finishes")
        if maintenance not in {"loading", "serving", "stopping", "failed"}:
            raise AccountingDrainError(f"engine cannot prepare stop from state {maintenance!r}")

        # This transition and FrontendManager.new_user's check run on the same event loop, making
        # the generation gate atomic with respect to every protocol adapter. Production state
        # also serializes it with the backend supervisor so a concurrent failure stays terminal;
        # the direct assignment preserves compatibility with focused/external state adapters.
        mark_stopping = getattr(state, "mark_stopping", None)
        if callable(mark_stopping):
            mark_stopping()
        else:
            state.maintenance_state = "stopping"

        stats = state.stats
        drained = await _wait_for_idle(stats, drain_timeout_s)
        if not drained:
            inflight = list(getattr(stats, "inflight_uids", ()))
            if not inflight:
                # Defensive: active>0 without identities cannot be aborted or proven terminal.
                raise AccountingDrainError(
                    f"accounting drain timed out with {stats.active} unidentified request(s)"
                )
            try:
                await asyncio.gather(*(state.abort_user(uid) for uid in inflight))
            except Exception as exc:  # noqa: BLE001 -- preserve engine on any abort transport error
                raise AccountingDrainError(f"failed to abort active requests: {exc}") from exc
            if not await _wait_for_idle(stats, abort_timeout_s):
                raise AccountingDrainError(
                    f"accounting abort barrier timed out with {stats.active} request(s) still active"
                )

        config = state.config
        ready_at = getattr(state, "ready_at", None)
        uptime_s = (
            max(0, int(time.monotonic() - ready_at)) if ready_at is not None else 0
        )
        sealed = {
            "instance_id": state.instance_id,
            "model_id": getattr(config, "served_model_name", None),
            "prompt_tokens_total": int(stats.prompt_tokens_total),
            "completion_tokens_total": int(stats.completion_tokens_total),
            "uptime_s": uptime_s,
            "drain_complete": True,
        }
        state._sealed_accounting = dict(sealed)
        return sealed


class PrepareStopBody(BaseModel):
    # Keep the total below the daemon's independent 15s upstream timeout.
    drain_timeout_s: float = Field(default=5.0, ge=0.0, le=10.0)
    abort_timeout_s: float = Field(default=3.0, ge=0.0, le=4.0)


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    try:
        address = ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    if address.is_loopback:
        return True
    return (
        isinstance(address, IPv6Address)
        and address.ipv4_mapped is not None
        and address.ipv4_mapped.is_loopback
    )


def register_accounting_routes(app: FastAPI, get_state: Callable[[], Any]) -> None:
    async def _admission_closed(_: Request, exc: AdmissionClosedError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": str(exc)})

    app.add_exception_handler(AdmissionClosedError, _admission_closed)

    @app.post("/v1/admin/prepare-stop")
    async def prepare_stop(request: Request, body: PrepareStopBody | None = None):
        # This endpoint closes admission and can abort live requests, so it must never be a
        # remotely callable part of an engine bound to 0.0.0.0. The daemon/Desktop control plane
        # is intentionally colocated and always reaches it over loopback.
        if not _is_loopback(request.client.host if request.client else None):
            return JSONResponse(status_code=403, content={"error": "loopback access required"})
        body = body or PrepareStopBody()
        try:
            return await prepare_stop_accounting(
                get_state(),
                drain_timeout_s=body.drain_timeout_s,
                abort_timeout_s=body.abort_timeout_s,
            )
        except AccountingDrainError as exc:
            return JSONResponse(
                status_code=503,
                content={"error": str(exc), "drain_complete": False, "engine_preserved": True},
            )
