"""Shared maintenance-gate helpers for API entrypoints."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse


def maintenance_state_of(state: Any) -> str:
    return getattr(state, "maintenance_state", "serving")


def maintenance_unavailable_detail(mstate: str) -> str | None:
    """Client-facing unavailable message, or None when serving."""
    if mstate == "serving":
        return None
    if mstate == "loading":
        return "model is still loading"
    if mstate == "failed":
        return "server unavailable: maintenance failed (restart required)"
    return "server unavailable: cache rebuild in progress"


def maintenance_gate(state: Any) -> JSONResponse | None:
    """503 while the engine is not serving. None when serving."""
    if (msg := maintenance_unavailable_detail(maintenance_state_of(state))) is None:
        return None
    return JSONResponse({"error": msg}, status_code=503)
