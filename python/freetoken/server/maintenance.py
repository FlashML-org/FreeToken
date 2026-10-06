"""What a request is told while the engine is not serving: one reason per lifecycle stage, shared by every
API's gate, each of which wraps it in its own error format."""

from __future__ import annotations

from typing import Any

_REFUSAL_REASONS = {
    "loading": "model is still loading",
    "rebuilding": "cache rebuild in progress",
    "stopping": "server is stopping",
    "failed": "maintenance failed (restart required)",
}


def refusal_reason(state: Any) -> str | None:
    """Why a request is refused at ``state``'s lifecycle stage, or None while the engine serves."""
    stage = getattr(state, "maintenance_state", "serving")
    return None if stage == "serving" else _REFUSAL_REASONS[stage]
