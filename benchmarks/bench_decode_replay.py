"""Teacher-forced replay identity helpers for ROCm comparisons.

Serving adapters may differ, but replay rows must carry same prompt/continuation identity and
route evidence. Sampled streams never enter this lane.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

LANE = "teacher_forced_replay"
REPLAY_SCHEMAS = frozenset({"freetoken-replay-manifest-v1", "freetoken-replay-manifest-v2"})
_HEX64 = set("0123456789abcdefABCDEF")


def _full_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def ids_sha256(ids: Iterable[int]) -> str:
    return hashlib.sha256(json.dumps([int(value) for value in ids], separators=(",", ":")).encode()).hexdigest()


def build_replay_record(
    manifest: dict[str, Any],
    *,
    prompt_ids: Iterable[int],
    continuation_ids: Iterable[int],
    route_digest: str,
    oracle_id: str,
    route_hash_status: str = "matched",
) -> dict[str, Any]:
    """Attach forced-ID and route identity to one already validated timing manifest."""
    prompt = [int(value) for value in prompt_ids]
    continuation = [int(value) for value in continuation_ids]
    if not prompt or not continuation:
        raise ValueError("replay prompt and continuation IDs must be non-empty")
    if not _full_sha(route_digest):
        raise ValueError("replay route digest must be a full SHA-256")
    if not isinstance(oracle_id, str) or not oracle_id.strip():
        raise ValueError("independent replay oracle ID is required")
    result = dict(manifest)
    result["timing"] = {**dict(manifest.get("timing", {})), "lane": LANE}
    result["replay"] = {
        "schema": "freetoken-replay-manifest-v2",
        "forced": True,
        "prompt_ids_sha256": ids_sha256(prompt),
        "continuation_ids_sha256": ids_sha256(continuation),
        "route_digest": route_digest,
        "oracle_id": oracle_id,
        "route_hash_status": route_hash_status,
    }
    return result


def validate_replay_record(value: object) -> list[str]:
    if not isinstance(value, dict):
        return ["replay record is not an object"]
    replay = value.get("replay")
    timing = value.get("timing")
    problems: list[str] = []
    if not isinstance(replay, dict) or replay.get("schema") not in REPLAY_SCHEMAS:
        problems.append("replay.schema must be a known replay manifest schema")
    if not isinstance(timing, dict) or timing.get("lane") != LANE:
        problems.append("timing.lane must be teacher_forced_replay")
    if not isinstance(replay, dict) or replay.get("forced") is not True:
        problems.append("replay.forced must be true")
    if not isinstance(replay, dict) or not _full_sha(replay.get("prompt_ids_sha256")):
        problems.append("replay prompt identity must be a full SHA-256")
    if not isinstance(replay, dict) or not _full_sha(replay.get("continuation_ids_sha256")):
        problems.append("replay continuation identity must be a full SHA-256")
    if not isinstance(replay, dict) or not _full_sha(replay.get("route_digest")):
        problems.append("replay route digest must be a full SHA-256")
    if not isinstance(replay, dict) or not str(replay.get("oracle_id", "")).strip():
        problems.append("replay independent oracle ID is missing")
    if not isinstance(replay, dict) or replay.get("route_hash_status") != "matched":
        problems.append("replay route hashes are not matched")
    return problems
