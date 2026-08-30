#!/usr/bin/env python3
"""Validate and summarize a retained LAN-223 Qwen GGUF endurance artifact.

The endurance wrapper stores one JSON result and one process-scoped memory
sample for each deterministic multi-turn conversation.  This program turns
those raw files into a single machine-readable conclusion without treating
unrelated whole-host swap as evidence that the model process was swapped.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


# Accept only the session naming contract emitted by the endurance shell driver.
SESSION_NAME = re.compile(r"session-(\d+)\.json$")
# Read the two swap fields deliberately rather than parsing unrelated telemetry.
SWAP_FIELD = re.compile(r"^(runner_swap_kib|whole_host_swap_kib)=(\d+)$", re.M)


def percentile(values: Iterable[float], fraction: float) -> float:
    """Return a nearest-rank percentile for a non-empty numeric collection."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty collection")
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def read_swap_fields(path: Path) -> dict[str, int]:
    """Extract the explicitly recorded per-runner and whole-host swap values."""

    values = {name: int(value) for name, value in SWAP_FIELD.findall(path.read_text())}
    if set(values) != {"runner_swap_kib", "whole_host_swap_kib"}:
        raise ValueError(f"missing swap fields in {path}")
    return values


def session_number(path: Path) -> int:
    """Return the numeric session label, rejecting unrelated JSON files."""

    match = SESSION_NAME.search(path.name)
    if not match:
        raise ValueError(f"unexpected session filename: {path.name}")
    return int(match.group(1))


def summarize(artifact_root: Path, expected_sessions: int) -> dict[str, Any]:
    """Validate every session and return portable summary metrics and failures."""

    sessions_dir = artifact_root / "sessions"
    session_paths = sorted(sessions_dir.glob("session-*.json"), key=session_number)
    failures: list[str] = []
    ttfts: list[float] = []
    gaps: list[float] = []
    runner_swaps: list[int] = []
    host_swaps: list[int] = []

    if len(session_paths) != expected_sessions:
        failures.append(f"expected {expected_sessions} sessions, found {len(session_paths)}")

    for session_path in session_paths:
        session = session_number(session_path)
        payload = json.loads(session_path.read_text())
        if payload.get("status") != "passed":
            failures.append(f"session {session:02d} status={payload.get('status')!r}")
        for turn in payload.get("results", []):
            if turn.get("status") != "passed":
                failures.append(
                    f"session {session:02d} turn {turn.get('id', '<unknown>')} "
                    f"status={turn.get('status')!r}"
                )
        tail = payload.get("tail_metrics", {})
        try:
            ttfts.append(float(tail["max_ttft_seconds"]))
            gaps.append(float(tail["max_token_gap_seconds"]))
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"session {session:02d} missing tail metric: {error}")

        telemetry = sessions_dir / f"session-{session:02d}-telemetry.txt"
        if not telemetry.is_file():
            failures.append(f"session {session:02d} missing telemetry")
            continue
        try:
            swap = read_swap_fields(telemetry)
        except ValueError as error:
            failures.append(str(error))
            continue
        runner_swaps.append(swap["runner_swap_kib"])
        host_swaps.append(swap["whole_host_swap_kib"])
        if swap["runner_swap_kib"] != 0:
            failures.append(
                f"session {session:02d} runner swap={swap['runner_swap_kib']} KiB"
            )

    return {
        "schema_version": 1,
        "artifact_root": str(artifact_root),
        "expected_sessions": expected_sessions,
        "observed_sessions": len(session_paths),
        "passed": not failures,
        "failures": failures,
        "max_turn_ttft_seconds": {
            "mean": mean(ttfts) if ttfts else None,
            "p95": percentile(ttfts, 0.95) if ttfts else None,
            "p99": percentile(ttfts, 0.99) if ttfts else None,
            "max": max(ttfts) if ttfts else None,
        },
        "max_visible_token_gap_seconds": {
            "mean": mean(gaps) if gaps else None,
            "p95": percentile(gaps, 0.95) if gaps else None,
            "p99": percentile(gaps, 0.99) if gaps else None,
            "max": max(gaps) if gaps else None,
        },
        "runner_swap_kib": {
            "min": min(runner_swaps) if runner_swaps else None,
            "max": max(runner_swaps) if runner_swaps else None,
        },
        "whole_host_swap_kib": {
            "min": min(host_swaps) if host_swaps else None,
            "max": max(host_swaps) if host_swaps else None,
        },
    }


def main() -> int:
    """Parse arguments, write the summary, and use exit status as the gate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--expected-sessions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.artifact_root, args.expected_sessions)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
