"""Capture or summarize one explicit ROCm decode trace.

No command runs unless supplied after ``--``. Missing route/lane/token evidence
produces an ``incomplete`` report; profiler gaps never become zero overhead.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

LANES = {"sampled_absolute", "greedy_correctness", "teacher_forced_replay"}
TRACE_FLAGS = (
    "--hip-trace",
    "--marker-trace",
    "--kernel-trace",
    "--memory-copy-trace",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, help="existing rocprof JSON artifact")
    parser.add_argument("--out", type=Path, required=True, help="profile report JSON output")
    parser.add_argument("--rocprof", default="rocprofv3")
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--route", help="observed runtime route, required for complete report")
    parser.add_argument("--graph-mode", choices=("eager", "replay", "disabled"))
    parser.add_argument("--lane", choices=sorted(LANES))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def _v3_timestamp_event(record: object, *, kind: str, name: str | None = None) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    start = record.get("start_timestamp")
    end = record.get("end_timestamp")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end < start:
        return None
    event: dict[str, Any] = {
        "kind": kind,
        "start_ns": start,
        "end_ns": end,
    }
    if name:
        event["name"] = name
    return event


def _normalise_rocprofv3(value: dict[str, Any]) -> dict[str, Any]:
    """Convert rocprofv3 SDK JSON into this report's small event schema.

    rocprofv3 emits one top-level tool record, with dispatches keyed by numeric
    kernel IDs. Keep conversion lossless enough for timing and route evidence;
    application token ranges and route metadata still come from the workload.
    """
    tools = value.get("rocprofiler-sdk-tool")
    if not isinstance(tools, list):
        raise ValueError("rocprofv3 artifact must contain a tool record list")
    kernels: list[dict[str, Any]] = []
    copies: list[dict[str, Any]] = []
    hip_api: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        symbols = {
            item.get("kernel_id"): (
                item.get("truncated_kernel_name")
                or item.get("demangled_kernel_name")
                or item.get("kernel_name")
                or f"kernel_{item.get('kernel_id')}"
            )
            for item in tool.get("kernel_symbols", [])
            if isinstance(item, dict) and item.get("kernel_id") is not None
        }
        buffers = tool.get("buffer_records")
        if not isinstance(buffers, dict):
            continue
        for record in buffers.get("kernel_dispatch", []):
            dispatch = record.get("dispatch_info", {}) if isinstance(record, dict) else {}
            kernel_id = dispatch.get("kernel_id") if isinstance(dispatch, dict) else None
            event = _v3_timestamp_event(
                record, kind="kernel", name=symbols.get(kernel_id, f"kernel_{kernel_id}")
            )
            if event is not None:
                kernels.append(event)
        for record in buffers.get("memory_copy", []):
            event = _v3_timestamp_event(record, kind="copy")
            if event is not None:
                copies.append(event)
        for record in buffers.get("hip_api", []):
            event = _v3_timestamp_event(record, kind="hip_api")
            if event is not None:
                operation = record.get("operation") if isinstance(record, dict) else None
                if operation is not None:
                    event["name"] = f"hip_operation_{operation}"
                hip_api.append(event)
        for agent in tool.get("agents", []):
            if not isinstance(agent, dict) or agent.get("type") != 2:
                continue
            identities.append(
                {
                    "gpu": agent.get("product_name") or agent.get("name"),
                    "target": agent.get("name"),
                    "gfx_target_version": agent.get("gfx_target_version"),
                }
            )
    return {
        "rocprofv3": True,
        "kernels": kernels,
        "copies": copies,
        "hip_api": hip_api,
        "identity": {"profiler": "rocprofv3", "agents": identities},
    }


def load_trace(path: Path) -> dict[str, Any]:
    # rocprofv3 7.2 can place a raw byte in a kernel/module string. Replacement
    # preserves JSON structure and timing fields while marking names lossy.
    value = json.loads(path.read_bytes().decode("utf-8", errors="replace"))
    if isinstance(value, list):
        return {"events": value}
    if not isinstance(value, dict):
        raise ValueError("rocprof artifact must be an object or event list")
    if "rocprofiler-sdk-tool" in value:
        return _normalise_rocprofv3(value)
    return value


def _resolve_generated_trace(path: Path) -> Path:
    """Find rocprof output, including rocprofv3's ``_results.json`` suffix."""
    candidates = [
        path,
        Path(f"{path}_results.json"),
        path.with_suffix(".json"),
        path.with_name(f"{path.stem}_results.json"),
    ]
    candidates.extend(sorted(path.parent.glob(f"{path.name}*_results.json")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"rocprof did not produce a JSON trace; checked: {checked}")


def _events(payload: dict[str, Any], key: str, kind: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    events = payload.get("events", [])
    if not isinstance(events, list):
        return []
    return [item for item in events if isinstance(item, dict) and item.get("kind") == kind]


def _interval(event: dict[str, Any]) -> tuple[float, float] | None:
    start = event.get("start_ns", event.get("start"))
    end = event.get("end_ns", event.get("end"))
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    if end < start:
        return None
    return float(start), float(end)


def _union_ns(events: list[dict[str, Any]], clip: tuple[float, float] | None = None) -> float:
    intervals = []
    for event in events:
        value = _interval(event)
        if value is None:
            continue
        start, end = value
        if clip is not None:
            start, end = max(start, clip[0]), min(end, clip[1])
        if end > start:
            intervals.append((start, end))
    intervals.sort()
    total = 0.0
    current: tuple[float, float] | None = None
    for start, end in intervals:
        if current is None:
            current = (start, end)
        elif start <= current[1]:
            current = (current[0], max(current[1], end))
        else:
            total += current[1] - current[0]
            current = (start, end)
    if current is not None:
        total += current[1] - current[0]
    return total


def _token_ledgers(
    tokens: list[dict[str, Any]], kernels: list[dict[str, Any]], copies: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    for token in tokens:
        clip = _interval(token)
        if clip is None:
            continue
        result.append(
            {
                "token": token.get("token"),
                "start_ns": clip[0],
                "end_ns": clip[1],
                "gpu_ns": _union_ns(kernels + copies, clip),
                "copy_ns": _union_ns(copies, clip),
                "unattributed_ns": max(0.0, clip[1] - clip[0] - _union_ns(kernels + copies, clip)),
            }
        )
    return result


def build_report(
    payload: dict[str, Any],
    *,
    route: str | None = None,
    graph_mode: str | None = None,
    lane: str | None = None,
) -> dict[str, Any]:
    kernels = _events(payload, "kernels", "kernel")
    copies = _events(payload, "copies", "copy")
    hip_api = _events(payload, "hip_api", "hip_api")
    observed = payload.get("observed") if isinstance(payload.get("observed"), dict) else {}
    timing = payload.get("timing") if isinstance(payload.get("timing"), dict) else {}
    route = route or observed.get("route") or payload.get("route")
    graph_mode = graph_mode or observed.get("graph_mode") or payload.get("graph_mode")
    lane = lane or payload.get("lane") or timing.get("lane")
    tokens = payload.get("tokens", payload.get("token_ranges", []))
    if not isinstance(tokens, list):
        tokens = []

    missing = []
    if not payload.get("clock_correlations"):
        missing.append("clock_correlations")
    if not route:
        missing.append("observed.route")
    if graph_mode not in {"eager", "replay", "disabled"}:
        missing.append("observed.graph_mode")
    if lane not in LANES:
        missing.append("timing.lane")
    if not tokens:
        missing.append("tokens")
    status = "complete" if not missing else "incomplete"
    return {
        "schema": "freetoken-rocm-profile-v1",
        "status": status,
        "missing": missing,
        "observed": {
            "route": route,
            "graph_mode": graph_mode,
            "lane": lane,
            "kernel_count": len(kernels),
            "copy_count": len(copies),
            "hip_api_count": len(hip_api),
        },
        "identity": payload.get("identity", {}),
        "ledgers": _token_ledgers(tokens, kernels, copies),
        "warm_offload": {
            "status": "measured" if any("ensure_experts" in str(item.get("name", "")) for item in kernels) else "missing",
            "ensure_experts_count": sum("ensure_experts" in str(item.get("name", "")) for item in kernels),
            "copy_missing_count": sum("copy_missing" in str(item.get("name", "")) for item in kernels),
            "copy_missing_bytes": payload.get("copy_missing_bytes"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trace = args.trace
    command = [item for item in args.command if item != "--"]
    if trace is None:
        if not command:
            raise SystemExit("provide --trace or an explicit command after --")
        trace = args.profile_output or args.out.with_suffix(".rocprof.json")
        subprocess.run(
            [
                args.rocprof,
                *TRACE_FLAGS,
                "--output-file",
                str(trace),
                "--output-format",
                "json",
                "--",
                *command,
            ],
            check=True,
        )
        trace = _resolve_generated_trace(trace)
    report = build_report(
        load_trace(trace), route=args.route, graph_mode=args.graph_mode, lane=args.lane
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
