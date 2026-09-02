"""Launch-mode ROCm trace wrapper and ledger converter.

No command is executed unless the caller supplies ``--`` followed by an explicit
runtime command. The implementation is safe to import and unit-test on CPU-only hosts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

try:
    from benchmarks.lib.rocm_trace import (
        calibrate_clocks, correlate_events, token_ledger, warm_offload_summary,
    )
except ModuleNotFoundError:  # direct ``python benchmarks/profile_decode_rocm.py``
    from lib.rocm_trace import calibrate_clocks, correlate_events, token_ledger, warm_offload_summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, help="existing rocprof JSON artifact")
    parser.add_argument("--out", type=Path, required=True, help="ledger JSON output")
    parser.add_argument("--rocprof", default="rocprofv3")
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def load_trace(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return {"events": payload}
    if not isinstance(payload, dict):
        raise ValueError("rocprof artifact must be an object or event list")
    return payload


def build_report(payload: dict) -> dict:
    records = payload.get("clock_correlations")
    if not records:
        raise ValueError("clock calibration records unavailable")
    calibration = calibrate_clocks(records)
    if not calibration.accepted:
        raise ValueError(f"clock calibration residual exceeds 10us: {calibration.max_residual_ns:.0f}ns")
    events = payload.get("events", [])
    hip = payload.get("hip_api", [event for event in events if event.get("kind") == "hip_api"])
    kernels = payload.get("kernels", [event for event in events if event.get("kind") == "kernel"])
    copies = payload.get("copies", [event for event in events if event.get("kind") == "copy"])
    correlated = correlate_events(hip, kernels, copies)
    return {
        "schema": "freetoken-rocm-ledger-v1",
        "clock": {
            "scale": calibration.scale,
            "offset_ns": calibration.offset_ns,
            "max_residual_ns": calibration.max_residual_ns,
            "accepted": calibration.accepted,
        },
        "ledgers": token_ledger(payload.get("tokens", payload.get("token_ranges", [])), correlated, payload.get("host_ranges", [])),
        "events": correlated,
        "warm_offload": warm_offload_summary(correlated),
        "missing": payload.get("missing", {}),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    trace = args.trace
    command = [item for item in args.command if item != "--"]
    if trace is None:
        if not command:
            raise SystemExit("provide --trace or an explicit command after --")
        profile_output = args.profile_output or args.out.with_suffix(".rocprof.json")
        run = [
            args.rocprof,
            "--hip-trace",
            "--marker-trace",
            "--kernel-trace",
            "--memory-copy-trace",
            "--output-file",
            str(profile_output),
            "--output-format",
            "json",
            "--",
            *command,
        ]
        subprocess.run(run, check=True)
        trace = profile_output
    report = build_report(load_trace(trace))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
