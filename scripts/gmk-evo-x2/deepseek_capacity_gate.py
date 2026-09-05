#!/usr/bin/env python3
"""Calculate a conservative, metadata-only DeepSeek capacity gate.

The script intentionally performs no model download and no model load.  It
compares a pinned checkpoint payload size with a caller-supplied live memory
observation, while reserving explicit headroom for the operating system,
runtime, KV cache, and recovery.  Strix Halo unified memory is treated as one
shared pool.  The ROCm-reported VRAM aperture is reported for context, but is
not added to the authoritative model budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GIB = 1024**3


def positive_float(value: str) -> float:
    """Parse a positive command-line quantity and reject unsafe values."""

    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def main() -> int:
    """Run the gate and write a machine-readable result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-bytes", type=int, required=True)
    parser.add_argument("--mem-available-gib", type=positive_float, required=True)
    parser.add_argument("--rocm-vram-gib", type=positive_float, required=True)
    parser.add_argument("--os-reserve-gib", type=positive_float, default=8.0)
    parser.add_argument("--runtime-reserve-gib", type=positive_float, default=4.0)
    parser.add_argument("--kv-reserve-gib", type=positive_float, default=4.0)
    parser.add_argument("--recovery-reserve-gib", type=positive_float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # The authoritative budget is available unified memory minus all declared
    # safety reservations.  The separate VRAM aperture is never double-counted.
    reserved_gib = (
        args.os_reserve_gib
        + args.runtime_reserve_gib
        + args.kv_reserve_gib
        + args.recovery_reserve_gib
    )
    authoritative_budget_gib = max(args.mem_available_gib - reserved_gib, 0.0)
    payload_gib = args.payload_bytes / GIB
    deficit_gib = payload_gib - authoritative_budget_gib

    # This optimistic number is diagnostic only.  It shows why adding the
    # reported ROCm aperture would still not make the host qualify, while the
    # authoritative decision remains based on the shared UMA pool.
    optimistic_budget_gib = args.mem_available_gib + args.rocm_vram_gib
    result = {
        "decision": "PASS_METADATA_ONLY" if deficit_gib <= 0 else "REJECT_FULL_LOAD",
        "payload_bytes": args.payload_bytes,
        "payload_gib": round(payload_gib, 3),
        "mem_available_gib": args.mem_available_gib,
        "rocm_vram_aperture_gib": args.rocm_vram_gib,
        "reserves_gib": {
            "os": args.os_reserve_gib,
            "runtime": args.runtime_reserve_gib,
            "kv_cache": args.kv_reserve_gib,
            "recovery": args.recovery_reserve_gib,
            "total": round(reserved_gib, 3),
        },
        "authoritative_model_budget_gib": round(authoritative_budget_gib, 3),
        "optimistic_budget_including_vram_gib": round(optimistic_budget_gib, 3),
        "authoritative_deficit_gib": round(max(deficit_gib, 0.0), 3),
        "optimistic_deficit_gib": round(max(payload_gib - optimistic_budget_gib, 0.0), 3),
        "interpretation": (
            "The full payload cannot be admitted with the declared headroom. "
            "Do not download or load it on this host."
            if deficit_gib > 0
            else "The metadata gate passes; a guarded tiny-slice test is permitted."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
