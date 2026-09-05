#!/usr/bin/env python3
"""Project the transfer-only lower bound for a DeepSeek routed token.

This is an analytical bound, not a model benchmark.  It uses measured
real-shape H2D bandwidth and the exact expert geometry to estimate the time
spent moving routed expert bytes when a chosen fraction of expert accesses miss
the GPU cache.  It excludes computation, routing, synchronization, attention,
KV state, allocator overhead, and all cache-management costs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GIB = 1024**3


def main() -> int:
    """Calculate and write the transfer-only projection table."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-bytes", type=int, default=13_369_344)
    parser.add_argument("--layers", type=int, default=43)
    parser.add_argument("--active-experts", type=int, default=6)
    parser.add_argument("--h2d-gib-per-second", type=float, required=True)
    parser.add_argument("--miss-rates", default="1.0,0.75,0.5,0.25")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if min(args.expert_bytes, args.layers, args.active_experts, args.h2d_gib_per_second) <= 0:
        parser.error("all geometry and bandwidth values must be positive")
    miss_rates = [float(item) for item in args.miss_rates.split(",")]
    if any(rate < 0 or rate > 1 for rate in miss_rates):
        parser.error("miss rates must be between 0 and 1")

    routed_bytes = args.expert_bytes * args.layers * args.active_experts
    rows = []
    for miss_rate in miss_rates:
        moved_bytes = routed_bytes * miss_rate
        seconds = moved_bytes / GIB / args.h2d_gib_per_second
        rows.append({
            "miss_rate": miss_rate,
            "moved_gib_per_token": moved_bytes / GIB,
            "transfer_seconds_per_token": seconds,
            "transfer_only_tokens_per_second": 1 / seconds if seconds else None,
        })

    result = {
        "scope": "analytical transfer-only lower bound",
        "expert_bytes": args.expert_bytes,
        "layers": args.layers,
        "active_experts_per_layer": args.active_experts,
        "routed_bytes_per_token_at_100_percent_miss": routed_bytes,
        "routed_gib_per_token_at_100_percent_miss": routed_bytes / GIB,
        "measured_h2d_gib_per_second": args.h2d_gib_per_second,
        "rows": rows,
        "excluded": [
            "matrix computation",
            "router and dispatch",
            "attention and recurrent state",
            "KV cache",
            "synchronization",
            "allocator overhead",
            "cache lookup and eviction",
            "D2H traffic",
        ],
        "full_model_serving_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
