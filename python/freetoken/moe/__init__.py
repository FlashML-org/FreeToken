from __future__ import annotations

# Where routed experts live. The offload family (offload / cpu / hybrid) serves experts from
# pinned host banks through an ``OffloadMoeCache`` -- the GPU only holds the two-layer prefill
# double buffer -- and differs only in how decode gets the experts: ``offload`` streams the
# missing experts over PCIe into a GPU slot cache, ``cpu`` computes them on the CPU from the
# host banks, ``hybrid`` fetches at most K missing experts per layer and computes the rest on
# the CPU, overlapped. ``fused`` keeps every expert resident on the GPU.
MOE_STRATEGIES = ("fused", "offload", "cpu", "hybrid")
OFFLOAD_MOE_STRATEGIES = frozenset({"offload", "cpu", "hybrid"})


def is_offload_moe_strategy(strategy: str) -> bool:
    return strategy in OFFLOAD_MOE_STRATEGIES


__all__ = ["MOE_STRATEGIES", "OFFLOAD_MOE_STRATEGIES", "is_offload_moe_strategy"]
