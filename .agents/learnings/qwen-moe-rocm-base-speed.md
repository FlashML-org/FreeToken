# Qwen MoE ROCm Base-Speed Optimization

## Context

2026-08-31: Qwen3.6-35B-A3B GGUF Q4_K_M base decode on RX 7900 XTX/gfx1100.
Plan compared FreeToken against Ollama with MTP excluded.

## Hardest decision

Keep legacy GGUF MMVQ as default after proving the gfx1100 rotated-wave
candidate 9.7% slower, and close cache/dense/handoff branches when matched
measurements showed no throughput win. Correctness and explicit target failure
outweighed speculative promotion.

## Alternatives rejected

- AMD `sdot4` intrinsic rewrite — ROCm target rejects it without `dot1-insts`,
  and candidate disassembly had no dot4 instruction.
- Smaller or fixed MoE cache — 4,352 slots fell to 36.875 tok/s; 8,000 slots
  did not beat auto; warmed runs had zero fetches.
- Dense MMQ or lm_head rewrite — native MMVQ won every exact bs=1 dense case.
- MTP/speculative path — outside requested base-speed scope.

## Least confident

The remaining roughly 13 tok/s gap is attributed to ROCm base execution cadence
and GGUF runtime cost, but profiler CPU totals include overlap attribution and
are not directly additive to API wall time. Revisit with a lower-overhead
timeline or newer ROCm/compiler before changing dispatch again.

## Reuse

Read before future Qwen ROCm speed work. Preserve benchmark contract, exact
completion validation, graph/finite-logit gates, cache hit/fetch telemetry, and
forced-only candidate dispatch in `benchmarks/`, `python/freetoken/kernel/gguf.py`,
and `.plans/qwen-moe-speed/`.
