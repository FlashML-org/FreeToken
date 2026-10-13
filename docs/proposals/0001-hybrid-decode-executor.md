# Hybrid decode orchestration

`HybridDecodeExecutor` extracts the CPU/GPU decode schedule from
`OffloadMoELayer`. It is a refactor, not a new cache policy or expert kernel.

## Boundaries

- `OffloadMoeCache` owns LRU and capped-fetch decisions. It rewrites raw expert IDs
  into GPU slot IDs, leaving `-1` for routes assigned to the CPU.
- `HybridDecodeExecutor` preserves raw IDs for CPU overflow, masks the two sets
  of routes, submits CPU work, copies GPU misses, invokes the GPU callback,
  waits for CPU work and adds the two partial results.
- `OffloadMoELayer` supplies the GPU callback and retains quantization dispatch.
  Kernels supporting inactive slots receive `-1`; others receive safe slot 0
  with zero weight. The executor does not choose kernels.
- `CpuMoeExecutor.decode_submit()` / `decode_sync()` own CPU compute, buffers,
  and platform synchronization. They are unchanged by this extraction.

The cache creates one hybrid executor when `set_cpu_executor()` is called, so
the engine's initialization order is unchanged. Pure CPU layers still bypass
hybrid scheduling, including CPU-only layers within an otherwise hybrid model.
Reattaching a CPU executor also replaces the hybrid executor. Its cache reference
is weak, so ownership does not form a cycle that delays releasing expert banks.

## Preserved contracts

The caller's `topk_ids` is mutable and rewritten in place; hidden states and
router weights are forwarded to the existing kernels. Statistics remain optional.
CPU submit precedes GPU copies and math, and CPU sync follows them.
`FREETOKEN_HYBRID_OVERLAP=0`, read at startup, moves the wait before GPU work for
serial A/B measurements. Even an all-GPU step submits and waits for the empty
CPU side, keeping the schedule independent of the runtime routing split.

## CUDA and ROCm

Both platforms share the orchestration above. Graph support is inherited from
the underlying cache, kernels and CPU executor; extracting the schedule does
not make an unsupported synchronization path graph-safe. In particular, the
ROCm CPU/Hybrid graph fix in PR #378 is separate and is not included here.
This refactor applies directly to `main` and does not depend on that fix.

Tests in `tests/moe/test_hybrid_decode_executor.py` check routing against an
independent weighted-expert reference, ordering, and the layer callback's slot
contract. Real CPU/GPU integration tests exercise the same layer entry point.
These are not full-model throughput measurements.
