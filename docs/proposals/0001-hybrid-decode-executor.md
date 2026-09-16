# Proposal 0001: Extract Hybrid MoE Decode Orchestration

- Status: Draft
- Date: 2026-09-16
- Base: `fix/issue-350-rocm-cpu-moe-graph` (`e67f08e`)
- Initial platform scope: NVIDIA CUDA and AMD ROCm

## Summary

Move Hybrid MoE decode orchestration out of `OffloadMoELayer` into a
platform-neutral `HybridDecodeExecutor`.

The first version deliberately preserves the existing device synchronization
implementations:

- CUDA continues to use the current mapped-pinned flag/memop or host-callback
  path in `CpuMoeExecutor`;
- ROCm continues to use the signal-memory and graph batch-memop path introduced
  for CPU/Hybrid graph replay safety;
- model- and quantization-specific GPU expert kernels remain behind a narrow
  callback supplied by `OffloadMoELayer`.

This is the first step of a larger separation between Hybrid scheduling, expert
cache management, CPU expert compute, and device-specific host-task bridges. It
does not redesign those later boundaries in the same change.

## Motivation

Before this proposal, `OffloadMoELayer._decode_hybrid` performs all of the
following:

1. preserves raw routed expert IDs;
2. runs capped-fetch LRU and rewrites IDs to cache slots or `-1`;
3. splits routes between CPU and GPU;
4. submits CPU expert work;
5. copies missing experts and runs the GPU expert kernel;
6. waits for CPU work;
7. merges the two partial outputs.

None of that ordering is model-specific. The model-specific part is only the
expert kernel selected by `_expert_gemm`. Keeping the schedule in the model
layer forces each future model to understand cache internals, CPU overflow
markers, and overlap ordering.

It also makes later device work harder: platform synchronization and Hybrid
scheduling appear to be one problem even though they are separate concerns.

## Goals

- Give Hybrid decode one reusable execution boundary.
- Keep the exact current tensor operations and operation order.
- Keep CPU submission before PCIe fetch and GPU expert compute when overlap is
  enabled.
- Preserve the in-place rewrite of `topk_ids`.
- Preserve the fixed submit/wait structure even when all routes hit the GPU.
- Keep CUDA and ROCm synchronization behind the existing CPU executor API.
- Let a model provide only a narrow GPU expert runner.
- Add CPU-only contract tests for orchestration and ordering.

## Non-goals

- Rewriting LRU or capped-fetch policy.
- Changing Hybrid fetch selection or bandwidth balancing.
- Changing any BF16, NVFP4, MXFP4, DS-FP4, or Q4_0 math kernel.
- Replacing `torch.cuda`; ROCm PyTorch intentionally exposes that namespace.
- Refactoring the native CPU MoE extension.
- Introducing a complete platform abstraction in this proposal.
- Changing CUDA or ROCm graph capability and fallback behavior.

## Design

### Request and GPU callback

The new module defines a request containing only the data required by Hybrid
decode:

```python
@dataclass(frozen=True)
class HybridDecodeRequest:
    layer_id: int
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
```

The model supplies a narrow GPU callback:

```python
class GpuExpertRunner(Protocol):
    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> torch.Tensor: ...
```

`OffloadMoELayer._run_cached_decode_experts` implements this callback by invoking
the existing `_expert_gemm` dispatch over the cache's current bank views and
per-slot scales. Subclasses that already specialize `_expert_gemm` continue to
work without copying the Hybrid schedule.

### Execution order

With overlap enabled, the executor preserves this sequence:

```text
ensure_experts_hybrid
record_stats (optional)
cpu.decode_submit
cache.copy_missing
gpu_expert_runner
cpu.decode_sync
gpu_result + cpu_result
```

With `FREETOKEN_HYBRID_OVERLAP=0`, only the wait position changes:

```text
ensure_experts_hybrid
record_stats (optional)
cpu.decode_submit
cpu.decode_sync
cache.copy_missing
gpu_expert_runner
gpu_result + cpu_result
```

No `.item()`, host-side route branch, stream synchronize, or GPU synchronize is
introduced.

### Route partition

`ensure_experts_hybrid` rewrites the caller-owned `topk_ids` in place:

- non-negative values are GPU cache slot IDs;
- `-1` marks an overflow route assigned to the CPU.

The executor keeps a clone of the original expert IDs and constructs:

- CPU IDs: raw expert ID for CPU routes, `-1` for GPU routes;
- GPU slot IDs: rewritten slot ID for GPU routes, slot `0` for CPU routes;
- GPU weights: original weight for GPU routes, zero for CPU routes.

The CPU kernel skips negative IDs and the GPU receives zero weight for CPU
routes, so every route is computed exactly once.

### Temporary ownership

For this first change, `OffloadMoeCache.set_cpu_executor` constructs and stores
the `HybridDecodeExecutor` beside the already cache-owned `cpu_executor`. This
preserves the current Engine initialization order and avoids mixing an ownership
migration into the scheduling extraction.

This is an explicit transition seam. A follow-up proposal will introduce
`OffloadMoeRuntime` and move ownership of both execution objects out of the
cache.

## CUDA and ROCm compatibility

`HybridDecodeExecutor` contains no vendor or runtime checks. It only calls the
existing CPU executor contract:

```text
decode_submit(...)
decode_sync(pending)
```

The platform behavior therefore remains:

| Platform | Existing synchronization retained by v1 |
|---|---|
| CUDA | mapped-pinned ready/done flags and stream memory operations, with the existing host-callback fallback |
| ROCm | HIP signal memory, eager stream write/wait, and explicit graph batch-memory-op nodes when the runtime probe succeeds |

ROCm graph replay still fails closed to the correct eager path when the native
handshake is unavailable. This proposal neither weakens nor duplicates that
probe.

## Alternatives considered

### Put the schedule in `BaseMoeBackend.forward`

The existing backend interface assumes resident weights and does not naturally
carry cache, host-bank, or model-specific expert runner state. Expanding it now
would make the first extraction much larger.

### Add a global `Platform` class first

The immediate coupling is semantic, not an API spelling problem. A broad
platform wrapper would move `is_rocm()` checks without separating Hybrid route
policy from host-task synchronization.

### Rewrite LRU for AMD

LRU is not the source of this coupling. The device-side cache policy is already
shared. Rewriting it would add graph, correctness, and performance risk without
creating the missing execution boundary.

## Testing

The new platform-independent tests use fake cache, CPU executor, and GPU runner
objects. They verify:

- overlapped call ordering;
- serialized call ordering;
- CPU and GPU route masks are complementary;
- GPU receives clamped slot IDs and zeroed CPU-route weights;
- CPU receives raw expert IDs only for overflow routes;
- `topk_ids` keeps its existing in-place rewrite behavior;
- all-GPU steps retain the fixed CPU submit/wait structure;
- stats update exactly once when enabled;
- `OffloadMoELayer` delegates instead of owning the schedule.

Existing CUDA and ROCm CPU/Hybrid numerical and graph tests remain the platform
regression suite.

## Rollout and follow-ups

The intended sequence after this proposal is:

1. introduce `OffloadMoeRuntime` and typed decode modes;
2. replace the pending tuple with a named opaque pending object;
3. introduce a `HostTaskBridge` capability contract;
4. move CUDA/ROCm selection into the bridge factory;
5. split CPU compute and device synchronization into separate native translation
   units.

Each step must preserve both CUDA and ROCm before the compatibility seam from
the previous step is removed.

## Acceptance criteria

- `OffloadMoELayer` no longer contains the Hybrid CPU/GPU overlap schedule.
- `HybridDecodeExecutor` contains no CUDA, HIP, or vendor checks.
- Model-specific behavior is limited to `GpuExpertRunner`.
- CUDA and ROCm keep their existing synchronization and graph behavior.
- Platform-independent orchestration tests pass without a GPU.
- Existing platform regression tests remain unchanged.
- The structural move adds no measurable decode throughput regression.
