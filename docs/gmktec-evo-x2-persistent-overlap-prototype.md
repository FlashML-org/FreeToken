# GMKtec EVO-X2 persistent grouped-transfer overlap prototype

This prototype tested a lower-overhead overlap design than the earlier
per-group-event experiment. It uses one persistent transfer stream,
double-buffered pinned host staging, and one completion event per reusable
buffer. It does not load a large model or alter the protected service.

## Method

- GPU: AMD Radeon 8060S, gfx1151.
- PyTorch: `2.13.0+rocm10.0.0`.
- HIP: `7.15.26333`.
- Four groups per round, each containing sixteen random 64 KiB blocks.
- Two reusable pinned host buffers and two device buffers.
- Eight measured rounds after three warmups per condition.
- Serial baseline synchronizes after each group.
- Persistent pipeline synchronizes only when a reusable buffer is needed and
  once at the end of the round.

## Results

| Synthetic compute | Serial mean | Persistent overlap mean | Relative speed |
|---|---:|---:|---:|
| None | 0.317 ms | 1.931 ms | 0.164x |
| One device add per group | 0.450 ms | 1.424 ms | 0.316x |

The persistent design remained slower than serial in both conditions. It was
approximately 6.1 times slower without compute and 3.2 times slower with the
synthetic device operation.

## Decision

This candidate is rejected for promotion. Persistent streams and buffer reuse
alone do not hide the Python-side staging and scheduling cost in this test.
The result does not rule out a native fused runtime path. It does rule out
continuing to add Python-level stream and event orchestration as the primary
optimization strategy.

Future work should move batching into the runtime or a compiled HIP kernel,
where expert indexing, gather, transfer scheduling, and compute can be fused or
queued with substantially fewer host interventions. Any such implementation
must still pass deterministic quality, tail-latency, recovery, and API gates.

