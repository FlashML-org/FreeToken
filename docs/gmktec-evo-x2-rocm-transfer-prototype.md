# GMKtec EVO-X2 ROCm transfer prototype

This read-only prototype measures contiguous host and device copies in the
existing FreeToken Python environment. It is a lower-bound systems datapoint
for the DeepSeek offload decision, not a model benchmark. It does not download
weights, start a model, or change the protected service.

## Environment

| Field | Observation |
|---|---|
| GPU | AMD Radeon 8060S Graphics |
| Architecture | gfx1151 |
| PyTorch | 2.13.0+rocm10.0.0 |
| HIP runtime | 7.15.26333 |
| Transfer size | 64 MiB per copy |
| Repetitions | 20 measured copies after 5 warmups |
| Synchronization | `torch.cuda.synchronize()` after every copy |

## Measured copies

| Direction | Mean time | Minimum time | Effective rate |
|---|---:|---:|---:|
| Host to device, pageable | 0.841 ms | 0.825 ms | 79.79 GB/s |
| Host to device, pinned | 0.842 ms | 0.818 ms | 79.67 GB/s |
| Device to host, pageable | 0.955 ms | 0.947 ms | 70.24 GB/s |

Pinned memory was available, but it did not materially change the result for
this small contiguous copy. The source and destination were single contiguous
64 MiB tensors, so these values should not be interpreted as the bandwidth of
small, scattered expert-block fetches.

## Implication for the paper model

At the measured host-to-device contiguous-copy rate, moving the paper's stated
approximately 140 GB routed-expert volume once has an ideal transfer floor of
approximately 1.75 seconds. This is close to the paper's 80 GB/s reference
scale, but it excludes tensor slicing, page faults, format conversion, cache
miss scheduling, synchronization, and repeated decode fetches.

The result therefore removes one uncertainty: the EVO-X2 memory fabric can
reach roughly the same raw bandwidth class as the paper's stated host-side
bandwidth examples. It does not remove the dominant capacity problem. The
official checkpoint is approximately 148.66 GiB while the live available host
memory was approximately 18 GiB, so most of the model would still need to be
reloaded or streamed repeatedly.

## Next measurement

The next useful prototype should use a synthetic expert-block access pattern:
many small, non-contiguous blocks with the same sizes and batching as the
runtime's miss path. It should report throughput, launch overhead, and p95/p99
copy latency. A full checkpoint download remains gated on that result and on a
resident-memory budget that preserves the protected service.

