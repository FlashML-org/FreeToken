# GMKtec EVO-X2 mapped-host descriptor gather

This prototype compiled a HIP kernel that consumes one device-side descriptor
list and gathers expert-like blocks directly from mapped pinned host memory. It
is the first test of a device-side miss path with one kernel submission rather
than one host synchronization per block. It does not load a model or alter the
protected service.

## Method

- GPU: AMD Radeon 8060S, gfx1151.
- Compiler: ROCm 10 `hipcc`, target `gfx1151`.
- Mapped host source: 64 MiB allocated with `hipHostMallocMapped`.
- Descriptor list: 64 random 64 KiB-aligned block offsets.
- Output: contiguous 4 MiB device buffer.
- Kernel: one HIP launch with 64 blocks and 256 threads per block.
- Warmup: one synchronized launch.
- Measurement: 50 launches timed with HIP events.

## Result

| Block size | Descriptors per launch | Bytes per launch | Effective mapped-host gather rate |
|---:|---:|---:|---:|
| 64 KiB | 64 | 4 MiB | 112.908 GB/s |

The measured 50 launches completed in approximately 1.857 ms.

## Interpretation

The result is substantially better than the earlier serialized host-transfer
path, which ranged from 0.167 to 16.876 GB/s depending on block size, and it
approaches the 79.79 GB/s contiguous host-to-device copy ceiling measured in a
separate test. It demonstrates that reducing host intervention to one
descriptor-driven device operation is a credible optimization direction on
Strix Halo.

This is not yet a model result. Mapped host reads use the unified-memory fabric
directly and do not prove that a 148.66 GiB checkpoint can remain resident or
that real expert tensors will have the same locality. The kernel also omits
format conversion, cache eviction, routing, KV state, and model computation.

## Next gate

Integrate a descriptor-list gather into a small synthetic MoE layer with the
actual expert tensor shapes and FP4 or FP8 conversion path. Measure quality,
per-token latency, and p95/p99 miss behavior before considering any full
DeepSeek checkpoint download.

