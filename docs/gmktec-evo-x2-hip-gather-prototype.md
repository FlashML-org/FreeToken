# GMKtec EVO-X2 compiled HIP gather prototype

This prototype compiled a small device-side gather kernel with `hipcc` and ran
it on the Radeon 8060S. It gathers 64 randomly spaced 64 KiB blocks from a
64 MiB device-resident source buffer into a contiguous output buffer. It does
not access model files or change the protected service.

## Build and runtime

- Compiler: ROCm `hipcc` 7.15.26333 from ROCm 10.0.
- Target: `gfx1151`.
- Runtime: PyTorch environment reported HIP 7.15.26333 for the surrounding
  host, while the kernel was launched through the HIP runtime directly.
- Kernel launch: 64 blocks, 256 threads per block.
- Warmup: one synchronized launch.
- Measurement: 100 launches timed with HIP events.

## Result

| Block size | Blocks per launch | Bytes per launch | Effective device gather rate |
|---:|---:|---:|---:|
| 64 KiB | 64 | 4 MiB | 281.916 GB/s |

The kernel completed 400 MiB of gathered output in approximately 1.488 ms.
This is a device-to-device gather result, not a host-transfer or model-TPS
result.

## Interpretation

The result shows that compiled device-side indexing and gather can be much
faster than the host-orchestrated miss path. It does not by itself solve model
offload because the source expert blocks would still need to arrive in device
memory. It does, however, identify a credible implementation direction:
transfer and expert selection should be represented as persistent device-side
work, with host intervention reduced to batched descriptor submission.

The 281.9 GB/s figure must not be compared directly with the earlier 79.8
GB/s contiguous host-to-device measurement. They measure different links and
different operations. The useful comparison is against the 0.167 to 16.876
GB/s scattered host-transfer results, which are dominated by launch and
synchronization overhead.

## Next gate

The next prototype should use mapped or pinned host memory and a HIP kernel
that consumes a descriptor list, then compare one descriptor submission for 64
blocks against the serialized Python path. That will measure how much host
intervention can be removed while retaining the real host-to-device boundary.

