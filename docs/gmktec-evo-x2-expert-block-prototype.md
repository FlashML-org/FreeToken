# GMKtec EVO-X2 expert-block transfer prototype

This isolated prototype approximates MoE expert-cache misses with random,
non-contiguous host slices copied to a device tensor. Each block is
synchronized before the next block, making the result intentionally closer to
a serialized miss path than to an ideal bulk copy. It does not download or
load DeepSeek-V4-Flash and does not modify the protected service.

## Environment and method

- GPU: AMD Radeon 8060S, gfx1151.
- PyTorch: `2.13.0+rocm10.0.0`.
- HIP: `7.15.26333`.
- Host source buffer: 64 MiB float32 tensor.
- Each round: 64 randomly selected, 4 KiB-aligned blocks.
- Three warmup rounds and ten measured rounds per block size.
- `torch.cuda.synchronize()` after every block copy.

## Results

| Block size | Bytes per round | Effective host-to-device rate | Mean block latency |
|---:|---:|---:|---:|
| 4 KiB | 256 KiB | 0.167 GB/s | 24.6 microseconds |
| 16 KiB | 1 MiB | 0.914 GB/s | 17.9 microseconds |
| 64 KiB | 4 MiB | 5.009 GB/s | 13.1 microseconds |
| 256 KiB | 16 MiB | 16.876 GB/s | 15.5 microseconds |

The earlier contiguous 64 MiB prototype measured 79.79 GB/s host-to-device.
The contrast shows that launch and synchronization overhead, not only the
memory fabric, dominates small scattered transfers.

## Interpretation for DeepSeek offload

These are synthetic lower-level measurements, not a prediction of model TPS.
They nevertheless bound the cost of a cache policy that services many small
expert misses individually. A policy that transfers 4 KiB blocks one at a time
would operate at roughly 0.21 percent of the contiguous-copy rate. Even 64 KiB
blocks reach only approximately 6.3 percent of that rate.

The result strengthens the capacity decision: a 148.66 GiB checkpoint cannot be
made interactive merely by relying on fast contiguous unified-memory copies.
The runtime would need to batch and coalesce expert transfers, retain a very
high-locality working set, or accept much lower throughput. A full checkpoint
download remains unjustified until the actual runtime miss granularity and
coalescing behavior are demonstrated on a small synthetic model.

## Next gate

The next useful experiment is a batched version that copies a fixed total byte
count using one grouped operation per layer, then compares it with the
serialized result above. This will quantify how much batching the runtime must
provide before a large-model offload attempt can be considered technically
credible.

