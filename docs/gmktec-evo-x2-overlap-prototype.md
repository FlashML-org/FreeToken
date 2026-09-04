# GMKtec EVO-X2 grouped-transfer overlap prototype

This prototype tested whether a naive two-stream pipeline could hide grouped
expert staging and transfer behind synthetic GPU work. It is an isolated
systems experiment. It does not load a large model or alter the protected
service.

## Method

- GPU: AMD Radeon 8060S, gfx1151.
- PyTorch: `2.13.0+rocm10.0.0`.
- HIP: `7.15.26333`.
- Four groups per round, each containing sixteen random 64 KiB blocks.
- Pinned host staging buffers and a separate transfer stream.
- Separate compute stream with an event dependency after each copy.
- Comparison against a synchronized serial implementation.

## Results

| Synthetic compute per group | Serial mean | Overlap mean | Overlap speedup |
|---:|---:|---:|---:|
| 0 matrix multiplications | 0.331 ms | 2.723 ms | 0.122x |
| 1 matrix multiplication | 1.793 ms | 10.039 ms | 0.179x |

The naive overlap pipeline was slower in both cases. With no compute it was
approximately 8.2 times slower, and with one matrix multiplication it was
approximately 5.6 times slower.

## Interpretation and rejection reason

This is not evidence that overlap is impossible in the production runtime. The
prototype intentionally used many small stream and event operations and a
single reusable compute tensor, so it exposes orchestration overhead that a
fused production scheduler might avoid. It does show that simply adding a
transfer stream and per-group events is not a valid optimization.

The candidate is rejected for promotion because it regressed end-to-end wall
time in both measured cases. Future overlap work must use persistent streams,
event pools, larger fused batches, and scheduler-level pipelining, then pass
the same quality and tail-latency gates as the current validated path.

