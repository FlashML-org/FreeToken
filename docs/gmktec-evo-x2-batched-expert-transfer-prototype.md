# GMKtec EVO-X2 batched expert-transfer prototype

This isolated prototype uses the same 64 randomly selected 64 KiB blocks as
the serialized expert-block test, but gathers blocks into pinned staging
buffers and performs one device copy per group. It separates CPU staging time
from host-to-device transfer time. It does not download or load a large model
and does not modify the protected service.

## Method

- GPU: AMD Radeon 8060S, gfx1151.
- PyTorch: `2.13.0+rocm10.0.0`.
- HIP: `7.15.26333`.
- Total payload per round: 4 MiB.
- Block size: 64 KiB.
- Three warmup rounds and ten measured rounds per grouping.
- Device synchronization after each grouped copy.

## Results

| Blocks per group | Groups per round | Total round rate | Transfer-only rate | Staging mean |
|---:|---:|---:|---:|---:|
| 1 | 64 | 4.67 GB/s | 5.83 GB/s | 0.171 ms |
| 4 | 16 | 9.44 GB/s | 15.28 GB/s | 0.168 ms |
| 16 | 4 | 12.84 GB/s | 29.79 GB/s | 0.185 ms |
| 64 | 1 | 9.46 GB/s | 33.64 GB/s | 0.318 ms |

The serialized 64 KiB benchmark reached approximately 5.01 GB/s under its
different round and synchronization setup. Grouping 16 blocks improved the
transfer-only rate to approximately 29.79 GB/s, about 5.1 times the serialized
transfer-only result. The best end-to-end round rate was 12.84 GB/s at group
size 16 because CPU staging and synchronization remain part of the path.

## Interpretation

Batching and coalescing are necessary to make scattered expert movement
credible, but they do not recover the 79.79 GB/s contiguous-copy ceiling by
themselves. A production miss path must overlap staging with computation,
reuse pinned buffers, and choose a group size that avoids excessive CPU gather
cost. These measurements are synthetic and must not be converted directly to
model TPS.

For the 284B feasibility question, this result means that a full-checkpoint
offload path would need a high locality cache plus grouped transfers. A design
that services every expert miss as an independent small copy is ruled out by
the earlier prototype. A design that batches misses has a plausible systems
direction, but still faces the approximately 148.66 GiB payload versus the
approximately 18 GiB live available-memory constraint.

## Next gate

The next optimization experiment should overlap grouped staging and device
copies with a synthetic compute kernel. It should report whether overlap hides
the approximately 0.17 to 0.32 ms staging cost without increasing tail latency.

