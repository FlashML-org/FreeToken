# GMKtec EVO-X2 fused MoE expert prototype

## Purpose

This bounded native HIP experiment measures a representative expert-row
operation on the GMKtec EVO-X2 without downloading or loading a large model
checkpoint. It is intended to identify whether a device kernel can consume
multiple routed expert rows from mapped host memory while performing the dot
product and reduction on the GPU.

The prototype is not a model benchmark and must not be reported as FreeToken
tokens per second. Its packed signed-int4 data is deliberately simpler than
the production Qwen NVFP4 format. The result is therefore a kernel-path
baseline for the next implementation step, not proof of model equivalence.

## Configuration

| Field | Value |
| --- | --- |
| Backend | Native HIP, compiled with `hipcc` |
| Target | `gfx1151` |
| Experts per launch | 64 |
| Values per expert row | 16,384 |
| Packed bytes per row | 8,192 |
| Weight source | HIP mapped pinned host memory |
| Activation type | FP32 |
| Workgroup | 256 threads per expert row |
| Timed launches | 100, after one warmup launch |
| Measurement | HIP events around the timed launch loop |

Each byte contains two signed four-bit weights. One workgroup processes one
expert row, dequantizes the nibbles in the kernel, multiplies by the resident
activation vector, and reduces to one output value.

## Result

The remote compile and execution completed successfully:

```text
experts=64 values=16384 packed_bytes=8192 rounds=100 elapsed_ms=1.604692 effective_weight_GBps=32.672189
```

The measured effective packed-weight read rate was **32.672 GB/s** for this
serialized mapped-host expert-row workload. The compiler emitted only unused
return-value warnings for the intentionally compact prototype; the kernel
completed and returned exit code zero.

## Interpretation

This result confirms that a fused HIP kernel can perform expert-row address
selection, mapped-host reads, on-device signed-int4 unpacking, multiply, and
reduction in one launch. It does not establish that the production NVFP4
kernel will reach this rate, because NVFP4 metadata, scaling, tensor layout,
routing, and the production hidden dimensions are different.

The result is also not directly comparable to the earlier 281.916 GB/s
device-resident gather or 112.908 GB/s mapped-host gather microbenchmarks.
Those tests measured bulk gather bandwidth without the dequantization,
dot-product, and reduction work. The present experiment intentionally includes
that compute to expose the combined path that a fused MoE implementation must
optimize.

## Next action

Replace the synthetic signed-int4 row format with the exact production NVFP4
metadata and tensor dimensions, then compare the fused kernel against the
current FreeToken expert implementation using deterministic output hashes.
Only a complete API run that preserves quality, TTFT, decode TPS, tail
latency, recovery, and concurrency can promote a fused candidate.
