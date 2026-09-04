# llama.cpp b10434 single-token source contract

Pinned source commit:

```text
7e4c0a96880dae4fc4268ad441f8a6446bd5460a
```

This directory records the narrow single-token ABI used by FreeToken. Active quant block and
vec-dot declarations live in the sibling `ggml-common*`, `vecdotq*`, and `moe_vec*` headers so
CUDA/HIP builds share one source surface. Multi-token MMID compaction and `small_k` are excluded.

ABI choices:

- Q8_1 activation rows: `ceil(H / 32) * 36` bytes, unpadded contract requires `H % 32 == 0`.
- Output: caller-owned FP32 `[channels, Nrows]`.
- Every workspace region is aligned to 256 bytes.
- PDL calls are no-ops; stream order already serializes the caller-owned workspace during graph
  replay. Native PDL mapping requires a later capture/replay proof.
