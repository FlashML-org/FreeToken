# KV Cache Quantization

> User-facing reference for the `--kv-cache-dtype` family. For the
> implementation rationale (why mag=8 not 7, why per-block 32 not
> super-block, why the dual-plane Q6 layout), see the PR bodies
> (`pr-bodies/PR1_q4_0.md` and `pr-bodies/PR2_q6_0.md`).

## What is `--kv-cache-dtype`?

The KV cache stores the K (key) and V (value) tensors the attention
layers read from on every decode step. By default it lives in bf16
(2 bytes per element). Quantizing the cache shrinks the per-element
storage cost at the cost of some numerical precision on the
attention computation.

FreeToken supports the following `--kv-cache-dtype` values:

| value | bytes/elem | effective context @ 8 GB | precision vs bf16 |
|---|---|---|---|
| `auto` (or unset) | 2.000 (bf16) | ~110K | - |
| `q8_0` | 1.0625 | ~160K | ~0.6% kernel rel_err |
| `q6_0` | 0.8125 | ~190K | ~2.4% kernel rel_err |
| `q4_0` | 0.5625 | ~220K | ~9.4% kernel rel_err |
| `fp8_e4m3` | 1.0625 | ~160K | ~0.6% kernel rel_err (float path) |

"Effective context" assumes the engine's hybrid MoE backend is
enabled (the MoE offload cache lives in its own pool sized by
`--moe-cache-auto`). On 8 GB consumer GPUs the Q4 path is the
only one that pushes the context window past 200K.

## How to launch

The simplest form:

```bash
ft serve \
  --model Qwen/Qwen3.6-35B-A3B \
  --kv-cache-dtype q4_0 \
  --kv-reserve-tokens 220000 \
  --moe-backend hybrid \
  --moe-cpu-threads 12 \
  --memory-ratio 0.97 \
  --moe-cache-auto
```

`--kv-cache-dtype auto` is the same as not setting it (bf16).
`--kv-reserve-tokens N` is the size of the K/V pool; pick N to
match the longest conversation you intend to serve. `--moe-cpu-threads
12` should be calibrated on the target machine (`ft bench bw` to
find the best value).

The first request after startup will spend ~3-5 s JIT-compiling the
quantized store / load kernels. Subsequent requests are at full
throughput.

## When to pick which dtype

- **Default / when in doubt**: `q8_0` is the upstream PR#103 default
  and a safe bet; near-bf16 precision, 47% memory savings over bf16.
- **Need maximum context** (long documents, full-book Q&A): `q4_0`
  gives 3.5x the bf16 context on the same VRAM, at the cost of
  ~9% kernel rel_err. Empirically: GSM8K-CoT 97.3% (no loss), MMLU
  90.8% (no loss), GPQA Diamond 73.2% (Q4 vs bf16 89.2%: -16pp
  on hard reasoning).
- **Precision-first sub-byte**: `q6_0` is between Q4 and Q8: ~4x
  better kernel precision than Q4, 24% more bytes. Use when Q4
  loses too much on your workload and Q8's context window is
  too small.
- **bf16 only**: `auto` (or unset). Required if you see model-output
  drift on hard reasoning and need the canonical baseline.

The Q4/Q6 paths do **not** require any model quantization: weights
stay in bf16, only the K/V cache is sub-byte. The Q4/Q6 sub-byte
path is orthogonal to NVFP4 / FP8 weight quantization -- both
can be active at the same time.

## How it works (one paragraph)

The K/V pool's last axis is `head_dim`. Quantized schemes pack
multiple values per byte along that axis:

- **q8_0** / **fp8_e4m3** -- 1 byte per element. One int8 (or fp8)
  value per slot, plus one fp16 scale per 32 values along head_dim.
- **q6_0** -- 0.75 byte per element. 32 values are packed into
  16 bytes (low plane: low 4 bits of each 6-bit value, packed the
  same as Q4) plus 8 bytes (high plane: top 2 bits of each value,
  packed four-per-byte at bit positions 0, 2, 4, 6), plus one fp16
  scale per block.
- **q4_0** -- 0.5 byte per element. 32 values are packed into
  16 bytes: byte `j` holds `val[2j]` in the low nibble and
  `val[2j+1]` in the high nibble, both as unsigned 4-bit. One fp16
  scale per block.

The attention kernel is told the logical `head_dim` and unpacks
inside the load. The store kernel packs on the write path. Both
operations are transparent to the model code.

```
         logical head_dim (e.g. 128)
         =======================
bf16     [v0][v1] ... [v127]            256 bytes per token per layer
q8_0     [v0][v1] ... [v127]            128 bytes +  8 bytes scale = 136
q6_0     [v0/lo][v1/lo] ... [v127/lo]   96 bytes +  8 bytes scale = 104
         [v0/hi, v1/hi, v2/hi, v3/hi] ... (8 bytes, 4 values each)
q4_0     [v0/lo|v1/hi][v2/lo|v3/hi] ...  64 bytes +  8 bytes scale =  72
```

## What the Q4/Q6 paths do NOT change

- **Model weights** are still bf16 (or NVFP4 / FP8 if you set the
  weight quantization separately). Only the K/V cache is sub-byte.
- **Linear-attention (GatedDeltaNet) layers** are not affected.
  Hybrid models (e.g. Qwen3.5-35B-A3B's 4 linear + 32 full attention
  layers) get the full context-length win because the paged pool is
  what hits the wall, but the linear layers' state pool is untouched.
- **The MoE offload cache** lives in its own pool sized by
  `--moe-cache-auto`. Sub-byte KV does not change the MoE cache
  budget solve.
- **The OpenAI-compatible API surface** is unchanged. Tokens/s,
  request formats, response formats, and the streaming protocol are
  all identical across dtypes; the only knob is the new context
  budget.

## Hybrid model note

For hybrid models (Qwen3.5 / Qwen3.6 MoE with linear attention
layers), the K/V pool is sized for the **full-attention** layers
only. The linear layers' state is held in a separate pool that
this PR does not touch. Empirically on Qwen3.5-35B-A3B the linear
layers account for 4 of the 36 layers, so the effective Q4 KV
context is still the Q4 number from the table; the linear layers'
state is on top of that, sized separately by the engine.

## Compatibility with the GGUF Q4_0 spec

The byte layout (low-nibble-even, high-nibble-odd, 16 bytes per
32-value block, 1 fp16 scale) matches the GGUF Q4_0 spec, **except**
for the `max_magnitude` constant: we use 8 (range `[-8, 7]`) where
GGUF uses 7 (range `[-7, 7]`). The 8-bound is empirically 5% better
on K/V-shaped data because the distribution tail biases the per-
block scale upward, leaving the +7 boundary the more frequent side.
A Q4_0 cache produced by a tool that uses the GGUF 7-bound will
round-trip through our dequant with ~5% rel_err; we do not read
pre-quantized caches from disk, so this only matters if a user
later writes a converter.

## How to verify it's working

```bash
# Start the service
ft serve --model Qwen/Qwen3.6-35B-A3B --kv-cache-dtype q4_0 \
  --kv-reserve-tokens 220000 --moe-backend hybrid \
  --moe-cpu-threads 12 --memory-ratio 0.97 --moe-cache-auto

# In another terminal, check the startup log for "Allocating ... tokens
# for KV cache, K + V = <X> GiB". Q4_0 yields ~1.18 GiB at 160K tokens;
# Q6_0 yields ~1.24 GiB; q8_0 yields ~1.24 GiB; bf16 yields ~3.20 GiB.
```

A clean run will also report the per-kernel compile lines on the
first request; these can be ignored after the first decode.

## How to recover

Reverting to bf16 is one flag change:

```bash
ft serve ... --kv-cache-dtype auto
```

There is no data loss across dtype changes -- the K/V cache is
ephemeral (regenerated on every request) and a session-started
flag controls the pool allocation at startup. The CLI rejects
mismatched configurations at startup; if you change `--kv-cache-
dtype` mid-session, restart the service.

## See also

- `pr-bodies/PR1_q4_0.md` -- the Q4 PR body, with kernel-level
  numbers, A/B test results, and "why mag=8" rationale
- `pr-bodies/PR2_q6_0.md` -- the Q6 PR body, with the dual-plane
  layout, kernel-level numbers, and the "why two PRs" rationale
- `tests/kvcache/test_subbyte_quant.py` -- spec round-trip tests
- `tests/kernels/test_attention_subbyte.py` -- kernel parity tests
- `WALKTHROUGH.md` (in the upload package) -- review-prep doc with
  the 3 most likely reviewer questions
