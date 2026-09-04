# Quantized KV-cache branch delta

This is a compact recovery map for the KV-cache work on
`feat/reliable-quantized-kv-cache`, compared with `main`. It is intended to make
the implementation reconstructable without copying engine source into a second,
divergent tree.

Reference points when this document was created:

- `main`: `4b94bdc` (`docs: add SECURITY.md`)
- last known-good quantized-KV implementation: `716c1a2`
- implementation commits: `c086b28..716c1a2`

Recreate the exact source diff with:

```bash
git diff main...feat/reliable-quantized-kv-cache -- \
  python/freetoken/attention/triton.py \
  python/freetoken/engine/config.py \
  python/freetoken/engine/engine.py \
  python/freetoken/kernel/triton/attention.py \
  python/freetoken/kernel/triton/kv_quant.py \
  python/freetoken/kvcache \
  python/freetoken/server/args.py
```

## What the branch adds

The branch adds one shared `--kv-cache-dtype` setting for both K and V. Its
values are `auto`, `q8_0`, `fp8_e4m3`, `q4_0`, and `q6_0`. It does **not** add
independent K/V formats or TurboQuant layouts.

The storage formats use a 32-element quantization block and one fp16 scale per
block:

| Format | Payload per 32 values | Scale | Total bytes/value |
| --- | ---: | ---: | ---: |
| bf16 (`auto`) | 64 bytes | none | 2.0000 |
| `q8_0` / `fp8_e4m3` | 32 bytes | 2 bytes | 1.0625 |
| `q6_0` | 24 bytes | 2 bytes | 0.8125 |
| `q4_0` | 16 bytes | 2 bytes | 0.5625 |

`q4_0` packs even values into the low nibble and odd values into the high
nibble. `q6_0` uses a 16-byte low-four-bit plane followed by an 8-byte
high-two-bit plane.

## Source ownership and data flow

1. `python/freetoken/server/args.py` exposes the CLI setting.
2. `python/freetoken/engine/config.py` resolves it to a `KVQuantSpec`.
3. `python/freetoken/engine/engine.py` rejects unsupported backends, latent-KV
   pools, and head dimensions not divisible by 32 before model execution.
4. `python/freetoken/kvcache/base.py` includes packed payload and scale slabs in
   cache-budget calculations.
5. `python/freetoken/kvcache/quant.py` is the format specification and CPU
   correctness oracle. Its `quantize`/`dequantize` methods define the reference
   behavior independently of Triton.
6. `python/freetoken/kvcache/quant_storage.py` allocates scale slabs and routes
   writes to either the ordinary cache store or the quantized Triton store.
7. `python/freetoken/kvcache/mha_pool.py` and
   `python/freetoken/kvcache/hybrid_swa_pool.py` allocate the physical payload
   geometry and expose K/V scales.
8. `python/freetoken/kernel/triton/kv_quant.py` quantizes and packs K/V while
   storing new tokens.
9. `python/freetoken/attention/triton.py` passes payloads, scales, logical head
   dimensions, and layout identifiers into attention.
10. `python/freetoken/kernel/triton/attention.py` loads and dequantizes cached K/V
    inside paged decode and extend kernels.

The key invariant is that attention operates on the **logical** head dimension,
while cache tensors may have a smaller **physical** last dimension. Scale shapes
are derived from the logical dimension, never from packed payload width.

## Supported boundary

Quantized KV is intentionally limited to the Triton attention backend and the
MHA/hybrid-SWA pools. MLA, DSA, DSV4, BSA, FlashInfer, and TRT-LLM cache layouts
are rejected rather than allowed to reinterpret packed bytes.

`auto` preserves the original compute-dtype path. The quantized and unquantized
paths share attention wrappers, so any change to payload geometry, scale
presence, logical dimensions, or Triton constexprs must be tested in both modes.

## Verification map

- `tests/kvcache/test_subbyte_quant.py`: CPU format round trips, packing, sizing,
  and dtype resolution.
- `tests/kvcache/test_pool_sizing_surface.py`: budget accounting and pool sizing.
- `tests/kernels/test_attention_subbyte.py`: quantized attention parity.
- `tests/kernels/test_triton_attention.py`: paged/decode/extend and scale-plumbing
  coverage; CUDA-dependent cases skip on CPU-only hosts.
- `tests/server/test_kv_cli_args.py`: capacity overrides compose with
  `--kv-cache-dtype`, while `--num-pages` and `--num-tokens` remain exclusive.

For forensic review, start with the CPU oracle in `kvcache/quant.py`, compare its
output with `kernel/triton/kv_quant.py`, and only then inspect attention loads.
This isolates format corruption from cache allocation and attention math.
