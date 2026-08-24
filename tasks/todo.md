# KV-cache int4 support

- [x] Fix packed-pool store routing to preserve the logical head dimension instead of using the byte-packed slab dimension.
- [x] Finish the public `int4` surface: exports plus dtype-neutral source and CLI documentation with the 0.5625-byte payload-and-scale cost.
- [x] Extend pool/config/kernel tests with int4 physical-layout, nibble-packing, rebuild, and KV-cost assertions.
- [x] Run the complete KV-quantization regression suite and record results.

## Review
- `uv run pytest tests/engine/test_kv_cache_dtype_gating.py tests/kernels/test_kv_quant.py tests/kvcache/test_kv_quant_pool.py tests/kernels/test_triton_attention.py -q`: 107 passed.
- `uv run ruff check` passed on the changed KV-support files; `server/args.py` passes with its two pre-existing E712 assertions ignored.
- `uv run ft serve --help` lists `--kv-cache-dtype {auto,q8_0,fp8_e4m3,int4}` and the compact-storage contract.

## E2E validation (2026-08-24) — closes the step-9 gate
Live serve of `Laguna-XS-2.1-APEX-I-Mini.gguf` with `--kv-cache-dtype int4 --num-tokens 262144`:
- **KV @ 262,144 tokens = 4.65 GiB** (vs fp8 8.79 GiB) — storage exactly halved.
- **NIAH 3/3 exact** at ~250k tokens: passcode `7391-ALPHA` recovered at 10%/50%/90% depth.
- Worker log clean; no traceback/worker-gone.
