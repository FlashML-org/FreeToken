# Adding a GGUF quant type

Checklist for wiring a new native ggml block quant into FreeToken. Follow it top to bottom;
each step names the symbol and the test that proves it.

## 0. Get the bit layout from ground truth
- Struct + CPU dequant in llama.cpp / ik_llama.cpp `ggml/src/ggml-common.h` and
  `ggml-quants.c` (`block_<type>`, `dequantize_row_<type>`), and the gguf-py reference in
  `gguf-py/gguf/quants.py`. Note the block size and `type_size` (bytes per block).
- Check the vendored kernels first (`python/freetoken/kernel/csrc/gguf/`):
  `dequantize.cuh:ggml_get_to_cuda` and `gguf_kernel.cu` (`ggml_mul_mat_vec_a8`,
  `ggml_mul_mat_a8`, `ggml_moe_a8_vec`). Their `case` sets decide which of steps 2-3 are
  possible.

## 1. Reference dequant (`models/gguf/dequant.py`)
- `GGML_<NAME> = <id>`, `BLOCK_SHAPE[type] = (block, type_size)`, `GGML_NAME[type]`.
- `dequant_<type>(raw, out_dtype)`: reshape to `(-1, type_size)`, vectorize, return flat
  storage order. Copy the style of `dequant_q4_0` (linear) or `dequant_q6_k` (super-block).
- Register in `_DEQUANT` and `__all__`; extend the module docstring.

## 2. Dense dispatch (`layers/gguf.py`)
- Add to `_MMVQ` if `ggml_mul_mat_vec_a8` has a case; `_MMQ` if `ggml_mul_mat_a8` has one;
  `_DEQUANT` for the fallback. Only add where the kernel switch really has the case.

## 3. Routed experts (only if the type appears on `ffn_*_exps`)
- `moe/gguf_experts.py`: add to `GGUF_EXPERT_QUANTS` (tag -> id). Per-role differences are
  expressed by a composite `"<gate_up>+<down>"` tag (`gguf_expert_role_types`).
- `moe/offload_cache.py`: add the block/size to `_GGUF_BANK_TYPES` (used for sizing a
  composite bank). Single-type schemas/bytes are picked up automatically for a new tag.
- `moe/expert_banks._gguf_banks` already accepts any tag via the role-types helper; nothing
  else to add. `layers/moe._expert_gemm` dispatches on the tag prefix automatically.

## 4. CPU MoE (optional; kernel/csrc/cpu_moe/cpu_moe_ext.cpp)
- Extend `WFmt` (must match `moe/cpu_executor._WFMT_IDS`), add scalar + AVX2 + AVX-VNNI
  `_dot_i8_*` kernels, wire `select_gguf_dot` and `gguf_row_bytes`/`is_gguf_w4a8`.
- Add the format to `_GGUF_W4A8_FORMATS`; `_resolve_gguf_banks` validates per-role row bytes.
- Note the C++ executor picks **one** dot per executor, so composite formats are GPU-only.

## 5. Tests
- `tests/models/test_gguf_dequant.py`: add to `EXPECTED_SHAPE`, write a `_ref_<type>` literal
  transcription of the C, register in `REFERENCE`.
- `tests/moe/test_gguf_experts.py`: add the tag to `_CASES` (bank packing, role shapes).
- `tests/layers/test_gguf_dispatch.py`: extend the dispatch-set assertions if the type is
  (or must not be) in `_MMQ`/`_MMVQ`.
- `tests/moe/test_cpu_moe_gguf_quants.py`: add to `_CASES` for the CPU/GPU parity test.
- Cross-check the reference against gguf-py (`gguf.quants.dequantize`) on random blocks, and
  against the CUDA kernels on a GPU (`ggml_dequantize`, `ggml_mul_mat_vec_a8`,
  `ggml_moe_a8_vec`).

## Gotchas learned
- `_MMQ` has no `default:` — a type it does not implement returns **NaNs**, not an error.
- A quant whose block does not divide the row width cannot be used for that role; e.g. a
  640-wide intermediate cannot host a 256-block type, which is why `iq4_xs+iq4_nl` exists.
- Per-role ≠ per-layer: the model currently allows one format for the whole expert set.
- Bump the AOT entry (`kernel/aot_models.py`) if a new release/format should be prebuilt.
