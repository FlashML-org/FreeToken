# GGUF support

FreeToken loads GGUF checkpoints natively (no conversion) for the architectures registered
in `models/gguf/config.py:GGUF_ARCH_TO_REGISTRY`:

| GGUF `general.architecture` | Registry spec |
|---|---|
| `gemma4` | `Gemma4GGUFForCausalLM` |
| `qwen4exp` (Qwen3.8-Flash-Next) | `Qwen4ExpGGUFForCausalLM` |

The shared plumbing lives in `models/gguf/`:
- `reader.py` — metadata, split-shard resolution, `GgufTensor` (torch shape + ggml type +
  packed `[rows, row_bytes]` bytes), `write_metadata_gguf` for `ft checkpoint`.
- `dequant.py` — pure-torch reference dequant + the blocked-shape metadata the packed paths use.
- `tokenizer.py` — builds a HF fast tokenizer from `tokenizer.ggml.*`.
- `config.py` — the `GgufConfigShim` the registry sees, and the arch -> registry map.

## Supported quant types

| ggml type | block / bytes | dense | experts | CPU MoE |
|---|---|---|---|---|
| F32 / F16 / BF16 | — | dense fallback | — | — |
| Q4_0 | 32 / 18 | MMVQ, MMQ, dequant | yes | yes |
| Q5_K | 256 / 176 | MMVQ, MMQ, dequant | yes | yes |
| Q6_K | 256 / 210 | MMVQ, MMQ, dequant | yes | no |
| Q8_0 | 32 / 34 | MMVQ, MMQ, dequant | yes | no |
| IQ4_NL | 32 / 18 | MMVQ, dequant | yes | yes |
| IQ4_XS | 256 / 136 | MMVQ, dequant | yes | yes |

Dense dispatch (`layers/gguf.py`): `_MMVQ` (small-batch GEMV) and `_DEQUANT`
(dequant-then-matmul) cover every type above; `_MMQ` (large-batch) only covers the types
the vendored `ggml_mul_mat_a8` actually implements. **Never add a type to a dispatch set
whose kernel switch lacks it** — the MMQ entry point has no default and silently returns
NaNs (IQ4_NL/IQ4_XS are deliberately MMVQ/dequant only).

## Routed experts

Experts stay as packed block bytes (`moe/gguf_experts.py`) and are dequantized inside the
borrowed `ggml_moe_a8_vec` kernel. Bank layout is one `gate_up [E, 2I, row_bytes(H)]` and one
`down [E, H, row_bytes(I)]` per layer, so the checkpoint may quantize gate/up and down
differently; that is a **composite format tag** `<gate_up>+<down>` (e.g. `iq4_xs+iq4_nl`).
`expert_quant` / `moe_weight_format` hold that tag, and the engine/cache/kernel dispatch read
the per-role types from it.

The expert banks are read once at startup into **pinned host RAM** and streamed to a
GPU LRU slot cache (`--moe-cache-size` / `--moe-cache-auto`). There is no disk-backed expert
source yet, so the full packed expert set must fit pinned host RAM; on plain Linux the pin
budget is 90% of `MemAvailable` (`FREETOKEN_PIN_BUDGET_GB` overrides) and an oversized model
stops with a clear error instead of OOM-crashing the host.

## Qwen3.8-Flash-Next (`qwen4exp`)

- Config from the `qwen4exp.*` KVs (`models/qwen4_exp/gguf.py:parse_gguf_config`); 48 layers,
  36 GDN + 12 QSA, 512 routed experts (releases vary).
- Tiny tensors (norms, routers, hyper-connection mixes, conv weights) become dense bf16;
  packed projections become `GGUFLinear`/`GGUFEmbedding` (`convert_qwen4_exp_to_gguf`). A
  group whose parts differ in quant type (e.g. QSA q=Q6_K with k/v=Q8_0) stays dense bf16.
- GDN is built split: packed `in_proj_qkvz` + dense `in_proj_ba` (the checkpoint ships b/a as
  F32) and `ssm_*` recurrences.
- **PLE**: the fp8 n-gram table is not taken from the GGUF. It is loaded from the original
  safetensors with `--ple-source <repo-or-dir>` (`--ple-backend disk|pinned`); the source's
  own `ngram_embedding.shard_<i>.weight` count defines the table. `ple.layer_multipliers` and
  the n-gram head sizes/offsets are derived and reproduce the GGUF's own values.
- How many IQ4_XS experts fit (52 GiB available):
  Base geometry (H=2560, I=640; gate/up IQ4_XS + down IQ4_NL) = 2.54 MiB/expert, 48 layers, so banks scale as:
  N experts   banks
   128        15.2 GiB
   256        30.5 GiB
   288        34.3 GiB
   320        38.1 GiB
   352        41.9 GiB
   384        45.7 GiB
   512        60.9 GiB   <- current, won't fit
  With ~56 GiB MemAvailable and ~4–6 GiB for the process/dense/KV, the practical ceiling is ~352–384 experts; 320 is comfortable. expert_used_count (top-k=10) must stay ≤ N.

## Verification

- `tests/models/test_gguf_dequant.py` — dequant bit-exact vs a literal `ggml-quants.c`
  transcription and gguf-py (synthetic + real checkpoint).
- `tests/models/test_gguf_reader_split.py` — split shards (synthetic + real).
- `tests/moe/test_gguf_experts.py`, `tests/layers/test_gguf_dispatch.py` — expert/service
  wiring and dispatch-set guards.
- `tests/moe/test_cpu_moe_gguf_quants.py` — CPU W4A8 vs GPU.
- `tests/models/test_gguf_qwen4exp_config.py` — qwen4exp config/plan/PLE-source (synthetic;
  real checkpoint behind `FREETOKEN_QWEN4EXP_GGUF`).

## Known limits / TODOs

- Offload requires the whole expert set in pinned host RAM; see
  `mmap-expert-tiering.md` for the planned hot/warm/cold tiers.
- Expert format must be uniform across layers (one tag per model). Unsloth "dynamic" quants
  that vary the type per layer are rejected for now.
- FTW conversion is not wired for `qwen4exp`: a metadata-only GGUF has no tensor table, so the
  quant plan cannot be derived; it would need the plan persisted at convert time.
- CPU/hybrid MoE only supports single-type (non-composite) expert formats.
- `moe/bench_profile.py` / `benchbw.py` have no entries for the new formats, so
  `--moe-strategy auto` stays on GPU offload unless overridden.
- CPU Q5_K GEMV recomputes the per-32 activation sum inside every output-row dot
  (`cpu_moe_ext.cpp: gguf_asum32`), so it does ~`2I + H` redundant sums per token/route.
  Precomputing the sums in the activation-quantization pass would remove it; left undone
  because it changes the W4A8 dot signature and needs A/B numbers.
- Two qwen4_exp tests assert exact equality where only closeness holds, so they fail
  deterministically (measured; both are test-hygiene, not runtime bugs):
  - `test_qsa_backend.py::test_chunked_prefill_matches_one_shot` compares a split prefill to
    a one-shot with `torch.equal`; the dual-source compress reduces the pooled group in a
    different order, giving max abs diff ~1.5e-3 at output scale ~0.1 (~1.5%), inside the
    `rtol=2e-2` the same file uses against the fp32 oracle. Should be a closeness check.
  - `test_ple.py::test_track_snapshot_equals_a_prefill_stopped_at_the_boundary` compares the
    boundary snapshot to a truncated prefill with `torch.equal`, at ~1.3e-7 relative fp32
    noise. Should be `allclose`.
- The two `needs_weights` tests in `tests/models/test_gguf_qwen4exp_config.py` hardcode the
  full Qwen3.8-Flash-Next checkpoint (512 experts, PLE on layer 1, layer-0 `in_proj_qkvz`
  fused as `qweight`), so they fail when `FREETOKEN_QWEN4EXP_GGUF` points at a variant such as
  the 160-expert no-PLE coder finetune (its layer-0 qkv/z types differ, so `in_proj_qkvz`
  stays dense). Assert derived properties or gate the tests to the matching file.
