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
