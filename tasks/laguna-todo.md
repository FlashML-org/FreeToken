# Laguna-S-2.1 GGUF support (unsloth Laguna-S-2.1-UD-IQ1_S.gguf)

Goal: `ft serve --model Laguna-S-2.1-UD-IQ1_S.gguf --kv-cache-dtype q8_0 --num-tokens 65536`
loads and generates correctly on the RTX 5080 (16 GB), experts on the MoE offload cache.

## Ground truth (verified from the file header + llama.cpp origin/master)

Single unsplit GGUF, 33,766,781,984 bytes. **No split-reader work needed for this file.**

Metadata (`laguna.*`):
- block_count 48, embedding_length 3072, vocab 100352, context_length 262144
- head_count per layer: `[48,72,72,72] * 12` (48 ⇒ full attention at `il%4==0`, 72 ⇒ SWA)
- head_count_kv 8 (uniform), key/value_length 128, sliding_window 512
- rope full layers: dim 64 (partial), θ=500000, YaRN factor 32, orig_ctx 8192,
  attn_factor 1.0, beta_fast 32, beta_slow 1
- rope SWA layers: dim 128, θ=10000, **plain rope, no YaRN** (freq_scale 1.0)
- rms_eps 1e-6; leading_dense_block_count 1; expert_count 256, used 10,
  expert_ff 1024, shared_expert_ff 1024, weights_norm true, weights_scale 2.5,
  gating_func 2 (sigmoid); dense ffn 12288
- tokenizer: gpt2 BPE, pre="laguna", bos 2, eos 2, eot 24, add_bos true,
  chat template embedded
- Tensor quant types by role (per-tensor, "Unsloth Dynamic" = mixed):
  - token_embd / output: Q4_K (untied head)
  - attn q/k/v/o, attn_gate, dense ffn, shexp gate/up: Q5_K (layer 47: Q6_K)
  - ffn_down (dense), down_shexp: Q6_K (one Q8_0)
  - expert banks: gate/up IQ1_S (33 layers) or IQ2_XXS (14), down IQ3_XXS (45) or IQ4_XS (2)
  - norms, router (ffn_gate_inp), exp_probs_b: F32
- Layer schedule: layer 0 dense SwiGLU; layers 1..47 MoE (expert bank index = layer_id-1)

Reference semantics (llama.cpp `src/models/laguna.cpp`, copy in
`/tmp/claude-1000/-home-lucas-ai-FreeToken/61016eb3-71fd-4d7b-97c6-816d5fa0839a/scratchpad/llamacpp-laguna.cpp`):
- QK RMSNorm at head_dim level (weight shape [128]) before rope, Qwen3-style
- attn_gate: `g = softplus(g_proj(pre-norm hidden))`, per-head ([3072, n_head_il]);
  multiply attention output per head (broadcast over head_dim) **before** o_proj
- router: fp32 logits → sigmoid → +exp_probs_b bias for top-10 selection only →
  gather **unbiased** sigmoid scores → renormalize → ×2.5 → weighted expert sum;
  shared expert always added
- pre-attn and pre-ffn RMSNorm only (no post norms); final norm + untied lm_head

CUDA kernel coverage (verified in `python/freetoken/kernel/csrc/gguf/`): mmvq, mmq
(Q4_K/Q5_K), dequantize and moe_vec for **all** the types above already exist and are
dispatched by ggml type id in `gguf_kernel.cu`. Only the Python-side tables gate them.

Blockers found in Python side:
- `models/gguf/dequant.py`: only F32/F16/BF16/Q4_0/Q8_0/Q6_K in tables
- `layers/gguf.py`: `_MMVQ/_MMQ/_DEQUANT = {Q4_0, Q8_0, Q6_K}`
- `models/weight.py::load_q4_0_moe_expert_sources`: Q4_0-only expert banks
- `ModelConfig`: scalar `num_qo_heads` (Laguna needs per-layer 48/72)
- GGUF registry: only gemma4

Model download running in background task `bp45xxl7b` →
`~/.cache/huggingface/hub/models--unsloth--Laguna-S-2.1-GGUF/...`.
Header fixture: scratchpad `laguna_sparse.gguf` (metadata + tensor table, sparse data).

## Orchestration

Implementer: `pc-gpt-5-3-codex-spark` (weak — tasks kept small, exact files/symbols given).
Escalation (user 2026-08-23): on the next Spark failure (context death, usage limit,
botched task), switch the implementer seat to `pc-gpt-5-6-luna` permanently.
Reviewer per phase: `pc-gpt-5-6-terra` `[[effort: medium]]` — findings **with fixes**.
If a phase needs >2 review rounds, the lead (Fable) reviews and fixes directly.

## Phase 1 — GGML quant-type plumbing (no model code)

- [x] 1a. `models/gguf/dequant.py`: add ids Q4_K=12, Q5_K=13, IQ2_XXS=16, IQ3_XXS=18,
      IQ1_S=19, IQ4_XS=23; BLOCK_SHAPE (256,144)/(256,176)/(256,66)/(256,98)/(256,50)/(256,136);
      GGML_NAME entries. Reference dequant: delegate to gguf-py `gguf.quants.dequantize`
      (verify installed gguf-py handles these types; else port only what tests need).
- [x] 1b. `layers/gguf.py`: extend `_MMVQ` with all six; `_MMQ` with Q4_K, Q5_K only;
      `_DEQUANT` per actual `ggml_dequantize` switch coverage in `gguf_kernel.cu`
      (verify each case id before adding). IQ types have no MMQ → prefill path must
      fall back to dequant+bf16 matmul; confirm `fused_mul_mat_gguf` already routes that.
- [x] 1c. Tests `tests/kernels/test_gguf_quant_types.py`: per type — CUDA
      `ggml_dequantize` vs gguf-py reference on random packed blocks; `mmvq` matvec vs
      `F.linear` on dequantized weight; `moe_vec` for IQ1_S/IQ2_XXS/IQ3_XXS/IQ4_XS.
- [x] R1. Terra review round(s) → fixes → tests green.

Phase 1 notes: gguf-py cannot quantize K/IQ formats, so tests use random
safe-scaled packed bytes with gguf-py dequant as reference; matmul refs model
q8_1 activation quantization; moe_vec asserted bit-exact vs mmvq. MMQ exists
for Q4_K/Q5_K only among the new types (IQ prefill falls back to dequant).
Lead (Fable) fixed tests after 2 implementer rounds.

## Phase 2 — config, registry, tokenizer

- [x] 2a. `models/config.py`: add optional per-layer qo-head counts
      (`num_qo_heads_per_layer: tuple[int, ...] | None = None`) with accessor defaulting
      to scalar `num_qo_heads`; audit consumers that assume the scalar for Q width
      (KV geometry is uniform: 8 KV heads × 128 — KV pools unaffected).
- [x] 2b. New `models/laguna/` package: `config.py::parse_gguf_config` building
      ModelConfig from the metadata above (full/SWA schedule from head-count array;
      per-layer rope params; MoE fields; eos {2,24}); registry entries
      (`gguf/config.py::GGUF_ARCH_TO_REGISTRY["laguna"]`, `register.py` ModelSpec
      `LagunaGGUFForCausalLM`).
- [x] 2c. `models/gguf/tokenizer.py`: make it arch-generic where gemma-specific
      (eos ids from `tokenizer.ggml.eos_token_id` + `eot_token_id`; verify transformers
      converts gpt2/"laguna" pre BPE from GGUF; else load via tokenizer.ggml.* fields).
- [x] 2d. Tests: config parse from a metadata dict fixture; registry dispatch;
      tokenizer eos/bos/chat-template presence (uses scratchpad header fixture).
- [x] R2. Terra review → fixes.

Phase 2 notes: yarn scaling needs explicit attention_factor=1.0 (metadata
yarn_attn_factor) or freetoken defaults to ggml-incompatible mscale. Tokenizer
routes laguna->gpt2 converter with bos/eos strings materialized from ids.
Committed metadata-only fixture tests/fixtures/laguna-s-2.1-metadata.gguf (3.5MB).
Registry entry live; model stub raises a clear in-progress error until Phase 3/4.
Review: 1 round (blocker attention_factor + hardening), fixes by Spark.

## Phase 3 — model modules (torch, quant-agnostic bf16 reference first)

Phase 3 wiring notes (from Terra's R2 review): FlashInfer plans a single global
qo_head count and LinearQKVMerged/LinearOProj take one Q/O width — attention
modules must be built with `config.qo_heads(layer_id)` and pass it to QKV/O
projections, reshapes, gate logic, and the GQA/tensor-core decision; Triton
decode scratch may keep max-72 allocation but gets the real per-layer count at
invocation. attn backend for SWA is triton-only.

- [x] 3a. `models/laguna/attention.py`: per-layer head count, QK head-dim RMSNorm,
      partial-rope (dim 64 yarn) / full-rope (dim 128 plain) per layer type, per-head
      softplus gate (fp32) applied before o_proj. Reuse `layers/rotary.py::get_rope`
      (verify partial+yarn support), existing attention backend plumbing (SWA hybrid
      pool as in gemma4).
- [x] 3b. `models/laguna/moe.py` + `mlp`: router semantics above; reuse existing MoE
      offload machinery (find sigmoid+bias router precedent, e.g. glm/minimax/afmoe
      style in repo); shared expert; dense layer 0. Expert bank index = layer_id-1.
- [x] 3c. `models/laguna/model.py`: decoder wiring, final norm, untied head;
      `convert_laguna_to_gguf` pass swapping Linear/Embedding → GGUFLinear/GGUFEmbedding
      **keeping each tensor's own ggml type** (unlike gemma4's uniform assumption).
- [x] 3d. Unit tests vs handwritten torch reference: one full layer, one SWA layer,
      gate math, router top-10 selection/bias/renorm/scale, dense vs MoE layer.
- [x] R3. Terra review — semantics clean; loader-contract findings folded into Phase 4;
      test-quality findings fixed in round 1. Module tests: 6, all green.

## Phase 4 — weight loading

- [x] 4a. `models/laguna/gguf.py::iter_gguf_weights`: name map (`blk.N.attn_q` etc. →
      module params, table in ground truth above); norms/router/bias to f32/bf16;
      quantized tensors kept packed with per-tensor type.
      R3 contract (blocker): Engine collects model.state_dict() BEFORE iterating
      weights, so DeferredGGUFLinear must be materialized earlier — add
      `gguf_model_path` to ModelConfig (set from shim.model_path), have
      convert_laguna_to_gguf read the tensor table (name→ggml type) and
      materialize every swapped module up front using the same name map 4a uses
      (share one mapping helper). Also (major): the untied lm_head must select
      last-token rows on prefill like ParallelLMHead/GGUFTiedLMHead
      (batch.attn_metadata.get_last_indices(batch.size)) — wrap DeferredGGUFLinear
      in a LagunaGGUFLMHead that does the gather before the fused matmul.
- [x] 4b. Generalize `models/weight.py::load_q4_0_moe_expert_sources` → type-aware
      expert-bank loader (per-layer, per-projection ggml type — gate/up/down differ);
      wire type ids through the offload cache to `moe_vec` dispatch.
- [x] 4c. Test: synthetic tiny laguna GGUF (write with gguf-py) loads end-to-end with
      dummy weights; every tensor consumed exactly once; unknown tensor → clear error.
- [x] R4. Sol (medium) review — clean except FTW-conversion blocker; laguna FTW
      now rejects loudly (documented limitation). Kernel scope independently
      cleared by a Terra sub-lens before the reviewer switch.

## Phase 5 — e2e validation (lead-driven)

- [ ] 5a. Load real file; `ft serve --kv-cache-dtype q8_0 --num-tokens 65536`;
      one-token decode, short prefill; VRAM/expert-cache occupancy sane.
- [ ] 5b. Greedy next-token comparison vs llama.cpp on fixed prompts.
- [ ] 5c. Short needle + SWA-boundary (>512 tok) prompts; eos 2/24 stop behavior.
- [ ] Review section here.

## Initial limitations

TP=1 only; FTW conversion unsupported (serve the .gguf directly); text-only; no speculative decoding; split-GGUF variants (BF16 etc.)
out of scope; prefill for IQ expert types goes through moe_vec/dequant fallback
(optimize later if slow).
