# Open bug: qwen4exp GGUF loads and serves but generates incoherent text

Status: **open** (as of 2026-09-27). Branch: `gguf-quants`. Everything below is verified
against the source checkpoint with byte-range reads, so it can be picked up in a fresh
session without the 85 GB download.

## Symptom

`ft serve` on a native Qwen3.8-Flash-Next (`qwen4exp`) GGUF loads weights + expert banks,
captures CUDA graphs, and answers requests — but the text is multilingual gibberish from the
**first generated token**. The same file on ik_llama.cpp generates correctly, so the file and
the GGUF are sound; the fault is inside FreeToken's qwen4exp runtime.

| prompt | ik_llama (same file) | FreeToken |
|---|---|---|
| `The capital of France is` | ` Paris.` | `in无限yingting無限irl...` |
| `2+2=` | `4` | `_message\n收到...` |
| `def reverse_string(s):` | valid code | `a do,gnore...` |

## Artifacts

- GGUF (single file, 21.9 GiB, IQ4_XS experts): `/media/b/Hyena/gguf/mradermacher/qwen3.8-flash-coder-85gb-bf16-i1-GGUF/qwen3.8-flash-coder-85gb-bf16.i1-IQ4_XS.gguf`
- Source HF checkpoint used as ground truth (no full download needed; fetch by byte range):
  `Jab1718/qwen3.8-flash-coder-85gb-bf16` (`model-00001/00002-of-00002.safetensors`,
  `model.safetensors.index.json`, `tokenizer.json`, `config.json`). An index dump is at
  `/tmp/kilo/coder_idx.json` (may be gone after a reboot).
- Reference runtime: `/home/b/ai/ik_llama.cpp/build/bin/llama-cli` (has native `qwen4exp`).
- No PLE: this finetune has none, so PLE is not in the loop for this repro.

## Reproduction

```bash
cd ~/ai/FreeToken && source .venv/bin/activate
FREETOKEN_ALLOW_CUDA_MISMATCH=1 .venv/bin/ft serve \
  --model /media/b/Hyena/gguf/mradermacher/qwen3.8-flash-coder-85gb-bf16-i1-GGUF/qwen3.8-flash-coder-85gb-bf16.i1-IQ4_XS.gguf \
  --dtype bfloat16 --moe-strategy offload --text-model-only \
  --max-running-requests 1 --max-seq-len-override 4096 --cuda-graph-max-bs 1 \
  --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-coder-85gb-bf16.i1-IQ4_XS.gguf","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
```

Build/env gotchas are in `docs/build-troubleshooting.md` (system Python venv, gcc for
`setup.py`, clang for the nvcc JIT, `FREETOKEN_ALLOW_CUDA_MISMATCH=1` since only nvcc 12.8 is
installed while torch is cu130).

## Fixed along the way (committed; these were real loader bugs)

`f036201` mapped the four converter transforms the loader wasn't undoing. All are verified
against the source HF tensors:

1. **Zero-centred norms stored as `(1+w)`** — subtract 1 for `attn_q_norm`/`attn_k_norm`,
   indexer `q/k_layernorm`, `hc_*_norm`, `output_hc_norm`, `ple_norm_*` (`ssm_norm` is *not*
   zero-centred; leave raw).
2. **GDN value heads reordered** by the converter (`gguf[j] = hf[i]`, `j = i//R + K*(i%R)`)
   — gather back for `ssm_a`, `ssm_dt.bias`, `in_proj_b/a`, `attn_gate`, the value block of
   `attn_qkv`, `conv1d` value channels, and `ssm_out` (emitted dense: its input axis packs
   across 128-wide heads).
3. **`ssm_a` = `-exp(A_log)`** — invert with `log(-A)` in fp32; load `dt_bias` fp32.
4. **Text mRoPE** with the GGUF `rope.dimension_sections` (interleaved), not plain rope.

Config robustness: missing/all-zero `compress_ratios` -> 4 (source HF says 4); missing
`expert_shared_feed_forward_length` -> routed width when a shared expert is present; no-PLE
placeholder kept non-degenerate.

Earlier commits in the branch: quant types + generic expert banks (`0240db5`), split reader
(`a4f0247`), qwen4exp GGUF support (`ce30bcc`), pin-budget fail-fast (`3b7ff96`), review fixes
(`94e9205`), follow-ups (`13c8c4d`), split-in_proj test (`e65d7a9`).

## Ruled out (each verified against a reference)

| Area | How verified | Result |
|---|---|---|
| All weight mappings (incl. top-level mixer, `token_embd`/`output`, experts) | byte-range read of HF tensors vs our dequantized values | exact for F32/BF16; corr >= 0.996 for packed |
| GDN `in_proj_qkv/z/b/a`, `out_proj`, `conv1d`, `A_log`, `dt_bias` | HF tensors | match |
| GDN **split** `in_proj_qkvz` + `in_proj_ba` (the op-swap layout) | `test_split_in_proj_matches_fused`, also at real geometry (hidden 2560, 16/48 heads) | corr 1.0000 |
| GDN op math | `tests/models/qwen4_exp/test_gdn.py` vs `gdn_reference.py` | pass |
| QSA indexer/attention | `tests/models/qwen4_exp/test_qsa_hf.py` vs HF math (index geometry already shipping: head_dim 256, idx 128, 4/1 heads, budget 2048, ratio 4, page 64) | pass |
| QSA config + indexer/q/k norms | HF config + tensors | equal / exact |
| rope + mRoPE (incl. text degeneracy), section tables | `tests/kernels/test_mrope.py` + independent GPU check | bit-exact / exact |
| HC mix/gate/combine Triton kernels | vs `hc.py` torch path, T = 1,3,16,17,66,67,160 | bf16-level |
| HC/`input_mix`/top mixer tensors | HF | exact / quant |
| Expert banks (gate/up/down order + content) | HF `experts.gate_up_proj`/`down_proj` | corr >= 0.996 |
| MoE forward wiring | code inspection (`qwen3_5_moe/moe.py`: `routed + sigmoid(shared_gate)*shared`, `renormalize=norm_topk_prob`) | looks correct |
| Tokenizer ids | vs HF `tokenizer.json` | identical |
| Model composition (embed -> repeat(hc) -> layers -> top mixer -> lm_head) | code inspection (`models/qwen4_exp/model.py`) | matches documented HF contract |
| `test_qsa_backend` 3 failures | chunked vs one-shot `torch.equal` Triton flake | pre-existing/environmental |

## Not yet checked (next candidates, ranked)

1. **Manual prefill forward, engine bypassed** (build the real model, run
   `Qwen4ExpDecoderLayer`/`Qwen4ExpModel.forward` on a fixed prompt, decode `lm_head` logits,
   dump per-layer hidden stats). This separates engine plumbing from the model and localizes
   the divergence. Sketch below.
2. **QSA main attention at `num_q = 24` / `num_kv = 2`** — the HF-reference test uses
   `num_q = 4` (head_dim and index geometry are already real). GQA repeat 12 is untested.
3. **MoE forward numerics at real geometry** (`num_experts=160`, top-10) against a written
   HF reference — only wiring was inspected, not the numbers.
4. Engine plumbing: `Batch.get_attn_positions`/`mrope_positions` for QSA windows,
   `graph.py` decode buffer, `kvcache/qsa_pool.py` addressing, `LinearStatePool` slots.
5. Any dtype/precision path: `--dtype float32` run (rules out bf16 accumulation if it
   becomes coherent).

Note: a single bad component would not explain "everything matches its reference but output is
garbage", so favour (1) first — it tells you whether the model itself or the engine wrapping
is at fault.

## Plan / harness sketch for the next session

Build the model without the server and run one prefill on `The capital of France is`:

```python
# set_tp_info(0, 1) first
from freetoken.utils import cached_load_hf_config
from freetoken.models.register import get_model_spec, _load_attr
from freetoken.models.weight import load_weight
from freetoken.engine.engine import _materialize_loaded_weight_state_dict, _weight_load_context
# build config+model (moe_strategy="offload"), then:
with _weight_load_context():
    sd = _materialize_loaded_weight_state_dict(model.state_dict(),
        load_weight(GGUF, dev, include_moe_experts=False, include_vision=False), device=dev)
    model.load_state_dict(sd)
# ctx: freetoken.core.Context(page_size=?), create_kv_pool(...), LinearStatePool(group, slots, bf16, dev),
#      attention backend (attention.create_attention_backend), set_global_ctx, page tables
# batch: prefill with input_ids = tokenizer.encode(prompt), positions, mrope_positions [3,n] (equal rows),
#        out_loc; then `with ctx.forward_batch(batch): hidden = model.model.forward(input_ids, batch)`
# logits = model.lm_head.forward(hidden); top-1 -> token
```

The ctx wiring is easiest copied from `tests/models/qwen4_exp/common.py:fixture` (pool, backend,
`set_global_ctx`) scaled to the real ModelConfig from `parse_gguf_config`. Compare:
- top-1/top-5 logits vs ik_llama on the same prompt;
- per-layer `R` stats (min/max/std, any NaN/inf) to see where it blows up.

If the manual forward is **also** gibberish, bisect layers: feed the HF-transcribed reference
for layer0 (GDN) and layer3 (QSA) with the same `R` and compare `R'`; the tests
(`test_gdn.py`, `test_qsa_hf.py`) already contain the reference math to extend.

If the manual forward is **coherent** but the server is not, the bug is in the engine plumbing
(batch/positions/graph/pool) — then compare `forward` under `ctx.forward_batch` vs the engine's
batch construction in `scheduler/scheduler.py` (`_make_positions`/`_make_mrope_positions`) and
`engine/graph.py`.

## Key code pointers

- Loader/mapping/op-swap: `python/freetoken/models/qwen4_exp/gguf.py` (`parse_gguf_config`,
  `_gdn_head_perm`, `_ZERO_CENTERED_NORMS`, `iter_gguf_weights`, `convert_qwen4_exp_to_gguf`).
- Model wiring: `python/freetoken/models/qwen4_exp/model.py`.
- QSA: `python/freetoken/models/qwen4_exp/attention.py`, `python/freetoken/attention/qsa_sparse.py`,
  `python/freetoken/kvcache/qsa_pool.py`, kernels in `python/freetoken/kernel/triton/qsa/`.
- GDN: `python/freetoken/models/qwen4_exp/gdn.py` (+ `gdn_reference.py`).
- HC: `python/freetoken/models/qwen4_exp/hc.py`, kernels in `python/freetoken/kernel/triton/hc.py`.
- MoE: `python/freetoken/models/qwen3_5_moe/moe.py`, offload in `python/freetoken/layers/moe.py`.
- Batches/positions: `python/freetoken/core.py` (`Batch.get_attn_positions`),
  `python/freetoken/scheduler/scheduler.py` (`_make_positions`/`_make_mrope_positions`),
  `python/freetoken/engine/graph.py`.
- Expert banks: `python/freetoken/moe/gguf_experts.py`, `python/freetoken/moe/expert_banks.py`.
