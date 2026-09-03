"""Glm5NextForCausalLM wiring smoke test (tiny random model, dense MLPs).

The per-op math is covered elsewhere (KDA kernels/op, kpool backend, mHC); this
test checks the ASSEMBLY: a 2-layer hybrid (KDA + DSA) model with mHC threading
runs prefill and decode through the real backends/pools, and the strongest
cache invariant holds -- decoding token T after prefilling [0, T) produces the
same logits as prefilling [0, T] outright (state handoff across the KDA
recurrent pool, the MLA latent pool, and the kpool indexer cache).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HIDDEN, VOCAB = 64, 128
KDA_H, KDA_D = 2, 128  # KDA kernels specialize on D=128
IDX_H, IDX_D = 16, 64
LATENT = 32
DEV = "cuda"


def _hf_config():
    from freetoken.utils.hf import RawConfigShim

    text = {
        "hidden_size": HIDDEN, "intermediate_size": 96, "num_hidden_layers": 2,
        "num_attention_heads": 2, "vocab_size": VOCAB, "hidden_act": "silu",
        "rms_norm_eps": 1e-5, "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        "q_lora_rank": 48, "kv_lora_rank": LATENT, "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 0, "v_head_dim": 32, "mla_use_nope": True,
        "index_n_heads": IDX_H, "index_head_dim": IDX_D, "index_topk": 32,
        "indexer_types": ["full", "full"], "indexer_rope_interleave": True,
        "index_kpool": 4, "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
        "linear_attn_config": {
            "num_heads": KDA_H, "head_dim": KDA_D,
            "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
        },
        "layer_types": ["linear_attention", "deepseek_sparse_attention"],
        "mlp_layer_types": ["dense", "dense"],  # no MoE machinery in this test
        "first_k_dense_replace": 2,
        "mhc": True, "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
        "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 1,
        "moe_intermediate_size": 32, "norm_topk_prob": True,
        "routed_scaling_factor": 2.5, "scoring_func": "sigmoid",
        "n_group": 1, "topk_group": 1, "swiglu_limit": 10.0,
        "attention_bias": False, "model_type": "glm5_next_text",
    }
    return RawConfigShim({
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next", "text_config": text,
    })


@pytest.fixture()
def rig(monkeypatch):
    from freetoken.attention.dsa_indexer_kpool import Glm5NextDSABackend
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.dsa_pool import KpoolDSAKVCache
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.glm5_next.config import parse_config
    from freetoken.models.glm5_next.model import Glm5NextForCausalLM

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    config = parse_config(_hf_config())

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    prev_dev = torch.get_default_device()
    torch.set_default_device(DEV)
    try:
        model = Glm5NextForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.set_default_device(prev_dev)

    # Random weights via the state-dict round trip (keeps shapes/dtypes honest).
    torch.manual_seed(0)
    sd = model.state_dict()
    rand = {}
    for k, v in sd.items():
        t = torch.randn(v.shape, dtype=torch.float32, device=DEV) * 0.05
        if k.endswith("norm.weight") or ".o_norm.weight" in k:
            t = t.abs() + 0.5
        rand[k] = t.to(v.dtype)
    model.load_state_dict(rand)

    kv = KpoolDSAKVCache(
        latent_dim=LATENT, num_layers=2, num_pages=4, page_size=64,
        dtype=torch.bfloat16, device=torch.device(DEV),
        index_head_dim=IDX_D, num_index_layers=1,
        index_ratio=4, num_req_slots=4,
    )
    page_table = torch.full((2, 256), -1, dtype=torch.int32, device=DEV)
    page_table[0] = torch.arange(256, dtype=torch.int32, device=DEV)
    linear_pool = LinearStatePool(
        config.linear_attention_group(), num_slots=4,
        dtype=torch.bfloat16, device=torch.device(DEV), tp_size=1,
    )

    ctx = SimpleNamespace(
        kv_cache=kv, page_table=page_table, linear_state_pool=linear_pool,
        attn_backend=None, batch=None,
    )
    for mod in (
        "freetoken.attention.dsa.get_global_ctx",
        "freetoken.models.glm5_next.kda.get_global_ctx",
        "freetoken.models.glm5_next.attention.get_global_ctx",
        "freetoken.models.glm5_next.model.get_global_ctx",
        "freetoken.layers.embedding.get_global_ctx",
    ):
        monkeypatch.setattr(mod, lambda: ctx)
    ctx.attn_backend = Glm5NextDSABackend(config)
    return model, ctx


def _req(device_len, cached_len):
    return SimpleNamespace(
        table_idx=0, device_len=device_len, extend_len=device_len - cached_len,
        cached_len=cached_len, linear_slot_idx=1, mamba_ping_pong=None,
    )


def _batch(ctx, ids, t0, phase):
    from freetoken.attention.linear import FLAMetadata

    t1 = t0 + len(ids)
    is_decode = phase == "decode"
    batch = SimpleNamespace(
        phase=phase,
        is_prefill=not is_decode, is_decode=is_decode, size=1,
        reqs=[_req(t1, t0)], padded_reqs=[_req(t1, t0)],
        input_ids=torch.tensor(ids, device=DEV),
        positions=torch.arange(t0, t1, device=DEV),
        out_loc=torch.arange(t0, t1, device=DEV),
        active_table_idx=torch.tensor([0], device=DEV) if is_decode else None,
        fla_metadata=FLAMetadata(
            cu_seqlens=torch.tensor([0, len(ids)], dtype=torch.int32, device=DEV),
            cache_indices=torch.tensor([1], dtype=torch.int32, device=DEV),
            has_initial_state=None if is_decode else torch.tensor([t0 > 0], device=DEV),
            fresh_state_indices=(
                None if (is_decode or t0 > 0)
                else torch.tensor([1], dtype=torch.int64, device=DEV)
            ),
        ),
        mm_embeds=None,
    )
    ctx.batch = batch
    ctx.attn_backend.prepare_metadata(batch)
    return batch


def _reset(ctx):
    ctx.linear_state_pool.reset(1)
    ctx.kv_cache._kv_buffer.zero_()
    ctx.kv_cache._index_k_buffer.zero_()
    ctx.kv_cache._tail_k.zero_()
    ctx.kv_cache._tail_gate.zero_()


def test_prefill_decode_consistency(rig):
    model, ctx = rig
    torch.manual_seed(1)
    total = 24
    ids = torch.randint(0, VOCAB, (total,)).tolist()

    # One-shot prefill over the full sequence: last-token logits per position
    # are only produced for the final token, so run it twice at different splits.
    _reset(ctx)
    _batch(ctx, ids, 0, "prefill")
    full_logits = model.forward()  # [1, VOCAB] logits of the last position
    assert full_logits.shape == (1, VOCAB)
    assert torch.isfinite(full_logits.float()).all()

    # Prefill [0, total-1) then decode the last token: must match the one-shot run.
    _reset(ctx)
    _batch(ctx, ids[:-1], 0, "prefill")
    model.forward()
    _batch(ctx, ids[-1:], total - 1, "decode")
    dec_logits = model.forward()
    err = (dec_logits.float() - full_logits.float()).abs().max().item()
    scale = full_logits.float().abs().max().item() + 1e-8
    assert err / scale < 3e-2, f"decode/prefill divergence: {err} (scale {scale})"


def test_chunked_prefill_consistency(rig):
    model, ctx = rig
    torch.manual_seed(2)
    total = 28  # split 16 + 12; chunk boundary pool-aligned (16 % 4 == 0)
    ids = torch.randint(0, VOCAB, (total,)).tolist()

    _reset(ctx)
    _batch(ctx, ids, 0, "prefill")
    full_logits = model.forward()

    _reset(ctx)
    _batch(ctx, ids[:16], 0, "prefill")
    model.forward()
    _batch(ctx, ids[16:], 16, "prefill")
    chunk_logits = model.forward()
    err = (chunk_logits.float() - full_logits.float()).abs().max().item()
    scale = full_logits.float().abs().max().item() + 1e-8
    assert err / scale < 3e-2, f"chunked/one-shot divergence: {err} (scale {scale})"
