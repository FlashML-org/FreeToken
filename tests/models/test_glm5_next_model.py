"""Glm5NextForCausalLM wiring with dense BF16 or offloaded NVFP4 experts.

The per-op math is covered elsewhere (KDA kernels/op, kpool backend, mHC); this
test checks the assembly: a hybrid KDA + DSA model with mHC threading
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


def _hf_config(nvfp4=False):
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
    data = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next", "text_config": text,
    }
    if nvfp4:
        from tests.models.test_glm5_next_config import _CT_MIXED_QUANT

        # Preserve the published regex's layer IDs and feed a routed MoE output into KDA.
        text.update(
            num_hidden_layers=5,
            layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention", "linear_attention"],
            mlp_layer_types=["dense"] * 3 + ["sparse"] * 2,
            first_k_dense_replace=3, indexer_types=["full"] * 5,
            moe_intermediate_size=64,
        )
        data["quantization_config"] = _CT_MIXED_QUANT
    return RawConfigShim(data)


@pytest.fixture(params=["dense-bf16", "RedHatAI-nvfp4"])
def rig(monkeypatch, request, tmp_path):
    from freetoken.attention.dsa_indexer_kpool import Glm5NextDSABackend
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.layers.quantization import QuantBackend, QuantKind, finalize_quant
    from freetoken.models.glm5_next.config import parse_config
    from freetoken.models.glm5_next.model import Glm5NextForCausalLM
    from freetoken.models.register import checkpoint_quant_config, get_model_spec

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    nvfp4 = request.param == "RedHatAI-nvfp4"
    hf = _hf_config(nvfp4)
    config = parse_config(hf)
    quant = checkpoint_quant_config(str(tmp_path), hf, get_model_spec(hf.architectures[0]))
    object.__setattr__(config, "quant", quant)
    object.__setattr__(config, "moe_strategy", "offload" if nvfp4 else "resident")
    monkeypatch.setattr(
        "freetoken.layers.quantization.quant_backend._QUANT_BACKEND",
        QuantBackend.parse("moe.nvfp4=triton"),
    )

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
    assert finalize_quant(model) > 0
    model.prepare_for_runtime()

    kv = create_kvcache_pool(
        config, num_pages=4, page_size=64,
        dtype=torch.bfloat16, device=torch.device(DEV),
        num_req_slots=4, kv_quant="nvfp4" if nvfp4 else "none",
    )
    page_table = torch.full((2, 256), -1, dtype=torch.int32, device=DEV)
    page_table[0] = torch.arange(256, dtype=torch.int32, device=DEV)
    linear_pool = LinearStatePool(
        config.linear_attention_group(), num_slots=4,
        dtype=torch.bfloat16, device=torch.device(DEV), tp_size=1,
    )

    ctx = SimpleNamespace(
        kv_cache=kv, page_table=page_table, linear_state_pool=linear_pool,
        attn_backend=None, batch=None, moe_offload_cache=None,
    )
    for mod in (
        "freetoken.attention.dsa.get_global_ctx",
        "freetoken.models.glm5_next.kda.get_global_ctx",
        "freetoken.models.glm5_next.attention.get_global_ctx",
        "freetoken.models.glm5_next.model.get_global_ctx",
        "freetoken.layers.embedding.get_global_ctx",
        "freetoken.layers.moe.get_global_ctx",
    ):
        monkeypatch.setattr(mod, lambda: ctx)
    ctx.attn_backend = Glm5NextDSABackend(config)
    if nvfp4:
        from freetoken.moe.expert_banks import build_expert_banks
        from freetoken.moe.offload_cache import (
            OffloadMoeCache, attach_offload_moe_cache, iter_offload_moe_layers,
        )

        experts = list(iter_offload_moe_layers(model))
        assert len(experts) == 2
        for expert in experts:
            method = expert.quant_method
            assert method.kind is QuantKind.NVFP4
            assert method.kernel.name == "triton"
            assert method.cfg.activation == "swiglu_clamp" and method.cfg.limit == 10.0
            assert method.cfg.strategy == "offload" and method.cfg.decode_target == "gpu"
        banks = build_expert_banks(
            experts[0].quant_method, len(experts), None, device=torch.device(DEV), dummy=True,
        )
        cache = OffloadMoeCache(
            num_layers=len(experts), num_experts=config.num_experts,
            cache_size=config.num_experts, device=torch.device(DEV),
            quant_format=banks.quant_format, layout=banks.layout, prefill_overlap=False,
            max_slots=experts[0].quant_method.slot_limit(),
        )
        cache.set_bank_sources(banks.sources)
        cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
        cache.reset()
        assert len(attach_offload_moe_cache(model, cache)) == len(experts)
        ctx.moe_offload_cache = cache
        assert kv.num_layers == 1
        assert kv.latent_rows(3).shape == (256, LATENT // 2)
        assert kv.latent_rows(3).dtype == torch.uint8
        assert kv.latent_block_scale(3).shape == (256, LATENT // 16)
        assert kv.latent_block_scale(3).dtype == torch.uint8
        assert kv.latent_scale(3).dtype == torch.float32
        assert kv.index_k_cache(0).dtype == torch.bfloat16
        assert kv.tail_k(0).dtype == kv.tail_gate(0).dtype == torch.bfloat16
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
    if ctx.kv_cache.kv_quant == "nvfp4":
        ctx.kv_cache._scale_buffer.zero_()
        ctx.kv_cache._block_scale_buffer.zero_()
        ctx.moe_offload_cache.reset()


def _state(ctx):
    pool = ctx.linear_state_pool
    return pool.recurrent_states[:, 1].clone(), pool.conv_states[:, 1].clone()


def _assert_state_close(actual, expected):
    for name, got, ref in zip(("recurrent", "convolution"), actual, expected):
        assert torch.isfinite(got).all()
        for layer, (got_layer, ref_layer) in enumerate(zip(got, ref)):
            scale = ref_layer.float().abs().max().item()
            assert scale > 0
            err = (got_layer.float() - ref_layer.float()).abs().max().item()
            assert err / scale < 3e-2, f"{name} layer {layer} divergence: {err} (scale {scale})"


def test_prefill_decode_consistency(rig):
    model, ctx = rig
    torch.manual_seed(1)
    nvfp4 = ctx.kv_cache.kv_quant == "nvfp4"
    total, decode_tokens = (40, 3) if nvfp4 else (24, 1)
    ids = torch.randint(0, VOCAB, (total,)).tolist()

    # One-shot prefill over the full sequence: last-token logits per position
    # are only produced for the final token, so run it twice at different splits.
    _reset(ctx)
    _batch(ctx, ids, 0, "prefill")
    full_logits = model.forward()  # [1, VOCAB] logits of the last position
    assert full_logits.shape == (1, VOCAB)
    assert torch.isfinite(full_logits.float()).all()
    full_state = _state(ctx)

    # Continue through incomplete and completed kpool tails using the same request state.
    _reset(ctx)
    prefix = total - decode_tokens
    _batch(ctx, ids[:prefix], 0, "prefill")
    model.forward()
    if nvfp4:
        packed_prefix = ctx.kv_cache.latent_rows(3)[:prefix].clone()
        scale_prefix = ctx.kv_cache.latent_scale(3)[:prefix].clone()
        block_prefix = ctx.kv_cache.latent_block_scale(3)[:prefix].clone()
        assert packed_prefix.any() and block_prefix.any()
        assert torch.isfinite(scale_prefix).all() and (scale_prefix > 0).all()
    for pos in range(prefix, total):
        _batch(ctx, ids[pos:pos + 1], pos, "decode")
        dec_logits = model.forward()
    err = (dec_logits.float() - full_logits.float()).abs().max().item()
    scale = full_logits.float().abs().max().item() + 1e-8
    assert err / scale < 3e-2, f"decode/prefill divergence: {err} (scale {scale})"
    _assert_state_close(_state(ctx), full_state)
    if nvfp4:
        assert torch.equal(ctx.kv_cache.latent_rows(3)[:prefix], packed_prefix)
        assert torch.equal(ctx.kv_cache.latent_scale(3)[:prefix], scale_prefix)
        assert torch.equal(ctx.kv_cache.latent_block_scale(3)[:prefix], block_prefix)
        assert (ctx.kv_cache.latent_scale(3)[prefix:total] > 0).all()


def test_chunked_prefill_consistency(rig):
    model, ctx = rig
    torch.manual_seed(2)
    nvfp4 = ctx.kv_cache.kv_quant == "nvfp4"
    total, split = (44, 17) if nvfp4 else (28, 16)
    ids = torch.randint(0, VOCAB, (total,)).tolist()

    _reset(ctx)
    _batch(ctx, ids, 0, "prefill")
    full_logits = model.forward()
    full_state = _state(ctx)

    _reset(ctx)
    _batch(ctx, ids[:split], 0, "prefill")
    model.forward()
    _batch(ctx, ids[split:], split, "prefill")
    chunk_logits = model.forward()
    err = (chunk_logits.float() - full_logits.float()).abs().max().item()
    scale = full_logits.float().abs().max().item() + 1e-8
    assert err / scale < 3e-2, f"chunked/one-shot divergence: {err} (scale {scale})"
    _assert_state_close(_state(ctx), full_state)
