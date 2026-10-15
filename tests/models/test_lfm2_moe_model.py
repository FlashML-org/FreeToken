"""Lfm2MoeForCausalLM assembly smoke test (tiny synthetic checkpoint, resident experts).

The per-op math is covered elsewhere (conv, routing, weight reader); this test checks the
ASSEMBLY: a 3-layer hybrid (conv + attention + conv, dense MLP then MoE) with the real pools
and the triton attention backend, and the strongest cache invariant holds -- decoding token T
after prefilling [0, T) produces the same logits as prefilling [0, T], and so does a chunked
prefill. That exercises the conv state handoff through ``LinearStatePool`` and the paged KV
remap to the full-attention layers only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .test_lfm2_moe_weight import E, V, _raw, _write

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = "cuda"
MAX_SEQ = 128  # page_size 1: one page per token
NUM_SLOTS = 4


class _Rig:
    def __init__(self, model, ctx):
        self.model = model
        self.ctx = ctx

    def reset(self) -> None:
        self.ctx.linear_state_pool.reset(1)

    def forward(self, ids, t0=0, phase="prefill"):
        from freetoken.attention.linear import FLAMetadata

        t1 = t0 + len(ids)
        device = torch.device(DEV)
        is_decode = phase == "decode"
        positions = torch.arange(t0, t1, device=device)
        req = SimpleNamespace(
            table_idx=0, device_len=t1, extend_len=len(ids), cached_len=t1 - len(ids),
            linear_slot_idx=1, mamba_ping_pong=None,
        )
        if is_decode:
            fla = FLAMetadata(
                cu_seqlens=torch.arange(2, dtype=torch.int32, device=device),
                cache_indices=torch.tensor([1], dtype=torch.int32, device=device),
            )
        else:
            slot = torch.tensor([1], dtype=torch.int64, device=device)
            fla = FLAMetadata(
                cu_seqlens=torch.tensor([0, len(ids)], dtype=torch.int32, device=device),
                cache_indices=slot.to(torch.int32),
                has_initial_state=torch.tensor([t0 > 0], device=device),
                fresh_state_indices=None if t0 > 0 else slot,
            )
        batch = SimpleNamespace(
            phase=phase, is_prefill=not is_decode, is_decode=is_decode, size=1,
            reqs=[req], padded_reqs=[req],
            input_ids=torch.tensor(ids, device=device),
            positions=positions,
            out_loc=torch.arange(t0, t1, device=device),
            active_table_idx=torch.tensor([0], device=device) if is_decode else None,
            fla_metadata=fla, mm_embeds=None, mm_rows=None,
            get_attn_positions=lambda: positions,
        )
        self.ctx.attn_backend.prepare_metadata(batch)
        with self.ctx.forward_batch(batch), torch.inference_mode():
            return self.model.forward()


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    from freetoken.core import Context
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _materialize_loaded_weight_state_dict
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.layers import iter_moe_layers, rotary, set_rope_device
    from freetoken.layers.quantization.method import finalize_quant
    from freetoken.models import create_model
    from freetoken.moe.expert_banks import attach_resident_banks, load_expert_banks
    from freetoken.models.lfm2_moe.weight import iter_weights
    from freetoken.utils.torch_utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    folder = _write(tmp_path, "bf16", _raw("bf16"))
    config = EngineConfig(
        model_path=folder, tp_info=try_get_tp_info(), dtype=torch.bfloat16, moe_strategy="fused"
    )
    mc = config.model_config
    device = torch.device(DEV)
    saved_rope = rotary._ROPE_DEVICE
    # the engine builds on meta with the rope table on the device; get_rope only consults
    # _ROPE_DEVICE from under a meta default device
    set_rope_device(device)
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            model = create_model(mc)
        state = _materialize_loaded_weight_state_dict(
            model.state_dict(),
            iter_weights(folder, device, include_moe_experts=False, include_non_moe=True),
            device=device,
        )
        model.load_state_dict(state)
        finalize_quant(model)
        # resident experts load as banks, not through the state dict (as Engine._load_resident_experts)
        layers = list(iter_moe_layers(model))
        attach_resident_banks(layers, load_expert_banks(
            folder, mc, method=layers[0].quant_method, device=device,
            dtype=torch.bfloat16, parallel=False, resident=True,
        ))
        model.eval() if hasattr(model, "eval") else None

        from freetoken.attention.triton import TritonAttentionBackend

        pool = create_kvcache_pool(
            model_config=mc, num_pages=MAX_SEQ, page_size=1, dtype=torch.bfloat16,
            device=device, num_req_slots=2,
        )
        page_table = torch.full((2, MAX_SEQ), -1, dtype=torch.int32, device=device)
        page_table[0] = torch.arange(MAX_SEQ, dtype=torch.int32, device=device)
        ctx = Context(page_size=1)
        ctx.kv_cache = pool
        ctx.page_table = page_table
        ctx.linear_state_pool = LinearStatePool(
            mc.state_group(), num_slots=NUM_SLOTS, dtype=torch.bfloat16, device=device, tp_size=1
        )
        monkeypatch.setattr("freetoken.core._GLOBAL_CTX", ctx)
        ctx.attn_backend = TritonAttentionBackend(mc)
        yield _Rig(model, ctx)
    finally:
        set_rope_device(saved_rope)
        rotary.get_rope.cache_clear()


def _assert_same_logits(a, b, tag):
    err = (a.float() - b.float()).abs().max().item()
    scale = a.float().abs().max().item() + 1e-8
    assert err / scale < 3e-2, f"{tag}: max abs err {err:.5f} (scale {scale:.3f})"


def test_decode_after_prefill_matches_a_one_shot_prefill(rig):
    torch.manual_seed(1)
    total = 24
    ids = torch.randint(0, V, (total,)).tolist()

    rig.reset()
    full = rig.forward(ids, t0=0, phase="prefill")
    assert full.shape == (1, V) and torch.isfinite(full.float()).all()

    rig.reset()
    rig.forward(ids[:-1], t0=0, phase="prefill")
    step = rig.forward(ids[-1:], t0=total - 1, phase="decode")
    _assert_same_logits(step, full, "decode vs one-shot prefill")


def test_chunked_prefill_matches_a_one_shot_prefill(rig):
    torch.manual_seed(2)
    total = 28
    ids = torch.randint(0, V, (total,)).tolist()

    rig.reset()
    full = rig.forward(ids, t0=0, phase="prefill")

    rig.reset()
    rig.forward(ids[:16], t0=0, phase="prefill")
    chunked = rig.forward(ids[16:], t0=16, phase="prefill")
    _assert_same_logits(chunked, full, "chunked vs one-shot prefill")


def test_every_moe_layer_owns_banks(rig):
    """The leading dense layer runs a plain MLP; the two trailing MoE layers own one bank set each."""
    from freetoken.layers import iter_moe_layers

    moe = list(iter_moe_layers(rig.model))
    assert [layer.layer_id for layer in moe] == [0, 1]  # bank ids: layers 1 and 2 minus the dense prefix
    assert all(layer._expert_banks["gate_up"].shape[0] == E for layer in moe)
    layers = rig.model.model.layers.op_list
    assert [type(layer.feed_forward).__name__ for layer in layers] == [
        "Lfm2DenseMLP", "Lfm2MoeSparseBlock", "Lfm2MoeSparseBlock"
    ]
    assert [hasattr(layer, "self_attn") for layer in layers] == [False, True, False]
    assert rig.model.lm_head.tied_embedding is rig.model.model.embed_tokens
