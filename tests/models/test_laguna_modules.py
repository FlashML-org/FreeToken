from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "laguna-s-2.1-metadata.gguf"


@pytest.fixture(scope="module", autouse=True)
def _tp_one():
    from freetoken.distributed import set_tp_info

    set_tp_info(rank=0, size=1)


def _tiny_config():
    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.models.laguna.gguf import parse_gguf_config
    from freetoken.models.config import FullAttentionGroupConfig, RotaryConfig, SWAAttentionGroupConfig

    cfg = parse_gguf_config(build_gguf_shim(str(FIXTURE)))
    full_rope = replace(cfg.attention_groups[0].rotary_config, head_dim=32, rotary_dim=32)
    swa_rope = replace(cfg.attention_groups[1].rotary_config, head_dim=32, rotary_dim=32)
    groups = (
        FullAttentionGroupConfig("full", (0, 4), 2, 32, full_rope),
        SWAAttentionGroupConfig("swa", (1, 2, 3, 5, 6, 7), 2, 32, swa_rope, 512),
    )
    return replace(cfg, num_layers=8, num_qo_heads=6,
                   num_qo_heads_per_layer=(4, 6, 6, 6, 4, 6, 6, 6), num_kv_heads=2,
                   head_dim=32, hidden_size=64, intermediate_size=96, moe_intermediate_size=16,
                   shared_expert_intermediate_size=16, vocab_size=128, num_experts=8,
                   num_experts_per_tok=3, gguf_embed_quant=None,
        gguf_model_path=None, rotary_config=replace(cfg.rotary_config, head_dim=32, rotary_dim=32),
                   attention_groups=groups)


def test_attention_geometry(monkeypatch):
    import freetoken.models.laguna.attention as attention_mod
    from freetoken.models.laguna.attention import LagunaAttention
    monkeypatch.setattr(attention_mod, "get_rope", lambda **kw: SimpleNamespace(rotary_dim=kw["rotary_dim"]))
    cfg = _tiny_config(); full = LagunaAttention(cfg, 0); swa = LagunaAttention(cfg, 1)
    assert full.num_qo_heads == 4 and full.q_proj.weight.shape == (128, 64)
    assert full.k_proj.weight.shape == (64, 64) and full.v_proj.weight.shape == (64, 64)
    assert full.rotary.rotary_dim == 32 and full.attn_spec.sliding_window is None
    assert swa.num_qo_heads == 6 and swa.attn_spec.sliding_window == 512
    assert swa.gate_proj.weight.shape[0] == 6


def test_attention_forward_applies_gate_independently(monkeypatch):
    import freetoken.models.laguna.attention as attention_mod
    from freetoken.models.laguna.attention import LagunaAttention
    monkeypatch.setattr(attention_mod, "get_rope", lambda **kw: SimpleNamespace(rotary_dim=kw["rotary_dim"], forward=lambda p, q, k: (q, k)))
    cfg = _tiny_config(); x = torch.randn(3, 64)
    for layer_id, heads in ((0, 4), (1, 6)):
        attn = LagunaAttention(cfg, layer_id); seen = {}
        qv = torch.randn(3, heads * 32); kv = torch.randn(3, 2 * 32); vv = torch.randn(3, 2 * 32)
        gate = torch.randn(3, heads); backend = torch.randn(3, heads, 32, dtype=torch.float16)
        def proj(key, out):
            class P:
                def forward(self, z): seen[key] = z; return out
            return P()
        class O:
            def forward(self, z): seen["o"] = z; return z
        class N:
            def forward_inplace(self, z): return z
        attn.q_proj, attn.k_proj, attn.v_proj = proj("q", qv), proj("k", kv), proj("v", vv)
        attn.gate_proj, attn.o_proj = proj("gate", gate), O(); attn.q_norm = attn.k_norm = N()
        ctx = SimpleNamespace(batch=SimpleNamespace(positions=torch.arange(3)), attn_backend=SimpleNamespace(forward=lambda *a, **k: backend))
        monkeypatch.setattr(attention_mod, "get_global_ctx", lambda: ctx)
        got = attn.forward(x)
        assert seen["q"] is x and seen["k"] is x and seen["v"] is x and seen["gate"] is x
        expected = backend.view(3, heads, 32) * F.softplus(gate.float()).unsqueeze(-1).to(backend.dtype)
        torch.testing.assert_close(seen["o"], expected.reshape(3, heads * 32)); torch.testing.assert_close(got, seen["o"])


def test_router_matches_reference_and_full_forward():
    from freetoken.models.laguna.moe import LagunaSparseMoeBlock
    torch.manual_seed(0); blk = LagunaSparseMoeBlock.__new__(LagunaSparseMoeBlock)
    blk.top_k, blk.num_experts, blk.norm_topk_prob, blk.routed_scaling_factor = 3, 8, True, 2.5
    blk.gate = SimpleNamespace(weight=torch.randn(8, 64)); blk.e_score_correction_bias = torch.randn(8)
    x = torch.randn(5, 64); scores = (x @ blk.gate.weight.T).sigmoid(); ids = torch.topk(scores + blk.e_score_correction_bias, 3, dim=-1).indices
    weights = scores.gather(-1, ids); weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * 2.5
    got_w, got_ids = blk._route(x); torch.testing.assert_close(got_ids, ids.to(torch.int32)); torch.testing.assert_close(got_w, weights)
    blk.experts = SimpleNamespace(routed_forward=lambda h, w, i: h * w.sum(-1, keepdim=True)); blk.shared_experts = SimpleNamespace(forward=lambda h: h + 7)
    torch.testing.assert_close(blk.forward(x), x * got_w.sum(-1, keepdim=True) + x + 7)


def test_decoder_residual_semantics(monkeypatch):
    import freetoken.models.laguna.attention as attention_mod
    from freetoken.models.laguna.model import LagunaDecoderLayer
    monkeypatch.setattr(attention_mod, "get_rope", lambda **kw: SimpleNamespace(rotary_dim=kw["rotary_dim"]))
    layer = LagunaDecoderLayer(_tiny_config(), 0); x = torch.randn(3, 64)
    layer.input_layernorm = layer.ffn_norm = SimpleNamespace(forward=lambda z: z * 2)
    layer.self_attn = SimpleNamespace(forward=lambda z: z + 1); layer.mlp = SimpleNamespace(forward=lambda z: z * 3)
    h = x + (x * 2 + 1); torch.testing.assert_close(layer.forward(x), h + h * 2 * 3)


def test_laguna_mlp_fused_layout(monkeypatch):
    import freetoken.models.laguna.moe as moe_mod
    from freetoken.models.laguna.moe import LagunaMLP
    monkeypatch.setattr(moe_mod, "silu_and_mul", lambda z: F.silu(z[..., :4]) * z[..., 4:])
    m = LagunaMLP(8, 4); torch.manual_seed(2); m.gate_up_proj.weight.copy_(torch.randn_like(m.gate_up_proj.weight)); m.down_proj.weight.copy_(torch.randn_like(m.down_proj.weight))
    x = torch.randn(3, 8); wg, wu = m.gate_up_proj.weight[:4], m.gate_up_proj.weight[4:]
    m.gate_up_proj.forward = lambda z: F.linear(z, m.gate_up_proj.weight); m.down_proj.forward = lambda z: F.linear(z, m.down_proj.weight)
    expected = F.linear(F.silu(F.linear(x, wg)) * F.linear(x, wu), m.down_proj.weight)
    torch.testing.assert_close(m.forward(x), expected)


def test_deferred_gguf_linear_q8():
    from freetoken.models.gguf.dequant import GGML_Q8_0, row_bytes
    from freetoken.models.laguna.gguf import DeferredGGUFLinear
    layer = DeferredGGUFLinear(64, 32)
    with pytest.raises(AssertionError): layer.forward(torch.randn(2, 64))
    layer.materialize(GGML_Q8_0); assert layer.qweight.shape == (32, row_bytes(64, GGML_Q8_0))
    if not torch.cuda.is_available(): pytest.skip("CUDA required for fused GGUF forward")
    import gguf
    rng = np.random.default_rng(1); weight = rng.standard_normal((32, 64), dtype=np.float32); packed = gguf.quants.quantize(weight, gguf.GGMLQuantizationType.Q8_0)
    layer.qweight = torch.from_numpy(np.ascontiguousarray(packed)).cuda(); x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16); got = layer.forward(x).float(); blocks = x.float().reshape(2, -1, 32); scale = (blocks.abs().amax(dim=-1, keepdim=True) / 127).half().float(); aq = torch.where(scale > 0, (blocks / scale).round().clamp(-127, 127), blocks).mul(scale).reshape_as(x); ref = F.linear(aq, torch.from_numpy(gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.Q8_0)).float().cuda()); assert (got - ref).abs().max() <= 5e-3 * ref.abs().max().clamp(min=1.0)
