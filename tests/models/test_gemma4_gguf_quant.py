from __future__ import annotations

import copy
from types import SimpleNamespace

import torch


def test_cuda_quant_types_have_packed_row_layouts():
    from freetoken.layers.gguf import _DEQUANT, _MMQ, _MMVQ
    from freetoken.models.gguf.dequant import BLOCK_SHAPE

    expected = {
        2: (32, 18),
        3: (32, 20),
        6: (32, 22),
        7: (32, 24),
        8: (32, 34),
        10: (256, 84),
        11: (256, 110),
        12: (256, 144),
        13: (256, 176),
        14: (256, 210),
        16: (256, 66),
        17: (256, 74),
        18: (256, 98),
        19: (256, 50),
        20: (32, 18),
        21: (256, 110),
        22: (256, 82),
        23: (256, 136),
        29: (256, 56),
    }

    assert {quant_type: BLOCK_SHAPE[quant_type] for quant_type in expected} == expected
    assert _MMVQ == set(expected)
    assert _DEQUANT == set(expected)
    assert _MMQ == {2, 3, 6, 7, 8, 10, 11, 12, 13, 14}


def test_gguf_config_shim_can_be_masked_like_hf_config():
    from freetoken.models.gguf.config import GgufConfigShim

    shim = GgufConfigShim(
        architectures=["Gemma4GGUFForCausalLM"],
        model_path="model.gguf",
        model_type="gemma4",
        metadata={},
        vocab_size=4,
        tie_word_embeddings=True,
    )
    masked = copy.copy(shim)
    masked.audio_config = None
    assert masked.audio_config is None


class _Tensor:
    def __init__(self, name: str, ggml_type: int, packed: torch.Tensor):
        self.name = name
        self.ggml_type = ggml_type
        self._packed = packed

    def packed(self) -> torch.Tensor:
        return self._packed


def test_quant_layout_detects_dense_and_per_layer_expert_types(monkeypatch):
    from freetoken.models.gemma4 import gguf
    from freetoken.models.gguf import reader

    tensors = [
        _Tensor("token_embd.weight", 8, torch.empty(1, 34, dtype=torch.uint8)),
        _Tensor("blk.0.attn_q.weight", 8, torch.empty(1, 34, dtype=torch.uint8)),
        _Tensor("blk.0.attn_k.weight", 8, torch.empty(1, 34, dtype=torch.uint8)),
        _Tensor("blk.0.attn_v.weight", 8, torch.empty(1, 34, dtype=torch.uint8)),
        _Tensor("blk.0.attn_output.weight", 8, torch.empty(1, 34, dtype=torch.uint8)),
        _Tensor("blk.0.ffn_gate.weight", 8, torch.empty(1, 34, dtype=torch.uint8)),
        _Tensor("blk.0.ffn_up.weight", 8, torch.empty(1, 34, dtype=torch.uint8)),
        _Tensor("blk.0.ffn_down.weight", 8, torch.empty(1, 34, dtype=torch.uint8)),
        _Tensor("blk.0.ffn_gate_up_exps.weight", 21, torch.empty(4, 11, dtype=torch.uint8)),
        _Tensor("blk.0.ffn_down_exps.weight", 20, torch.empty(4, 13, dtype=torch.uint8)),
        _Tensor("blk.1.attn_q.weight", 8, torch.empty(1, 34, dtype=torch.uint8)),
        _Tensor("blk.1.attn_k.weight", 8, torch.empty(1, 34, dtype=torch.uint8)),
        _Tensor("blk.1.attn_output.weight", 8, torch.empty(1, 34, dtype=torch.uint8)),
        _Tensor("blk.1.ffn_gate.weight", 8, torch.empty(1, 34, dtype=torch.uint8)),
        _Tensor("blk.1.ffn_up.weight", 8, torch.empty(1, 34, dtype=torch.uint8)),
        _Tensor("blk.1.ffn_down.weight", 8, torch.empty(1, 34, dtype=torch.uint8)),
        _Tensor("blk.1.ffn_gate_up_exps.weight", 23, torch.empty(4, 17, dtype=torch.uint8)),
        _Tensor("blk.1.ffn_down_exps.weight", 8, torch.empty(4, 19, dtype=torch.uint8)),
    ]
    monkeypatch.setattr(reader, "iter_gguf_tensors", lambda _path: iter(tensors))

    layout = gguf._gguf_quant_layout("model.gguf", 2, 4)

    assert layout["embedding"] == 8
    assert layout["qkv"] == (8, 8)
    assert layout["shared_gate_up"] == (8, 8)
    assert layout["expert_gate_up"] == (21, 23)
    assert layout["expert_down"] == (20, 8)
    assert layout["expert_gate_up_bytes"] == (11, 17)
    assert layout["expert_down_bytes"] == (13, 19)


def test_mixed_expert_loader_pads_slots_but_preserves_packed_bytes(monkeypatch):
    from freetoken.models.gemma4 import gguf
    from freetoken.models.gguf import reader

    config = SimpleNamespace(
        num_layers=2,
        num_experts=2,
        hidden_size=32,
        moe_intermediate_size=32,
        gguf_quant_types={
            "expert_gate_up": (21, 23),
            "expert_down": (20, 8),
            "expert_gate_up_bytes": (7, 11),
            "expert_down_bytes": (5, 13),
        },
    )
    gu0 = torch.arange(14, dtype=torch.uint8).reshape(2, 7)
    dn0 = torch.arange(10, dtype=torch.uint8).reshape(2, 5)
    gu1 = torch.arange(22, dtype=torch.uint8).reshape(2, 11)
    dn1 = torch.arange(26, dtype=torch.uint8).reshape(2, 13)
    tensors = [
        _Tensor("blk.0.ffn_gate_up_exps.weight", 21, gu0),
        _Tensor("blk.0.ffn_down_exps.weight", 20, dn0),
        _Tensor("blk.1.ffn_gate_up_exps.weight", 23, gu1),
        _Tensor("blk.1.ffn_down_exps.weight", 8, dn1),
    ]
    monkeypatch.setattr(reader, "iter_gguf_tensors", lambda _path: iter(tensors))
    monkeypatch.setattr(gguf, "_require_tp1", lambda _what: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    banks = gguf.load_q4_0_expert_sources("model.gguf", config)

    assert [tuple(t.shape) for t in banks["gate_up"]] == [(2, 11), (2, 11)]
    assert [tuple(t.shape) for t in banks["down"]] == [(2, 13), (2, 13)]
    torch.testing.assert_close(banks["gate_up"][0][:, :7], gu0)
    torch.testing.assert_close(banks["down"][0][:, :5], dn0)
    torch.testing.assert_close(banks["gate_up"][1], gu1)
    torch.testing.assert_close(banks["down"][1], dn1)


def test_gguf_expert_gemm_passes_independent_quant_types(monkeypatch):
    from freetoken.kernel import gguf as kernel
    from freetoken.moe import fused_q4_0
    from freetoken.moe.fused_q4_0 import fused_experts_gguf_q4_0

    calls = []

    def fake_moe(x, weight, ids, top_k, quant_type, row, tokens):
        calls.append((quant_type, tuple(weight.shape), top_k, row, tokens))
        return torch.ones(tokens * top_k, row, dtype=x.dtype)

    monkeypatch.setattr(kernel, "ggml_moe_a8_vec", fake_moe)
    monkeypatch.setitem(fused_q4_0._ACT, "silu", lambda x: x[:, : x.shape[1] // 2])
    hidden = torch.ones(1, 4)
    gate_up = torch.empty(2, 32, dtype=torch.uint8)
    down = torch.empty(2, 48, dtype=torch.uint8)
    weights = torch.ones(1, 1)
    ids = torch.zeros(1, 1, dtype=torch.int32)

    out = fused_experts_gguf_q4_0(
        hidden,
        gate_up,
        down,
        weights,
        ids,
        "silu",
        gate_up_quant_type=21,
        down_quant_type=20,
        intermediate_size=3,
        hidden_size=4,
    )

    assert out.shape == (1, 4)
    assert [call[0] for call in calls] == [21, 20]
