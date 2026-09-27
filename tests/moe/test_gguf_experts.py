"""Native-GGUF expert banks/builds, generic over the block-quant type.

Exercises the loader and the grouped-GEMM dispatch without a real checkpoint or a GPU
by feeding synthetic packed tensors; the kernel-level numerical equivalence lives in
``tests/models/test_gguf_dequant.py`` (dequant) and was checked on-device against the
borrowed ``ggml_moe_a8_vec``.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.gguf.dequant import (
    GGML_IQ4_NL,
    GGML_IQ4_XS,
    GGML_Q5_K,
    row_bytes,
)
from freetoken.moe.gguf_experts import (
    GGUF_EXPERT_QUANTS,
    dummy_gguf_expert_sources,
    expert_specs,
    fused_experts_gguf,
    load_gguf_expert_sources,
)

_CASES = [(GGML_Q5_K, "q5_K"), (GGML_IQ4_NL, "iq4_nl"), (GGML_IQ4_XS, "iq4_xs")]
E, H, I = 2, 256, 256


@dataclass
class _FakeTensor:
    name: str
    ggml_type: int
    _data: torch.Tensor

    def packed(self) -> torch.Tensor:
        return self._data.reshape(-1)


def _bytes(n: int) -> torch.Tensor:
    return (torch.arange(n, dtype=torch.int64) % 251).to(torch.uint8)


def _init_tp() -> None:
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


@pytest.fixture(autouse=True)
def _cpu_banks(monkeypatch):
    _init_tp()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _patch_tensors(monkeypatch, tensors) -> None:
    from freetoken.models.gguf import reader

    monkeypatch.setattr(reader, "iter_gguf_tensors", lambda path: iter(tensors))


@pytest.mark.parametrize("quant_type,name", _CASES)
def test_expert_specs_use_the_type_block_size(quant_type: int, name: str) -> None:
    specs = expert_specs(H, I, E, quant_type, quant_type)
    assert specs["gate_up"] == ((E, 2 * I, row_bytes(H, quant_type)), torch.uint8)
    assert specs["down"] == ((E, H, row_bytes(I, quant_type)), torch.uint8)


@pytest.mark.parametrize("quant_type,name", _CASES)
def test_load_fused_gate_up_and_down_match_source_bytes(monkeypatch, quant_type: int, name: str) -> None:
    hb, ib = row_bytes(H, quant_type), row_bytes(I, quant_type)
    gate_up = _bytes(E * 2 * I * hb)
    down = _bytes(E * H * ib)
    _patch_tensors(monkeypatch, [
        _FakeTensor("blk.0.ffn_gate_up_exps.weight", quant_type, gate_up.reshape(E, 2 * I, hb)),
        _FakeTensor("blk.0.ffn_down_exps.weight", quant_type, down.reshape(E, H, ib)),
    ])
    banks = load_gguf_expert_sources(
        "x", num_layers=1, num_experts=E, hidden_size=H, intermediate_size=I,
        gate_up_type=quant_type, down_type=quant_type
    )
    assert torch.equal(banks["gate_up"][0].reshape(-1), gate_up), name
    assert torch.equal(banks["down"][0].reshape(-1), down), name


def test_load_separate_gate_up_packs_gate_then_up(monkeypatch) -> None:
    quant_type = GGML_IQ4_XS
    hb, ib = row_bytes(H, quant_type), row_bytes(I, quant_type)
    gate, up = _bytes(E * I * hb), _bytes(E * I * hb) + 1
    down = _bytes(E * H * ib)
    _patch_tensors(monkeypatch, [
        _FakeTensor("blk.0.ffn_gate_exps.weight", quant_type, gate.reshape(E, I, hb)),
        _FakeTensor("blk.0.ffn_up_exps.weight", quant_type, up.reshape(E, I, hb)),
        _FakeTensor("blk.0.ffn_down_exps.weight", quant_type, down.reshape(E, H, ib)),
    ])
    banks = load_gguf_expert_sources(
        "x", num_layers=1, num_experts=E, hidden_size=H, intermediate_size=I,
        gate_up_type=quant_type, down_type=quant_type
    )
    assert torch.equal(banks["gate_up"][0][:, :I].reshape(-1), gate)
    assert torch.equal(banks["gate_up"][0][:, I:].reshape(-1), up)


def test_load_rejects_a_tensor_of_the_wrong_type(monkeypatch) -> None:
    other = GGML_Q5_K
    _patch_tensors(monkeypatch, [
        _FakeTensor(
            "blk.0.ffn_gate_up_exps.weight", other,
            _bytes(E * 2 * I * row_bytes(H, other)).reshape(E, 2 * I, row_bytes(H, other)),
        ),
        _FakeTensor(
            "blk.0.ffn_down_exps.weight", other,
            _bytes(E * H * row_bytes(I, other)).reshape(E, H, row_bytes(I, other)),
        ),
    ])
    with pytest.raises(ValueError, match="does not match"):
        load_gguf_expert_sources(
            "x", num_layers=1, num_experts=E, hidden_size=H, intermediate_size=I,
            gate_up_type=GGML_IQ4_XS, down_type=GGML_IQ4_XS,
        )


def test_load_rejects_a_missing_layer(monkeypatch) -> None:
    quant_type = GGML_Q5_K
    hb = row_bytes(H, quant_type)
    _patch_tensors(monkeypatch, [
        _FakeTensor(
            "blk.0.ffn_gate_up_exps.weight", quant_type,
            _bytes(E * 2 * I * hb).reshape(E, 2 * I, hb),
        ),
    ])
    with pytest.raises(AssertionError, match="missing GGUF expert layers"):
        load_gguf_expert_sources(
            "x", num_layers=1, num_experts=E, hidden_size=H, intermediate_size=I,
        gate_up_type=quant_type, down_type=quant_type
        )


def test_fused_experts_passes_each_bank_type_to_the_kernel(monkeypatch) -> None:
    from freetoken.kernel import gguf as kernel

    calls: list[int] = []

    def fake(w, x, ids, topk, qt, row, tokens):
        calls.append(qt)
        return torch.zeros(tokens * topk, row, dtype=w.dtype)

    monkeypatch.setattr(kernel, "ggml_moe_a8_vec", fake)
    from freetoken.moe import gguf_experts

    monkeypatch.setattr(gguf_experts, "_ACT", {"silu": lambda t: t})

    hidden = torch.zeros(3, H)
    gate_up_q = torch.zeros(4, 2 * I, row_bytes(H, GGML_Q5_K), dtype=torch.uint8)
    down_q = torch.zeros(4, H, row_bytes(I, GGML_IQ4_XS), dtype=torch.uint8)
    topk_ids = torch.zeros(3, 2, dtype=torch.long)
    out = fused_experts_gguf(
        hidden, gate_up_q, down_q, torch.ones(3, 2), topk_ids, "silu",
        GGML_Q5_K, GGML_IQ4_XS,
    )
    assert calls == [GGML_Q5_K, GGML_IQ4_XS]
    assert out.shape == (3, H)


def test_provider_dummy_uses_the_format_schema(monkeypatch) -> None:
    from freetoken.moe.expert_banks import _gguf_banks

    cfg = SimpleNamespace(num_experts=E, num_layers=1, hidden_size=H, moe_intermediate_size=I)
    for quant_type, tag in _CASES:
        banks = _gguf_banks(
            "x", cfg, torch.device("cpu"), torch.bfloat16, True, quant_format=tag
        )
        assert banks.quant_format == tag
        assert quant_type == GGUF_EXPERT_QUANTS[tag]
        assert set(banks.sources) == {"gate_up", "down"}
        assert banks.sources["gate_up"][0].dtype == torch.uint8


def test_dummy_sources_shapes_match_the_loader(monkeypatch) -> None:
    for quant_type, _ in _CASES:
        banks = dummy_gguf_expert_sources(
            num_layers=1, num_experts=E, hidden_size=H, intermediate_size=I,
            gate_up_type=quant_type, down_type=quant_type
        )
        assert banks["gate_up"][0].shape == (E, 2 * I, row_bytes(H, quant_type))
        assert banks["down"][0].shape == (E, H, row_bytes(I, quant_type))


def test_composite_format_resolves_in_and_getitem() -> None:
    """A per-role tag must satisfy both `in` and `[]` (dict.__contains__ skips __missing__)."""
    from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT, _BANK_SCHEMAS

    assert "iq4_xs+iq4_nl" in _BANK_SCHEMAS
    assert _BANK_SCHEMAS["iq4_xs+iq4_nl"] == ("gate_up", "down")
    assert _BANK_BYTES_PER_EXPERT.get("iq4_xs+iq4_nl") is not None
    assert "bogus+nope" not in _BANK_SCHEMAS


def test_expert_format_detects_a_fused_gate_up_tensor(monkeypatch) -> None:
    from freetoken.models.gguf import reader
    from freetoken.moe.gguf_experts import gguf_expert_format

    fused = _FakeTensor("blk.0.ffn_gate_up_exps.weight", GGML_Q5_K, torch.zeros(0))
    down = _FakeTensor("blk.0.ffn_down_exps.weight", GGML_IQ4_NL, torch.zeros(0))
    monkeypatch.setattr(reader, "iter_gguf_tensors", lambda path: iter([fused, down]))
    assert gguf_expert_format("x") == "q5_K+iq4_nl"


def test_gguf_block_size_and_format_tables_agree() -> None:
    """The GGUF block sizes and expert format tags are restated in several places (two of
    them intentionally, to stay importable in narrow build envs); drift mis-sizes banks."""
    from freetoken.kernel.aot_models import _GGUF_ROLE_BLOCK
    from freetoken.models.gguf.dequant import (
        BLOCK_SHAPE,
        GGML_IQ4_NL,
        GGML_IQ4_XS,
        GGML_Q4_0,
        GGML_Q5_K,
    )
    from freetoken.moe.cpu_executor import _GGUF_W4A8_FORMATS, _WFMT_IDS
    from freetoken.moe.expert_banks import _PROVIDERS
    from freetoken.moe.gguf_experts import GGUF_EXPERT_QUANTS, gguf_expert_role_types
    from freetoken.moe.offload_cache import _GGUF_BANK_TYPES

    expected = {
        "q4_0": BLOCK_SHAPE[GGML_Q4_0],
        "q5_K": BLOCK_SHAPE[GGML_Q5_K],
        "iq4_nl": BLOCK_SHAPE[GGML_IQ4_NL],
        "iq4_xs": BLOCK_SHAPE[GGML_IQ4_XS],
    }
    assert _GGUF_BANK_TYPES == expected
    assert _GGUF_ROLE_BLOCK == expected
    assert set(GGUF_EXPERT_QUANTS) == set(_GGUF_W4A8_FORMATS) == set(_PROVIDERS)
    assert set(GGUF_EXPERT_QUANTS) <= set(_WFMT_IDS)
    for tag, ggml_type in GGUF_EXPERT_QUANTS.items():
        assert gguf_expert_role_types(tag) == (ggml_type, ggml_type)
