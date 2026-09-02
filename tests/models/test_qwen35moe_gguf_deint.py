"""qwen35moe GGUF GDN value-head de-interleaving.

llama.cpp stores the GDN *value* projections with the ``mrope_interleaved`` head order
(even heads 0..nv/2-1 first, then odd heads nv/2..nv-1). FreeToken uses the HF contiguous
head order, so the loader de-interleaves the value projections on load. These tests pin the
permutation and the packed/dense de-interleave helpers (no model weights required).
"""

import torch
import pytest

from freetoken.models.qwen3_5_moe.gguf import (
    _gdn_head_perm,
    _deint_dense_rows,
    _deint_q8_cols,
    _deint_q8_rows,
    _gguf_down_quant_types,
)
from freetoken.models.gguf.dequant import (
    GGML_Q5_K,
    GGML_Q6_K,
    dequantize,
    quantize_q8_0,
)


def _interleave(x: torch.Tensor, nv: int, rows_per_head: int = 1) -> torch.Tensor:
    """Build the GGUF head-interleaved layout from an HF-contiguous tensor.

    GGUF position ``perm[h]`` holds HF head ``h`` (so interleaved[perm[h]] = x[h]).
    """
    m = x.reshape(nv, rows_per_head, -1)
    perm = _gdn_head_perm(nv)
    out = m.clone()
    for h in range(nv):
        out[perm[h]] = m[h]
    return out.reshape(x.shape)


def test_head_permutation_is_a_bijection():
    nv = 32
    perm = _gdn_head_perm(nv)
    assert sorted(perm) == list(range(nv))  # valid permutation
    # GGUF layout: even HF heads occupy positions 0..15, odd heads 16..31.
    even = sorted(perm[h] for h in range(0, nv, 2))
    odd = sorted(perm[h] for h in range(1, nv, 2))
    assert even == list(range(nv // 2))
    assert odd == list(range(nv // 2, nv))


def test_deint_dense_rows_recovers_contiguous():
    nv, hd = 32, 128
    x = torch.randn(nv, hd)
    inter = _interleave(x, nv, rows_per_head=1)
    rec = _deint_dense_rows(inter, nv)
    assert torch.allclose(rec, x, atol=1e-6)


def test_deint_q8_rows_recovers_contiguous():
    nv, hd = 32, 128
    rph = 64  # rows per value head in the projection output dim
    x = torch.randn(nv * rph, 4)  # packed rows x (row_bytes mocked)
    inter = _interleave(x, nv, rows_per_head=rph)
    rec = _deint_q8_rows(inter, nv, rph)
    assert torch.allclose(rec, x, atol=1e-6)


def test_deint_q8_cols_recovers_contiguous():
    nv = 32
    rows, blocks_per_head, bb = 16, 4, 34
    x = torch.randn(rows, nv * blocks_per_head * bb)
    # interleave column head-groups: inter[:, perm[h]*bbp:(perm[h]+1)*bbp] = x[:, h*bbp:(h+1)*bbp]
    bbp = blocks_per_head * bb
    inter = torch.empty_like(x)
    perm = _gdn_head_perm(nv)
    for h in range(nv):
        inter[:, perm[h] * bbp:(perm[h] + 1) * bbp] = x[:, h * bbp:(h + 1) * bbp]
    rec = _deint_q8_cols(inter, nv, blocks_per_head)
    assert torch.allclose(rec, x, atol=1e-6)


def _dequant_q8_0_reference(raw: torch.Tensor) -> torch.Tensor:
    """Independent Q8_0 unpack used to check the cache's re-quantized bytes."""
    blocks = raw.reshape(-1, 34)
    scales = blocks[:, :2].contiguous().view(torch.float16).float()
    values = blocks[:, 2:].contiguous().view(torch.int8).float()
    return (values * scales).reshape(raw.shape[0], -1)


@pytest.mark.parametrize(
    ("ggml_type", "row_bytes"),
    [(GGML_Q5_K, 176), (GGML_Q6_K, 210)],
)
def test_k_quant_source_requantizes_to_q8_cache_within_tolerance(ggml_type, row_bytes):
    """FreeToken's Q8_0 expert cache preserves Q5_K/Q6_K source values closely."""
    generator = torch.Generator().manual_seed(100 + ggml_type)
    packed = torch.randint(0, 256, (4, row_bytes), dtype=torch.uint8, generator=generator)
    scales = torch.tensor([0.0, -0.5, 0.75, 1.25], dtype=torch.float16)
    if ggml_type == GGML_Q5_K:
        # Q5_K stores (dall, dmin) in its first two fp16 values.
        packed[:, :4] = torch.stack((scales, scales.abs() / 2), dim=1).view(torch.uint8)
    else:
        packed[:, 208:210] = scales.view(torch.uint8).reshape(4, 2)

    source = dequantize(packed, ggml_type, torch.float32).view(4, 256)
    cache_bytes = quantize_q8_0(source)
    cached = _dequant_q8_0_reference(cache_bytes)
    assert torch.isfinite(source).all()
    assert torch.isfinite(cached).all()
    assert torch.equal(cached[0], torch.zeros_like(cached[0]))
    relative_rmse = (cached - source).square().mean().sqrt() / source.square().mean().sqrt()
    assert relative_rmse < 0.01


def test_down_quant_type_reader_preserves_mixed_layer_types(monkeypatch):
    from freetoken.models.gguf import reader

    class Tensor:
        def __init__(self, name, quant_type):
            self.name = name
            self.tensor_type = quant_type

    class FakeReader:
        tensors = [
            Tensor("blk.0.ffn_down_exps.weight", GGML_Q5_K),
            Tensor("blk.1.ffn_down_exps.weight", GGML_Q6_K),
            Tensor("blk.2.ffn_down_exps.weight", GGML_Q5_K),
        ]

    monkeypatch.setattr(reader, "_reader", lambda _path: FakeReader())
    assert _gguf_down_quant_types("synthetic.gguf") == (
        GGML_Q5_K, GGML_Q6_K, GGML_Q5_K
    )
