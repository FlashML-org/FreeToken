"""Numeric checks for the upstream int8-MMA MMQ extension (Q4_K/Q6_K).

Same strategy as test_gguf_quant_types: random-but-safe packed bytes (fp16
scale fields masked small), gguf-py's decode of the same bytes as reference.
Runs only where the dispatch would actually select the MMA path (sm_120+).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if torch.cuda.get_device_capability() < (12, 0):
    pytest.skip("int8-MMA MMQ path is sm_120-gated", allow_module_level=True)

import gguf

from freetoken.models.gguf.dequant import BLOCK_SHAPE, GGML_Q4_K, GGML_Q6_K


def _packed_rows(qtype: int, rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (rows, BLOCK_SHAPE[qtype][1]), dtype=np.uint8)
    u16 = raw.view(np.uint16)
    u16 &= np.uint16(0x3BFF)  # clears sign, caps exponent -> |value| < 1
    return raw


@pytest.mark.parametrize("qtype", [GGML_Q4_K, GGML_Q6_K])
@pytest.mark.parametrize("tokens", [7, 16, 129])
def test_mma_matches_reference(qtype, tokens):
    from freetoken.kernel.gguf import ggml_mul_mat_a8_mma

    out_features = 320  # not a multiple of 128 -> exercises the fallback kernel
    block, type_size = BLOCK_SHAPE[qtype]
    in_features = 2 * block
    raw = _packed_rows(qtype, out_features * in_features // block, seed=qtype * 100 + tokens)
    weight = torch.from_numpy(raw.reshape(out_features, -1).copy()).cuda()
    ref_w = torch.from_numpy(
        gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(qtype))
    ).float().reshape(out_features, in_features).cuda()

    torch.manual_seed(0)
    x = torch.randn(tokens, in_features, dtype=torch.float32, device="cuda")
    got = ggml_mul_mat_a8_mma(weight, x, qtype, out_features)
    ref = x @ ref_w.T
    rel = ((got - ref).norm() / (ref.norm() + 1e-12)).item()
    assert got.shape == (tokens, out_features)
    assert rel < 0.02, rel


@pytest.mark.parametrize("qtype", [GGML_Q4_K, GGML_Q6_K])
@pytest.mark.parametrize("broadcast", [True, False])
def test_moe_mma_matches_reference(qtype, broadcast):
    from freetoken.kernel.gguf import ggml_moe_a8_mma

    experts, out_f, tokens, top_k = 16, 320, 9, 4
    block, type_size = BLOCK_SHAPE[qtype]
    in_f = 2 * block
    pad = 2 * type_size  # padded slots, still block-aligned
    payload = out_f * (in_f // block) * type_size
    stride = payload + pad

    rng = np.random.default_rng(3)
    bank = torch.zeros(experts, stride, dtype=torch.uint8)
    ws = []
    for e in range(experts):
        raw = _packed_rows(qtype, out_f * in_f // block, seed=e)
        bank[e, :payload] = torch.from_numpy(raw.reshape(-1))
        ws.append(
            torch.from_numpy(gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(qtype)))
            .float().reshape(out_f, in_f)
        )
    bank = bank.cuda()
    w = torch.stack(ws).cuda()
    topk_ids = torch.from_numpy(
        np.stack([rng.permutation(experts)[:top_k] for _ in range(tokens)])
    ).int().cuda()

    torch.manual_seed(2)
    rows_x = tokens if broadcast else tokens * top_k
    x = torch.randn(rows_x, in_f, dtype=torch.float32, device="cuda")
    if broadcast:
        ref = torch.stack(
            [x[t] @ w[topk_ids[t, k]].T for t in range(tokens) for k in range(top_k)]
        )
    else:
        ref = torch.stack(
            [x[t * top_k + k] @ w[topk_ids[t, k]].T for t in range(tokens) for k in range(top_k)]
        )
    got = ggml_moe_a8_mma(x, bank, topk_ids, top_k, qtype, out_f, tokens, stride, broadcast)
    rel = ((got - ref).norm() / (ref.norm() + 1e-12)).item()
    assert got.shape == (tokens * top_k, out_f)
    assert rel < 0.02, rel


def test_mma_multiple_of_128_rows():
    from freetoken.kernel.gguf import ggml_mul_mat_a8_mma

    qtype = GGML_Q4_K
    out_features = 256  # multiple of 128 -> non-fallback kernel
    block, type_size = BLOCK_SHAPE[qtype]
    in_features = 2 * block
    raw = _packed_rows(qtype, out_features * in_features // block, seed=7)
    weight = torch.from_numpy(raw.reshape(out_features, -1).copy()).cuda()
    ref_w = torch.from_numpy(
        gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(qtype))
    ).float().reshape(out_features, in_features).cuda()

    torch.manual_seed(1)
    x = torch.randn(33, in_features, dtype=torch.float32, device="cuda")
    got = ggml_mul_mat_a8_mma(weight, x, qtype, out_features)
    ref = x @ ref_w.T
    rel = ((got - ref).norm() / (ref.norm() + 1e-12)).item()
    assert rel < 0.02, rel
