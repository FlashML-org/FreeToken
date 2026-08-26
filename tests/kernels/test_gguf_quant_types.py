"""Coverage tests for the GGUF quant types added for Laguna (Q4_K/Q5_K/IQ*).

Strategy: gguf-py cannot *quantize* K/IQ formats, but it can *dequantize* any
packed bytes. So every test builds a weight from random-but-safe packed bytes
(fp16 scale fields masked small so the kernels' fp16 intermediates cannot
overflow -- real weights are O(1), random fp16 scales are not) and compares the
CUDA kernels against gguf-py's decode of the SAME bytes.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

import gguf

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_IQ1_S,
    GGML_IQ2_S,
    GGML_IQ2_XXS,
    GGML_IQ3_XXS,
    GGML_IQ4_XS,
    GGML_Q3_K,
    GGML_Q4_K,
    GGML_Q5_K,
    dequantize,
)

TYPES = [
    GGML_Q3_K, GGML_Q4_K, GGML_Q5_K,
    GGML_IQ1_S, GGML_IQ2_S, GGML_IQ2_XXS, GGML_IQ3_XXS, GGML_IQ4_XS,
]


def _packed_rows(qtype: int, rows: int, seed: int) -> np.ndarray:
    """Random packed rows with every 16-bit field masked to a small positive
    fp16 (exponent forced below 1.0) so any scale interpretation stays tiny and
    fp16 kernel intermediates cannot overflow. Payload bits keep plenty of
    entropy for the sub-block codes."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (rows, BLOCK_SHAPE[qtype][1]), dtype=np.uint8)
    u16 = raw.view(np.uint16) if raw.shape[1] % 2 == 0 else None
    if u16 is None:  # odd row_bytes (IQ1_S is 50 -> even; guard anyway)
        raw[:, 1::2] &= 0x3B
        return raw
    u16 &= np.uint16(0x3BFF)  # clears sign, caps exponent -> |value| < 1
    return raw


def _reference(raw: np.ndarray, qtype: int) -> torch.Tensor:
    return torch.from_numpy(
        gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(qtype))
    ).float()


def _q8_1_activations(x: torch.Tensor) -> torch.Tensor:
    """Model the kernels' q8_1 activation quantization (int8 per 32-block with an
    fp16 absmax/127 scale) so matmul references carry the same rounding."""
    blocks = x.float().reshape(x.shape[0], -1, 32)
    scale = (blocks.abs().amax(dim=-1, keepdim=True) / 127.0).half().float()
    q = torch.where(scale > 0, (blocks / scale).round().clamp(-127, 127), blocks)
    return (q * scale).reshape(x.shape)


def _randn(shape, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("qtype", TYPES)
def test_python_reference_matches_gguf(qtype):
    """The freetoken reference dequant (used by non-CUDA callers) agrees with
    gguf-py on identical bytes."""
    raw = _packed_rows(qtype, rows=2, seed=qtype)
    ours = dequantize(torch.from_numpy(raw), qtype, torch.float32).reshape(2, -1)
    ref = _reference(raw, qtype).reshape(2, -1)
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("qtype", TYPES)
def test_bytes_cuda_matches_gguf_reference(qtype):
    from freetoken.kernel.gguf import ggml_dequantize

    raw = _packed_rows(qtype, rows=4, seed=qtype)
    packed = torch.from_numpy(raw).cuda()
    got = ggml_dequantize(packed, qtype, 4, BLOCK_SHAPE[qtype][0], torch.float32).cpu()
    ref = _reference(raw, qtype).reshape(4, -1)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("qtype", TYPES)
def test_mmvq_matches_linear(qtype):
    from freetoken.kernel.gguf import ggml_mul_mat_vec_a8

    block = BLOCK_SHAPE[qtype][0]
    rows, cols = 8, 2 * block
    raw = _packed_rows(qtype, rows=rows * (cols // block), seed=qtype + 1)
    raw = np.ascontiguousarray(raw.reshape(rows, -1))
    packed = torch.from_numpy(raw).cuda()
    w = _reference(raw, qtype).reshape(rows, cols).cuda()
    x = _randn((1, cols), seed=qtype + 10)
    got = ggml_mul_mat_vec_a8(packed, x, qtype, rows).float()
    ref = F.linear(_q8_1_activations(x), w)
    tol = 5e-3 * ref.abs().max().clamp(min=1.0)
    assert (got.reshape(-1) - ref.reshape(-1)).abs().max() <= tol


@pytest.mark.parametrize("qtype", [GGML_Q3_K, GGML_Q4_K, GGML_Q5_K])
def test_mmq_matches_linear(qtype):
    from freetoken.kernel.gguf import ggml_mul_mat_a8

    block = BLOCK_SHAPE[qtype][0]
    rows, cols, batch = 8, 2 * block, 8
    raw = _packed_rows(qtype, rows=rows * (cols // block), seed=qtype + 2)
    raw = np.ascontiguousarray(raw.reshape(rows, -1))
    packed = torch.from_numpy(raw).cuda()
    w = _reference(raw, qtype).reshape(rows, cols).cuda()
    x = _randn((batch, cols), seed=qtype + 20)
    got = ggml_mul_mat_a8(packed, x, qtype, rows).float()
    ref = F.linear(_q8_1_activations(x), w)
    tol = 5e-3 * ref.abs().max().clamp(min=1.0)
    assert (got - ref).abs().max() <= tol


@pytest.mark.parametrize("qtype", [GGML_Q4_K, GGML_Q5_K])
def test_dense_gguf_prefill_uses_dequantized_cublas_result(qtype, monkeypatch):
    import freetoken.layers.gguf as layers_gguf
    from freetoken.kernel.gguf import ggml_dequantize
    from freetoken.layers.gguf import fused_mul_mat_gguf

    # On sm_120 Q4_K/Q6_K prefill dispatches to int8-MMA MMQ instead (different
    # rounding; covered by test_gguf_mma). This test validates the dequant path.
    monkeypatch.setattr(layers_gguf, "_use_mma_mmq", lambda qt, cc: False)

    block = BLOCK_SHAPE[qtype][0]
    rows, cols, batch = 32, 2 * block, 32
    raw = _packed_rows(qtype, rows=rows * (cols // block), seed=qtype + 70)
    raw = np.ascontiguousarray(raw.reshape(rows, -1))
    packed = torch.from_numpy(raw).cuda()
    x = _randn((batch, cols), seed=qtype + 71)

    got = fused_mul_mat_gguf(x, packed, qtype)
    dense = ggml_dequantize(packed, qtype, rows, cols, x.dtype)
    expected = x @ dense.T
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "qtype", [GGML_Q3_K, GGML_Q4_K, GGML_IQ2_S, GGML_IQ2_XXS, GGML_IQ3_XXS, GGML_IQ1_S, GGML_IQ4_XS]
)
def test_moe_vec_matches_mmvq(qtype):
    """moe_vec shares mmvq's vec_dot; per selected expert it must reproduce the
    (reference-validated) mmvq result on that expert's rows bit-exactly."""
    from freetoken.kernel.gguf import ggml_moe_a8_vec, ggml_mul_mat_vec_a8

    block = BLOCK_SHAPE[qtype][0]
    experts, rows, cols, top_k = 4, 4, block, 2
    raw = _packed_rows(qtype, rows=experts * rows, seed=qtype + 3)
    bank = np.ascontiguousarray(raw.reshape(experts, rows, -1))
    packed = torch.from_numpy(bank).cuda()
    x = _randn((1, cols), seed=qtype + 30)
    topk_ids = torch.tensor([[1, 3]], device="cuda", dtype=torch.int32)
    out = ggml_moe_a8_vec(x, packed, topk_ids, top_k, qtype, rows, 1)
    out = out.reshape(top_k, rows)
    for j, e in enumerate([1, 3]):
        one = torch.from_numpy(np.ascontiguousarray(bank[e])).cuda()
        ref = ggml_mul_mat_vec_a8(one, x, qtype, rows).reshape(rows)
        torch.testing.assert_close(out[j], ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("qtype", [GGML_IQ1_S, GGML_IQ2_S, GGML_IQ3_XXS])
def test_moe_vec_expert_stride_padded_bank(qtype):
    """Mixed-quant banks store each expert's payload in the leading bytes of a
    padded flat slot; expert_stride_bytes must reproduce the dense-bank result."""
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    block = BLOCK_SHAPE[qtype][0]
    experts, rows, cols, top_k = 4, 4, block, 2
    raw = _packed_rows(qtype, rows=experts * rows, seed=qtype + 40)
    dense = torch.from_numpy(np.ascontiguousarray(raw.reshape(experts, rows, -1))).cuda()
    payload = rows * raw.shape[1] // 1  # bytes per expert (row_bytes * rows)
    payload = rows * dense.shape[2]
    stride = payload + 64  # pad each expert slot
    flat = torch.zeros(experts, stride, dtype=torch.uint8, device="cuda")
    flat[:, :payload] = dense.reshape(experts, payload)
    x = _randn((1, cols), seed=qtype + 41)
    topk_ids = torch.tensor([[1, 3]], device="cuda", dtype=torch.int32)
    ref = ggml_moe_a8_vec(x, dense, topk_ids, top_k, qtype, rows, 1)
    got = ggml_moe_a8_vec(x, flat, topk_ids, top_k, qtype, rows, 1, stride)
    torch.testing.assert_close(got, ref, rtol=0.0, atol=0.0)


def test_moe_mmq_skips_capacity_block_after_live_prefix():
    """The align buffers have spare capacity beyond ``num_tokens_post_padded``.

    A launch block can start exactly at the live-prefix boundary.  It must return
    before reading the intentionally invalid spare entries; otherwise the old
    strict-``>`` guard uses -1 as an activation/output offset and faults.
    """
    from freetoken.kernel.gguf import ggml_moe_a8

    qtype = GGML_Q4_K
    cols, rows = BLOCK_SHAPE[qtype][0], 32
    raw = _packed_rows(qtype, rows=rows, seed=91)
    packed = torch.from_numpy(np.ascontiguousarray(raw.reshape(1, rows, -1))).cuda()
    x = _randn((1, cols), seed=92)
    # MMQ_X_Q4_K == 4.  The first block is the entire live aligned prefix;
    # the second block exists only because the buffers have spare capacity.
    sorted_ids = torch.tensor(
        [0, 1, 1, 1, -1, -1, -1, -1], dtype=torch.int32, device="cuda"
    )
    expert_ids = torch.zeros((2,), dtype=torch.int32, device="cuda")
    num_tokens_post_padded = torch.tensor([4], dtype=torch.int32, device="cuda")

    got = ggml_moe_a8(
        x,
        packed,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        qtype,
        rows,
        1,
        1,
    )
    torch.cuda.synchronize()
    assert got.shape == (1, rows)
    assert torch.isfinite(got).all()
