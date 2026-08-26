"""Arch-aware GGUF kernel dispatch thresholds (sm_120 vs Ada/default).

The threshold *functions* are pure and tested on both archs without a GPU; the
dispatch-site tests stub the CUDA kernels and fake the device capability, so
they only need any CUDA device (branch selection, not numerics).
"""
from __future__ import annotations

import pytest
import torch

import freetoken.layers.gguf as layers_gguf
import freetoken.moe.fused_gguf as moe_gguf
from freetoken.layers.gguf import dequant_gemm_min_rows
from freetoken.models.gguf.dequant import BLOCK_SHAPE, GGML_Q4_K
from freetoken.moe.fused_gguf import mmq_min_tokens


@pytest.mark.parametrize(
    "capability,expected",
    [(None, 32), ((8, 9), 32), ((9, 0), 32), ((12, 0), 24), ((12, 1), 24)],
)
def test_dequant_gemm_min_rows(capability, expected):
    assert dequant_gemm_min_rows(capability) == expected


@pytest.mark.parametrize(
    "capability,expected",
    [(None, 32), ((8, 9), 32), ((9, 0), 32), ((12, 0), 16), ((12, 1), 16)],
)
def test_mmq_min_tokens(capability, expected):
    assert mmq_min_tokens(capability) == expected


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


@pytest.mark.parametrize(
    "capability,rows,expected_branch",
    [
        ((12, 0), 24, "dequant"),
        ((12, 0), 23, "mmq"),
        ((8, 9), 24, "mmq"),
        ((8, 9), 32, "dequant"),
    ],
)
def test_dense_dispatch_branch(cuda, monkeypatch, capability, rows, expected_branch):
    import freetoken.kernel.gguf as kernel_gguf

    monkeypatch.setattr(layers_gguf, "_device_capability", lambda i: capability)
    # The MMA path (tested separately) sits above the dequant/DP4A crossover.
    monkeypatch.setattr(layers_gguf, "_use_mma_mmq", lambda qt, cc: False)
    block, type_size = BLOCK_SHAPE[GGML_Q4_K]
    in_features, out_features = 256, 8
    qweight = torch.zeros(
        (out_features, in_features // block * type_size), dtype=torch.uint8, device="cuda"
    )
    x = torch.zeros((rows, in_features), dtype=torch.float16, device="cuda")
    called = []
    monkeypatch.setattr(
        kernel_gguf,
        "ggml_dequantize",
        lambda *a, **k: called.append("dequant")
        or torch.zeros((out_features, in_features), dtype=x.dtype, device="cuda"),
    )
    monkeypatch.setattr(
        kernel_gguf,
        "ggml_mul_mat_a8",
        lambda *a, **k: called.append("mmq")
        or torch.zeros((rows, out_features), dtype=x.dtype, device="cuda"),
    )
    out = layers_gguf.fused_mul_mat_gguf(x, qweight, GGML_Q4_K)
    assert called == [expected_branch]
    assert out.shape == (rows, out_features)


@pytest.mark.parametrize(
    "capability,expect_mma",
    [((12, 0), True), ((12, 1), True), ((8, 9), False), ((9, 0), False)],
)
def test_dense_dispatch_mma_branch(cuda, monkeypatch, capability, expect_mma):
    """On sm_120, Q4_K rows above the MMVQ band route to int8-MMA MMQ."""
    import freetoken.kernel.gguf as kernel_gguf

    monkeypatch.setattr(layers_gguf, "_device_capability", lambda i: capability)
    monkeypatch.setattr(layers_gguf, "_mma_mmq_ok", lambda: True)
    block, type_size = BLOCK_SHAPE[GGML_Q4_K]
    in_features, out_features, rows = 256, 8, 64
    qweight = torch.zeros(
        (out_features, in_features // block * type_size), dtype=torch.uint8, device="cuda"
    )
    x = torch.zeros((rows, in_features), dtype=torch.float16, device="cuda")
    called = []
    monkeypatch.setattr(
        kernel_gguf,
        "ggml_mul_mat_a8_mma",
        lambda *a, **k: called.append("mma")
        or torch.zeros((rows, out_features), dtype=torch.float32, device="cuda"),
    )
    monkeypatch.setattr(
        kernel_gguf,
        "ggml_dequantize",
        lambda *a, **k: called.append("dequant")
        or torch.zeros((out_features, in_features), dtype=x.dtype, device="cuda"),
    )
    out = layers_gguf.fused_mul_mat_gguf(x, qweight, GGML_Q4_K)
    assert called == (["mma"] if expect_mma else ["dequant"])
    assert out.dtype == x.dtype


@pytest.mark.parametrize(
    "capability,tokens,expect_mmq",
    [
        ((12, 0), 16, True),
        ((12, 0), 15, False),
        ((8, 9), 16, False),
        ((8, 9), 32, True),
    ],
)
def test_moe_dispatch_branch(cuda, monkeypatch, capability, tokens, expect_mmq):
    import freetoken.kernel.gguf as kernel_gguf

    monkeypatch.setattr(layers_gguf, "_device_capability", lambda i: capability)
    monkeypatch.setattr(moe_gguf, "_use_mma_moe", lambda *a: False)
    considered = []
    # Returning 0 makes the MMQ branch fall through to the (stubbed) vec path,
    # so the sentinel records threshold crossing without running any kernel.
    monkeypatch.setattr(
        kernel_gguf,
        "ggml_moe_get_block_size",
        lambda qt: considered.append("mmq") or 0,
    )
    sentinel = torch.zeros((tokens, 4))
    monkeypatch.setattr(moe_gguf, "_moe_vec_chunked", lambda *a, **k: sentinel)
    x = torch.zeros((tokens, 8), dtype=torch.float16, device="cuda")
    topk_ids = torch.zeros((tokens, 2), dtype=torch.int32, device="cuda")
    weight = torch.zeros((4, 16), dtype=torch.uint8, device="cuda")
    out = moe_gguf._moe_matmul(
        x, weight, topk_ids, 2, GGML_Q4_K, rows=4, tokens=tokens, stride=16
    )
    assert (considered == ["mmq"]) is expect_mmq
    assert out is sentinel


def test_moe_use_mma_gate(monkeypatch):
    """_use_mma_moe: sm_120 + supported type + block-aligned stride only."""
    monkeypatch.setattr(layers_gguf, "_mma_mmq_ok", lambda: True)
    block, type_size = BLOCK_SHAPE[GGML_Q4_K]
    x = torch.zeros(4, 8)
    assert moe_gguf._use_mma_moe(GGML_Q4_K, 4 * type_size, x, (12, 0))
    assert not moe_gguf._use_mma_moe(GGML_Q4_K, 4 * type_size, x, (8, 9))
    assert not moe_gguf._use_mma_moe(GGML_Q4_K, 4 * type_size + 1, x, (12, 0))
    assert not moe_gguf._use_mma_moe(2, 64, x, (12, 0))  # Q4_0: not instantiated


@pytest.mark.parametrize(
    "tokens,broadcast,expect_mma",
    [(320, True, True), (320, False, True), (319, True, False), (16385, True, False)],
)
def test_moe_dispatch_mma_branch(cuda, monkeypatch, tokens, broadcast, expect_mma):
    import freetoken.kernel.gguf as kernel_gguf

    monkeypatch.setattr(layers_gguf, "_device_capability", lambda i: (12, 0))
    monkeypatch.setattr(moe_gguf, "_use_mma_moe", lambda *a: True)
    top_k = 2
    called = []
    mma_out = torch.zeros(tokens * top_k, 4)
    monkeypatch.setattr(
        kernel_gguf, "ggml_moe_a8_mma", lambda *a, **k: called.append("mma") or mma_out
    )
    vec_out = torch.zeros(tokens * top_k, 4)
    monkeypatch.setattr(moe_gguf, "_moe_vec_chunked", lambda *a, **k: vec_out)
    import freetoken.kernel.gguf as kg

    monkeypatch.setattr(kg, "ggml_moe_get_block_size", lambda qt: 0)
    rows_x = tokens if broadcast else tokens * top_k
    x = torch.zeros((rows_x, 8), dtype=torch.float16, device="cuda")
    topk_ids = torch.zeros((tokens, top_k), dtype=torch.int32, device="cuda")
    weight = torch.zeros((4, 16), dtype=torch.uint8, device="cuda")
    out = moe_gguf._moe_matmul(
        x, weight, topk_ids, top_k, GGML_Q4_K, rows=4, tokens=tokens, stride=16,
        broadcast=broadcast,
    )
    assert (called == ["mma"]) is expect_mma
    if expect_mma:
        assert out.dtype == x.dtype  # fp32 kernel output cast back
    else:
        assert out is vec_out
