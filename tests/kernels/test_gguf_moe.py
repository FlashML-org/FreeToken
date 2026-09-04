"""Reference and finite-output gate for the gfx1100 GGUF MoE candidate."""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.gguf import (
    ggml_dequantize,
    ggml_moe_a8_vec,
    ggml_moe_a8_vec_strided,
    ggml_moe_gate_up_swiglu_id,
    ggml_moe_mmvdq_id,
    ggml_moe_mmvq_id,
)
from freetoken.models.gguf.dequant import GGML_Q5_K, GGML_Q6_K
from freetoken.moe.fused_gguf import MoeDecodeWork


def test_weighted_route_reduce_is_fixed_order_on_cpu():
    from freetoken.moe.fused_gguf import _reduce_routes

    routes = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
    output = _reduce_routes(routes, torch.empty(2, 3), weights)
    torch.testing.assert_close(output, (routes * weights[..., None]).sum(dim=1))


ROCM_GFX1100 = pytest.mark.skipif(
    torch.version.hip is None
    or not torch.cuda.is_available()
    or getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") != "gfx1100",
    reason="needs ROCm gfx1100",
)


def test_moe_decode_work_has_explicit_id_space_and_reuses_buffers():
    work = MoeDecodeWork("moe_decode")
    hidden = torch.zeros((1, 256), dtype=torch.bfloat16)
    gate_up = torch.empty((4, 16, 144), dtype=torch.uint8)
    down = torch.empty((4, 8, 210), dtype=torch.uint8)
    ids = torch.tensor([[0, 3]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)

    work.bind(hidden, gate_up, down, weights, ids, id_space="slot", down_quant_type=GGML_Q6_K)
    first = work.reserve("output", (1, 8), torch.bfloat16, torch.device("cpu"))
    second = work.reserve("output", (1, 8), torch.bfloat16, torch.device("cpu"))

    assert work.id_space == "slot"
    assert work.gate_expert_stride_bytes == gate_up.stride(0)
    assert work.down_row_stride_bytes == down.stride(1)
    assert first.data_ptr() == second.data_ptr()


def test_moe_decode_work_rejects_raw_route_dtype_mismatch():
    work = MoeDecodeWork("moe_decode")
    tensors = (
        torch.zeros((1, 256), dtype=torch.bfloat16),
        torch.empty((4, 16, 144), dtype=torch.uint8),
        torch.empty((4, 8, 210), dtype=torch.uint8),
        torch.ones((1, 2), dtype=torch.float32),
        torch.ones((1, 2), dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="IDs must be int32"):
        work.bind(*tensors, id_space="raw", down_quant_type=GGML_Q6_K)


def test_native_mixed_down_types_preserve_dispatch_and_routes(monkeypatch):
    import freetoken.moe.fused_gguf as fused

    calls = []

    def fake_dispatch(phase, quant_type, rows, cols, tokens, arch):
        calls.append((phase, quant_type, rows, cols, tokens, arch))
        return {"implementation": "ggml_moe_a8_vec"}

    def fake_matmul(x, weights, ids, quant_type, row, dispatch, output=None):
        calls.append(("matmul", quant_type, tuple(x.shape), tuple(ids.shape), row, output))
        if output is not None:
            output.zero_()
            return output
        return torch.zeros((ids.shape[0] * ids.shape[1], row), dtype=x.dtype)

    monkeypatch.setattr(fused, "_gguf_moe_matmul", fake_matmul)
    def fake_act(x, out=None):
        value = x[..., : x.shape[-1] // 2]
        if out is not None:
            out.copy_(value)
            return out
        return value

    monkeypatch.setattr(fused, "_ACT", {"test": fake_act})
    monkeypatch.setattr("freetoken.kernel.gguf.gguf_runtime_metadata", lambda: {"arch": "gfx1100"})
    monkeypatch.setattr("freetoken.kernel.gguf.gguf_dispatch", fake_dispatch)

    hidden = torch.ones((2, 4), dtype=torch.bfloat16)
    gate_up = torch.empty((3, 8, 1), dtype=torch.uint8)
    topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)
    for down_type in (GGML_Q5_K, GGML_Q6_K):
        calls.clear()
        workspace = {}
        out = fused.fused_experts_gguf_native(
            hidden, gate_up, torch.empty((3, 4, 1), dtype=torch.uint8),
            topk_weights, topk_ids, "test", down_quant_type=down_type,
            workspace=workspace,
        )
        assert out.shape == (2, 4)
        assert torch.isfinite(out).all()
        assert topk_ids.tolist() == [[2, 0], [1, 2]]
        dispatch_types = [row[1] for row in calls if row[0] == "moe_decode"]
        assert dispatch_types == [12, down_type]
        assert calls[-1][-1] is workspace["down"]


def test_grouped_helper_preserves_aligned_route_contract(monkeypatch):
    import freetoken.moe.fused_gguf as fused

    calls = {}

    def fake_align(ids, block_size, experts):
        calls["align"] = (ids.clone(), block_size, experts)
        return (torch.tensor([0, 1, 4, 4], dtype=torch.int32),
                torch.tensor([1, 2], dtype=torch.int32),
                torch.tensor([4], dtype=torch.int32))

    def fake_vec(*args):
        calls["vec"] = True
        return torch.empty((2, 4))

    def fake_grouped(x, weights, sorted_ids, expert_ids, padded, quant_type, row, top_k, tokens):
        calls["grouped"] = (tuple(x.shape), tuple(weights.shape), tuple(sorted_ids.shape),
                             tuple(expert_ids.shape), int(padded.item()), quant_type, row,
                             top_k, tokens)
        return torch.zeros((tokens * top_k, row), dtype=x.dtype)

    monkeypatch.setattr("freetoken.moe.fused.moe_align_block_size", fake_align)
    monkeypatch.setattr("freetoken.kernel.gguf.ggml_moe_get_block_size", lambda _qt: 32)
    monkeypatch.setattr("freetoken.kernel.gguf.ggml_moe_a8_vec", fake_vec)
    monkeypatch.setattr("freetoken.kernel.gguf.ggml_moe_a8", fake_grouped)
    out = fused._gguf_moe_matmul(
        torch.zeros((2, 4)), torch.zeros((3, 4, 1), dtype=torch.uint8),
        torch.tensor([[1, 2], [0, 1]], dtype=torch.int32), 12, 4,
        {"implementation": "ggml_moe_a8"},
    )
    assert out.shape == (4, 4)
    assert calls["align"][0].tolist() == [[1, 2], [0, 1]]
    assert calls["grouped"][-3:] == (4, 2, 2)


@ROCM_GFX1100
@pytest.mark.slow
def test_native_kquant_stride_matches_compact_rows():
    """Q5_K rows padded to the Q6_K cache stride retain native GEMV output."""
    device = torch.device("cuda")
    for quant_type, row_bytes in ((GGML_Q5_K, 176), (GGML_Q6_K, 210)):
        x = torch.randn((2, 256), dtype=torch.bfloat16, device=device)
        compact = torch.zeros((3, 2, row_bytes), dtype=torch.uint8, device=device)
        if quant_type == GGML_Q5_K:
            compact[..., :4] = torch.tensor([128, 63, 128, 63], dtype=torch.uint8, device=device)
            compact[..., 4:16] = torch.randint(1, 64, (3, 2, 12), dtype=torch.uint8, device=device)
            compact[..., 16:] = torch.randint(0, 255, (3, 2, row_bytes - 16), dtype=torch.uint8, device=device)
        else:
            compact[..., :208] = torch.randint(0, 255, (3, 2, 208), dtype=torch.uint8, device=device)
            compact[..., 208:] = torch.tensor([128, 63], dtype=torch.uint8, device=device)
        padded = torch.zeros((3, 2, 210), dtype=torch.uint8, device=device)
        padded[..., :row_bytes].copy_(compact)
        ids = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32, device=device)
        expected = ggml_moe_a8_vec(x, compact, ids, 2, quant_type, 2, 2)
        actual = ggml_moe_a8_vec_strided(
            x, padded, ids, 2, quant_type, 2, 2,
            int(padded.stride(0)), int(padded.stride(1)),
        )
        torch.cuda.synchronize(device)
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


@ROCM_GFX1100
@pytest.mark.slow
def test_gfx1100_moe_matches_legacy(monkeypatch):
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(2026)
    ids = torch.arange(8, dtype=torch.int32, device=device).reshape(1, 8)

    x_gate = torch.randn(1, 256, generator=generator, dtype=torch.bfloat16, device="cpu").to(device)
    q4 = torch.zeros((8, 16, 144), dtype=torch.uint8, device=device)
    q4[..., :4] = torch.tensor([128, 63, 128, 63], dtype=torch.uint8, device=device)
    q4[..., 4:16] = torch.randint(1, 64, (8, 16, 12), generator=generator, dtype=torch.uint8).to(device)
    q4[..., 16:] = torch.randint(0, 255, (8, 16, 128), generator=generator, dtype=torch.uint8).to(device)
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_IMPL", "legacy")
    legacy_gate = ggml_moe_a8_vec(x_gate, q4, ids, 8, 12, 16, 1)
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_IMPL", "gfx1100")
    candidate_gate = ggml_moe_a8_vec(x_gate, q4, ids, 8, 12, 16, 1)

    x_down = torch.randn(8, 128, generator=generator, dtype=torch.bfloat16, device="cpu").to(device)
    q8_blocks = torch.zeros((8, 16, 4, 34), dtype=torch.uint8, device=device)
    q8_blocks[..., :2] = torch.tensor([128, 63], dtype=torch.uint8, device=device)
    q8_blocks[..., 2:] = torch.randint(
        0, 255, (8, 16, 4, 32), generator=generator, dtype=torch.uint8
    ).to(device)
    q8 = q8_blocks.reshape(8, 16, 136)
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_IMPL", "legacy")
    legacy_down = ggml_moe_a8_vec(x_down, q8, ids, 1, 8, 16, 8)
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_IMPL", "gfx1100")
    candidate_down = ggml_moe_a8_vec(x_down, q8, ids, 1, 8, 16, 8)
    torch.cuda.synchronize(device)

    assert torch.isfinite(candidate_gate).all()
    assert torch.isfinite(candidate_down).all()
    torch.testing.assert_close(candidate_gate, legacy_gate, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(candidate_down, legacy_down, rtol=5e-2, atol=5e-2)


@ROCM_GFX1100
@pytest.mark.slow
def test_rdna3_mmvdq_and_fused_gate_up_match_dequant_reference(monkeypatch):
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(2027)
    hidden, intermediate, top_k, experts = 256, 256, 2, 8
    ids = torch.tensor([[1, 7]], dtype=torch.int32, device=device)
    x = torch.randn((1, hidden), generator=generator, dtype=torch.float32, device="cpu").to(device)
    gate = torch.randint(
        0, 255, (experts, 2 * intermediate, 144), generator=generator, dtype=torch.uint8
    ).to(device)
    gate[..., :4] = torch.tensor([128, 63, 128, 63], dtype=torch.uint8, device=device)
    gate[..., 4:16] = torch.randint(
        1, 4, gate[..., 4:16].shape, generator=generator, dtype=torch.uint8
    ).to(device)
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_IMPL", "rdna3_mmvdq")
    mmvdq = ggml_moe_mmvdq_id(
        x, gate, ids, top_k, 12, 2 * intermediate, 1,
        int(gate.stride(0)), int(gate.stride(1)), "raw",
    )
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_IMPL", "rdna3_mmid")
    mmvq = ggml_moe_mmvq_id(
        x, gate, ids, top_k, 12, 2 * intermediate, 1,
        int(gate.stride(0)), int(gate.stride(1)), "raw",
    )
    fused = ggml_moe_gate_up_swiglu_id(
        x, gate, ids, top_k, intermediate, 1,
        int(gate.stride(0)), int(gate.stride(1)), "raw",
    )
    reference = torch.cat(
        [x @ ggml_dequantize(gate[expert], 12, 2 * intermediate, hidden, torch.float32).t()
         for expert in ids[0].tolist()],
        dim=0,
    )
    mmvq_fused = (
        mmvq[:, :intermediate] /
        (1.0 + torch.exp(-mmvq[:, :intermediate])) * mmvq[:, intermediate:]
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(mmvdq, reference, rtol=3e-4, atol=3e-3)
    torch.testing.assert_close(fused, mmvq_fused, rtol=3e-4, atol=3e-3)

    q6 = torch.randint(0, 255, (experts, hidden, 210), generator=generator, dtype=torch.uint8).to(device)
    q6[..., -2:] = torch.tensor([128, 63], dtype=torch.uint8, device=device)
    down_x = torch.randn((top_k, intermediate), generator=generator, dtype=torch.float32, device="cpu").to(device)
    route_ids = ids.reshape(-1, 1)
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_IMPL", "rdna3_mmvdq")
    down = ggml_moe_mmvdq_id(
        down_x, q6, route_ids, 1, GGML_Q6_K, hidden, top_k,
        int(q6.stride(0)), int(q6.stride(1)), "raw",
    )
    down_reference = torch.cat(
        [down_x[i:i + 1] @ ggml_dequantize(q6[expert], GGML_Q6_K, hidden, intermediate, torch.float32).t()
         for i, expert in enumerate(ids[0].tolist())],
        dim=0,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(down, down_reference, rtol=3e-4, atol=3e-3)
