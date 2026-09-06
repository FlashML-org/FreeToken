"""GGUF MoE route contracts and native K-quant expert parity gates."""

from __future__ import annotations

import pytest
import torch

from freetoken.models.gguf.dequant import (
    GGML_Q4_K,
    GGML_Q5_K,
    GGML_Q6_K,
    GGML_Q8_0,
    dequantize,
    row_bytes,
)


def _packed_bank(
    experts: int, rows: int, cols: int, quant_type: int, *, device="cpu"
) -> torch.Tensor:
    raw = torch.zeros(
        (experts, rows, row_bytes(cols, quant_type)), dtype=torch.uint8, device=device
    )
    flat = raw.reshape(-1, raw.shape[-1])
    if quant_type == GGML_Q8_0:
        blocks = flat.reshape(-1, 34)
        blocks[:, :2] = torch.tensor([0, 48], dtype=torch.uint8, device=device)
        blocks[:, 2:] = torch.randint(0, 255, blocks[:, 2:].shape, device=device)
    elif quant_type == GGML_Q6_K:
        flat[..., :192] = torch.randint(1, 255, flat[..., :192].shape, device=device)
        flat[..., 192:208] = torch.randint(1, 5, flat[..., 192:208].shape, device=device)
        flat[..., 208:210] = torch.tensor([0, 48], dtype=torch.uint8, device=device)
    else:
        flat[..., :4] = torch.tensor([0, 48, 0, 48], dtype=torch.uint8, device=device)
        flat[..., 4:16] = torch.randint(1, 5, flat[..., 4:16].shape, device=device)
        flat[..., 16:] = torch.randint(0, 255, flat[..., 16:].shape, device=device)
    return raw


def _q8_1_activation_reference(x: torch.Tensor) -> torch.Tensor:
    padded = ((x.shape[1] + 511) // 512) * 512
    padded_x = torch.nn.functional.pad(x.float(), (0, padded - x.shape[1]))
    grouped = padded_x.reshape(x.shape[0], -1, 32)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / 127.0
    scale = scale.to(torch.float16).float()
    return (torch.round(grouped / scale) * scale).reshape_as(padded_x)[:, : x.shape[1]]


def test_gguf_moe_route_reduction_matches_reference_and_reuses_output():
    from freetoken.moe.fused_gguf import _reduce_routes

    routes = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    weights = torch.tensor([[0.2, 0.3, 0.5], [0.5, 0.25, 0.25]])
    output = torch.empty((2, 4))

    reduced = _reduce_routes(routes, output, weights)

    expected = (routes * weights.unsqueeze(-1)).sum(dim=1)
    assert reduced is output
    torch.testing.assert_close(reduced, expected)


def test_gguf_moe_work_declares_id_space_and_reuses_shape_stable_buffers():
    from freetoken.moe.fused_gguf import MoeDecodeWork

    hidden = torch.zeros((2, 8), dtype=torch.bfloat16)
    gate_up = torch.zeros((3, 8, 1), dtype=torch.uint8)
    down = torch.zeros((3, 4, 1), dtype=torch.uint8)
    ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    weights = torch.ones((2, 2), dtype=torch.float32) / 2
    work = MoeDecodeWork("moe_decode")

    work.bind(hidden, gate_up, down, weights, ids, id_space="raw", down_quant_type=GGML_Q5_K)
    first = work.reserve("inter", (4, 4), torch.bfloat16, hidden.device)
    second = work.reserve("inter", (4, 4), torch.bfloat16, hidden.device)

    assert work.id_space == "raw"
    assert work.down_quant_type == GGML_Q5_K
    assert first.data_ptr() == second.data_ptr()
    with pytest.raises(ValueError, match="expert ID space"):
        work.bind(hidden, gate_up, down, weights, ids, id_space="unknown", down_quant_type=GGML_Q5_K)


def test_gguf_moe_matmul_requires_bound_work_id_space():
    from freetoken.moe.fused_gguf import MoeDecodeWork, _gguf_moe_matmul

    work = MoeDecodeWork("moe_decode")
    x = torch.zeros((2, 8))
    packed = torch.zeros((2, 4, 1), dtype=torch.uint8)
    ids = torch.zeros((2, 1), dtype=torch.int32)

    with pytest.raises(ValueError, match="bound before GGUF dispatch"):
        _gguf_moe_matmul(x, packed, ids, GGML_Q5_K, 4, {"implementation": "ggml_moe_a8_vec"}, work=work)


def test_gguf_moe_grouped_dispatch_preserves_explicit_route_contract(monkeypatch):
    import freetoken.kernel.gguf as kernel
    import freetoken.moe.fused as fused
    from freetoken.moe.fused_gguf import _gguf_moe_matmul

    calls = {}

    def align(topk_ids, block_size, num_experts):
        calls["align"] = (block_size, num_experts)
        return (
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
        )

    def grouped(x, weight, sorted_ids, expert_ids, count, quant_type, row, top_k, tokens):
        calls["grouped"] = (quant_type, row, top_k, tokens)
        return torch.zeros((tokens * top_k, row), dtype=x.dtype)

    monkeypatch.setattr(fused, "moe_align_block_size", align)
    monkeypatch.setattr(kernel, "ggml_moe_get_block_size", lambda _quant_type: 32)
    monkeypatch.setattr(kernel, "ggml_moe_a8", grouped)

    x = torch.zeros((2, 8))
    packed = torch.zeros((2, 4, 1), dtype=torch.uint8)
    ids = torch.tensor([[0], [1]], dtype=torch.int32)
    output = _gguf_moe_matmul(
        x, packed, ids, GGML_Q5_K, 4, {"implementation": "ggml_moe_a8"}
    )

    assert output.shape == (2, 4)
    assert calls["align"] == (32, 2)
    assert calls["grouped"] == (GGML_Q5_K, 4, 1, 2)


@pytest.mark.parametrize("down_type", [GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0])
def test_gguf_moe_native_mixed_down_types_keep_quant_route(monkeypatch, down_type):
    import freetoken.kernel.gguf as kernel
    import freetoken.moe.fused_gguf as fused

    dispatches = []

    def dispatch(phase, quant_type, rows, cols, tokens, arch):
        dispatches.append((phase, quant_type, rows, cols, tokens, arch))
        return {"implementation": "test"}

    def matmul(x, _weights, ids, _quant_type, row, _dispatch, output=None, **_kwargs):
        value = torch.arange(ids.shape[0] * ids.shape[1] * row, dtype=x.dtype).reshape(-1, row)
        if output is not None:
            output.copy_(value)
            return output
        return value

    def activation(x, out=None):
        value = x[..., : x.shape[-1] // 2]
        if out is not None:
            out.copy_(value)
            return out
        return value

    monkeypatch.setattr(kernel, "gguf_dispatch", dispatch)
    monkeypatch.setattr(kernel, "gguf_runtime_metadata", lambda: {"arch": "gfx1100"})
    monkeypatch.setattr(fused, "_gguf_moe_matmul", matmul)
    monkeypatch.setitem(fused._ACT, "silu", activation)

    hidden = torch.zeros((2, 8), dtype=torch.float32)
    gate_up = torch.zeros((2, 8, 1), dtype=torch.uint8)
    down = torch.zeros((2, 2, 1), dtype=torch.uint8)
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    workspace = {}

    output = fused.fused_experts_gguf_native(
        hidden, gate_up, down, weights, ids, "silu", down_quant_type=down_type, workspace=workspace
    )

    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    assert [item[1] for item in dispatches] == [GGML_Q4_K, down_type]
    assert {"gate_up", "inter", "down"} <= workspace.keys()


ROCM_DEVICE = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="needs ROCm device",
)


@ROCM_DEVICE
@pytest.mark.parametrize("down_type", [GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0])
def test_gguf_moe_native_k_quant_matches_dequant_reference(down_type):
    from freetoken.moe.fused_gguf import fused_experts_gguf_native

    torch.manual_seed(19 + down_type)
    experts, hidden_size, intermediate_size, top_k = 2, 256, 256, 2
    hidden = torch.randn((1, hidden_size), dtype=torch.bfloat16, device="cuda")
    gate_up = _packed_bank(
        experts, intermediate_size * 2, hidden_size, GGML_Q4_K, device="cuda"
    )
    down = _packed_bank(experts, hidden_size, intermediate_size, down_type, device="cuda")
    ids = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    weights = torch.tensor([[0.4, 0.6]], dtype=torch.float32, device="cuda")

    output = fused_experts_gguf_native(
        hidden, gate_up, down, weights, ids, "silu", down_quant_type=down_type
    ).float()

    gate_dense = dequantize(gate_up.cpu(), GGML_Q4_K, torch.float32).reshape(
        experts, intermediate_size * 2, hidden_size
    ).to("cuda")
    down_dense = dequantize(down.cpu(), down_type, torch.float32).reshape(
        experts, hidden_size, intermediate_size
    ).to("cuda")
    hidden_quant = _q8_1_activation_reference(hidden)
    expected_routes = []
    for expert, weight in zip(ids[0].tolist(), weights[0].tolist()):
        gate_up_value = (hidden_quant @ gate_dense[expert].T).to(torch.bfloat16)
        gate, up = gate_up_value.chunk(2, dim=-1)
        intermediate = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        intermediate = _q8_1_activation_reference(intermediate)
        route = (intermediate @ down_dense[expert].T).to(torch.bfloat16)
        expected_routes.append(
            route * torch.tensor(weight, dtype=torch.bfloat16, device="cuda")
        )
    expected = torch.stack(expected_routes, dim=1).sum(dim=1).float()
    torch.cuda.synchronize()

    assert torch.isfinite(output).all()
    # Native output is BF16; large packed-GEMV values therefore carry up to one
    # BF16 ulp of absolute rounding error even when relative quantization parity holds.
    torch.testing.assert_close(output, expected, rtol=1.2e-1, atol=4096)
