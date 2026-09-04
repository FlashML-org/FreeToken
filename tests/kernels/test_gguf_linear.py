"""Independent finite/reference gate for native GGUF dense GEMV."""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.gguf import ggml_mul_mat_vec_a8
from freetoken.layers.gguf import fused_mul_mat_gguf
from freetoken.models.gguf.dequant import (
    GGML_Q4_K,
    GGML_Q5_K,
    GGML_Q6_K,
    GGML_Q8_0,
    dequantize,
    dequant_q6_k,
    row_bytes,
)


@pytest.mark.parametrize("quant_type", [GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0])
def test_native_k_quant_reference_shape(quant_type):
    raw = torch.zeros(row_bytes(256, quant_type), dtype=torch.uint8)
    decoded = dequantize(raw, quant_type, torch.float32)
    assert decoded.shape == (256,)
    assert torch.equal(decoded, torch.zeros(256))


@pytest.mark.parametrize("operation", ["dense", "lm_head"])
def test_dense_policy_covers_lm_head_operation(monkeypatch, operation):
    import freetoken.kernel.gguf as gguf

    monkeypatch.setattr(gguf, "gguf_runtime_metadata", lambda: {"arch": "gfx1100"})
    monkeypatch.setattr(gguf, "_runtime_backend", lambda: "rocm")
    seen = {}

    def dispatch(op, quant_type, rows, cols, tokens, arch):
        seen.update(op=op, quant_type=quant_type, rows=rows, cols=cols, tokens=tokens, arch=arch)
        return {"implementation": "unsupported"}

    monkeypatch.setattr(gguf, "gguf_dispatch", dispatch)
    monkeypatch.setattr(gguf, "ggml_dequantize", lambda *args: torch.zeros((args[2], args[3])))
    x = torch.zeros((1, 256), dtype=torch.float32)
    qweight = torch.zeros((7, 210), dtype=torch.uint8)
    out = fused_mul_mat_gguf(x, qweight, GGML_Q6_K, operation)
    assert out.shape == (1, 7)
    assert seen == {
        "op": operation, "quant_type": GGML_Q6_K, "rows": 7, "cols": 256,
        "tokens": 1, "arch": "gfx1100",
    }


ROCM_DEVICE = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="needs CUDA/ROCm device",
)


def _q8_weight(rows: int, cols: int) -> torch.Tensor:
    blocks = torch.zeros((rows, cols // 32, 34), dtype=torch.uint8, device="cuda")
    blocks[..., :2] = torch.tensor([128, 63], dtype=torch.uint8, device="cuda")
    blocks[..., 2:] = torch.randint(0, 255, blocks[..., 2:].shape, dtype=torch.uint8, device="cuda")
    return blocks.reshape(rows, row_bytes(cols, GGML_Q8_0))


def _q6_weight(rows: int, cols: int) -> torch.Tensor:
    blocks = torch.zeros((rows, cols // 256, 210), dtype=torch.uint8, device="cuda")
    blocks[..., 192:208] = torch.randint(
        1, 255, blocks[..., 192:208].shape, dtype=torch.uint8, device="cuda"
    )
    blocks[..., 208:210] = torch.tensor([0, 60], dtype=torch.uint8, device="cuda")
    return blocks.reshape(rows, row_bytes(cols, GGML_Q6_K))


def _q8_reference(weight: torch.Tensor) -> torch.Tensor:
    blocks = weight.reshape(weight.shape[0], -1, 34)
    scale = blocks[..., :2].contiguous().view(torch.float16).float()
    quant = blocks[..., 2:].contiguous().view(torch.int8).float()
    return (scale * quant).reshape(weight.shape[0], -1)


def _q8_1_activation_reference(x: torch.Tensor) -> torch.Tensor:
    """Mirror gguf_kernel.cu quantize_q8_1, including per-32-block scales."""
    padded = ((x.shape[1] + 511) // 512) * 512
    padded_x = torch.nn.functional.pad(x.float(), (0, padded - x.shape[1]))
    grouped = padded_x.reshape(x.shape[0], -1, 32)
    d = grouped.abs().amax(dim=-1, keepdim=True) / 127.0
    d = d.clamp_min(1e-9).to(torch.float16).float()
    return (torch.round(grouped / d) * d).reshape_as(padded_x)[:, : x.shape[1]]


@ROCM_DEVICE
@pytest.mark.parametrize("quant_type", [GGML_Q8_0, GGML_Q6_K])
def test_gguf_linear_matches_independent_dequant(quant_type):
    rows, cols = (7, 256) if quant_type == GGML_Q8_0 else (5, 256)
    weight = _q8_weight(rows, cols) if quant_type == GGML_Q8_0 else _q6_weight(rows, cols)
    x = torch.randn(1, cols, generator=torch.Generator(device="cpu").manual_seed(2026), dtype=torch.bfloat16)
    x = x.to("cuda")
    output = ggml_mul_mat_vec_a8(weight, x, quant_type, rows).float()
    dense = _q8_reference(weight) if quant_type == GGML_Q8_0 else dequant_q6_k(weight, torch.float32).reshape(rows, cols)
    reference = _q8_1_activation_reference(x) @ dense.T
    assert torch.isfinite(output).all()
    # Q6_K GEMV accumulates packed sub-blocks in device precision/order; keep
    # this reference gate aligned with the MoE packed-kernel gate.
    torch.testing.assert_close(output, reference, rtol=5e-2, atol=0.5)
