"""Dense GGUF K-quant native/reference parity gates."""

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


@pytest.mark.parametrize("quant_type", [GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0])
def test_k_quant_zero_rows_have_valid_reference_shape(quant_type):
    raw = torch.zeros((2, row_bytes(256, quant_type)), dtype=torch.uint8)
    output = dequantize(raw, quant_type, torch.float32)
    assert output.shape == (512,)
    assert torch.isfinite(output).all()


ROCM_DEVICE = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="needs ROCm device",
)


def _packed_rows(rows: int, cols: int, quant_type: int) -> torch.Tensor:
    raw = torch.zeros((rows, row_bytes(cols, quant_type)), dtype=torch.uint8, device="cuda")
    if quant_type == GGML_Q8_0:
        blocks = raw.reshape(rows, cols // 32, 34)
        blocks[..., :2] = torch.tensor([128, 63], dtype=torch.uint8, device="cuda")
        blocks[..., 2:] = torch.randint(0, 255, blocks[..., 2:].shape, device="cuda")
    elif quant_type == GGML_Q6_K:
        raw[..., 192:208] = torch.randint(1, 255, raw[..., 192:208].shape, device="cuda")
        raw[..., 208:210] = torch.tensor([128, 63], dtype=torch.uint8, device="cuda")
    elif quant_type == GGML_Q5_K:
        raw[..., :4] = torch.tensor([128, 63, 128, 63], dtype=torch.uint8, device="cuda")
        raw[..., 4:16] = torch.randint(1, 64, raw[..., 4:16].shape, device="cuda")
        raw[..., 16:] = torch.randint(0, 255, raw[..., 16:].shape, device="cuda")
    else:
        raw[..., :4] = torch.tensor([128, 63, 128, 63], dtype=torch.uint8, device="cuda")
        raw[..., 4:16] = torch.randint(1, 64, raw[..., 4:16].shape, device="cuda")
        raw[..., 16:] = torch.randint(0, 255, raw[..., 16:].shape, device="cuda")
    return raw


def _q8_1_activation_reference(x: torch.Tensor) -> torch.Tensor:
    padded = ((x.shape[1] + 511) // 512) * 512
    padded_x = torch.nn.functional.pad(x.float(), (0, padded - x.shape[1]))
    grouped = padded_x.reshape(x.shape[0], -1, 32)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / 127.0
    scale = scale.to(torch.float16).float()
    return (torch.round(grouped / scale) * scale).reshape_as(padded_x)[:, : x.shape[1]]


@ROCM_DEVICE
@pytest.mark.parametrize("quant_type", [GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0])
def test_native_k_quant_matvec_matches_reference(quant_type):
    from freetoken.kernel.gguf import ggml_mul_mat_vec_a8

    rows, cols = 3, 256
    weight = _packed_rows(rows, cols, quant_type)
    hidden = torch.randn((1, cols), dtype=torch.bfloat16, device="cuda")
    output = ggml_mul_mat_vec_a8(weight, hidden, quant_type, rows).float()
    dense = dequantize(weight.cpu(), quant_type, torch.float32).reshape(rows, cols).to("cuda")
    reference = _q8_1_activation_reference(hidden) @ dense.T
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=8e-2, atol=0.75)
