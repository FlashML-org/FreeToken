from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize("ggml_type", [2, 8, 14])
def test_cpu_moe_packed_kquant_rows_use_declared_stride(ggml_type):
    from freetoken.models.gguf.dequant import BLOCK_SHAPE, row_bytes

    block, type_size = BLOCK_SHAPE[ggml_type]
    rows = torch.zeros((7, row_bytes(block * 2, ggml_type)), dtype=torch.uint8)
    assert rows.shape == (7, 2 * type_size)


def test_cpu_moe_reference_q8_0_has_finite_values():
    from freetoken.models.gguf.dequant import GGML_Q8_0, dequantize

    raw = torch.zeros(34, dtype=torch.uint8)
    raw[:2] = torch.tensor([0.5], dtype=torch.float16).view(torch.uint8)
    raw[2:] = torch.arange(-16, 16, dtype=torch.int8).view(torch.uint8)
    result = dequantize(raw, GGML_Q8_0, torch.bfloat16)
    assert result.shape == (32,)
    assert torch.isfinite(result.float()).all()
