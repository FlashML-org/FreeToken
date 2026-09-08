"""qwen3_5_moe dense weight-loader helpers: the fp8 ``weight_scale`` shape contract.

``_per_row_scale`` is what turns whatever ``weight_scale`` a compressed-tensors or modelopt
checkpoint puts on disk into the per-output-row fp32 vector ``Fp8PerTensorLinear.weight_scale``
declares. Both on-disk granularities are exercised, plus the refusal that keeps a mis-shaped
scale from being applied to the wrong output rows.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.qwen3_5_moe.weight import _per_row_scale

_ROWS = 7


@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_per_tensor_scale_broadcasts_to_every_row(shape):
    """``strategy: "tensor"`` -- one scalar for the whole weight."""
    out = _per_row_scale(torch.full(shape, 0.25), _ROWS)
    assert out.shape == (_ROWS,)
    assert out.dtype == torch.float32
    assert out.tolist() == [0.25] * _ROWS


@pytest.mark.parametrize("shape", [(_ROWS, 1), (1, _ROWS), (_ROWS,)])
def test_per_channel_scale_keeps_row_order(shape):
    """``strategy: "channel"`` -- already one scalar per row; order must survive the reshape.

    Order is asserted with distinct values rather than a set/sum: permuting the scales would
    keep every aggregate identical while silently scaling each output row by another row's
    factor.
    """
    values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    out = _per_row_scale(torch.tensor(values).reshape(shape), _ROWS)
    assert out.shape == (_ROWS,)
    assert out.dtype == torch.float32
    assert out.tolist() == values


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_scale_is_promoted_to_fp32_from_any_storage_dtype(dtype):
    out = _per_row_scale(torch.ones(_ROWS, 1, dtype=dtype), _ROWS)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("shape", [(3, 1), (_ROWS + 1, 1), (2, 3)])
def test_a_mismatched_scale_raises_instead_of_broadcasting(shape):
    """Neither 1 nor ``rows`` elements: raise. Broadcasting row 0 over every output row would
    load without error and serve fluent, wrong tokens."""
    with pytest.raises(ValueError, match="expected either 1"):
        _per_row_scale(torch.ones(shape), _ROWS)
