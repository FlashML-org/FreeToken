"""GGUF Linear kernel dispatch must match what the borrowed kernels actually implement.

In particular IQ4_NL/IQ4_XS have an MMVQ (and dequant) case but no large-batch MMQ case
-- the MMQ entry point has no default, so routing them there would silently return NaNs.
"""
from __future__ import annotations

import pytest
import torch

from freetoken.layers.gguf import _DEQUANT, _MMQ, _MMVQ, fused_mul_mat_gguf
from freetoken.models.gguf.dequant import (
    GGML_IQ4_NL,
    GGML_IQ4_XS,
    GGML_Q4_0,
    GGML_Q5_K,
    GGML_Q6_K,
    GGML_Q8_0,
)


def test_mmvq_covers_all_supported_block_quants() -> None:
    assert {GGML_Q4_0, GGML_Q8_0, GGML_Q5_K, GGML_Q6_K, GGML_IQ4_NL, GGML_IQ4_XS} <= _MMVQ


def test_mmq_excludes_the_non_linear_quants() -> None:
    assert GGML_Q5_K in _MMQ
    assert GGML_Q5_K in _DEQUANT
    assert GGML_IQ4_NL not in _MMQ
    assert GGML_IQ4_XS not in _MMQ
    assert GGML_IQ4_NL in _DEQUANT
    assert GGML_IQ4_XS in _DEQUANT


def test_unknown_type_raises_instead_of_running_an_empty_kernel() -> None:
    with pytest.raises(NotImplementedError):
        fused_mul_mat_gguf(torch.zeros(1, 32), torch.zeros(4, 18, dtype=torch.uint8), 999)
