"""GGUF Linear kernel dispatch must match what the borrowed kernels actually implement.

In particular IQ4_NL/IQ4_XS have an MMVQ (and dequant) case but no large-batch MMQ case
-- the MMQ entry point has no default, so routing them there would silently return NaNs.

Also covers the LM head's prefill contract: the engine reads ``logits[:batch.size]``, so the
head must select each request's last prompt row itself.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.layers.gguf import _DEQUANT, _MMQ, _MMVQ, GGUFUntiedLMHead, fused_mul_mat_gguf
from freetoken.models.gguf.dequant import (
    GGML_IQ4_NL,
    GGML_IQ4_XS,
    GGML_Q4_0,
    GGML_Q5_K,
    GGML_Q6_K,
    GGML_Q8_0,
)


def test_untied_lm_head_gathers_the_last_prefill_row(monkeypatch) -> None:
    """A prefill must project each request's last prompt row, not its first."""
    import freetoken.core as core
    from freetoken.layers import gguf as gguf_layers

    captured = {}

    def fake(x, qweight, quant_type):
        captured["x"] = x.clone()
        return torch.zeros(x.shape[0], 16)

    monkeypatch.setattr(gguf_layers, "fused_mul_mat_gguf", fake)
    head = GGUFUntiedLMHead(32, 16, GGML_Q4_0)
    x = torch.arange(3 * 32, dtype=torch.float32).reshape(3, 32)

    batch = SimpleNamespace(
        is_prefill=True,
        size=1,
        attn_metadata=SimpleNamespace(get_last_indices=lambda bs: torch.tensor([2])[:bs]),
    )
    monkeypatch.setattr(core, "get_global_ctx", lambda: SimpleNamespace(batch=batch))
    head.forward(x)
    assert torch.equal(captured["x"], x[[2]])

    batch.is_prefill = False
    head.forward(x)
    assert torch.equal(captured["x"], x)


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
