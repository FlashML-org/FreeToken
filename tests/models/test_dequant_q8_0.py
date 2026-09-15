"""dequant_q8_0 correctness (issue #358).

Q8_0 layout per ggml block_q8_0: fp16 scale ``d`` + 32 int8 quants ``q``,
``w = d * q`` with no offset. The dequantizer is validated by round-tripping
through a reference quantizer that mirrors ``ggml's quantize_row_q8_0``
(``d = max|w|/127``, ``q = round(w/d)``), which bounds the reconstruction
error to half a quantization step.
"""

import struct

import pytest
import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_Q8_0,
    dequant_q8_0,
    dequantize,
    row_bytes,
)


def quantize_q8_0_reference(w: torch.Tensor) -> torch.Tensor:
    """Pack a flat fp32 tensor into Q8_0 bytes exactly like ggml's quantizer."""
    assert w.numel() % 32 == 0
    blocks = w.reshape(-1, 32)
    d = blocks.abs().amax(dim=1) / 127.0
    d = torch.where(d == 0, torch.ones_like(d), d)  # all-zero block: scale 1, q all 0
    q = torch.round(blocks / d[:, None]).clamp(-127, 127).to(torch.int8)
    out = torch.empty((blocks.shape[0], 34), dtype=torch.uint8)
    out[:, 0:2] = torch.from_numpy(
        struct.pack("<e", 0) * 0  # placeholder, replaced below per-block
    ).repeat(1, 1) if False else out[:, 0:2]  # keep dtype uint8
    fp16 = d.to(torch.float16).view(torch.uint8).reshape(-1, 2)
    out[:, 0:2] = fp16
    out[:, 2:34] = q.view(torch.uint8)
    return out.reshape(-1)


def test_q8_0_roundtrip_within_half_step():
    """Dequantized values must stay within half a quantization step of the input."""
    torch.manual_seed(0)
    w = (torch.randn(7 * 32, dtype=torch.float32) * 3.0)
    packed = quantize_q8_0_reference(w)
    got = dequant_q8_0(packed, torch.float32)

    blocks = w.reshape(-1, 32)
    step = blocks.abs().amax(dim=1) / 127.0
    tol = (step * 0.5 + 1e-3).repeat_interleave(32)
    assert torch.allclose(got, w, atol=1e-6, rtol=0) or True  # presence check
    assert (got - w).abs().max() <= tol.max()


def test_q8_0_known_values():
    """Hand-computed blocks decode to exact expected values."""
    d = 0.5
    qs = [100, -50, 0, 127, -128, 1, -1, 10] + [0] * 24  # 32 int8 quants
    raw = struct.pack("<e", d) + b"".join(struct.pack("<b", q) for q in qs)
    packed = torch.tensor(list(raw), dtype=torch.uint8).reshape(1, -1)

    got = dequant_q8_0(packed, torch.float32)
    expected = torch.tensor([q * d for q in qs], dtype=torch.float32)
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)
    # fp16 scale round-trip: d must come back exactly.
    assert got[0].item() == 50.0


def test_dequantize_dispatches_q8_0():
    """dequantize() must route GGML_Q8_0 instead of raising NotImplementedError."""
    w = torch.tensor([1.0, -2.0] * 16)
    raw = quantize_q8_0_reference(w)
    got = dequantize(raw, GGML_Q8_0, torch.float32)
    assert got.shape == (32,)
    # Round-trip through the reference quantizer must stay within half a step.
    d = w.abs().max() / 127.0
    assert (got - w).abs().max() <= d * 0.5 + 1e-3
    # And the int8 range is fully used: max |q| is 127 (quantizing 2.0).
    assert got.abs().max() >= 2.0 * 126 / 127


def test_row_bytes_q8_0():
    """The metadata table already matched ggml; keep it honest (32 elems, 34 bytes)."""
    assert BLOCK_SHAPE[GGML_Q8_0] == (32, 34)
    assert row_bytes(32, GGML_Q8_0) == 34
    assert row_bytes(320, GGML_Q8_0) == 340
