"""Reference checks for the Q4_K GGUF format used by Q4_K_M model releases."""

import torch

from freetoken.models.gguf.dequant import GGML_Q4_K, dequant_q4_k, row_bytes


def _half_bytes(value: float) -> torch.Tensor:
    """Return the two little-endian bytes of one IEEE fp16 scalar."""
    return torch.tensor([value], dtype=torch.float16).view(torch.uint8)


def test_q4_k_row_size_matches_the_ggml_block_layout():
    """A 256-value Q4_K super-block occupies 144 bytes in the GGUF tensor table."""
    assert row_bytes(256, GGML_Q4_K) == 144
    assert row_bytes(2048, GGML_Q4_K) == 8 * 144


def test_q4_k_reference_decoder_handles_scale_minimum_and_nibble_order():
    """Known packed bytes decode by the same affine rule used in llama.cpp."""
    raw = torch.zeros((1, 144), dtype=torch.uint8)
    raw[0, 0:2] = _half_bytes(2.0)
    raw[0, 2:4] = _half_bytes(0.5)
    # The first packed 32-byte pair encodes group 0 in its low nibbles and
    # group 1 in its high nibbles.  Their scale/minimum fields are separate.
    raw[0, 4] = 3
    raw[0, 8] = 4
    raw[0, 5] = 3
    raw[0, 9] = 4
    raw[0, 16:32] = 0xF1  # low nibble 1, high nibble 15 for the first 32-value group.

    decoded = dequant_q4_k(raw, torch.float32)
    # group 0: 2 * 3 * q - 0.5 * 4. The first 32 values use low nibbles.
    assert decoded[0].item() == 4.0
    assert decoded[31].item() == 4.0
    # Group 1 uses the high nibbles from the same packed byte range.
    assert decoded[32].item() == 88.0
    assert decoded[63].item() == 88.0
    # The remaining groups have zero scale/minimum and therefore decode to zero.
    assert torch.count_nonzero(decoded[64:]) == 0
