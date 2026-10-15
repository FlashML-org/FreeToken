"""The torch GGUF dequantizers back every load path and the padded-row bank geometry.

gguf-py carries an independent implementation (``gguf.dequantize``), so each type this repo
models is checked against it. The K-quant fixtures are hand-encoded with the trivial
"d = 1, scales = 1" convention, so the expected values follow in closed form from the block
layout (a nibble / 6-bit code minus its offset) instead of being read back from another
decoder, and they double as the reference the borrowed CUDA kernels are compared against.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from freetoken.models.gguf.dequant import (
    GGML_Q4_0,
    GGML_Q4_K,
    GGML_Q6_K,
    GGML_Q8_0,
    dequantize,
    padded_row_bytes,
    row_numel,
)

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _torch_bytes(raw: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(raw).reshape(-1).view(np.uint8))


def q4k_codes(codes: np.ndarray) -> np.ndarray:
    """Q4_K blocks of 4-bit values: ``d = 1``, ``dmin = 0``, every 6-bit scale/min 1/0.

    With those scales the dequantized values are the nibbles themselves, so the expected
    output is the input. ``codes`` is ``[..., 256]``-shaped in ggml's storage order (the low
    nibbles of each 64-element group first, then its high nibbles).
    """
    codes = np.asarray(codes, dtype=np.uint8).reshape(-1, 256)
    n = codes.shape[0]
    qs = np.empty((n, 128), dtype=np.uint8)
    for g in range(4):
        low, high = codes[:, 64 * g : 64 * g + 32], codes[:, 64 * g + 32 : 64 * g + 64]
        assert codes.shape[1] == 256 and low.max(initial=0) < 16 and high.max(initial=0) < 16
        qs[:, 32 * g : 32 * g + 32] = low | (high << 4)
    scales = np.zeros((n, 12), dtype=np.uint8)
    scales[:, :4] = 1  # scales 0-3
    scales[:, 8:12] = 1  # scales 4-7 and mins 4-7 high bits
    one = np.ones(n, dtype=np.float16).view(np.uint8).reshape(n, 2)
    zero = np.zeros(n, dtype=np.float16).view(np.uint8).reshape(n, 2)
    return np.concatenate([one, zero, scales, qs], axis=1)


def q6k_codes(codes: np.ndarray) -> np.ndarray:
    """Q6_K blocks of 6-bit values: ``d = 1``, every scale 1, so the values are ``code - 32``.

    ``codes`` is ``[..., 256]`` in ggml's storage order: within each 128-element half, the
    low nibbles of both 32-element halves carry bits 0-3 and ``qh`` bits 4-5, high nibbles
    the same for the next two 32-element groups (see ``dequant_q6_k``).
    """
    codes = np.asarray(codes, dtype=np.uint8).reshape(-1, 256)
    n = codes.shape[0]
    assert codes.max(initial=0) < 64
    ql = np.zeros((n, 128), dtype=np.uint8)
    qh = np.zeros((n, 64), dtype=np.uint8)
    for half in range(2):
        for j in range(32):
            for src, dst, shift, nibble in (
                (half * 128 + j, half * 64 + j, 0, "low"),
                (half * 128 + j + 32, half * 64 + j + 32, 2, "low"),
                (half * 128 + j + 64, half * 64 + j, 4, "high"),
                (half * 128 + j + 96, half * 64 + j + 32, 6, "high"),
            ):
                code = codes[:, src]
                if nibble == "low":
                    ql[:, dst] |= code & 0x0F
                else:
                    ql[:, dst] |= (code & 0x0F) << 4
                qh[:, half * 32 + j] |= (code >> 4) << shift
    scales = np.ones((n, 16), dtype=np.uint8)
    one = np.ones(n, dtype=np.float16).view(np.uint8).reshape(n, 2)
    return np.concatenate([ql, qh, scales, one], axis=1)


def test_q4_0_and_q8_0_match_gguf_py():
    import gguf

    rng = np.random.default_rng(0)
    for qtype in (GGML_Q4_0, GGML_Q8_0):
        name = {GGML_Q4_0: "Q4_0", GGML_Q8_0: "Q8_0"}[qtype]
        ggml_type = getattr(gguf.GGMLQuantizationType, name)
        values = (rng.standard_normal(512).astype(np.float32) * 0.05)
        raw = gguf.quantize(values, ggml_type)
        got = dequantize(_torch_bytes(raw), qtype, torch.float32)
        want = torch.from_numpy(gguf.dequantize(raw, ggml_type).reshape(-1).astype(np.float32))
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


def test_q4_k_decodes_its_nibbles_and_matches_gguf_py():
    import gguf

    rng = np.random.default_rng(1)
    codes = rng.integers(0, 16, size=(2, 256), dtype=np.uint8)
    raw = q4k_codes(codes)
    got = dequantize(_torch_bytes(raw), GGML_Q4_K, torch.float32)
    want = torch.from_numpy(codes.reshape(-1).astype(np.float32))
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    reference = gguf.dequantize(raw, gguf.GGMLQuantizationType.Q4_K).reshape(-1)
    torch.testing.assert_close(got, torch.from_numpy(reference.astype(np.float32)), rtol=1e-6, atol=1e-6)


def test_q6_k_decodes_its_codes_and_matches_gguf_py():
    import gguf

    rng = np.random.default_rng(2)
    codes = rng.integers(0, 64, size=(2, 256), dtype=np.uint8)
    raw = q6k_codes(codes)
    got = dequantize(_torch_bytes(raw), GGML_Q6_K, torch.float32)
    want = torch.from_numpy(codes.reshape(-1).astype(np.float32) - 32.0)
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    reference = gguf.dequantize(raw, gguf.GGMLQuantizationType.Q6_K).reshape(-1)
    torch.testing.assert_close(got, torch.from_numpy(reference.astype(np.float32)), rtol=1e-6, atol=1e-6)


@_CUDA
def test_cuda_dequant_kernel_matches_torch_for_k_quants():
    """The borrowed kernels dequantize in place; the load path's reference is the torch one."""
    from freetoken.kernel.gguf import ggml_dequantize

    rng = np.random.default_rng(3)
    cases = (
        (GGML_Q4_K, q4k_codes(rng.integers(0, 16, size=(3, 256), dtype=np.uint8)), 256),
        (GGML_Q6_K, q6k_codes(rng.integers(0, 64, size=(3, 256), dtype=np.uint8)), 256),
    )
    for qtype, raw, numel in cases:
        rows = torch.from_numpy(np.ascontiguousarray(raw)).cuda()
        got = ggml_dequantize(rows, qtype, rows.shape[0], numel, torch.float32)
        want = dequantize(_torch_bytes(raw), qtype, torch.float32).reshape(rows.shape[0], numel).cuda()
        torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)


def test_padded_row_bytes_aligns_and_round_trips():
    # Q4_K is 144 bytes per 256 elements, already a 16-byte multiple; Q6_K is 210, so a
    # 1536-wide row needs two extra super-blocks to become 1680 (the offload cache moves rows
    # with 16-byte-granular kernels).
    assert padded_row_bytes(2048, GGML_Q4_K) == (1152, 2048)
    assert padded_row_bytes(1536, GGML_Q4_K) == (864, 1536)
    assert padded_row_bytes(1536, GGML_Q6_K) == (1680, 2048)
    assert padded_row_bytes(2048, GGML_Q6_K) == (1680, 2048)
    assert padded_row_bytes(128, GGML_Q4_0) == (144, 256)
    for numel in (256, 512, 1024, 1536, 2048):
        for qtype in (GGML_Q4_K, GGML_Q6_K, GGML_Q4_0, GGML_Q8_0):
            row, padded = padded_row_bytes(numel, qtype)
            assert row % 16 == 0
            assert padded >= numel and padded % 256 == 0
            assert row_numel(row, qtype) == padded
    with pytest.raises(ValueError):
        padded_row_bytes(100, GGML_Q4_K)  # not a whole number of blocks
