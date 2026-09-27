"""Numerical equivalence for the native GGUF codebook / K-quant dequantizers.

The reference here is a literal transcription of ggml's CPU dequant loops
(``ggml-quants.c``), independent of the vectorized torch implementation, so a
layout or bit-packing regression in :mod:`freetoken.models.gguf.dequant` shows up
as a mismatch rather than being masked by a shared helper.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_IQ4_NL,
    GGML_IQ4_XS,
    GGML_Q5_K,
    GGML_Q8_0,
    IQ4NL_KVALUES,
    dequantize,
    row_bytes,
)

# ggml_type -> (block numel, bytes per block) as it must appear in BLOCK_SHAPE.
# ggml's IQ4 codebook, spelled out so the reference does not share the production constant.
_IQ4_KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)

EXPECTED_SHAPE = {
    GGML_Q5_K: (256, 176),
    GGML_IQ4_NL: (32, 18),
    GGML_IQ4_XS: (256, 136),
    GGML_Q8_0: (32, 34),
}


def _from_f16(raw: bytes) -> np.float32:
    return np.frombuffer(raw, np.float16)[0].astype(np.float32)


def _safe_blocks(ggml_type: int, n: int, seed: int) -> np.ndarray:
    """``n`` random packed blocks with the fp16 scale fields forced finite."""
    rng = np.random.default_rng(seed)
    size = EXPECTED_SHAPE[ggml_type][1]
    blocks = rng.integers(0, 256, size=(n, size), dtype=np.uint8)
    scales = (0.01 + rng.random(n)).astype(np.float16)  # positive, so no NaN/inf when re-read
    raw = np.frombuffer(scales.tobytes(), np.uint8).reshape(n, 2)
    blocks[:, 0:2] = raw
    if ggml_type in (GGML_Q5_K, GGML_IQ4_XS):
        blocks[:, 2:4] = raw
    return blocks


def _ref_q5_k(blocks: np.ndarray) -> np.ndarray:
    out = np.empty((blocks.shape[0], 256), dtype=np.float32)
    for i, b in enumerate(blocks):
        d = _from_f16(b[0:2].tobytes())
        dmin = _from_f16(b[2:4].tobytes())
        q = b[4:16]
        sc = np.empty(8, np.int64)
        mn = np.empty(8, np.int64)
        for j in range(8):
            if j < 4:
                sc[j] = q[j] & 63
                mn[j] = q[j + 4] & 63
            else:
                sc[j] = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
                mn[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
        qh, qs = b[16:48], b[48:176]
        for g in range(4):
            ql = qs[32 * g : 32 * g + 32]
            u1, u2 = 1 << (2 * g), 1 << (2 * g + 1)
            d1, m1 = d * sc[2 * g], dmin * mn[2 * g]
            d2, m2 = d * sc[2 * g + 1], dmin * mn[2 * g + 1]
            for l in range(32):
                out[i, 64 * g + l] = d1 * ((ql[l] & 0xF) + (16 if qh[l] & u1 else 0)) - m1
                out[i, 64 * g + 32 + l] = d2 * ((ql[l] >> 4) + (16 if qh[l] & u2 else 0)) - m2
    return out


def _ref_iq4_nl(blocks: np.ndarray) -> np.ndarray:
    out = np.empty((blocks.shape[0], 32), dtype=np.float32)
    for i, b in enumerate(blocks):
        d = _from_f16(b[0:2].tobytes())
        qs = b[2:18]
        for j in range(16):
            out[i, j] = d * _IQ4_KVALUES[qs[j] & 0xF]
            out[i, j + 16] = d * _IQ4_KVALUES[qs[j] >> 4]
    return out


def _ref_q8_0(blocks: np.ndarray) -> np.ndarray:
    out = np.empty((blocks.shape[0], 32), dtype=np.float32)
    for i, b in enumerate(blocks):
        d = _from_f16(b[0:2].tobytes())
        out[i] = d * b[2:34].view(np.int8)
    return out


def _ref_iq4_xs(blocks: np.ndarray) -> np.ndarray:
    out = np.empty((blocks.shape[0], 256), dtype=np.float32)
    for i, b in enumerate(blocks):
        d = _from_f16(b[0:2].tobytes())
        scales_h = int.from_bytes(b[2:4].tobytes(), "little")
        scales_l, qs = b[4:8], b[8:136]
        for ib in range(8):
            ls = ((int(scales_l[ib // 2]) >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4)
            dl = d * (ls - 32)
            for j in range(16):
                out[i, 32 * ib + j] = dl * _IQ4_KVALUES[qs[16 * ib + j] & 0xF]
                out[i, 32 * ib + j + 16] = dl * _IQ4_KVALUES[qs[16 * ib + j] >> 4]
    return out


REFERENCE = {
    GGML_Q5_K: _ref_q5_k,
    GGML_IQ4_NL: _ref_iq4_nl,
    GGML_IQ4_XS: _ref_iq4_xs,
    GGML_Q8_0: _ref_q8_0,
}
GGML_NAME = {GGML_Q5_K: "Q5_K", GGML_IQ4_NL: "IQ4_NL", GGML_IQ4_XS: "IQ4_XS"}


@pytest.mark.parametrize("ggml_type", sorted(REFERENCE))
def test_block_shape_matches_the_on_disk_format(ggml_type: int) -> None:
    assert BLOCK_SHAPE[ggml_type] == EXPECTED_SHAPE[ggml_type]
    block, size = EXPECTED_SHAPE[ggml_type]
    assert row_bytes(block * 3, ggml_type) == size * 3


@pytest.mark.parametrize("ggml_type", sorted(REFERENCE))
def test_dequantize_matches_the_c_reference(ggml_type: int) -> None:
    blocks = _safe_blocks(ggml_type, n=64, seed=ggml_type)
    got = dequantize(torch.from_numpy(blocks.copy()).reshape(-1), ggml_type, torch.float32)
    expected = REFERENCE[ggml_type](blocks).reshape(-1)
    assert torch.equal(got, torch.from_numpy(expected))


GGUF_QUANTS_MODEL = os.environ.get("FREETOKEN_GGUF_QUANTS_MODEL", "")


@pytest.mark.needs_weights
@pytest.mark.skipif(
    not os.path.isfile(GGUF_QUANTS_MODEL),
    reason="FREETOKEN_GGUF_QUANTS_MODEL not set to a local GGUF carrying Q5_K/IQ4_NL/IQ4_XS",
)
def test_dequantize_matches_the_c_reference_on_a_real_checkpoint() -> None:
    from freetoken.models.gguf.reader import iter_gguf_tensors

    # Split-aware reader: the types are spread across the shards.
    seen = set()
    for t in iter_gguf_tensors(GGUF_QUANTS_MODEL):
        ggml_type = t.ggml_type
        if ggml_type not in REFERENCE or ggml_type in seen:
            continue
        seen.add(ggml_type)
        size = EXPECTED_SHAPE[ggml_type][1]
        raw = np.ascontiguousarray(t.packed().numpy()).reshape(-1)
        blocks = raw[: 256 * size].reshape(256, size)
        got = dequantize(torch.from_numpy(blocks.copy()).reshape(-1), ggml_type, torch.float32)
        expected = REFERENCE[ggml_type](blocks).reshape(-1)
        assert torch.equal(got, torch.from_numpy(expected)), GGML_NAME[ggml_type]
    assert seen == set(REFERENCE), sorted(GGML_NAME[t] for t in set(REFERENCE) - seen)


def test_iq4_codebook_matches_the_reference_literal() -> None:
    assert tuple(IQ4NL_KVALUES) == _IQ4_KVALUES
