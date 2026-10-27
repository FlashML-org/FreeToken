"""Packed GGUF ops against gguf-py's CPU dequantization and PyTorch math."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel import ggml
from freetoken.layers.quantization.ggml_types import pad_rows
from freetoken.third_party.gguf import GGMLQuantizationType as QType
from freetoken.third_party.gguf import quants

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _q4_weights(shape, seed=690):
    values = np.random.default_rng(seed).normal(0, 0.1, shape).astype(np.float32)
    packed = quants.quantize(values, QType.Q4_0)
    reference = torch.from_numpy(quants.dequantize(packed, QType.Q4_0))
    # ggml's last MMQ tile may read an extra partial row.
    storage = torch.zeros(*packed.shape[:-2], packed.shape[-2] + 1, packed.shape[-1], dtype=torch.uint8, device="cuda")
    storage[..., :-1, :].copy_(torch.from_numpy(packed))
    return storage[..., :-1, :], reference


def _nvfp4_weights(shape, seed=765):
    """Random NVFP4 blocks over every fp4 code and every UE4M3 scale code below NaN (0x7F), padded for MMQ."""
    rng = np.random.default_rng(seed)
    rows, k = shape
    blocks = k // 64
    d = rng.integers(0, 0x7F, (rows, blocks, 4), dtype=np.uint8)
    qs = rng.integers(0, 256, (rows, blocks, 32), dtype=np.uint8)
    packed = np.concatenate([d, qs], axis=-1).reshape(rows, blocks * 36)
    reference = torch.from_numpy(quants.dequantize(packed, QType.NVFP4))
    storage = torch.zeros(rows + pad_rows(int(QType.NVFP4), k), packed.shape[-1], dtype=torch.uint8, device="cuda")
    storage[:rows].copy_(torch.from_numpy(packed))
    return storage, reference


def _iq1_m_weights(shape, exact_sums, seed=1729):
    """Random IQ1_M blocks over every grid index, shift bit and 3-bit scale, padded for MMQ.

    Super-scales are random fp16 in [0.005, 0.02], every bit of which the loader decodes. With ``exact_sums`` they are
    m / 1024 (m in 5..20) instead: every product with ``_activation`` is then a multiple of 2^-19 and the sums are exact
    in fp32, so the kernel and the reference round a bf16 / fp16 result alike.
    """
    rng = np.random.default_rng(seed)
    *lead, rows, k = shape
    blocks = (*lead, rows, k // 256)
    qs = rng.integers(0, 256, (*blocks, 32), dtype=np.uint8)
    qh = rng.integers(0, 256, (*blocks, 16), dtype=np.uint8)
    d = rng.integers(5, 21, blocks) / 1024 if exact_sums else rng.uniform(0.005, 0.02, blocks)
    d = d.astype(np.float16).view(np.uint16)
    # the super-scale's four nibbles sit in the top nibble of the four uint16 scale words
    scales = rng.integers(0, 1 << 12, (*blocks, 4), dtype=np.uint16) | (((d[..., None] >> (4 * np.arange(4, dtype=np.uint16))) & 0xF) << 12)
    packed = np.concatenate([qs, qh, scales.astype("<u2").view(np.uint8)], axis=-1).reshape(*lead, rows, -1)
    reference = torch.from_numpy(quants.dequantize(packed, QType.IQ1_M))
    storage = torch.zeros(*lead, rows + 1, packed.shape[-1], dtype=torch.uint8, device="cuda")
    storage[..., :-1, :].copy_(torch.from_numpy(packed))
    return storage[..., :-1, :], reference


def _activation(shape, dtype, seed=19):
    rng = np.random.default_rng(seed)
    half = rng.integers(-127, 128, (*shape[:-1], shape[-1] // 32, 16))
    half[..., 0] = 127
    # Exactly representable q8_1 inputs with zero block sums isolate weight math.
    values = np.concatenate([half, -half], axis=-1).reshape(shape).astype(np.float32) / 64
    return torch.from_numpy(values).to("cuda", dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_q4_dequantize_and_rows(dtype):
    w, ref = _q4_weights((64, 256))
    actual = ggml.dequantize(w, int(QType.Q4_0), 256, dtype)
    torch.testing.assert_close(actual.cpu(), ref.to(dtype), atol=0, rtol=0)
    ids = torch.tensor([63, 0, 17, 17], dtype=torch.int32, device="cuda")
    rows = ggml.get_rows(w, int(QType.Q4_0), 256, ids, dtype)
    torch.testing.assert_close(rows.cpu(), ref[ids.cpu().long()].to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17])
def test_q4_matrix_matches_cpu(dtype, batch):
    w, ref = _q4_weights((64, 256))
    x = _activation((batch, 256), dtype)
    actual = ggml.mul_mat(w, int(QType.Q4_0), x)
    expected = F.linear(x.cpu().float(), ref).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17])
def test_q4_moe_routes_match_cpu(dtype, batch):
    w, ref = _q4_weights((4, 64, 256))
    x = _activation((batch, 256), dtype)
    ids = torch.tensor([[i % 4, (i + 1) % 4] for i in range(batch)], dtype=torch.int32, device="cuda")
    actual = ggml.mul_mat_id(w, int(QType.Q4_0), x, ids)
    expected = torch.einsum("tk, tjok -> tjo", x.cpu().float(), ref[ids.cpu().long()]).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17, 300])
def test_iq1_m_matrix_and_moe_routes_match_cpu(dtype, batch):
    # llama.cpp has no IQ1_M MMQ: above the vector kernel's batch this runs FreeToken's IQ1_M tile loader
    exact_sums = dtype != torch.float32
    w, ref = _iq1_m_weights((64, 512), exact_sums)
    x = _activation((batch, 512), dtype)
    actual = ggml.mul_mat(w, int(QType.IQ1_M), x)
    torch.testing.assert_close(actual.cpu(), F.linear(x.cpu().float(), ref).to(dtype), atol=2e-4, rtol=1e-3)
    w, ref = _iq1_m_weights((4, 64, 512), exact_sums)
    ids = torch.tensor([[i % 4, (i + 1) % 4] for i in range(batch)], dtype=torch.int32, device="cuda")
    actual = ggml.mul_mat_id(w, int(QType.IQ1_M), x, ids)
    expected = torch.einsum("tk, tjok -> tjo", x.cpu().float(), ref[ids.cpu().long()]).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_q4_shared_q8_matches_internal_quantization(dtype):
    w, _ = _q4_weights((64, 256))
    x = _activation((1, 256), dtype)
    expected = ggml.mul_mat(w, int(QType.Q4_0), x)
    actual = ggml.mul_mat(w, int(QType.Q4_0), x, xq=ggml.quantize_q8_1(x))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_q4_nondefault_stream():
    w, ref = _q4_weights((64, 256))
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = _activation((1, 256), torch.float32)
        actual = ggml.mul_mat(w, int(QType.Q4_0), x)
        copied = actual.clone()
    stream.synchronize()
    torch.testing.assert_close(copied.cpu(), F.linear(x.cpu(), ref), atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_nvfp4_dequantize_and_rows(dtype):
    # 320 values: five NVFP4 blocks, not a multiple of the 256-value super-block the k-quant get_rows path needs
    w, ref = _nvfp4_weights((64, 320))
    actual = ggml.dequantize(w[:64], int(QType.NVFP4), 320, dtype)
    torch.testing.assert_close(actual.cpu(), ref.to(dtype), atol=0, rtol=0)
    ids = torch.tensor([63, 0, 17, 17], dtype=torch.int32, device="cuda")
    rows = ggml.get_rows(w, int(QType.NVFP4), 320, ids, dtype)
    torch.testing.assert_close(rows.cpu(), ref[ids.cpu().long()].to(dtype), atol=0, rtol=0)

