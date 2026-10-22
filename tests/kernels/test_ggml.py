"""Packed GGUF ops against gguf-py's CPU dequantization and PyTorch math."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel import ggml, gguf
from freetoken.layers.gguf import GGUFLinear
from freetoken.models.gguf.dequant import GGML_Q6_K, row_bytes
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
    compat = gguf.ggml_dequantize(w, int(QType.Q4_0), 64, 256, dtype)
    torch.testing.assert_close(compat.cpu(), ref.to(dtype), atol=0, rtol=0)
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
def test_q4_shared_q8_matches_internal_quantization(dtype):
    w, _ = _q4_weights((64, 256))
    x = _activation((1, 256), dtype)
    expected = ggml.mul_mat(w, int(QType.Q4_0), x)
    actual = ggml.mul_mat(w, int(QType.Q4_0), x, xq=ggml.quantize_q8_1(x))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_q6_k_mmq_adds_missing_zero_tail():
    weight = torch.zeros(
        64, row_bytes(256, GGML_Q6_K), dtype=torch.uint8, device="cuda"
    )
    x = _activation((17, 256), torch.bfloat16)
    expected_storage_bytes = weight.numel() + ggml._mmq_tail_bytes(GGML_Q6_K, 256)

    output = ggml.mul_mat(weight, GGML_Q6_K, x)

    assert weight.untyped_storage().nbytes() >= expected_storage_bytes
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)


def test_gguf_linear_pads_mmq_weight_before_forward():
    linear = GGUFLinear(256, 64, GGML_Q6_K)
    weight = torch.zeros(
        64, row_bytes(256, GGML_Q6_K), dtype=torch.uint8, device="cuda"
    )
    linear.load_state_dict({"qweight": weight})
    expected_storage_bytes = weight.numel() + ggml._mmq_tail_bytes(GGML_Q6_K, 256)
    assert linear.qweight.untyped_storage().nbytes() >= expected_storage_bytes

    output = linear.forward(_activation((17, 256), torch.bfloat16))

    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17])
def test_gguf_compat_wrappers_use_shared_ggml_kernels(dtype, batch):
    w, ref = _q4_weights((64, 256))
    x = _activation((batch, 256), dtype)
    actual = gguf.ggml_mul_mat_a8(w, x, int(QType.Q4_0), 64)
    expected = F.linear(x.cpu().float(), ref).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gguf_moe_vector_wrapper_routes_via_shared_ggml_kernel(dtype):
    w, ref = _q4_weights((4, 64, 256))
    x = _activation((3, 256), dtype)
    ids = torch.tensor([[0, 1], [2, 3], [3, 0]], dtype=torch.int32, device="cuda")
    actual = gguf.ggml_moe_a8_vec(x, w, ids, 2, int(QType.Q4_0), 64, 3)
    expected = torch.einsum("tk,tjok->tjo", x.cpu().float(), ref[ids.cpu().long()]).reshape(
        6, 64
    ).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=1e-3)


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
