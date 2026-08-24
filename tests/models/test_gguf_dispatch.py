"""Unit tests for GGUF linear dispatch routing (fused_mul_mat_gguf).

Tests the 3-tier dispatch strategy without requiring CUDA or the compiled kernel extension.
Uses monkeypatch to replace the C kernel functions with mocks that record which dispatch
path was taken and return correctly-shaped CPU tensors.

Dispatch order (from python/freetoken/layers/gguf.py:46-75):
1. Empty input early return
2. Unquantized (F32/F16/BF16) → torch matmul
3. Small batch (<= _MMVQ_SAFE) AND MMVQ_TYPES → ggml_mul_mat_vec_a8
4. MMQ_TYPES → ggml_mul_mat_a8
5. DEQUANT_TYPES (but not MMQ_TYPES) → ggml_dequantize + torch matmul
6. Else → NotImplementedError
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from freetoken.models.gguf.dequant import (
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_IQ1_M,
    GGML_IQ2_S,
    GGML_Q2_K,
    GGML_Q4_K,
    GGML_Q6_K,
    BLOCK_SHAPE,
)
from freetoken.layers.gguf import _MMVQ_SAFE, fused_mul_mat_gguf


@pytest.fixture
def mock_kernel_module(monkeypatch):
    """Replace freetoken.kernel.gguf with a mock that tracks kernel calls.

    Returns a dict tracking which kernels were called and with what arguments.
    """
    call_log = {
        "ggml_mul_mat_vec_a8": None,
        "ggml_mul_mat_a8": None,
        "ggml_dequantize": None,
    }

    def make_mmvq_kernel(call_log):
        """Mock ggml_mul_mat_vec_a8: GEMV kernel for small batch."""
        def kernel(qweight, x, qweight_type, out_features):
            call_log["ggml_mul_mat_vec_a8"] = {
                "qweight_shape": qweight.shape,
                "x_shape": x.shape,
                "qweight_type": qweight_type,
                "out_features": out_features,
            }
            batch_size = x.shape[0]
            return torch.randn(batch_size, out_features, dtype=x.dtype)
        return kernel

    def make_mmq_kernel(call_log):
        """Mock ggml_mul_mat_a8: MMQ kernel for large batch."""
        def kernel(qweight, x, qweight_type, out_features):
            call_log["ggml_mul_mat_a8"] = {
                "qweight_shape": qweight.shape,
                "x_shape": x.shape,
                "qweight_type": qweight_type,
                "out_features": out_features,
            }
            batch_size = x.shape[0]
            return torch.randn(batch_size, out_features, dtype=x.dtype)
        return kernel

    def make_dequant_kernel(call_log):
        """Mock ggml_dequantize: materializes weight into BF16."""
        def kernel(qweight, qweight_type, out_features, in_features, out_dtype):
            call_log["ggml_dequantize"] = {
                "qweight_shape": qweight.shape,
                "qweight_type": qweight_type,
                "out_features": out_features,
                "in_features": in_features,
                "out_dtype": out_dtype,
            }
            return torch.randn(out_features, in_features, dtype=out_dtype)
        return kernel

    mock_module = ModuleType("freetoken.kernel.gguf")
    mock_module.ggml_mul_mat_vec_a8 = make_mmvq_kernel(call_log)
    mock_module.ggml_mul_mat_a8 = make_mmq_kernel(call_log)
    mock_module.ggml_dequantize = make_dequant_kernel(call_log)

    monkeypatch.setitem(sys.modules, "freetoken.kernel.gguf", mock_module)

    yield call_log

    # Cleanup: ensure mock is removed so it doesn't interfere with other tests
    if "freetoken.kernel.gguf" in sys.modules:
        del sys.modules["freetoken.kernel.gguf"]


def make_qweight(out_features: int, in_features: int, qweight_type: int) -> torch.Tensor:
    """Create a mock qweight tensor in packed format for the given type."""
    block, type_size = BLOCK_SHAPE[qweight_type]
    row_bytes_val = in_features // block * type_size
    return torch.randint(0, 256, (out_features, row_bytes_val), dtype=torch.uint8)


class TestEmptyInput:
    """Test early return for zero-row input."""

    def test_empty_input_short_circuits(self, mock_kernel_module):
        """Zero-row input returns shape (0, out_features) with no kernel call."""
        out_features = 4096
        in_features = 4096
        qweight_type = GGML_Q4_K

        x = torch.randn(0, in_features)
        qweight = make_qweight(out_features, in_features, qweight_type)

        result = fused_mul_mat_gguf(x, qweight, qweight_type)

        assert result.shape == (0, out_features)
        assert mock_kernel_module["ggml_mul_mat_vec_a8"] is None
        assert mock_kernel_module["ggml_mul_mat_a8"] is None
        assert mock_kernel_module["ggml_dequantize"] is None


class TestUnquantized:
    """Test unquantized paths (F32, F16, BF16)."""

    @pytest.mark.parametrize("qweight_type,dtype", [
        (GGML_F32, torch.float32),
        (GGML_F16, torch.float16),
        (GGML_BF16, torch.bfloat16),
    ])
    def test_unquantized_never_calls_kernels(self, mock_kernel_module, qweight_type, dtype):
        """F32/F16/BF16 go through plain torch matmul, no kernel call."""
        out_features = 4096
        in_features = 4096
        batch_size = 32

        x = torch.randn(batch_size, in_features, dtype=dtype)
        # For unquantized, qweight is stored as-is (1 element per 1-4 bytes)
        qweight = torch.randn(out_features, in_features, dtype=dtype)

        result = fused_mul_mat_gguf(x, qweight, qweight_type)

        # Check no kernels were called
        assert mock_kernel_module["ggml_mul_mat_vec_a8"] is None
        assert mock_kernel_module["ggml_mul_mat_a8"] is None
        assert mock_kernel_module["ggml_dequantize"] is None
        # Result should match torch matmul
        expected = x @ qweight.T
        assert result.shape == expected.shape


class TestSmallBatchMMVQ:
    """Test small-batch dispatch to ggml_mul_mat_vec_a8 (MMVQ)."""

    @pytest.mark.parametrize("qweight_type", [
        GGML_Q2_K,   # K-quant
        GGML_Q4_K,   # K-quant
        GGML_Q6_K,   # K-quant
        GGML_IQ2_S,  # I-quant
        GGML_IQ1_M,  # I-quant
    ])
    def test_small_batch_uses_mmvq(self, mock_kernel_module, qweight_type):
        """Batch <= _MMVQ_SAFE in MMVQ_TYPES calls ggml_mul_mat_vec_a8."""
        out_features = 4096
        in_features = 4096
        batch_size = 1  # Small batch (well below _MMVQ_SAFE)

        x = torch.randn(batch_size, in_features, dtype=torch.bfloat16)
        qweight = make_qweight(out_features, in_features, qweight_type)

        result = fused_mul_mat_gguf(x, qweight, qweight_type)

        # MMVQ kernel should have been called
        assert mock_kernel_module["ggml_mul_mat_vec_a8"] is not None
        assert mock_kernel_module["ggml_mul_mat_a8"] is None
        assert mock_kernel_module["ggml_dequantize"] is None

        call_info = mock_kernel_module["ggml_mul_mat_vec_a8"]
        assert call_info["x_shape"] == (batch_size, in_features)
        assert call_info["qweight_type"] == qweight_type
        assert call_info["out_features"] == out_features
        assert result.shape == (batch_size, out_features)


class TestLargeBatchStandardQuants:
    """Test large-batch K-quants and standard quants dispatch to ggml_mul_mat_a8 (MMQ)."""

    @pytest.mark.parametrize("qweight_type", [
        GGML_Q2_K,
        GGML_Q4_K,
        GGML_Q6_K,
    ])
    def test_kquant_large_batch_takes_mmq(self, mock_kernel_module, qweight_type):
        """K-quants at large batch call ggml_mul_mat_a8."""
        out_features = 4096
        in_features = 4096
        batch_size = _MMVQ_SAFE + 1  # Large batch (above threshold)

        x = torch.randn(batch_size, in_features, dtype=torch.bfloat16)
        qweight = make_qweight(out_features, in_features, qweight_type)

        result = fused_mul_mat_gguf(x, qweight, qweight_type)

        # MMQ kernel should have been called
        assert mock_kernel_module["ggml_mul_mat_a8"] is not None
        assert mock_kernel_module["ggml_mul_mat_vec_a8"] is None
        assert mock_kernel_module["ggml_dequantize"] is None

        call_info = mock_kernel_module["ggml_mul_mat_a8"]
        assert call_info["x_shape"] == (batch_size, in_features)
        assert call_info["qweight_type"] == qweight_type
        assert call_info["out_features"] == out_features
        assert result.shape == (batch_size, out_features)


class TestIQuantDispatch:
    """Test I-quant dispatch: they have MMVQ but NO MMQ kernels."""

    @pytest.mark.parametrize("qweight_type", [
        GGML_IQ2_S,  # enum=22
        GGML_IQ1_M,  # enum=29
    ])
    def test_iquant_small_batch_takes_mmvq(self, mock_kernel_module, qweight_type):
        """I-quants with batch <= _MMVQ_SAFE call ggml_mul_mat_vec_a8."""
        out_features = 4096
        in_features = 4096
        batch_size = 1  # Small batch

        x = torch.randn(batch_size, in_features, dtype=torch.bfloat16)
        qweight = make_qweight(out_features, in_features, qweight_type)

        result = fused_mul_mat_gguf(x, qweight, qweight_type)

        # MMVQ (GEMV) kernel should have been called
        assert mock_kernel_module["ggml_mul_mat_vec_a8"] is not None
        assert mock_kernel_module["ggml_mul_mat_a8"] is None
        assert mock_kernel_module["ggml_dequantize"] is None

        call_info = mock_kernel_module["ggml_mul_mat_vec_a8"]
        assert call_info["qweight_type"] == qweight_type
        assert result.shape == (batch_size, out_features)

    @pytest.mark.parametrize("qweight_type", [
        GGML_IQ2_S,  # enum=22
        GGML_IQ1_M,  # enum=29
    ])
    def test_iquant_large_batch_takes_dequant_path(self, mock_kernel_module, qweight_type):
        """I-quants have no MMQ kernel, so large batch falls back to dequant + matmul.

        Rationale: I-quants have MMVQ and dequant kernels but no MMQ kernel. Routing them
        to ggml_mul_mat_a8 (which doesn't support them) would return uninitialized memory.
        Instead, we dequantize and use plain torch matmul.
        """
        out_features = 4096
        in_features = 4096
        batch_size = _MMVQ_SAFE + 1  # Large batch (above threshold)

        x = torch.randn(batch_size, in_features, dtype=torch.bfloat16)
        qweight = make_qweight(out_features, in_features, qweight_type)

        result = fused_mul_mat_gguf(x, qweight, qweight_type)

        # Dequant path should have been used
        assert mock_kernel_module["ggml_dequantize"] is not None
        assert mock_kernel_module["ggml_mul_mat_a8"] is None
        assert mock_kernel_module["ggml_mul_mat_vec_a8"] is None

        call_info = mock_kernel_module["ggml_dequantize"]
        assert call_info["qweight_type"] == qweight_type
        assert call_info["out_features"] == out_features
        assert call_info["in_features"] == in_features
        assert result.shape == (batch_size, out_features)


class TestUnsupportedTypes:
    """Test error handling for unsupported types."""

    def test_unknown_type_raises(self, mock_kernel_module):
        """Unsupported type (e.g. 99) raises NotImplementedError with type info."""
        out_features = 4096
        in_features = 4096
        batch_size = 32
        unsupported_type = 99

        x = torch.randn(batch_size, in_features, dtype=torch.bfloat16)
        qweight = torch.randint(0, 256, (out_features, 128), dtype=torch.uint8)

        with pytest.raises(NotImplementedError) as excinfo:
            fused_mul_mat_gguf(x, qweight, unsupported_type)

        # Error message should contain the type (either the enum value or a name)
        error_msg = str(excinfo.value)
        assert "99" in error_msg or "unsupported" in error_msg.lower()


class TestMMVQThresholdTracking:
    """Test that _MMVQ_SAFE constant is properly used in dispatch."""

    def test_mmvq_threshold_boundary(self, mock_kernel_module):
        """At batch == _MMVQ_SAFE, MMVQ path is taken; at +1, MMQ path is taken."""
        out_features = 4096
        in_features = 4096
        qweight_type = GGML_Q4_K

        # Test at threshold: should use MMVQ
        x_at_threshold = torch.randn(_MMVQ_SAFE, in_features, dtype=torch.bfloat16)
        qweight = make_qweight(out_features, in_features, qweight_type)

        result = fused_mul_mat_gguf(x_at_threshold, qweight, qweight_type)
        assert mock_kernel_module["ggml_mul_mat_vec_a8"] is not None

        # Reset call log
        mock_kernel_module["ggml_mul_mat_vec_a8"] = None
        mock_kernel_module["ggml_mul_mat_a8"] = None

        # Test above threshold: should use MMQ
        x_above_threshold = torch.randn(_MMVQ_SAFE + 1, in_features, dtype=torch.bfloat16)
        result = fused_mul_mat_gguf(x_above_threshold, qweight, qweight_type)
        assert mock_kernel_module["ggml_mul_mat_a8"] is not None
        assert mock_kernel_module["ggml_mul_mat_vec_a8"] is None
