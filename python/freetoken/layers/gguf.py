"""Native-GGUF quantized layers: weights stay in their packed block layout and are
dequantized *inside* the borrowed llama.cpp kernels (no bf16 copy ever materialized).

Mirrors vLLM/sglang's ``GGUFLinearMethod`` / ``GGUFEmbeddingMethod`` dispatch, ported
onto FreeToken's ``BaseOP``. FreeToken keeps fused projections (qkv, gate_up) as a
single tensor: because Q4_0/K-quants pack each *output row* independently over the
input dim, the loader can concatenate the per-shard packed rows along dim 0 (they
share an input dim, hence the same ``row_bytes``), so a fused layer is still one
``[out, row_bytes]`` qweight -- no per-shard padding bookkeeping needed.

TP is assumed to be 1 (the gemma4 GGUF path restricts to TP=1, like the HF path).
"""

from __future__ import annotations

import functools
import os

import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_NAME,
    GGML_IQ1_S,
    GGML_IQ2_S,
    GGML_IQ2_XXS,
    GGML_IQ3_XXS,
    GGML_IQ4_XS,
    GGML_Q3_K,
    GGML_Q4_0,
    GGML_Q4_K,
    GGML_Q5_K,
    GGML_Q6_K,
    GGML_Q8_0,
    row_bytes,
)

from .base import BaseOP

# ggml type groups for kernel dispatch (subset we build kernels for).
_UNQUANTIZED = {GGML_F32, GGML_F16, GGML_BF16}
# standard + k-quants: both an MMVQ (small-batch GEMV) and MMQ (large-batch) kernel exist.
_MMVQ = {
    GGML_Q4_0, GGML_Q8_0, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K,
    GGML_IQ1_S, GGML_IQ2_S, GGML_IQ2_XXS, GGML_IQ3_XXS, GGML_IQ4_XS,
}
# The vendored CUDA MMQ switch covers the standard + K-quants only (no IQ cases);
# IQ types take the dequant fallback for large batches.
_MMQ = {GGML_Q4_0, GGML_Q8_0, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K}
_DEQUANT = {
    GGML_Q4_0, GGML_Q8_0, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K,
    GGML_IQ1_S, GGML_IQ2_S, GGML_IQ2_XXS, GGML_IQ3_XXS, GGML_IQ4_XS,
}

# Below this token count, the MMVQ GEMV kernel wins (matches vLLM's heuristic).
_MMVQ_SAFE = 6

# The donated GGUF MMQ kernel does not use tensor cores.  For real prefill
# batches it is much faster to dequantize the comparatively small dense weight
# once and hand the matrix product to cuBLAS.  On sm_89 with Ornith's largest
# Q6_K projection the crossover is 32 rows (0.40 ms vs 0.64 ms) and at 8192
# rows it is 6.1 ms vs 132 ms.  This is transient -- packed weights remain the
# persistent representation, so long-context KV/expert capacity is unchanged.
_DEQUANT_GEMM_MIN_ROWS = 32
# On sm_120 (RTX 5080) the crossover moves down: dequant wins from 24 rows on
# Ornith's Q4_K attention shapes (0.0645 vs 0.0778 ms) while MMQ still wins the
# Q6_K lm_head at 16 (2.08 vs 2.93 ms), so 24 is the safe arch-wide value.
_DEQUANT_GEMM_MIN_ROWS_SM120 = 24


def dequant_gemm_min_rows(compute_capability: tuple[int, int] | None) -> int:
    """Row count from which transient dequant+cuBLAS beats the DP4A MMQ kernel."""
    if compute_capability is not None and compute_capability >= (12, 0):
        return _DEQUANT_GEMM_MIN_ROWS_SM120
    return _DEQUANT_GEMM_MIN_ROWS


@functools.lru_cache(maxsize=8)
def _device_capability(device_index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


@functools.lru_cache(maxsize=None)
def _mma_mmq_ok() -> bool:
    """JIT-build the upstream int8-MMA MMQ extension once; fall back on failure.

    ``FREETOKEN_GGUF_DISABLE_MMA=1`` forces the DP4A/dequant path -- an escape
    hatch if the JIT build misbehaves on a given toolchain, and the A/B control
    for benchmarking the port.
    """
    import logging

    log = logging.getLogger(__name__)
    if os.environ.get("FREETOKEN_GGUF_DISABLE_MMA", "") not in ("", "0"):
        log.info("int8-MMA MMQ disabled by FREETOKEN_GGUF_DISABLE_MMA")
        return False

    from freetoken.kernel.gguf import _mma_module

    try:
        _mma_module()
    except Exception as exc:  # build/toolchain failure -> DP4A/dequant path
        log.warning(
            "int8-MMA MMQ extension unavailable, using DP4A/dequant fallback: %s", exc
        )
        return False
    # One-time, greppable proof of which GEMM path a live server actually runs.
    log.info("int8-MMA MMQ ACTIVE for Q4_K/Q6_K (upstream llama.cpp mul_mat_q)")
    return True


def _use_mma_mmq(quant_type: int, capability: tuple[int, int] | None) -> bool:
    """int8-MMA MMQ replaces both the DP4A MMQ band and the dequant+cuBLAS band.

    Measured on sm_120 (RTX 5080, real Ornith tensors): faster than BOTH at
    every batch size >= 4 -- 8192-row Q4_K attn_q 1.79 ms vs dequant 2.40 vs
    DP4A 22.9; Q6_K lm_head @2048 tokens 17.5 vs 19.2 vs 262. Gated to sm_120
    for now (upstream supports sm_75+, but only Blackwell is measured here);
    MMVQ keeps the <= _MMVQ_SAFE decode band.
    """
    if capability is None or capability < (12, 0):
        return False
    from freetoken.kernel.gguf import mma_mmq_supported

    return mma_mmq_supported(quant_type) and _mma_mmq_ok()


def fused_mul_mat_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    """y = x @ dequant(qweight).T, dispatched by batch size and quant type."""
    from freetoken.kernel.gguf import (
        ggml_dequantize,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )

    out_features = qweight.shape[0]
    if x.shape[0] == 0:
        return x.new_empty((0, out_features))
    if qweight_type in _UNQUANTIZED:
        return x @ qweight.T
    if x.shape[0] <= _MMVQ_SAFE and qweight_type in _MMVQ:
        return ggml_mul_mat_vec_a8(qweight, x, qweight_type, out_features)
    if qweight_type in _MMQ:
        capability = _device_capability(x.device.index) if x.is_cuda else None
        if _use_mma_mmq(qweight_type, capability):
            from freetoken.kernel.gguf import ggml_mul_mat_a8_mma

            return ggml_mul_mat_a8_mma(qweight, x, qweight_type, out_features).to(x.dtype)
        if x.shape[0] >= dequant_gemm_min_rows(capability):
            block, type_size = BLOCK_SHAPE[qweight_type]
            in_features = qweight.shape[1] // type_size * block
            weight = ggml_dequantize(
                qweight, qweight_type, out_features, in_features, x.dtype
            )
            return x @ weight.T
        return ggml_mul_mat_a8(qweight, x, qweight_type, out_features)
    if qweight_type in _DEQUANT:
        block, type_size = BLOCK_SHAPE[qweight_type]
        in_features = qweight.shape[1] // type_size * block
        weight = ggml_dequantize(qweight, qweight_type, out_features, in_features, x.dtype)
        return x @ weight.T
    raise NotImplementedError(f"unsupported GGUF type {GGML_NAME.get(qweight_type, qweight_type)}")


class GGUFLinear(BaseOP):
    """Linear whose weight is a native GGUF block-quantized ``[out, row_bytes]`` tensor."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: int,
        has_bias: bool = False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self._quant_type = quant_type
        self.qweight = torch.empty(out_features, row_bytes(in_features, quant_type), dtype=torch.uint8)
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = fused_mul_mat_gguf(x, self.qweight, self._quant_type)
        if self.bias is not None:
            out = out + self.bias
        return out


class GGUFEmbedding(BaseOP):
    """Vocab embedding stored as a native GGUF block-quantized table.

    The full table is never dequantized: only the looked-up rows are gathered (in
    packed form) and dequantized per lookup, matching vLLM's ``_apply_gguf_embedding``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        quant_type: int,
        embed_scale: float | None = None,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._quant_type = quant_type
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8
        )
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.gguf import ggml_dequantize

        flat = x.flatten()
        rows = self.qweight.index_select(0, flat)  # [n, row_bytes] packed
        y = ggml_dequantize(rows, self._quant_type, flat.shape[0], self.embedding_dim, torch.bfloat16)
        y = y.view(*x.shape, self.embedding_dim)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(self._embed_scale, dtype=y.dtype, device=y.device)
            y = y * self._embed_scale_t
        return y


__all__ = ["GGUFLinear", "GGUFEmbedding", "fused_mul_mat_gguf"]
