"""Native-GGUF quantized layers: weights stay in their packed block layout and are
dequantized *inside* the borrowed llama.cpp CUDA kernels -- either fused into the matmul
(MMVQ/MMQ) or, for types with no MMQ kernel, by an explicit ``ggml_dequantize`` pass.

Mirrors vLLM/sglang's ``GGUFLinearMethod`` / ``GGUFEmbeddingMethod`` dispatch, ported
onto FreeToken's ``BaseOP``. FreeToken keeps fused projections (qkv, gate_up) as a
single tensor: because Q4_0/K-quants pack each *output row* independently over the
input dim, the loader can concatenate the per-shard packed rows along dim 0 (they
share an input dim, hence the same ``row_bytes``), so a fused layer is still one
``[out, row_bytes]`` qweight -- no per-shard padding bookkeeping needed.

**Matmul dispatch strategy** (4-tier, per fused_mul_mat_gguf):

1. **Unquantized (F32, F16, BF16)**: straight torch matmul ``x @ qweight.T``.
2. **Small-batch quantized (batch <= 6, MMVQ types)**: GEMV kernel via ``ggml_mul_mat_vec_a8``.
3. **Large-batch standard quants (MMQ types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, K-quants)**: MMQ kernel
   via ``ggml_mul_mat_a8``.
4. **Large-batch I-quants (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M)**:
   I-quants have MMVQ and dequant kernels but NO MMQ kernel. Prefill therefore falls back to
   ``ggml_dequantize`` + plain torch matmul. This materializes a transient BF16 copy of the weight
   (cost: ``out_features * in_features * 2 bytes``), which is a real tradeoff for memory-bound
   prefill on large I-quant weights.

TP is assumed to be 1 (the gemma4 GGUF path restricts to TP=1, like the HF path).
"""

from __future__ import annotations

import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    DEQUANT_TYPES,
    GGML_NAME,
    GGML_UNQUANTIZED,
    MMQ_TYPES,
    MMVQ_TYPES,
    row_bytes,
)

from .base import BaseOP

# Below this token count, the MMVQ GEMV kernel wins (matches vLLM's heuristic).
_MMVQ_SAFE = 6


def fused_mul_mat_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    """y = x @ dequant(qweight).T, dispatched by batch size and quant type.

    Dispatch order:
    1. Unquantized (F32/F16/BF16): plain torch matmul
    2. Small-batch quantized (batch <= 6, in MMVQ_TYPES): GEMV kernel
    3. Large-batch standard quants (in MMQ_TYPES): MMQ kernel
    4. Large-batch with I-quants (in DEQUANT_TYPES but not MMQ_TYPES): dequant + torch matmul
    """
    from freetoken.kernel.gguf import (
        ggml_dequantize,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )

    out_features = qweight.shape[0]
    if x.shape[0] == 0:
        return x.new_empty((0, out_features))
    if qweight_type in GGML_UNQUANTIZED:
        return x @ qweight.T
    if x.shape[0] <= _MMVQ_SAFE and qweight_type in MMVQ_TYPES:
        return ggml_mul_mat_vec_a8(qweight, x, qweight_type, out_features)
    if qweight_type in MMQ_TYPES:
        return ggml_mul_mat_a8(qweight, x, qweight_type, out_features)
    if qweight_type in DEQUANT_TYPES:
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
