"""Native-GGUF quantized layers: weights stay in their packed block layout and are
dequantized *inside* the borrowed llama.cpp CUDA kernels -- either fused into the matmul
(MMVQ/MMQ) or, for types with no MMQ kernel, by an explicit ``ggml_dequantize`` pass.

Mirrors vLLM/sglang's ``GGUFLinearMethod`` / ``GGUFEmbeddingMethod`` dispatch, ported
onto FreeToken's ``BaseOP``. FreeToken keeps fused projections (qkv, gate_up) as a
single tensor: because Q4_0/K-quants pack each *output row* independently over the
input dim, the loader can concatenate the per-shard packed rows along dim 0 (they
share an input dim, hence the same ``row_bytes``), so a fused layer is still one
``[out, row_bytes]`` qweight -- no per-shard padding bookkeeping needed.

**Merged vs. plain fused projections**:

When all output parts share the same quant type (the common case in gemma4), a plain
``GGUFLinear`` with concatenated packed rows is valid and efficient -- one kernel launch
dequantizes and multiplies. When parts use different quant types (as in Ornith's IQ3_M
checkpoint, where qkv_proj mixes IQ3_S and Q4_K), row_bytes differs per part, so torch.cat
would produce garbage. ``GGUFMergedLinear`` instead materializes the output of each part
separately via ``fused_mul_mat_gguf`` and concatenates the results along dim=-1 (equivalent
to the GEMM because all parts read the same input: ``cat([x @ W1.T, x @ W2.T]) == x @ cat([W1, W2], 0).T``).

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
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_NAME,
    GGML_UNQUANTIZED,
    MMQ_TYPES,
    MMVQ_TYPES,
    row_bytes,
)

# ggml type -> the dtype its raw bytes represent. Only the unquantized types appear here;
# everything else goes through a dequant kernel.
_UNQUANTIZED_DTYPE = {
    GGML_F32: torch.float32,
    GGML_F16: torch.float16,
    GGML_BF16: torch.bfloat16,
}

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
        # GGUFLinear/GGUFEmbedding store every type in a uint8 buffer of row_bytes width,
        # including the unquantized ones, where "packed" just means the raw F32/F16/BF16
        # bytes. Those must be reinterpreted before the matmul: multiplying the byte view
        # directly gives an in_features of row_bytes (2x too wide for F16) and fails with
        # "mat1 and mat2 shapes cannot be multiplied". A checkpoint only reaches this path
        # when it stores a projection unquantized -- Apodex-1.1-mini ships output.weight as
        # F16, which is how this surfaced; models whose lm_head is Q6_K never hit it.
        w = qweight
        if w.dtype == torch.uint8:
            w = w.view(_UNQUANTIZED_DTYPE[qweight_type])
        # Cast the ACTIVATION, not the weight. Converting the weight would copy the whole
        # matrix on every call -- about 1 GB per forward for a 248k-vocab lm_head -- and
        # allocating that during CUDA graph capture fails outright. x is [tokens, hidden],
        # so casting it is negligible, and computing in the stored precision is what
        # llama.cpp does for these tensors anyway.
        return (x.to(w.dtype) @ w.T).to(x.dtype)
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


class GGUFLMHead(GGUFLinear):
    """LM head over a native GGUF ``output.weight`` (untied embeddings).

    Identical to ``GGUFLinear`` except that during prefill it keeps only the last position
    of each sequence, exactly as ``ParallelLMHead`` (layers/embedding.py) and
    ``GGUFTiedLMHead`` (models/gemma4/gguf.py) already do.

    This is not an optimization, it is a memory correctness issue. Logits are
    [tokens, vocab], so on a large-vocabulary model the full-prefill tensor is enormous:
    Ornith-1.5's vocab is 248,320, which in bf16 is 486 KiB of logits PER TOKEN. A
    1,800-token prompt therefore asks for a single 894 MB allocation, which is more than the
    free VRAM left on an 8 GB card after weights and caches, and prefill dies with
    "CUDA driver error: device not ready" while decode is completely unaffected. Only the
    last position of each sequence is ever sampled, so every other row was computed and
    thrown away.

    The dense path never hit this because ``ParallelLMHead`` slices; the bug appears only
    when a GGUF checkpoint has untied embeddings and the head is swapped for a generic
    quantized Linear, which has no reason to know it is the head.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return super().forward(x)


class GGUFMergedLinear(BaseOP):
    """Merged linear projection with parts that have different quant types.

    Used when fusing output-parallel projections (qkv, gate_up) whose parts use different
    quantization types. Unlike GGUFLinear (which concatenates packed rows along dim 0 and
    requires all parts to share row_bytes), GGUFMergedLinear materializes the output of
    each part separately via fused_mul_mat_gguf, then concatenates the results.

    Mathematically equivalent to a single GEMM, since all parts read the same input x:
    cat([x @ W1.T, x @ W2.T]) == x @ cat([W1, W2], 0).T
    (source: llama.cpp's iq*_m mixed-quant strategy).
    """

    def __init__(
        self,
        in_features: int,
        output_sizes: list[int],
        quant_types: list[int],
        has_bias: bool = False,
    ):
        """Initialize a merged linear projection.

        Args:
            in_features: Input feature dimension (shared by all parts).
            output_sizes: List of output sizes for each part; must all be > 0.
            quant_types: List of GGML quant types, one per part; must match output_sizes length.
            has_bias: Whether to allocate a bias term.

        Raises:
            ValueError: If output_sizes and quant_types lengths do not match, or if any output_size <= 0.
            NotImplementedError: If any quant_type is not supported (not in MMVQ_TYPES or GGML_UNQUANTIZED).
        """
        if len(output_sizes) != len(quant_types):
            raise ValueError(
                f"output_sizes length {len(output_sizes)} != quant_types length {len(quant_types)}"
            )
        if not all(o > 0 for o in output_sizes):
            raise ValueError(f"all output_sizes must be > 0, got {output_sizes}")

        # Validate each quant type is supported.
        for qt in quant_types:
            if qt not in MMVQ_TYPES and qt not in GGML_UNQUANTIZED:
                raise NotImplementedError(
                    f"quant type {GGML_NAME.get(qt, qt)} not in MMVQ_TYPES or GGML_UNQUANTIZED"
                )

        self.in_features = in_features
        self.output_sizes = output_sizes
        self.out_features = sum(output_sizes)
        self._quant_types = quant_types
        self.part_names = []

        # Allocate packed weight buffers: one named tensor per part (qweight_0, qweight_1, ...).
        # Named (not underscore-prefixed) so they are discovered by state_dict.
        for i, (out_size, qt) in enumerate(zip(output_sizes, quant_types)):
            name = f"qweight_{i}"
            self.part_names.append(name)
            setattr(
                self,
                name,
                torch.empty(out_size, row_bytes(in_features, qt), dtype=torch.uint8),
            )

        self.bias = torch.empty(self.out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute each part's output and concatenate along dim=-1.

        Args:
            x: Input tensor of shape [..., in_features].

        Returns:
            Tensor of shape [..., out_features] with parts concatenated along dim=-1.
        """
        parts = []
        for name, qt in zip(self.part_names, self._quant_types):
            qweight = getattr(self, name)
            part_out = fused_mul_mat_gguf(x, qweight, qt)
            parts.append(part_out)

        out = torch.cat(parts, dim=-1)
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
        if self._quant_type in GGML_UNQUANTIZED:
            # Raw value bytes, not blocks: there is no dequant kernel for the unquantized
            # types (ggml_dequantize rejects type 1), so reinterpret the gathered rows.
            y = rows.view(_UNQUANTIZED_DTYPE[self._quant_type]).to(torch.bfloat16)
        else:
            y = ggml_dequantize(rows, self._quant_type, flat.shape[0], self.embedding_dim, torch.bfloat16)
        y = y.view(*x.shape, self.embedding_dim)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(self._embed_scale, dtype=y.dtype, device=y.device)
            y = y * self._embed_scale_t
        return y


def gguf_merged_or_plain(
    in_features: int,
    output_sizes: list[int],
    quant_types: list[int],
    has_bias: bool = False,
) -> GGUFLinear | GGUFMergedLinear:
    """Choose between GGUFLinear (uniform quant types) and GGUFMergedLinear (mixed types).

    When all output parts share the same quant type (the uniform case, common in gemma4),
    return a GGUFLinear with concatenated packed rows -- valid and cheaper since row_bytes
    is identical per part (one kernel launch instead of N).

    When quant types differ (the mixed case, produced by llama.cpp's IQ*_M / Q*_K_M),
    return a GGUFMergedLinear to avoid torch.cat garbage from misaligned row_bytes.

    Args:
        in_features: Input feature dimension.
        output_sizes: List of output sizes for each part.
        quant_types: List of GGML quant types, one per part.
        has_bias: Whether to allocate a bias term.

    Returns:
        GGUFLinear if all quant types are identical, else GGUFMergedLinear.
    """
    if len(set(quant_types)) == 1:
        # Uniform case: all parts use the same quant type.
        # Concatenate packed rows (they share row_bytes) into a single [sum(output_sizes), row_bytes] weight.
        out_features = sum(output_sizes)
        qt = quant_types[0]
        lin = GGUFLinear(in_features, out_features, qt, has_bias=has_bias)
        return lin
    else:
        # Mixed case: parts use different quant types.
        return GGUFMergedLinear(in_features, output_sizes, quant_types, has_bias=has_bias)


__all__ = [
    "GGUFLinear",
    "GGUFMergedLinear",
    "GGUFEmbedding",
    "fused_mul_mat_gguf",
    "gguf_merged_or_plain",
]
