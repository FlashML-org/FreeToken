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

from collections.abc import Sequence

import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_NAME,
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
_MMVQ = {GGML_Q4_0, GGML_Q8_0, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K}
_MMQ = {GGML_Q4_0, GGML_Q8_0, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K}
_DEQUANT = {GGML_Q4_0, GGML_Q8_0, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K}

# Below this token count, the MMVQ GEMV kernel wins (matches vLLM's heuristic).
_MMVQ_SAFE = 6


def fused_mul_mat_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    """y = x @ dequant(qweight).T, dispatched by batch size and quant type."""
    from freetoken.kernel.gguf import (
        ggml_dequantize,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
        gguf_dispatch,
        gguf_runtime_metadata,
    )

    out_features = qweight.shape[0]
    if x.shape[0] == 0:
        return x.new_empty((0, out_features))
    if qweight_type in _UNQUANTIZED:
        return x @ qweight.T
    if x.is_cuda and qweight_type in _MMVQ:
        runtime = gguf_runtime_metadata()
        dispatch = gguf_dispatch(
            "dense", qweight_type, out_features, x.shape[1], x.shape[0], runtime["arch"]
        )
        if dispatch["implementation"] == "ggml_mul_mat_vec_a8":
            return ggml_mul_mat_vec_a8(qweight, x, qweight_type, out_features)
        if dispatch["implementation"] == "ggml_mul_mat_a8":
            return ggml_mul_mat_a8(qweight, x, qweight_type, out_features)
        raise RuntimeError(
            "GGUF dense dispatch unavailable: "
            f"{dispatch.get('reason') or 'unknown reason'}"
        )
    if x.shape[0] <= _MMVQ_SAFE and qweight_type in _MMVQ:
        return ggml_mul_mat_vec_a8(qweight, x, qweight_type, out_features)
    if qweight_type in _MMQ:
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


class GGUFMergedLinear(GGUFLinear):
    """Output-concatenated GGUF linear.

    Packed rows can be concatenated when projections share quant type. GGUF exports
    may use different native types for each output slice; retain those slices separately
    and concatenate computed outputs, avoiding lossy dequantize/requantize conversion.
    """

    def __init__(
        self,
        in_features: int,
        output_sizes: Sequence[int],
        quant_type: int | Sequence[int],
        has_bias: bool = False,
    ):
        self.output_sizes = tuple(int(size) for size in output_sizes)
        if not self.output_sizes or any(size <= 0 for size in self.output_sizes):
            raise ValueError(f"output_sizes must contain positive values, got {output_sizes}")
        if isinstance(quant_type, int):
            types = (quant_type,) * len(self.output_sizes)
        else:
            types = tuple(int(item) for item in quant_type)
        if len(types) != len(self.output_sizes):
            raise ValueError("quant_type count must match output_sizes count")
        self.quant_types = types
        self._mixed = len(set(types)) != 1
        if not self._mixed:
            super().__init__(in_features, sum(self.output_sizes), types[0], has_bias=has_bias)
            return
        self.in_features = in_features
        self.out_features = sum(self.output_sizes)
        self._quant_type = None
        self.bias = torch.empty(self.out_features) if has_bias else None
        for i, (size, qtype) in enumerate(zip(self.output_sizes, types)):
            setattr(self, f"qweight_{i}", torch.empty(
                size, row_bytes(in_features, qtype), dtype=torch.uint8
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._mixed:
            return super().forward(x)
        parts = [
            fused_mul_mat_gguf(x, getattr(self, f"qweight_{i}"), qtype)
            for i, qtype in enumerate(self.quant_types)
        ]
        out = torch.cat(parts, dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out


class GGUFLMHead(BaseOP):
    """GGUF LM head with optional tied packed embedding and last-token gather."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        quant_type: int,
        tied_embedding: GGUFEmbedding | None = None,
    ):
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self._quant_type = int(quant_type)
        self.tied_embedding = tied_embedding
        self.qweight = (
            None
            if tied_embedding is not None
            else torch.empty(num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8)
        )

    def state_dict(self, *, prefix: str = "", result=None):
        if self.tied_embedding is not None:
            return result if result is not None else {}
        result = result if result is not None else {}
        result[f"{prefix}.qweight" if prefix else "qweight"] = self.qweight
        return result

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False):
        if self.tied_embedding is not None:
            state_dict.pop(f"{prefix}.weight" if prefix else "weight", None)
            state_dict.pop(f"{prefix}.qweight" if prefix else "qweight", None)
            return
        key = f"{prefix}.qweight" if prefix else "qweight"
        self.qweight = state_dict.pop(key)
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def forward(self, x: torch.Tensor, last_token_indices: torch.Tensor | None = None) -> torch.Tensor:
        if last_token_indices is None:
            try:
                from freetoken.core import get_global_ctx

                batch = get_global_ctx().batch
                if batch.is_prefill:
                    last_token_indices = batch.attn_metadata.get_last_indices(batch.size)
            except (AttributeError, RuntimeError):
                # Direct layer tests and non-engine callers do not have a global batch.
                pass
        if last_token_indices is not None:
            x = x[last_token_indices].contiguous()
        qweight = self.tied_embedding.qweight if self.tied_embedding is not None else self.qweight
        return fused_mul_mat_gguf(x, qweight, self._quant_type)


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


__all__ = [
    "GGUFLinear",
    "GGUFMergedLinear",
    "GGUFEmbedding",
    "GGUFLMHead",
    "fused_mul_mat_gguf",
]
