"""Quant-aware dense-linear factories, shared by the models that serve quantized
dense projections (qwen3_5_moe, muse_glimmer).

Maps the model's quant config (``expert_quant`` for the dense MLP / shared-expert path,
``attn_quant`` for attention + GatedDeltaNet projections; block-fp8 may be declared by
either, since a modelopt MIXED_PRECISION checkpoint block-quantizes only the dense side) to the right ``BaseOP`` linear:
block-FP8, per-tensor-FP8 and NVFP4 implementations live under ``freetoken.kernel.triton``;
the bf16 fallback is the framework's TP-aware ``freetoken.layers``. Only the *dispatch*
(config -> layer class) lives here.
"""

from __future__ import annotations


def make_col_merged_quant(expert_quant: str, attn_quant: str, in_f: int,
                          output_sizes: list[int], has_bias: bool = False):
    """Column-merged linear for a dense projection: block-fp8 / per-tensor-fp8 / nvfp4 / bf16."""
    if "fp8_block" in (expert_quant, attn_quant):
        from freetoken.kernel.triton.fp8_block_linear import Fp8BlockColMerged

        return Fp8BlockColMerged(in_f, output_sizes, has_bias)
    if attn_quant == "fp8_pertensor":
        from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged

        return Fp8PerTensorColMerged(in_f, output_sizes, has_bias)
    if attn_quant == "nvfp4":  # compressed-tensors W4A16 attention (q/k/v fused)
        from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged

        return Nvfp4DenseColMerged(in_f, output_sizes, has_bias)
    from freetoken.layers import LinearColParallelMerged

    return LinearColParallelMerged(in_f, output_sizes, has_bias=has_bias)


def make_replicated_quant(expert_quant: str, attn_quant: str, in_f: int, out_f: int,
                          has_bias: bool = False):
    """Replicated linear for a dense projection: block-fp8 / per-tensor-fp8 / nvfp4 / bf16."""
    if "fp8_block" in (expert_quant, attn_quant):
        from freetoken.kernel.triton.fp8_block_linear import Fp8BlockLinear

        return Fp8BlockLinear(in_f, out_f, has_bias)
    if attn_quant == "fp8_pertensor":
        from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorLinear

        return Fp8PerTensorLinear(in_f, out_f, has_bias)
    if attn_quant == "nvfp4":  # compressed-tensors W4A16 attention o_proj / GDN out_proj
        from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseLinear

        return Nvfp4DenseLinear(in_f, out_f, has_bias)
    from freetoken.layers import LinearReplicated

    return LinearReplicated(in_f, out_f, has_bias=has_bias)


def make_replicated(config, in_f: int, out_f: int, has_bias: bool = False):
    """Config-driven replicated linear: ``Fp8BlockLinear`` under block-fp8, ``Fp8PerTensorLinear``
    under per-tensor-fp8 attention, ``Nvfp4DenseLinear`` under nvfp4, else ``LinearReplicated``."""
    return make_replicated_quant(
        getattr(config, "expert_quant", "none"), getattr(config, "attn_quant", "none"),
        in_f, out_f, has_bias,
    )


def make_col_merged(config, in_f: int, output_sizes: list[int], has_bias: bool = False):
    """Config-driven column-merged linear: ``Fp8BlockColMerged`` under block-fp8,
    ``Fp8PerTensorColMerged`` under per-tensor-fp8 attention, ``Nvfp4DenseColMerged`` under
    nvfp4, else ``LinearColParallelMerged``."""
    return make_col_merged_quant(
        getattr(config, "expert_quant", "none"), getattr(config, "attn_quant", "none"),
        in_f, output_sizes, has_bias,
    )


__all__ = [
    "make_col_merged_quant",
    "make_replicated_quant",
    "make_replicated",
    "make_col_merged",
]
