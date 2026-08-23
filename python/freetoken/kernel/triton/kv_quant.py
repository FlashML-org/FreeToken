"""Quantizing store into an 8-bit KV pool.

The unquantized path stores K/V with ``kernel/store.py``'s CUDA kernel, which is a
pure byte copy parameterized by element size. Quantized storage has to compute a scale
per block of :data:`~freetoken.kvcache.quant.BLOCK` elements along ``head_dim`` on the
way in, so it gets its own kernel here.

One program handles one ``(token, kv_head)`` pair: it loads that head's ``head_dim``
values as a ``[head_dim // BLOCK, BLOCK]`` tile, reduces max-abs along the block, and
writes the quantized values plus one scale per block. K and V are done in the same
program -- they share the token's slot index and the tile geometry, so doing both
halves the launch count and the index math.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _store_kv_quant_kernel(
    k_ptr,  # [tokens, heads, D] source, compute dtype
    v_ptr,
    kc_ptr,  # [slots, heads, D] destination, storage dtype
    vc_ptr,
    ks_ptr,  # [slots, heads, D // BLOCK] scales, fp16
    vs_ptr,
    indices_ptr,  # [tokens] destination slot per token
    stride_kt,
    stride_kh,
    stride_ct,
    stride_ch,
    stride_st,
    stride_sh,
    MAX_MAG: tl.constexpr,
    IS_INT: tl.constexpr,
    BLOCK: tl.constexpr,
    NBLOCK: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(indices_ptr + tok).to(tl.int64)

    # [NBLOCK, BLOCK] tile over head_dim: rows are quant blocks, columns the elements
    # sharing one scale.
    offs = tl.arange(0, NBLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    scale_offs = tl.arange(0, NBLOCK)

    for is_v in tl.static_range(2):
        src_ptr = v_ptr if is_v else k_ptr
        dst_ptr = vc_ptr if is_v else kc_ptr
        sc_ptr = vs_ptr if is_v else ks_ptr

        x = tl.load(src_ptr + tok * stride_kt + head * stride_kh + offs).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=1)
        # An all-zero block quantizes to zeros under any positive scale; 1.0 keeps the
        # division finite.
        scale = tl.where(amax > 0, amax / MAX_MAG, 1.0)
        # Round to the stored precision before dividing, so the value written here and
        # the value the attention kernels read back are scaled by the identical number.
        scale = scale.to(sc_ptr.dtype.element_ty).to(tl.float32)
        # div_rn, not `/`: the plain operator is free to lower to a reciprocal multiply,
        # which disagrees with the torch reference on values sitting exactly between two
        # quantization steps. IEEE round-to-nearest divide makes the two bit-identical.
        q = tl.math.div_rn(x, scale[:, None])
        if IS_INT:
            # Round half away from zero (what GGUF's Q8_0 does), then clamp -- the
            # float->int cast truncates.
            q = tl.where(q >= 0, tl.floor(q + 0.5), tl.ceil(q - 0.5))
            q = tl.minimum(tl.maximum(q, -MAX_MAG), MAX_MAG)

        tl.store(
            dst_ptr + slot * stride_ct + head * stride_ch + offs,
            q.to(dst_ptr.dtype.element_ty),
        )
        tl.store(
            sc_ptr + slot * stride_st + head * stride_sh + scale_offs,
            scale.to(sc_ptr.dtype.element_ty),
        )


def store_kv_quant(
    k_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_cache: torch.Tensor,
    v_scale: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec,
) -> None:
    """Quantize ``k``/``v`` ``[tokens, heads, D]`` into the pool slots ``indices``.

    ``k_cache``/``v_cache`` are ``[slots, heads, D]`` in the spec's storage dtype and
    ``k_scale``/``v_scale`` ``[slots, heads, D // BLOCK]`` in fp16.
    """
    from freetoken.kvcache.quant import BLOCK

    num_tokens, num_heads, head_dim = k.shape
    if num_tokens == 0:
        return
    assert head_dim % BLOCK == 0, f"head_dim {head_dim} not a multiple of {BLOCK}"
    assert k_cache.shape[1:] == (num_heads, head_dim), (
        f"cache head geometry {tuple(k_cache.shape[1:])} != source {(num_heads, head_dim)}"
    )
    _store_kv_quant_kernel[(num_tokens, num_heads)](
        k,
        v,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        indices,
        k.stride(0),
        k.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        MAX_MAG=spec.max_magnitude,
        IS_INT=spec.is_integer,
        BLOCK=BLOCK,
        NBLOCK=head_dim // BLOCK,
        num_warps=4,
    )


__all__ = ["store_kv_quant"]
