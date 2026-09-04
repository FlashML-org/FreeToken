"""Graph-safe Q8_0 KV row store and its scalar reference.

Rows are quantized independently: one K/V head at one token slot. No row crosses a
head, page, layer, or K/V slab. The scale is FP16 and the payload is signed INT8,
matching llama.cpp b10434's ``quantize_row_q8_0_ref`` contract.
"""

from __future__ import annotations

import torch


Q8_BLOCK = 32


def _round_half_away_from_zero(value: torch.Tensor) -> torch.Tensor:
    """C ``roundf`` semantics, unlike torch.round's ties-to-even behavior."""
    return torch.where(value >= 0, torch.floor(value + 0.5), torch.ceil(value - 0.5))


def quantize_row_q8_0_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(payload[int8], scales[float16])`` for rows in ``[..., D]``."""
    if x.ndim < 1 or x.shape[-1] % Q8_BLOCK:
        raise ValueError(f"q8_0 requires last dimension divisible by 32, got {tuple(x.shape)}")
    source = x.to(torch.float32).reshape(-1, x.shape[-1])
    blocks = source.reshape(source.shape[0], -1, Q8_BLOCK)
    amax = blocks.abs().amax(dim=-1)
    scales = amax / 127.0
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quant = _round_half_away_from_zero(blocks / safe_scales.unsqueeze(-1))
    quant = quant.clamp(-127, 127).to(torch.int8)
    quant[amax == 0] = 0
    return quant.reshape_as(source), scales.to(torch.float16)


def validate_unique_destinations(indices: torch.Tensor) -> None:
    """Reject racy physical destinations before graph capture."""
    flat = indices.reshape(-1)
    if flat.ndim != 1 or flat.numel() == 0:
        raise ValueError("q8_0 store indices must be a non-empty 1-D tensor")
    if not flat.dtype in (torch.int32, torch.int64):
        raise TypeError(f"q8_0 store indices must be int32/int64, got {flat.dtype}")
    if torch.unique(flat).numel() != flat.numel():
        raise ValueError("q8_0 store has duplicate physical (page,offset) destinations")


def _store_reference(
    payload: torch.Tensor,
    scales: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
) -> None:
    quant, row_scales = quantize_row_q8_0_ref(values)
    payload.index_copy_(0, indices.to(torch.long), quant.reshape(-1, values.shape[-2], values.shape[-1]))
    scales.index_copy_(0, indices.to(torch.long), row_scales.reshape(-1, values.shape[-2], values.shape[-1] // Q8_BLOCK))


def _store_triton(
    payload: torch.Tensor,
    scales: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
) -> None:
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        x_ptr, out_ptr, scale_ptr, idx_ptr,
        sx0, sx1, sx2, so0, so1, so2, ss0, ss1, ss2,
        BLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        heads = tl.num_programs(0)
        token = row // heads
        head = row - token * heads
        slot = tl.load(idx_ptr + token)
        offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + token * sx0 + head * sx1 + offs * sx2).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=0)
        d = amax / 127.0
        safe_d = tl.where(d == 0, 1.0, d)
        scaled = x / safe_d
        q = tl.where(scaled >= 0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        q = tl.where(amax == 0, 0.0, q)
        tl.store(out_ptr + slot * so0 + head * so1 + offs * so2, q.to(tl.int8))
        tl.store(scale_ptr + slot * ss0 + head * ss1 + block * ss2, d.to(tl.float16))

    n, heads, dim = values.shape
    kernel[(n * heads, triton.cdiv(dim, Q8_BLOCK))](
        values, payload, scales, indices,
        values.stride(0), values.stride(1), values.stride(2),
        payload.stride(0), payload.stride(1), payload.stride(2),
        scales.stride(0), scales.stride(1), scales.stride(2),
        BLOCKS=dim // Q8_BLOCK,
        BLOCK_SIZE=Q8_BLOCK,
        num_warps=1,
    )


def store_q8_cache(
    *,
    k_payload: torch.Tensor,
    v_payload: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Quantize K and V directly into stable payload/scale page buffers."""
    if k.shape != v.shape or k.ndim != 3:
        raise ValueError(f"q8_0 K/V must be matching [tokens, heads, dim], got {k.shape}/{v.shape}")
    if not k.is_floating_point() or not v.is_floating_point():
        raise TypeError("q8_0 store inputs must be floating-point K/V tensors")
    if k.shape[-1] % Q8_BLOCK:
        raise ValueError(f"q8_0 head_dim must be divisible by 32, got {k.shape[-1]}")
    if indices.numel() != k.shape[0]:
        raise ValueError(f"q8_0 indices length {indices.numel()} != token count {k.shape[0]}")
    # ``torch.unique`` is an eager validation barrier and cannot run inside HIP graph
    # capture. Batch metadata validates destinations before capture; eager calls retain
    # the loud duplicate check here.
    capturing = False
    if indices.is_cuda:
        try:
            capturing = bool(torch.cuda.is_current_stream_capturing())
        except (AttributeError, RuntimeError):
            capturing = False
    if not capturing:
        validate_unique_destinations(indices)
    for payload, scales in ((k_payload, k_scales), (v_payload, v_scales)):
        if payload.dtype != torch.int8 or scales.dtype != torch.float16:
            raise TypeError("q8_0 cache buffers must be int8 payload and float16 scales")
        if payload.ndim != 3 or scales.shape != (*payload.shape[:2], payload.shape[-1] // Q8_BLOCK):
            raise ValueError("q8_0 cache payload/scale geometry mismatch")
    if not k.is_cuda:
        _store_reference(k_payload, k_scales, indices, k)
        _store_reference(v_payload, v_scales, indices, v)
        return
    _store_triton(k_payload, k_scales, indices, k)
    _store_triton(v_payload, v_scales, indices, v)


__all__ = [
    "Q8_BLOCK",
    "quantize_row_q8_0_ref",
    "store_q8_cache",
    "validate_unique_destinations",
]
