from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl


_MAX_KV_SPLITS = 8
_MIN_BLOCK_KV = 32


@triton.jit
def _load_kv(
    ptr,
    scale_ptr,
    slot_base,  # BYTE address of slot-head start (i.e. slots * stride_ks + kv_head * stride_kh)
    elem_offsets,  # LOGICAL element positions along head_dim (0..D-1, padded to BLOCK_D)
    elem_mask,
    scale_offsets,
    scale_mask,
    out_dtype: tl.constexpr,
    QUANT: tl.constexpr,
    QBLOCK: tl.constexpr,
    D_ON_ROWS: tl.constexpr,
    LAYOUT: tl.constexpr,  # "q8" | "q4" | "q6"
):
    """Load a K or V tile, dequantizing it when the pool stores quantized values.

    The cache is addressed as ``slot_base + byte_for_elem(elem_offsets)``. The caller
    passes the slot-head byte base (computed from per-slot / per-head strides) and
    the LOGICAL element offsets along head_dim; this function converts elem to byte
    positions internally based on LAYOUT and unpacks. The returned tile has one
    entry per logical element (so callers see the same shape they would for an
    unquantized bf16 load).

    Sub-byte layouts:
      * q4: 2 elements per byte, low nibble = even, high nibble = odd.
      * q6: 16-byte low plane + 8-byte high plane per 32-element block; low 4 bits
        via nibble layout, top 2 bits at bit positions 0, 2, 4, 6 of the hi plane.
    """
    if LAYOUT == "q8":
        # 1 byte per element. slot_base and elem_offsets are already broadcast
        # by the caller (their shapes must match: [BLOCK_N, BLOCK_D] for K with
        # D_ON_ROWS, [BLOCK_D, BLOCK_N] for V).
        vals = tl.load(ptr + slot_base + elem_offsets, mask=elem_mask, other=0.0)
    elif LAYOUT == "q4":
        # 2 elements per byte. byte_offs = elem // 2; is_odd picks the nibble.
        byte_offs = elem_offsets >> 1
        is_odd = (elem_offsets & 1).to(tl.int32)
        packed = tl.load(ptr + slot_base + byte_offs, mask=elem_mask, other=0).to(tl.uint8)
        lo = (packed & 0xF).to(tl.int32)
        hi = ((packed >> 4) & 0xF).to(tl.int32)
        lo = (lo ^ 0x8) - 0x8
        hi = (hi ^ 0x8) - 0x8
        vals = tl.where(is_odd == 0, lo, hi)
    elif LAYOUT == "q6":
        # 32-element blocks: 16-byte low plane + 8-byte high plane.
        block_idx = elem_offsets >> 5
        in_block = elem_offsets & 31
        lo_byte = block_idx * 24 + (in_block >> 1)
        hi_byte = block_idx * 24 + 16 + (in_block >> 2)
        is_odd = (in_block & 1).to(tl.int32)
        hi_pos = ((in_block & 3) * 2).to(tl.int32)

        lo_loaded = tl.load(ptr + slot_base + lo_byte, mask=elem_mask, other=0).to(tl.uint8)
        lo_nib = (lo_loaded & 0xF).to(tl.int32)
        lo_hi = ((lo_loaded >> 4) & 0xF).to(tl.int32)
        lo_val = tl.where(is_odd == 0, lo_nib, lo_hi)

        hi_loaded = tl.load(ptr + slot_base + hi_byte, mask=elem_mask, other=0).to(tl.int32)
        hi_val = ((hi_loaded >> hi_pos) & 0x3)

        raw6 = (lo_val | (hi_val << 4))
        se = (raw6 ^ 0x20) - 0x20
        vals = se
    elif LAYOUT == "nvfp4":
        # NVFP4: 16-element blocks, 2 codes per byte, same nibble layout as Q4
        # (low = even, high = odd). Codes 0..7 = +magnitude, 8..15 = -magnitude;
        # the magnitudes in order are 0, 0.5, 1, 1.5, 2, 3, 4, 6. We extract the
        # unsigned code, then expand it to its float magnitude with an
        # if-else ladder (Triton has no float4 lookup table).
        block_idx = elem_offsets >> 4  # 16 values per block
        in_block = elem_offsets & 15
        byte_offs = block_idx * 8 + (in_block >> 1)
        is_odd = (in_block & 1).to(tl.int32)
        packed = tl.load(ptr + slot_base + byte_offs, mask=elem_mask, other=0).to(tl.uint8)
        lo = (packed & 0xF).to(tl.int32)
        hi = ((packed >> 4) & 0xF).to(tl.int32)
        code = tl.where(is_odd == 0, lo, hi)
        sign_bit = (code >> 3) & 1
        mag_code = code & 0x7
        # Magnitude ladder: 0 -> 0, 1 -> 0.5, 2 -> 1, 3 -> 1.5, 4 -> 2, 5 -> 3,
        # 6 -> 4, 7 -> 6. We expand to fp32 by checking the code (the magnitude
        # itself is not the value; e.g. code 5 -> 3.0, not 5.0).
        mag = tl.where(
            mag_code == 0, 0.0,
            tl.where(
                mag_code == 1, 0.5,
                tl.where(
                    mag_code == 2, 1.0,
                    tl.where(
                        mag_code == 3, 1.5,
                        tl.where(
                            mag_code == 4, 2.0,
                            tl.where(
                                mag_code == 5, 3.0,
                                tl.where(
                                    mag_code == 6, 4.0, 6.0,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        vals = tl.where(sign_bit == 1, -mag, mag)
    else:
        tl.static_assert(False, f"unknown LAYOUT {LAYOUT!r}")

    if QUANT:
        scale = tl.load(scale_ptr + scale_offsets, mask=scale_mask, other=0.0)
        if D_ON_ROWS:
            nb: tl.constexpr = scale.shape[0]
            n: tl.constexpr = scale.shape[1]
            wide = tl.broadcast_to(scale[:, None, :], (nb, QBLOCK, n)).reshape(nb * QBLOCK, n)
        else:
            n: tl.constexpr = scale.shape[0]
            nb: tl.constexpr = scale.shape[1]
            wide = tl.broadcast_to(scale[:, :, None], (n, nb, QBLOCK)).reshape(n, nb * QBLOCK)
        return (vals.to(tl.float32) * wide.to(tl.float32)).to(out_dtype)
    # Both branches must yield the same type for Triton to compile the function, so the
    # unquantized path casts too. Callers pass the dtype the tile already has there
    # (float32 for the fp32 kernel, the cache's own dtype elsewhere), so it is a no-op.
    return vals.to(out_dtype)


@functools.lru_cache(maxsize=None)
def _optin_smem_bytes(device_index: int) -> int:
    """Per-block opt-in shared-memory budget for a CUDA device (0 if unavailable)."""
    props = torch.cuda.get_device_properties(device_index)
    return int(getattr(props, "shared_memory_per_block_optin", 0))


def _select_extend_tile(head_dim: int, block_d: int, smem_optin: int) -> tuple[int, int]:
    """Pick ``(BLOCK_M, BLOCK_N)`` for the extend/prefill kernel, shared-memory aware.

    Larger tiles run materially faster (~2x for head_dim 512 on H100) but their bf16
    q/k/v tiles need about ``(BLOCK_M + 2 * BLOCK_N) * BLOCK_D * 2`` bytes of shared
    memory, which overflows consumer GPUs (sm_89 ~99KB opt-in) once head_dim >= 256.
    Keep the fast tiles where the device's opt-in shared memory fits them (datacenter
    A100/H100); shrink only where it does not. ``smem_optin == 0`` (unknown) conservatively
    selects the small tiles, i.e. the prior consumer-safe behavior.
    """
    budget = smem_optin * 0.8  # headroom for scores/acc/alignment/triton scratch

    def fits(block_m: int, block_n: int) -> bool:
        return (block_m + 2 * block_n) * block_d * 2 <= budget

    if head_dim <= 128:
        return 128, 64
    if head_dim <= 256:
        return (128, 64) if fits(128, 64) else (64, 32)
    if head_dim <= 384:
        return (32, 64) if fits(32, 64) else (32, 32)
    return (32, 64) if fits(32, 64) else (16, 16)


@triton.jit
def _paged_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ks_ptr,
    vs_ptr,
    o_ptr,
    indptr_ptr,
    indices_ptr,
    q_to_req_ptr,
    q_pos_ptr,
    sm_scale,
    sinks_ptr,
    stride_qt,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_kss,
    stride_ksh,
    stride_vss,
    stride_vsh,
    stride_ot,
    stride_oh,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    QUANT: tl.constexpr,
    QBLOCK: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    q_tok = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // GROUP

    req = tl.load(q_to_req_ptr + q_tok)
    kv_start = tl.load(indptr_ptr + req)
    kv_end = tl.load(indptr_ptr + req + 1)
    kv_len = kv_end - kv_start
    q_pos = tl.load(q_pos_ptr + q_tok)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    # One scale per QBLOCK elements of head_dim: the tile's block axis.
    offs_nb = tl.arange(0, BLOCK_D // QBLOCK)
    mask_nb = offs_nb < D // QBLOCK
    # Byte offset along the head_dim axis. 8-bit: 1 byte per element. Q4: 2 per
    # byte. Q6/NVFP4: see _load_kv (it reads two planes / a packed nibble, so the
    # offset it consumes is the per-element raw position -- the byte address
    # differs per plane or per nibble).
    if LAYOUT == "q4":
        phys_offs_d = offs_d >> 1
    elif LAYOUT == "q6":
        phys_offs_d = offs_d
    else:
        phys_offs_d = offs_d
    q = tl.load(
        q_ptr + q_tok * stride_qt + q_head * stride_qh + offs_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)

    if HAS_SINKS:
        m_i = tl.load(sinks_ptr + q_head).to(tl.float32)
        l_i = 1
    else:
        m_i = -float("inf")
        l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start in range(0, kv_len, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_len
        k_pos = offs_n
        causal_mask = k_pos <= q_pos
        if SLIDING_WINDOW > 0:
            causal_mask = causal_mask & ((k_pos + SLIDING_WINDOW) > q_pos)
        mask_n = mask_n & causal_mask

        skip_tile = tl.max(mask_n.to(tl.int32), axis=0) == 0
        if not skip_tile:
            slots = tl.load(indices_ptr + kv_start + offs_n, mask=offs_n < kv_len, other=0)
            kv_mask = (offs_n[:, None] < kv_len) & mask_d[None, :]
            kv_scale_mask = (offs_n[:, None] < kv_len) & mask_nb[None, :]
            k = _load_kv(
                k_ptr,
                ks_ptr,
                (slots * stride_ks + kv_head * stride_kh)[:, None],  # [BLOCK_N, 1]
                phys_offs_d[None, :],  # [1, BLOCK_D] -> [BLOCK_N, BLOCK_D]
                kv_mask,
                (slots * stride_kss + kv_head * stride_ksh)[:, None] + offs_nb[None, :],
                kv_scale_mask,
                tl.float32,
                QUANT,
                QBLOCK,
                False,
                LAYOUT,
            ).to(tl.float32)
            scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
            scores = tl.where(mask_n, scores, -float("inf"))

            row_max = tl.max(scores, axis=0)
            row_max_fixed = tl.where(row_max == -float("inf"), -1e20, row_max)
            m_new = tl.maximum(row_max_fixed, m_i)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)

            v = _load_kv(
                v_ptr,
                vs_ptr,
                (slots * stride_vs + kv_head * stride_vh)[:, None],  # [BLOCK_N, 1]
                phys_offs_d[None, :],  # [1, BLOCK_D] -> [BLOCK_N, BLOCK_D]
                kv_mask,
                (slots * stride_vss + kv_head * stride_vsh)[:, None] + offs_nb[None, :],
                kv_scale_mask,
                tl.float32,
                QUANT,
                QBLOCK,
                False,
                LAYOUT,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

    out = tl.where(l_i == 0.0, 0.0, acc / l_i)
    tl.store(
        o_ptr + q_tok * stride_ot + q_head * stride_oh + offs_d,
        out.to(o_ptr.dtype.element_ty),
        mask=mask_d,
    )


@triton.jit
def _decode_grouped_stage1_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ks_ptr,
    vs_ptr,
    sm_scale,
    indptr_ptr,
    indices_ptr,
    q_pos_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    num_kv_splits_ptr,
    stride_qt,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_kss,
    stride_ksh,
    stride_vss,
    stride_vsh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    GROUP: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    VALID_BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    QUANT: tl.constexpr,
    QBLOCK: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_block_id = tl.program_id(1)
    split_id = tl.program_id(2)

    # VALID_BLOCK_H == min(cap, GROUP) is the number of query heads actually handled per program;
    # BLOCK_H is the power-of-two tile size for the head axis (tl.arange requires a power of two),
    # so a non-power-of-two GQA group (e.g. 24/4 == 6) rounds the tile up and masks the extra
    # lanes. Each kv head spans cdiv(GROUP, VALID_BLOCK_H) head blocks.
    kv_head = head_block_id // tl.cdiv(GROUP, VALID_BLOCK_H)
    q_heads = head_block_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = q_heads < (head_block_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (q_heads < NUM_Q_HEADS)

    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < D
    mask_dv = offs_dv < DV
    offs_nb = tl.arange(0, BLOCK_D // QBLOCK)
    offs_nbv = tl.arange(0, BLOCK_DV // QBLOCK)
    mask_nb = offs_nb < D // QBLOCK
    mask_nbv = offs_nbv < DV // QBLOCK

    kv_start = tl.load(indptr_ptr + batch_id)
    kv_len = tl.load(indptr_ptr + batch_id + 1) - kv_start
    q_pos = tl.load(q_pos_ptr + batch_id)
    effective_end = tl.minimum(kv_len, q_pos + 1)
    effective_start = 0
    if SLIDING_WINDOW > 0:
        effective_start = tl.maximum(0, q_pos - SLIDING_WINDOW + 1)
    effective_len = tl.maximum(0, effective_end - effective_start)

    kv_splits = tl.load(num_kv_splits_ptr + batch_id)
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(effective_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_start = kv_len_per_split * split_id
    split_end = tl.minimum(split_start + kv_len_per_split, effective_len)

    m_i = tl.zeros((BLOCK_H,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)

    q_offsets = batch_id * stride_qt + q_heads[:, None] * stride_qh + offs_d[None, :]
    k_base_offsets = kv_head * stride_kh + offs_d[:, None]
    v_base_offsets = kv_head * stride_vh + offs_dv[None, :]
    ks_base_offsets = kv_head * stride_ksh + offs_nb[:, None]
    vs_base_offsets = kv_head * stride_vsh + offs_nbv[None, :]

    if split_end > split_start:
        q = tl.load(q_ptr + q_offsets, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
        if not QUANT:
            # Unquantized: match the cache's dtype as before. Quantized: the cache is
            # int8/fp8 and casting q into it would destroy the query -- the dequantized
            # K/V tiles are produced in q's dtype instead.
            q = q.to(k_ptr.dtype.element_ty)

        for rel_start in tl.range(split_start, split_end, BLOCK_N):
            rel_offs = rel_start + tl.arange(0, BLOCK_N)
            mask_n = rel_offs < split_end
            logical_offs = effective_start + rel_offs
            slots = tl.load(indices_ptr + kv_start + logical_offs, mask=mask_n, other=0)
            # slot_base: byte address of each (slot, kv_head)'s head_dim start.
            # K with D_ON_ROWS expects [BLOCK_D, BLOCK_N] (D on rows, like the original
            # 8-bit path); V expects [BLOCK_N, BLOCK_D] (N on rows).
            k_slot_base = (slots * stride_ks + kv_head * stride_kh)  # [BLOCK_N]
            v_slot_base = (slots * stride_vs + kv_head * stride_vh)
            ks_slot_base = (slots * stride_kss + kv_head * stride_ksh)  # scales
            vs_slot_base = (slots * stride_vss + kv_head * stride_vsh)

            k = _load_kv(
                k_ptr,
                ks_ptr,
                k_slot_base[None, :],  # [1, BLOCK_N]
                offs_d[:, None],  # [BLOCK_D, 1] -> [BLOCK_D, BLOCK_N]
                mask_n[None, :] & mask_d[:, None],
                ks_slot_base[None, :] + offs_nb[:, None],
                mask_n[None, :] & mask_nb[:, None],
                q.dtype,
                QUANT,
                QBLOCK,
                True,
                LAYOUT,
            )
            scores = tl.dot(q, k) * sm_scale
            scores = tl.where(mask_h[:, None] & mask_n[None, :], scores, -float("inf"))

            v = _load_kv(
                v_ptr,
                vs_ptr,
                v_slot_base[:, None],  # [BLOCK_N, 1]
                offs_dv[None, :],  # [1, BLOCK_D] -> [BLOCK_N, BLOCK_D]
                mask_n[:, None] & mask_dv[None, :],
                vs_slot_base[:, None] + offs_nbv[None, :],
                mask_n[:, None] & mask_nbv[None, :],
                q.dtype,
                QUANT,
                QBLOCK,
                False,
                LAYOUT,
            )

            m_new = tl.maximum(tl.max(scores, axis=1), m_i)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        out = acc / l_i[:, None]
        mid_offsets = (
            batch_id * stride_mid_ob
            + q_heads[:, None] * stride_mid_oh
            + split_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(mid_o_ptr + mid_offsets, out, mask=mask_h[:, None] & mask_dv[None, :])

        lse_offsets = (
            batch_id * stride_lse_b
            + q_heads * stride_lse_h
            + split_id * stride_lse_s
        )
        tl.store(mid_lse_ptr + lse_offsets, m_i + tl.log(l_i), mask=mask_h)


@triton.jit
def _decode_stage2_kernel(
    mid_o_ptr,
    mid_lse_ptr,
    o_ptr,
    indptr_ptr,
    q_pos_ptr,
    num_kv_splits_ptr,
    sinks_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    stride_ot,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    DV: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
):
    batch_id = tl.program_id(0)
    q_head = tl.program_id(1)

    kv_len = tl.load(indptr_ptr + batch_id + 1) - tl.load(indptr_ptr + batch_id)
    q_pos = tl.load(q_pos_ptr + batch_id)
    effective_end = tl.minimum(kv_len, q_pos + 1)
    effective_start = 0
    if SLIDING_WINDOW > 0:
        effective_start = tl.maximum(0, q_pos - SLIDING_WINDOW + 1)
    effective_len = tl.maximum(0, effective_end - effective_start)

    kv_splits = tl.load(num_kv_splits_ptr + batch_id)
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(effective_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < DV
    if HAS_SINKS:
        m_i = tl.load(sinks_ptr + q_head).to(tl.float32)
        l_i = 1.0
    else:
        m_i = -float("inf")
        l_i = 0.0
    acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    mid_base = batch_id * stride_mid_ob + q_head * stride_mid_oh + offs_d
    lse_base = batch_id * stride_lse_b + q_head * stride_lse_h

    for split_id in tl.range(0, MAX_KV_SPLITS, num_stages=2):
        split_start = kv_len_per_split * split_id
        split_end = tl.minimum(split_start + kv_len_per_split, effective_len)

        if split_end > split_start:
            partial = tl.load(
                mid_o_ptr + mid_base + split_id * stride_mid_os,
                mask=mask_d,
                other=0.0,
            )
            partial_lse = tl.load(mid_lse_ptr + lse_base + split_id * stride_lse_s)
            m_new = tl.maximum(partial_lse, m_i)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(partial_lse - m_new)
            acc = acc * alpha + partial * beta
            l_i = l_i * alpha + beta
            m_i = m_new

    out = tl.where(l_i == 0.0, 0.0, acc / l_i)
    tl.store(
        o_ptr + batch_id * stride_ot + q_head * stride_oh + offs_d,
        out.to(o_ptr.dtype.element_ty),
        mask=mask_d,
    )


def _kv_scale_args(k_cache, v_cache, k_scale, v_scale, *, head_dim=None):
    """Scale tensors + strides + the QUANT/QBLOCK constexprs for a kernel launch.

    Unquantized pools pass ``k_scale=None``; the kernels then never touch the scale
    pointers, so the KV buffers themselves stand in and ``QUANT=False`` compiles the
    dequant away entirely -- the bf16 path emits the same code it did before.

    ``head_dim`` is the LOGICAL head_dim (post-quantization size of one element). For
    sub-byte schemes the cache's last axis is the packed byte count, not head_dim, so
    the QBLOCK has to be derived from the logical extent. Pass ``head_dim`` for those;
    we default to ``k_cache.shape[-1]`` for the 8-bit case where the two are equal.
    """
    if k_scale is None:
        assert v_scale is None, "k_scale and v_scale must be given together"
        return (k_cache, v_cache, 0, 0, 0, 0, False, 1)
    assert v_scale is not None, "k_scale and v_scale must be given together"
    assert k_scale.dim() == v_scale.dim() == 3, "scales are [slots, heads, D // block]"
    # QBLOCK = elements per scale (always 32 for our schemes). It is the ratio of
    # the LOGICAL head_dim to the scale's last-dim count -- not the physical cache
    # last-dim (which is smaller for sub-byte schemes).
    logical_d = head_dim if head_dim is not None else k_cache.shape[-1]
    block = logical_d // k_scale.shape[-1]
    assert logical_d == block * k_scale.shape[-1], (
        f"logical head_dim {logical_d} is not a whole number of {k_scale.shape[-1]} blocks"
    )
    return (
        k_scale,
        v_scale,
        k_scale.stride(0),
        k_scale.stride(1),
        v_scale.stride(0),
        v_scale.stride(1),
        True,
        block,
    )


def decode_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    q_positions: torch.Tensor,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale: float,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    layout: str = "q8",
) -> torch.Tensor:
    """SGLang-style split-k grouped decode attention for one query per request."""

    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.dim() == 3 and k_cache.dim() == 3 and v_cache.dim() == 3
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    # Scale strides + QBLOCK need the LOGICAL head_dim (sub-byte caches report a
    # packed byte count that must not feed into QBLOCK derivation).
    ks, vs, s_kss, s_ksh, s_vss, s_vsh, quant, qblock = _kv_scale_args(
        k_cache, v_cache, k_scale, v_scale, head_dim=head_dim
    )
    assert batch == indptr.numel() - 1
    assert v_cache.shape[1] == num_kv_heads
    # The cache's last-dim is the PACKED byte count for sub-byte schemes; for 8-bit
    # it equals head_dim. The kernel does the unpacking. We just sanity-check that
    # the head_dim we were given is the logical extent the caller intends.
    assert num_q_heads % num_kv_heads == 0
    assert attn_logits.shape[0] >= batch
    assert attn_logits.shape[1] >= num_q_heads
    assert attn_logits.shape[2] >= max_kv_splits
    assert attn_logits.shape[3] >= head_dim
    assert attn_lse.shape[0] >= batch
    assert attn_lse.shape[1] >= num_q_heads
    assert attn_lse.shape[2] >= max_kv_splits
    if sinks is not None:
        assert sinks.is_cuda
        assert sinks.dim() == 1
        assert sinks.numel() >= num_q_heads
        sinks = sinks.contiguous()

    o = out if out is not None else torch.empty_like(q)
    sinks_arg = sinks if sinks is not None else q
    group = num_q_heads // num_kv_heads
    # valid_block_h = heads computed per program (drives the grid + head indexing); block_h =
    # power-of-two tile size for tl.arange. They differ only for non-power-of-two GQA groups
    # (e.g. 6), where block_h rounds up and the kernel masks the extra lanes.
    valid_block_h = min(16, group)
    block_h = triton.next_power_of_2(valid_block_h)
    block_d = triton.next_power_of_2(head_dim)
    block_dv = triton.next_power_of_2(head_dim)

    _decode_grouped_stage1_kernel[
        (batch, triton.cdiv(num_q_heads, valid_block_h), max_kv_splits)
    ](
        q,
        k_cache,
        v_cache,
        ks,
        vs,
        sm_scale,
        indptr,
        indices,
        q_positions,
        attn_logits,
        attn_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        s_kss,
        s_ksh,
        s_vss,
        s_vsh,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        attn_lse.stride(0),
        attn_lse.stride(1),
        attn_lse.stride(2),
        GROUP=group,
        NUM_Q_HEADS=num_q_heads,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        BLOCK_N=32,
        BLOCK_H=block_h,
        VALID_BLOCK_H=valid_block_h,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        D=head_dim,
        DV=head_dim,
        SLIDING_WINDOW=sliding_window or 0,
        QUANT=quant,
        QBLOCK=qblock,
        LAYOUT=layout,
        num_warps=4,
        num_stages=2,
    )
    _decode_stage2_kernel[(batch, num_q_heads)](
        attn_logits,
        attn_lse,
        o,
        indptr,
        q_positions,
        num_kv_splits,
        sinks_arg,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        attn_lse.stride(0),
        attn_lse.stride(1),
        attn_lse.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=max_kv_splits,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=block_dv,
        DV=head_dim,
        SLIDING_WINDOW=sliding_window or 0,
        HAS_SINKS=sinks is not None,
        num_warps=4,
        num_stages=2,
    )
    return o


@triton.jit
def _extend_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ks_ptr,
    vs_ptr,
    o_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    prefix_lens_ptr,
    sm_scale,
    sinks_ptr,
    stride_qt,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_kss,
    stride_ksh,
    stride_vss,
    stride_vsh,
    stride_ot,
    stride_oh,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    QUANT: tl.constexpr,
    QBLOCK: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    seq_id = tl.program_id(0)
    q_head = tl.program_id(1)
    block_m_id = tl.program_id(2)
    kv_head = q_head // GROUP

    q_start = tl.load(qo_indptr_ptr + seq_id)
    q_end = tl.load(qo_indptr_ptr + seq_id + 1)
    q_len = q_end - q_start
    kv_start = tl.load(kv_indptr_ptr + seq_id)
    kv_len = tl.load(kv_indptr_ptr + seq_id + 1) - kv_start
    prefix_len = tl.load(prefix_lens_ptr + seq_id)

    offs_m = block_m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_m = offs_m < q_len
    mask_d = offs_d < D
    mask_dv = offs_dv < D
    offs_nb = tl.arange(0, BLOCK_D // QBLOCK)
    offs_nbv = tl.arange(0, BLOCK_DV // QBLOCK)
    mask_nb = offs_nb < D // QBLOCK
    mask_nbv = offs_nbv < D // QBLOCK
    # Byte offset into the KV cache. 8-bit: 1 byte per element; Q4: 2 per byte;
    # Q6: see _load_kv (it reads two planes; the offset consumed is the per-element
    # raw position).
    if LAYOUT == "q4":
        phys_offs_d = offs_d >> 1
        phys_offs_dv = offs_dv >> 1
    else:
        phys_offs_d = offs_d
        phys_offs_dv = offs_dv
    q_abs_pos = prefix_len + offs_m
    block_q_end = tl.minimum(q_len, (block_m_id + 1) * BLOCK_M)
    kv_loop_end = tl.minimum(kv_len, prefix_len + block_q_end)

    q = tl.load(
        q_ptr + (q_start + offs_m[:, None]) * stride_qt + q_head * stride_qh + offs_d[None, :],
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    if HAS_SINKS:
        sink = tl.load(sinks_ptr + q_head).to(tl.float32)
        m_i = tl.full((BLOCK_M,), sink, dtype=tl.float32)
        l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    else:
        m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)

    for start_n in tl.range(0, kv_loop_end, BLOCK_N):
        kv_offsets = start_n + offs_n
        mask_n = kv_offsets < kv_len
        key_pos = kv_offsets
        causal_mask = key_pos[None, :] <= q_abs_pos[:, None]
        if SLIDING_WINDOW > 0:
            causal_mask = causal_mask & ((key_pos[None, :] + SLIDING_WINDOW) > q_abs_pos[:, None])
        final_mask = mask_m[:, None] & mask_n[None, :] & causal_mask

        skip_tile = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0
        if not skip_tile:
            slots = tl.load(kv_indices_ptr + kv_start + kv_offsets, mask=mask_n, other=0)
            k = _load_kv(
                k_ptr,
                ks_ptr,
                (slots * stride_ks + kv_head * stride_kh)[None, :],  # [1, BLOCK_N]
                phys_offs_d[:, None],  # [BLOCK_D, 1] -> [BLOCK_D, BLOCK_N]
                mask_n[None, :] & mask_d[:, None],
                (slots * stride_kss + kv_head * stride_ksh)[None, :] + offs_nb[:, None],
                mask_n[None, :] & mask_nb[:, None],
                q.dtype,
                QUANT,
                QBLOCK,
                True,
                LAYOUT,
            )
            scores = tl.dot(q.to(k.dtype), k) * sm_scale
            scores = tl.where(final_mask, scores, -float("inf"))

            row_max = tl.max(scores, axis=1)
            row_max_fixed = tl.where(row_max == -float("inf"), -1e20, row_max)
            m_new = tl.maximum(row_max_fixed, m_i)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            v = _load_kv(
                v_ptr,
                vs_ptr,
                (slots * stride_vs + kv_head * stride_vh)[:, None],  # [BLOCK_N, 1]
                phys_offs_dv[None, :],  # [1, BLOCK_DV] -> [BLOCK_N, BLOCK_DV]
                mask_n[:, None] & mask_dv[None, :],
                (slots * stride_vss + kv_head * stride_vsh)[:, None] + offs_nbv[None, :],
                mask_n[:, None] & mask_nbv[None, :],
                q.dtype,
                QUANT,
                QBLOCK,
                False,
                LAYOUT,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

    out = tl.where(l_i[:, None] == 0.0, 0.0, acc / l_i[:, None])
    tl.store(
        o_ptr
        + (q_start + offs_m[:, None]) * stride_ot
        + q_head * stride_oh
        + offs_dv[None, :],
        out.to(o_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_dv[None, :],
    )


@triton.jit
def _extend_attention_split_kernel(
    q_ptr,
    k_extend_ptr,
    v_extend_ptr,
    k_cache_ptr,
    v_cache_ptr,
    ks_ptr,
    vs_ptr,
    o_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    prefix_lens_ptr,
    sm_scale,
    sinks_ptr,
    stride_qt,
    stride_qh,
    stride_ket,
    stride_keh,
    stride_vet,
    stride_veh,
    stride_kcs,
    stride_kch,
    stride_vcs,
    stride_vch,
    stride_kss,
    stride_ksh,
    stride_vss,
    stride_vsh,
    stride_ot,
    stride_oh,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    QUANT: tl.constexpr,
    QBLOCK: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    seq_id = tl.program_id(0)
    q_head = tl.program_id(1)
    block_m_id = tl.program_id(2)
    kv_head = q_head // GROUP

    q_start = tl.load(qo_indptr_ptr + seq_id)
    q_len = tl.load(qo_indptr_ptr + seq_id + 1) - q_start
    kv_start = tl.load(kv_indptr_ptr + seq_id)
    prefix_len = tl.load(prefix_lens_ptr + seq_id)

    offs_m = block_m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_m = offs_m < q_len
    mask_d = offs_d < D
    mask_dv = offs_dv < D
    offs_nb = tl.arange(0, BLOCK_D // QBLOCK)
    offs_nbv = tl.arange(0, BLOCK_DV // QBLOCK)
    mask_nb = offs_nb < D // QBLOCK
    mask_nbv = offs_nbv < D // QBLOCK
    # Byte offset into the cache for the head_dim axis (Q4 = elem // 2).
    if LAYOUT == "q4":
        phys_offs_d = offs_d >> 1
        phys_offs_dv = offs_dv >> 1
    else:
        phys_offs_d = offs_d
        phys_offs_dv = offs_dv
    q_abs_pos = prefix_len + offs_m

    q = tl.load(
        q_ptr
        + (q_start + offs_m[:, None]) * stride_qt
        + q_head * stride_qh
        + offs_d[None, :],
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    if HAS_SINKS:
        sink = tl.load(sinks_ptr + q_head).to(tl.float32)
        m_i = tl.full((BLOCK_M,), sink, dtype=tl.float32)
        l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    else:
        m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)

    for start_n in tl.range(0, prefix_len, BLOCK_N):
        kv_offsets = start_n + offs_n
        mask_n = kv_offsets < prefix_len
        key_pos = kv_offsets
        final_mask = mask_m[:, None] & mask_n[None, :]
        if SLIDING_WINDOW > 0:
            window_mask = (key_pos[None, :] + SLIDING_WINDOW) > q_abs_pos[:, None]
            final_mask = final_mask & window_mask

        skip_tile = False
        if SLIDING_WINDOW > 0:
            skip_tile = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not skip_tile:
            slots = tl.load(kv_indices_ptr + kv_start + kv_offsets, mask=mask_n, other=0)
            k = _load_kv(
                k_cache_ptr,
                ks_ptr,
                (slots * stride_kcs + kv_head * stride_kch)[None, :],  # [1, BLOCK_N]
                phys_offs_d[:, None],  # [BLOCK_D, 1] -> [BLOCK_D, BLOCK_N]
                mask_n[None, :] & mask_d[:, None],
                (slots * stride_kss + kv_head * stride_ksh)[None, :] + offs_nb[:, None],
                mask_n[None, :] & mask_nb[:, None],
                q.dtype,
                QUANT,
                QBLOCK,
                True,
                LAYOUT,
            )
            scores = tl.dot(q.to(k.dtype), k) * sm_scale
            scores = tl.where(final_mask, scores, -float("inf"))

            row_max = tl.max(scores, axis=1)
            row_max_fixed = tl.where(row_max == -float("inf"), -1e20, row_max)
            m_new = tl.maximum(row_max_fixed, m_i)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            v = _load_kv(
                v_cache_ptr,
                vs_ptr,
                (slots * stride_vcs + kv_head * stride_vch)[:, None],  # [BLOCK_N, 1]
                phys_offs_dv[None, :],  # [1, BLOCK_DV] -> [BLOCK_N, BLOCK_DV]
                mask_n[:, None] & mask_dv[None, :],
                (slots * stride_vss + kv_head * stride_vsh)[:, None] + offs_nbv[None, :],
                mask_n[:, None] & mask_nbv[None, :],
                q.dtype,
                QUANT,
                QBLOCK,
                False,
                LAYOUT,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

    current_end = tl.minimum(q_len, (block_m_id + 1) * BLOCK_M)
    for start_n in tl.range(0, current_end, BLOCK_N):
        local_kv_offsets = start_n + offs_n
        mask_n = local_kv_offsets < current_end
        local_q_pos = offs_m
        causal_mask = local_kv_offsets[None, :] <= local_q_pos[:, None]
        if SLIDING_WINDOW > 0:
            causal_mask = causal_mask & (
                (local_kv_offsets[None, :] + SLIDING_WINDOW) > local_q_pos[:, None]
            )
        final_mask = mask_m[:, None] & mask_n[None, :] & causal_mask

        skip_tile = False
        if SLIDING_WINDOW > 0:
            skip_tile = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not skip_tile:
            k = tl.load(
                k_extend_ptr
                + (q_start + local_kv_offsets[None, :]) * stride_ket
                + kv_head * stride_keh
                + offs_d[:, None],
                mask=mask_n[None, :] & mask_d[:, None],
                other=0.0,
            )
            scores = tl.dot(q.to(k.dtype), k) * sm_scale
            scores = tl.where(final_mask, scores, -float("inf"))

            row_max = tl.max(scores, axis=1)
            row_max_fixed = tl.where(row_max == -float("inf"), -1e20, row_max)
            m_new = tl.maximum(row_max_fixed, m_i)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            v = tl.load(
                v_extend_ptr
                + (q_start + local_kv_offsets[:, None]) * stride_vet
                + kv_head * stride_veh
                + offs_dv[None, :],
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

    out = tl.where(l_i[:, None] == 0.0, 0.0, acc / l_i[:, None])
    tl.store(
        o_ptr
        + (q_start + offs_m[:, None]) * stride_ot
        + q_head * stride_oh
        + offs_dv[None, :],
        out.to(o_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_dv[None, :],
    )


def extend_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_q_len: int,
    sm_scale: float,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    k_extend: torch.Tensor | None = None,
    v_extend: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    layout: str = "q8",
) -> torch.Tensor:
    """Block-tiled causal prefill/extend attention over paged KV cache."""

    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.dim() == 3 and k_cache.dim() == 3 and v_cache.dim() == 3
    num_q_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    ks, vs, s_kss, s_ksh, s_vss, s_vsh, quant, qblock = _kv_scale_args(
        k_cache, v_cache, k_scale, v_scale, head_dim=head_dim
    )
    assert qo_indptr.numel() == kv_indptr.numel()
    assert prefix_lens.numel() == qo_indptr.numel() - 1
    assert v_cache.shape[1] == num_kv_heads
    # k_cache.shape[-1] is the packed byte count for sub-byte schemes; the kernel
    # does the unpacking. We only check head_dim divisibility for sub-byte.
    assert num_q_heads % num_kv_heads == 0
    if sinks is not None:
        assert sinks.is_cuda
        assert sinks.dim() == 1
        assert sinks.numel() >= num_q_heads
        sinks = sinks.contiguous()

    o = out if out is not None else torch.empty_like(q)
    sinks_arg = sinks if sinks is not None else q
    block_d = triton.next_power_of_2(head_dim)
    block_dv = triton.next_power_of_2(head_dim)
    # Tile size is shared-memory bound: keep the fast (large) tiles on GPUs whose opt-in
    # shared memory fits them, shrink on consumer GPUs (sm_89 ~99KB) where the default
    # 128x64 overflows once head_dim >= 256 (e.g. gemma4: SWA 256, full-attention 512).
    block_m, block_n = _select_extend_tile(
        head_dim, block_d, _optin_smem_bytes(q.device.index)
    )
    grid = (qo_indptr.numel() - 1, num_q_heads, triton.cdiv(max_q_len, block_m))
    if k_extend is not None or v_extend is not None:
        assert k_extend is not None and v_extend is not None
        assert k_extend.is_cuda and v_extend.is_cuda
        assert k_extend.dim() == 3 and v_extend.dim() == 3
        assert k_extend.shape[0] == num_q_tokens and v_extend.shape[0] == num_q_tokens
        assert k_extend.shape[1] == num_kv_heads and v_extend.shape[1] == num_kv_heads
        assert k_extend.shape[-1] == head_dim and v_extend.shape[-1] == head_dim
        _extend_attention_split_kernel[grid](
            q,
            k_extend,
            v_extend,
            k_cache,
            v_cache,
            ks,
            vs,
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens,
            sm_scale,
            sinks_arg,
            q.stride(0),
            q.stride(1),
            k_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(0),
            v_extend.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            v_cache.stride(0),
            v_cache.stride(1),
            s_kss,
            s_ksh,
            s_vss,
            s_vsh,
            o.stride(0),
            o.stride(1),
            GROUP=num_q_heads // num_kv_heads,
            D=head_dim,
            BLOCK_D=block_d,
            BLOCK_DV=block_dv,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            SLIDING_WINDOW=sliding_window or 0,
            HAS_SINKS=sinks is not None,
            QUANT=quant,
            QBLOCK=qblock,
            LAYOUT=layout,
            num_warps=8,
            num_stages=1,
        )
        return o

    _extend_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        ks,
        vs,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        sm_scale,
        sinks_arg,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        s_kss,
        s_ksh,
        s_vss,
        s_vsh,
        o.stride(0),
        o.stride(1),
        GROUP=num_q_heads // num_kv_heads,
        D=head_dim,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SLIDING_WINDOW=sliding_window or 0,
        HAS_SINKS=sinks is not None,
        QUANT=quant,
        QBLOCK=qblock,
        LAYOUT=layout,
        num_warps=8,
        num_stages=1,
    )
    return o


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    q_to_req: torch.Tensor,
    q_positions: torch.Tensor,
    sm_scale: float,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_n: int = 32,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    layout: str = "q8",
) -> torch.Tensor:
    """Paged causal attention for one layer.

    ``q`` is ``[num_query_tokens, num_q_heads, head_dim]``. KV cache tensors are
    flattened to ``[num_slots, num_kv_heads, head_dim]``. ``indptr`` and
    ``indices`` describe each request's logical KV slots in order.
    """

    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.dim() == 3 and k_cache.dim() == 3 and v_cache.dim() == 3
    num_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    ks, vs, s_kss, s_ksh, s_vss, s_vsh, quant, qblock = _kv_scale_args(
        k_cache, v_cache, k_scale, v_scale, head_dim=head_dim
    )
    assert v_cache.shape[1] == num_kv_heads
    # k_cache.shape[-1] is the packed byte count for sub-byte schemes; the kernel
    # does the unpacking. No head_dim assert here.
    assert num_q_heads % num_kv_heads == 0
    if sinks is not None:
        assert sinks.is_cuda
        assert sinks.dim() == 1
        assert sinks.numel() >= num_q_heads
        sinks = sinks.contiguous()

    o = out if out is not None else torch.empty_like(q)
    sinks_arg = sinks if sinks is not None else q
    block_d = triton.next_power_of_2(head_dim)
    grid = (num_tokens, num_q_heads)
    _paged_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        ks,
        vs,
        o,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sinks_arg,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        s_kss,
        s_ksh,
        s_vss,
        s_vsh,
        o.stride(0),
        o.stride(1),
        GROUP=num_q_heads // num_kv_heads,
        D=head_dim,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        SLIDING_WINDOW=sliding_window or 0,
        HAS_SINKS=sinks is not None,
        QUANT=quant,
        QBLOCK=qblock,
        LAYOUT=layout,
        num_warps=8 if head_dim >= 256 else 4,
        num_stages=2,
    )
    return o
