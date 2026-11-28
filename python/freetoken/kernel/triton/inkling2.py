"""Triton kernels of Inkling 2 Mini (``models/inkling2``).

* ``extend_attention`` / ``decode_attention``: causal GQA over the paged KV cache with two
  additions no shared backend has: a per-query-token, per-head bias over the backward
  distance ``d = q_pos - k_pos`` (read from a ``[T, H, E]`` table, zero for ``d >= E``) and a
  per-query-token, per-head sink logit that only enlarges the softmax denominator. The
  output is gated by ``sigmoid(gate)`` before it is stored.
* ``short_conv``: the depthwise causal convolutions with a residual (``(conv(x) + x) * scale``),
  static or with per-token taps, whose left context comes from a per-request state row on the
  first rows of a chunk; optional per-head L2 clip / RMSNorm or a residual add on the output.
* ``attn_res``: the attention residual -- a softmax over RMS-normalized source rows, scored
  by learned queries, mixing the normalized rows; optionally RMS-normalized again.
* ``q_prep``, ``gate_inputs`` / ``gate_combine``: the per-head query / relative-state clips
  and the message gates' elementwise halves around their two projections.

Every bf16 rounding the reference makes between fused steps is reproduced in-kernel.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_u8_to_f32
from freetoken.kernel.triton.kv_rows import kv_scale_args, kv_tile, stage_tile
from freetoken.kvcache.format import NONE, KVFormat

_LOG2E = 1.4426950408889634
_DECODE_MIN_BLOCK_KV = 32
# a long request takes splits of at least this many keys
_DECODE_MIN_SPLIT_LEN = 512
_DECODE_MAX_SPLITS = 64
# stage 1 CTAs one SM holds for every KV format on sm_120 (8-bit: 3, bf16: 4)
_DECODE_CTAS_PER_SM = 3
# stage 2 holds at most this many fp32 partial elements per program
_DECODE_STAGE2_ELEMS = 2048


@triton.jit
def _gate_bf16(out, gate_ptr, mask):
    """``round(round(out) * sigmoid(gate))``: the reference gates the bf16 attention output in fp32."""
    o = out.to(tl.bfloat16).to(tl.float32)
    g = tl.load(gate_ptr, mask=mask, other=0.0).to(tl.float32)
    return o * tl.sigmoid(g)


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


@triton.jit
def _e4m3_bf16_bits(codes):
    # moves sign and exponent+mantissa into bf16 bits: the value comes out scaled by 2^-120
    return tl.inline_asm_elementwise(
        """{
        .reg .b32 a, b, s;
        prmt.b32 a, 0, $2, 0x5040;
        shr.b32 b, a, 4;
        and.b32 s, a, 0x80008000;
        and.b32 b, b, 0x07F007F0;
        or.b32 $0, b, s;
        prmt.b32 a, 0, $2, 0x7060;
        shr.b32 b, a, 4;
        and.b32 s, a, 0x80008000;
        and.b32 b, b, 0x07F007F0;
        or.b32 $1, b, s;
        }""",
        "=r,=r,r", [codes], dtype=tl.bfloat16, is_pure=True, pack=4,
    )


@triton.jit
def _int8_bf16_bits(codes):
    # 1024 + 128 + x in fp16, minus 1152, then the fp16 bits shifted into bf16: the value comes out scaled by 2^-112
    return tl.inline_asm_elementwise(
        """{
        .reg .b32 u, a, h, s, m, c;
        mov.b32 m, 0x64646464;
        mov.b32 c, 0x64806480;
        xor.b32 u, $2, 0x80808080;
        prmt.b32 a, u, m, 0x5140;
        sub.rn.f16x2 h, a, c;
        and.b32 s, h, 0x80008000;
        shr.b32 h, h, 3;
        and.b32 h, h, 0x0FFF0FFF;
        or.b32 $0, h, s;
        prmt.b32 a, u, m, 0x5342;
        sub.rn.f16x2 h, a, c;
        and.b32 s, h, 0x80008000;
        shr.b32 h, h, 3;
        and.b32 h, h, 0x0FFF0FFF;
        or.b32 $1, h, s;
        }""",
        "=r,=r,r", [codes], dtype=tl.bfloat16, is_pure=True, pack=4,
    )


@triton.jit
def _stage_bf16(x):
    # keeps the decode in the load layout: once per CTA, handed to the MMA through shared memory
    return tl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r", [x], dtype=tl.bfloat16, is_pure=False, pack=2)


@triton.jit
def _extend_kv_tile(ptrs, FMT: tl.constexpr, BIT_DECODE: tl.constexpr):
    """A K or V cache tile as the dot takes it; with BIT_DECODE an 8-bit tile is scaled by 2^-120 (fp8) or 2^-112 (int8)."""
    if FMT == "none":
        tile = tl.load(ptrs)
    elif BIT_DECODE:
        codes = tl.load(ptrs).to(tl.uint8, bitcast=True)
        if FMT == "fp8_per_token_head":
            tile = _stage_bf16(_e4m3_bf16_bits(codes))
        else:
            tile = _stage_bf16(_int8_bf16_bits(codes))
    else:
        codes = tl.load(ptrs)
        if FMT == "fp8_per_token_head":
            x = e4m3_u8_to_f32(codes)
        else:
            x = codes.to(tl.float32)
        tile = stage_tile(x).to(tl.bfloat16)
    return tile


@triton.jit
def _extend_attend_tile(
    q, m_i, l_i, acc, k, v, sk, sv, start, offs, key_hi, q_pos, bias_rows,
    KIND: tl.constexpr, SCALED: tl.constexpr, SLIDING_WINDOW: tl.constexpr, REL_EXTENT: tl.constexpr,
):
    """One key tile of the keys start + offs: KIND 0 has no bias and no mask, 1 a bias and no mask, 2 a bias and the key / causal / window masks."""
    if KIND != 0:
        keys = start + offs
        bias_ptrs = (bias_rows + (q_pos - start))[:, None] - offs[None, :]
    if KIND == 1:
        bias = tl.load(bias_ptrs)
    elif KIND == 2:
        dist = q_pos[:, None] - keys[None, :]
        key_ok = keys < key_hi
        if SLIDING_WINDOW > 0:
            vis = (dist.to(tl.uint32, bitcast=True) < SLIDING_WINDOW) & key_ok[None, :]
        else:
            vis = (dist >= 0) & key_ok[None, :]
        if SLIDING_WINDOW > 0 and SLIDING_WINDOW <= REL_EXTENT:
            in_ext = vis
        else:
            in_ext = (dist.to(tl.uint32, bitcast=True) < REL_EXTENT) & key_ok[None, :]
        bias = tl.load(bias_ptrs, mask=in_ext, other=0.0)
    s = tl.dot(q, k)
    if SCALED:
        s = s * sk[None, :]
    else:
        s = s * sk
    if KIND != 0:
        s = s + bias.to(tl.float32) * 1.4426950408889634
    if KIND == 2:
        s = tl.where(vis, s, -float("inf"))
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    alpha = tl.math.exp2(m_i - m_new)
    p = tl.math.exp2(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    if SCALED:
        p = p * sv[None, :]
    else:
        p = p * sv
    acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
    return m_new, l_i, acc


@triton.jit
def _extend_keys(
    q, m_i, l_i, acc, lo, hi, key_hi, k_base, v_base, ks_base, vs_base, kv_idx, ext_base, stride_k, stride_v,
    stride_kss, stride_vss, k_mul, v_mul, zero_ptr, zero_ref, first, kv_head, q_start, prefix_len, row_first, row_last,
    row_lo, bias_ptr, stride_bt, stride_bh,
    CACHE: tl.constexpr, KIND: tl.constexpr, BLOCK: tl.constexpr, STAGES: tl.constexpr, FMT: tl.constexpr,
    BIT_DECODE: tl.constexpr, GROUP: tl.constexpr, BLOCK_M: tl.constexpr, D: tl.constexpr, SLIDING_WINDOW: tl.constexpr,
    REL_EXTENT: tl.constexpr,
):
    """Keys [lo, hi) of one source: the cache prefix through kv_idx, or this chunk's rows ext_base + key. key_hi bounds the valid keys of masked tiles."""
    # a runtime zero keeps the row state and address offsets of each key loop out of the other loops' registers
    zero = tl.load(zero_ptr, volatile=True) - zero_ref
    r = tl.arange(0, BLOCK_M) + zero
    head = kv_head * GROUP + r % GROUP
    rows = tl.minimum(tl.maximum(q_start + first + r // GROUP, row_first), row_last)
    q_pos = prefix_len + rows - q_start
    bias_rows = bias_ptr + (rows - row_lo) * stride_bt + head * stride_bh
    offs = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK) + zero, BLOCK), BLOCK)
    offs_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, D) + zero, D), D)
    SCALED: tl.constexpr = CACHE and FMT != "none"
    for start in tl.range(lo, hi, BLOCK, num_stages=STAGES):
        keys = start + offs
        if CACHE:
            if KIND == 2:
                kv_rows = tl.load(kv_idx + tl.minimum(keys, key_hi - 1)).to(tl.int64)
            else:
                kv_rows = tl.load(kv_idx + keys).to(tl.int64)
        elif KIND == 2:
            kv_rows = ext_base + tl.minimum(keys, key_hi - 1)
        else:
            kv_rows = ext_base + keys
        if SCALED:
            sk = tl.load(ks_base + kv_rows * stride_kss) * k_mul
            sv = tl.load(vs_base + kv_rows * stride_vss) * v_mul
        else:
            sk = k_mul
            sv = v_mul
        if CACHE:
            k = _extend_kv_tile(k_base + kv_rows[None, :] * stride_k + offs_d[:, None], FMT, BIT_DECODE)
            v = _extend_kv_tile(v_base + kv_rows[:, None] * stride_v + offs_d[None, :], FMT, BIT_DECODE)
        else:
            k = tl.load(k_base + kv_rows[None, :] * stride_k + offs_d[:, None])
            v = tl.load(v_base + kv_rows[:, None] * stride_v + offs_d[None, :])
        m_i, l_i, acc = _extend_attend_tile(q, m_i, l_i, acc, k, v, sk, sv, start, offs, key_hi, q_pos, bias_rows, KIND,
                                            SCALED, SLIDING_WINDOW, REL_EXTENT)
    return m_i, l_i, acc


@triton.jit
def _extend_ranges(src_lo, src_hi, vis_lo, far_end, band_lo, band_hi, BLOCK_N: tl.constexpr, BLOCK_NB: tl.constexpr,
                   BLOCK_NE: tl.constexpr):
    """Tiles of the source keys [src_lo, src_hi): no-bias [s, a) below far_end, masked [a, b) up to band_lo, bias-only [b, c) below band_hi, masked [c, src_hi). Each range starts its tiles at its own lower bound."""
    s = tl.minimum(tl.maximum(src_lo, vis_lo), src_hi)
    a = s + tl.maximum(tl.minimum(far_end, src_hi) - s, 0) // BLOCK_N * BLOCK_N
    b = a + (tl.maximum(tl.minimum(band_lo, src_hi) - a, 0) + BLOCK_NE - 1) // BLOCK_NE * BLOCK_NE
    b = tl.minimum(b, src_hi)
    c = b + tl.maximum(tl.minimum(band_hi, src_hi) - b, 0) // BLOCK_NB * BLOCK_NB
    return s, a, b, c


@triton.jit
def _extend_attention_split_kernel(
    q_ptr,
    k_extend_ptr,
    v_extend_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    o_ptr,
    gate_ptr,
    bias_ptr,
    sink_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    prefix_lens_ptr,
    row_lo,
    row_hi,
    qk_scale,
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
    stride_vss,
    stride_ot,
    stride_gt,
    stride_bt,
    stride_bh,
    stride_st,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_NB: tl.constexpr,
    BLOCK_NE: tl.constexpr,
    FAR_STAGES: tl.constexpr,
    BAND_STAGES: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    KV_FMT: tl.constexpr,
    BIT_DECODE: tl.constexpr,
):
    """One program owns BLOCK_M rows = BLOCK_M // GROUP tokens x the GROUP q heads of one KV head, so each K/V tile is read and decoded once per KV head.

    The prefix comes from the cache, this chunk's keys from its own K/V rows. Per source the keys split into tiles with no bias and no mask (BLOCK_N, out of every row's extent), with a bias and no mask (BLOCK_NB) and masked ones (BLOCK_NE)."""
    TOKENS: tl.constexpr = BLOCK_M // GROUP
    block_m_id = tl.program_id(0)
    kv_head = tl.program_id(1)
    seq_id = tl.program_id(2)

    q_start = tl.load(qo_indptr_ptr + seq_id)
    q_len = tl.load(qo_indptr_ptr + seq_id + 1) - q_start
    first = block_m_id * TOKENS
    if first >= q_len or q_start + first >= row_hi or q_start + tl.minimum(q_len, first + TOKENS) <= row_lo:
        return
    kv_start = tl.load(kv_indptr_ptr + seq_id)
    prefix_len = tl.load(prefix_lens_ptr + seq_id)
    zero_ptr = qo_indptr_ptr + seq_id

    # rows outside the launch read the nearest valid row and are never stored
    row_first = tl.maximum(q_start + first, row_lo)
    row_last = tl.minimum(tl.minimum(q_start + q_len, row_hi) - 1, q_start + first + TOKENS - 1)
    r = tl.arange(0, BLOCK_M)
    head = kv_head * GROUP + r % GROUP
    rows = tl.minimum(tl.maximum(q_start + first + r // GROUP, row_first), row_last)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + rows[:, None] * stride_qt + head[:, None] * stride_qh + offs_d[None, :])
    if BIT_DECODE and KV_FMT != "none":
        # with q at 2^64 the products of the 2^-120 / 2^-112 scaled codes stay clear of fp32 subnormals
        q = (q.to(tl.float32) * 18446744073709551616.0).to(tl.bfloat16)
        if KV_FMT == "fp8_per_token_head":
            K_MUL = 72057594037927936.0  # 2^56
            V_MUL = 1.2676506002282294e30  # 2^100
        else:
            K_MUL = 281474976710656.0  # 2^48
            V_MUL = 4.951760157141521e27  # 2^92
        E_K = 5.421010862427522e-20  # 2^-64
        # acc holds the output at 2^-20, so p * v_scale * V_MUL stays inside the bf16 range
        E_V = 9.5367431640625e-07
    else:
        K_MUL = 1.0
        V_MUL = 1.0
        E_K = 1.0
        E_V = 1.0
    # the sink starts the running max with one unit of mass, so no row is ever empty
    m_i = tl.load(sink_ptr + rows * stride_st + head).to(tl.float32) * 1.4426950408889634
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    kc_base = k_cache_ptr + kv_head * stride_kch
    vc_base = v_cache_ptr + kv_head * stride_vch
    ks_base = k_scale_ptr + kv_head
    vs_base = v_scale_ptr + kv_head
    ke_base = k_extend_ptr + kv_head * stride_keh
    ve_base = v_extend_ptr + kv_head * stride_veh
    kv_idx = kv_indices_ptr + kv_start

    p0 = prefix_len + row_first - q_start
    p1 = prefix_len + row_last - q_start
    end = p1 + 1
    if SLIDING_WINDOW > 0:
        vis_lo = tl.maximum(p0 - SLIDING_WINDOW + 1, 0)
        far_end = vis_lo
        reach = tl.minimum(SLIDING_WINDOW, REL_EXTENT)
    else:
        vis_lo = 0
        far_end = tl.maximum(p0 - REL_EXTENT + 1, 0)
        reach = REL_EXTENT
    # keys every row of the program sees with a bias
    band_lo = p1 - reach + 1
    band_hi = p0 + 1
    ck = qk_scale * K_MUL
    ek = qk_scale * E_K
    # the pipeliner cannot predicate the staged decode's inline asm on every arch
    CACHE_FAR: tl.constexpr = FAR_STAGES if KV_FMT == "none" else 1
    CACHE_BAND: tl.constexpr = BAND_STAGES if KV_FMT == "none" else 1

    s0, a, b, c = _extend_ranges(0, prefix_len, vis_lo, far_end, band_lo, band_hi, BLOCK_N, BLOCK_NB, BLOCK_NE)
    if SLIDING_WINDOW == 0:
        m_i, l_i, acc = _extend_keys(
            q, m_i, l_i, acc, s0, a, prefix_len, kc_base, vc_base, ks_base, vs_base, kv_idx, 0, stride_kcs, stride_vcs,
            stride_kss, stride_vss, ck, V_MUL, zero_ptr, q_start, first, kv_head, q_start, prefix_len, row_first,
            row_last, row_lo, bias_ptr, stride_bt, stride_bh,
            True, 0, BLOCK_N, CACHE_FAR, KV_FMT, BIT_DECODE, GROUP, BLOCK_M, D, SLIDING_WINDOW, REL_EXTENT)
    m_i, l_i, acc = _extend_keys(
        q, m_i, l_i, acc, a, b, prefix_len, kc_base, vc_base, ks_base, vs_base, kv_idx, 0, stride_kcs, stride_vcs,
        stride_kss, stride_vss, ck, V_MUL, zero_ptr, q_start, first, kv_head, q_start, prefix_len, row_first,
        row_last, row_lo, bias_ptr, stride_bt, stride_bh,
        True, 2, BLOCK_NE, 1, KV_FMT, BIT_DECODE, GROUP, BLOCK_M, D, SLIDING_WINDOW, REL_EXTENT)
    m_i, l_i, acc = _extend_keys(
        q, m_i, l_i, acc, b, c, prefix_len, kc_base, vc_base, ks_base, vs_base, kv_idx, 0, stride_kcs, stride_vcs,
        stride_kss, stride_vss, ck, V_MUL, zero_ptr, q_start, first, kv_head, q_start, prefix_len, row_first,
        row_last, row_lo, bias_ptr, stride_bt, stride_bh,
        True, 1, BLOCK_NB, CACHE_BAND, KV_FMT, BIT_DECODE, GROUP, BLOCK_M, D, SLIDING_WINDOW, REL_EXTENT)
    m_i, l_i, acc = _extend_keys(
        q, m_i, l_i, acc, c, prefix_len, prefix_len, kc_base, vc_base, ks_base, vs_base, kv_idx, 0, stride_kcs,
        stride_vcs, stride_kss, stride_vss, ck, V_MUL, zero_ptr, q_start, first, kv_head, q_start, prefix_len,
        row_first, row_last, row_lo, bias_ptr, stride_bt, stride_bh,
        True, 2, BLOCK_NE, 1, KV_FMT, BIT_DECODE, GROUP, BLOCK_M, D, SLIDING_WINDOW, REL_EXTENT)

    # this chunk's keys [prefix_len, end) are rows q_start - prefix_len + key of k_extend / v_extend
    s0, a, b, c = _extend_ranges(prefix_len, end, vis_lo, far_end, band_lo, band_hi, BLOCK_N, BLOCK_NB, BLOCK_NE)
    eb = q_start - prefix_len
    if SLIDING_WINDOW == 0:
        m_i, l_i, acc = _extend_keys(
            q, m_i, l_i, acc, s0, a, end, ke_base, ve_base, ks_base, vs_base, kv_idx, eb, stride_ket, stride_vet,
            stride_kss, stride_vss, ek, E_V, zero_ptr, q_start, first, kv_head, q_start, prefix_len, row_first,
            row_last, row_lo, bias_ptr, stride_bt, stride_bh,
            False, 0, BLOCK_N, FAR_STAGES, "none", BIT_DECODE, GROUP, BLOCK_M, D, SLIDING_WINDOW, REL_EXTENT)
    m_i, l_i, acc = _extend_keys(
        q, m_i, l_i, acc, a, b, end, ke_base, ve_base, ks_base, vs_base, kv_idx, eb, stride_ket, stride_vet,
        stride_kss, stride_vss, ek, E_V, zero_ptr, q_start, first, kv_head, q_start, prefix_len, row_first,
        row_last, row_lo, bias_ptr, stride_bt, stride_bh,
        False, 2, BLOCK_NE, 1, "none", BIT_DECODE, GROUP, BLOCK_M, D, SLIDING_WINDOW, REL_EXTENT)
    m_i, l_i, acc = _extend_keys(
        q, m_i, l_i, acc, b, c, end, ke_base, ve_base, ks_base, vs_base, kv_idx, eb, stride_ket, stride_vet,
        stride_kss, stride_vss, ek, E_V, zero_ptr, q_start, first, kv_head, q_start, prefix_len, row_first,
        row_last, row_lo, bias_ptr, stride_bt, stride_bh,
        False, 1, BLOCK_NB, BAND_STAGES, "none", BIT_DECODE, GROUP, BLOCK_M, D, SLIDING_WINDOW, REL_EXTENT)
    m_i, l_i, acc = _extend_keys(
        q, m_i, l_i, acc, c, end, end, ke_base, ve_base, ks_base, vs_base, kv_idx, eb, stride_ket, stride_vet,
        stride_kss, stride_vss, ek, E_V, zero_ptr, q_start, first, kv_head, q_start, prefix_len, row_first,
        row_last, row_lo, bias_ptr, stride_bt, stride_bh,
        False, 2, BLOCK_NE, 1, "none", BIT_DECODE, GROUP, BLOCK_M, D, SLIDING_WINDOW, REL_EXTENT)

    o = (acc / (l_i * E_V)[:, None]).to(tl.bfloat16).to(tl.float32)
    zero = tl.load(zero_ptr, volatile=True) - q_start
    r = tl.arange(0, BLOCK_M) + zero
    tok = first + r // GROUP
    head = kv_head * GROUP + r % GROUP
    rows = q_start + tok
    mask_m = (r < TOKENS * GROUP) & (tok < q_len) & (rows >= row_lo) & (rows < row_hi)
    offs_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, D) + zero, D), D)
    cols = head[:, None] * D + offs_d[None, :]
    g = tl.load(gate_ptr + rows[:, None] * stride_gt + cols, mask=mask_m[:, None], other=0.0).to(tl.float32)
    tl.store(o_ptr + rows[:, None] * stride_ot + cols, (o * tl.sigmoid(g)).to(o_ptr.dtype.element_ty), mask=mask_m[:, None])


def extend_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    gate: torch.Tensor,
    bias: torch.Tensor,
    sinks: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_q_len: int,
    sm_scale: float,
    sliding_window: int | None,
    rel_extent: int,
    out: torch.Tensor,
    k_extend: torch.Tensor,
    v_extend: torch.Tensor,
    row_lo: int = 0,
    row_hi: int | None = None,
    kv_format: KVFormat = NONE,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prefill / extend attention of the query rows ``[row_lo, row_hi)`` (default all).

    ``q`` ``[T, Hq, D]``; the prefix comes from the flat ``[slots, Hkv, D]`` K/V caches, this chunk's keys from its bf16 rows ``k_extend`` / ``v_extend`` ``[T, Hkv, D]``. ``bias`` ``[row_hi - row_lo, Hq, rel_extent]`` holds the window's rows; ``sinks`` ``[T, Hq]`` fp32; ``gate`` / ``out`` ``[T, Hq * D]``. Query positions are ``prefix_len + row-in-request``.
    """
    T, Hq, D = q.shape
    Hkv = k_cache.shape[1]
    G = Hq // Hkv
    row_hi = T if row_hi is None else row_hi
    assert bias.shape == (row_hi - row_lo, Hq, rel_extent) and bias.stride(-1) == 1, bias.shape
    assert sinks.shape == (T, Hq) and sinks.stride(-1) == 1
    assert gate.shape == out.shape == (T, Hq * D) and gate.stride(-1) == 1 and out.stride(-1) == 1
    assert k_extend.shape == v_extend.shape == (T, Hkv, D) and k_extend.stride(-1) == v_extend.stride(-1) == 1
    # 128 rows = 42 tokens x 3 heads; 64-key tiles out of the bias extent, 32 in it, 16 at the masked edges (sm_120).
    # three stages on the far keys spill next to an 8-bit cache decode
    block_m, block_n, block_nb, block_ne = 128, 64, 32, 16
    far_stages = 1 if kv_format.quantized else 3
    grid = (triton.cdiv(max_q_len, block_m // G), Hkv, qo_indptr.numel() - 1)
    k_scale, v_scale, stride_kss, stride_vss = kv_scale_args(kv_format, k_cache, v_cache, k_scale, v_scale, q)
    _extend_attention_split_kernel[grid](
        q, k_extend, v_extend, k_cache, v_cache, k_scale, v_scale, out, gate, bias, sinks,
        qo_indptr, kv_indptr, kv_indices, prefix_lens,
        row_lo, row_hi, sm_scale * _LOG2E,
        q.stride(0), q.stride(1), k_extend.stride(0), k_extend.stride(1), v_extend.stride(0), v_extend.stride(1),
        k_cache.stride(0), k_cache.stride(1), v_cache.stride(0), v_cache.stride(1), stride_kss, stride_vss,
        out.stride(0), gate.stride(0), bias.stride(0), bias.stride(1), sinks.stride(0),
        GROUP=G, D=D, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_NB=block_nb, BLOCK_NE=block_ne,
        FAR_STAGES=far_stages, BAND_STAGES=3, SLIDING_WINDOW=sliding_window or 0, REL_EXTENT=rel_extent,
        KV_FMT=kv_format.name, BIT_DECODE=torch.version.hip is None,
        num_warps=8, num_stages=1,
    )
    return out


@triton.jit
def _request_split_log2(keys, max_kv_splits, short_splits, MIN_SPLIT_LEN: tl.constexpr, MIN_BLOCK_KV: tl.constexpr):
    """log2 of the request's split count: the largest power of two <= its length rule, within [1, max_kv_splits]."""
    by_len = tl.maximum(tl.cdiv(keys, MIN_SPLIT_LEN), tl.minimum(short_splits, tl.cdiv(keys, MIN_BLOCK_KV)))
    n = tl.minimum(by_len, max_kv_splits)
    return (n >= 2).to(tl.int32) + (n >= 4).to(tl.int32) + (n >= 8).to(tl.int32) + (n >= 16).to(tl.int32) + (n >= 32).to(tl.int32) + (n >= 64).to(tl.int32)


@triton.jit(do_not_specialize=["max_kv_splits", "short_splits"])
def _decode_attention_stage1(
    q_ptr,
    k_ptr,
    v_ptr,
    k_scale_ptr,
    v_scale_ptr,
    bias_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    part_acc_ptr,
    part_ml_ptr,
    qk_scale,
    max_kv_splits,
    short_splits,
    stride_qt,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_kss,
    stride_vss,
    stride_bt,
    stride_bh,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    MIN_SPLIT_LEN: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    KV_FMT: tl.constexpr,
):
    b = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)

    kv_start = tl.load(kv_indptr_ptr + b)
    kv_len = tl.load(kv_indptr_ptr + b + 1) - kv_start
    q_pos = kv_len - 1  # one query row per request: the newest key
    if SLIDING_WINDOW > 0:
        lo = tl.maximum(q_pos - SLIDING_WINDOW + 1, 0)
    else:
        lo = 0
    lg = _request_split_log2(kv_len - lo, max_kv_splits, short_splits, MIN_SPLIT_LEN, MIN_BLOCK_KV)
    kv_splits = 1 << lg
    per = tl.cdiv((kv_len - lo + kv_splits - 1) >> lg, MIN_BLOCK_KV) * MIN_BLOCK_KV
    s_lo = lo + split * per
    s_hi = tl.minimum(s_lo + per, kv_len)

    offs_g = tl.arange(0, GROUP_PAD)
    mask_g = offs_g < GROUP
    heads = kv_head * GROUP + offs_g
    if split >= kv_splits or s_lo >= kv_len:
        # stage 2 drops a split by its m = -inf
        part = (b * HQ + heads) * max_kv_splits + split
        tl.store(part_ml_ptr + part * 2, tl.full((GROUP_PAD,), -float("inf"), tl.float32), mask=mask_g)
        tl.store(part_ml_ptr + part * 2 + 1, tl.zeros((GROUP_PAD,), tl.float32), mask=mask_g)
        return

    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    q = tl.load(q_ptr + b * stride_qt + heads[:, None] * stride_qh + offs_d[None, :], mask=mask_g[:, None], other=0.0)
    bias_rows = bias_ptr + b * stride_bt + heads[:, None] * stride_bh

    m_i = tl.full((GROUP_PAD,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((GROUP_PAD,), dtype=tl.float32)
    acc = tl.zeros((GROUP_PAD, D), dtype=tl.float32)
    for start in tl.range(s_lo, s_hi, BLOCK_N):
        k_pos = start + offs_n
        mask_n = k_pos < s_hi
        slots = tl.load(kv_indices_ptr + kv_start + k_pos, mask=mask_n, other=0).to(tl.int64)
        k = kv_tile(k_ptr + slots[None, :] * stride_ks + kv_head * stride_kh + offs_d[:, None], mask_n[None, :], KV_FMT)
        s = tl.dot(q, k)
        if KV_FMT != "none":
            s = s * tl.load(k_scale_ptr + slots * stride_kss + kv_head, mask=mask_n, other=0.0)[None, :]
        s = s * qk_scale
        dist = q_pos - k_pos
        bias = tl.load(bias_rows + dist[None, :], mask=mask_g[:, None] & (mask_n & (dist < REL_EXTENT))[None, :], other=0.0)
        s = s + bias.to(tl.float32) * 1.4426950408889634
        s = tl.where(mask_n[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(s - m_safe[:, None])
        v = kv_tile(v_ptr + slots[:, None] * stride_vs + kv_head * stride_vh + offs_d[None, :], mask_n[:, None], KV_FMT)
        if KV_FMT == "none":
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        else:
            scale_v = tl.load(v_scale_ptr + slots * stride_vss + kv_head, mask=mask_n, other=0.0)
            acc = acc * alpha[:, None] + tl.dot((p * scale_v[None, :]).to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    part = (b * HQ + heads) * max_kv_splits + split
    tl.store(part_acc_ptr + part[:, None] * D + offs_d[None, :], acc, mask=mask_g[:, None])
    tl.store(part_ml_ptr + part * 2, m_i, mask=mask_g)
    tl.store(part_ml_ptr + part * 2 + 1, l_i, mask=mask_g)


@triton.jit(do_not_specialize=["max_kv_splits"])
def _decode_attention_stage2(
    part_acc_ptr,
    part_ml_ptr,
    sink_ptr,
    gate_ptr,
    o_ptr,
    max_kv_splits,
    stride_st,
    stride_gt,
    stride_ot,
    HQ: tl.constexpr,
    D: tl.constexpr,
    SPLIT_TILE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    TWO_ROUNDS: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_d = tl.program_id(2) * BLOCK_DV + tl.arange(0, BLOCK_DV)
    sink = tl.load(sink_ptr + b * stride_st + h).to(tl.float32) * 1.4426950408889634
    # the first round's acc rows load without waiting for m; a row that no split wrote gets weight 0
    FIRST: tl.constexpr = SPLIT_TILE // 2 if TWO_ROUNDS else SPLIT_TILE
    offs_a = tl.arange(0, FIRST)
    part_a = (b * HQ + h) * max_kv_splits + offs_a
    valid_a = offs_a < max_kv_splits
    m_a = tl.load(part_ml_ptr + part_a * 2, mask=valid_a, other=-float("inf"))
    l_a = tl.load(part_ml_ptr + part_a * 2 + 1, mask=valid_a, other=0.0)
    acc_a = tl.load(part_acc_ptr + part_a[:, None] * D + offs_d[None, :], mask=valid_a[:, None], other=0.0)
    m = tl.maximum(sink, tl.max(m_a, axis=0))
    if TWO_ROUNDS:
        # short requests leave second-round splits empty: read their acc rows only where m says they ran
        part_b = part_a + FIRST
        m_b = tl.load(part_ml_ptr + part_b * 2)
        l_b = tl.load(part_ml_ptr + part_b * 2 + 1)
        acc_b = tl.load(part_acc_ptr + part_b[:, None] * D + offs_d[None, :], mask=(m_b > -float("inf"))[:, None], other=0.0)
        m = tl.maximum(m, tl.max(m_b, axis=0))
    w_a = tl.math.exp2(m_a - m)
    acc = tl.sum(tl.where(w_a[:, None] > 0, acc_a * w_a[:, None], 0.0), axis=0)
    l = tl.math.exp2(sink - m) + tl.sum(l_a * w_a, axis=0)
    if TWO_ROUNDS:
        w_b = tl.math.exp2(m_b - m)
        acc += tl.sum(acc_b * w_b[:, None], axis=0)
        l += tl.sum(l_b * w_b, axis=0)
    out = _gate_bf16(acc / l, gate_ptr + b * stride_gt + h * D + offs_d, offs_d < D)
    tl.store(o_ptr + b * stride_ot + h * D + offs_d, out.to(o_ptr.dtype.element_ty))


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _pow2_floor(n: int) -> int:
    return 1 << (max(1, n).bit_length() - 1)


def _decode_split_caps(B: int, Hkv: int, window: int | None, device_index: int) -> tuple[int, int]:
    """(grid splits, short-request splits), both powers of two: short requests stay within one round of stage 1 CTAs; when that leaves fewer than 8 splits per request, long requests may take two rounds."""
    slots = _DECODE_CTAS_PER_SM * _num_sms(device_index)
    short = _pow2_floor(min(_DECODE_MAX_SPLITS, slots // (B * Hkv)))
    if window:
        short = min(short, _pow2_floor(triton.cdiv(window, _DECODE_MIN_BLOCK_KV)))
        return short, short
    return (short if short >= 8 else _pow2_floor(2 * slots // (B * Hkv))), short


def decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    gate: torch.Tensor,
    bias: torch.Tensor,
    sinks: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    sliding_window: int | None,
    rel_extent: int,
    out: torch.Tensor,
    kv_format: KVFormat = NONE,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """One query row per request (the request's newest key): split-K over the KV range, each request taking splits by its own length up to caps set by the batch, so a captured graph serves any length."""
    B, Hq, D = q.shape
    Hkv = k_cache.shape[1]
    group = Hq // Hkv
    cap, short = _decode_split_caps(B, Hkv, sliding_window, q.device.index)
    part_acc = torch.empty((B * Hq * cap, D), dtype=torch.float32, device=q.device)
    part_ml = torch.empty((B * Hq * cap, 2), dtype=torch.float32, device=q.device)
    k_scale, v_scale, stride_kss, stride_vss = kv_scale_args(kv_format, k_cache, v_cache, k_scale, v_scale, q)
    _decode_attention_stage1[(B, Hkv, cap)](
        q, k_cache, v_cache, k_scale, v_scale, bias, kv_indptr, kv_indices, part_acc, part_ml, sm_scale * _LOG2E, cap, short,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), v_cache.stride(0), v_cache.stride(1),
        stride_kss, stride_vss, bias.stride(0), bias.stride(1),
        GROUP=group, GROUP_PAD=max(16, triton.next_power_of_2(group)), HQ=Hq, D=D, BLOCK_N=32,
        MIN_BLOCK_KV=_DECODE_MIN_BLOCK_KV, MIN_SPLIT_LEN=_DECODE_MIN_SPLIT_LEN, SLIDING_WINDOW=sliding_window or 0,
        REL_EXTENT=rel_extent, KV_FMT=kv_format.name, num_warps=4, num_stages=2,
    )
    tile = triton.next_power_of_2(cap)
    block_dv = max(32, min(D, _DECODE_STAGE2_ELEMS // tile))
    _decode_attention_stage2[(B, Hq, triton.cdiv(D, block_dv))](
        part_acc, part_ml, sinks, gate, out, cap, sinks.stride(0), gate.stride(0), out.stride(0),
        HQ=Hq, D=D, SPLIT_TILE=tile, BLOCK_DV=block_dv, TWO_ROUNDS=cap > short, num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# query / relative-state preparation
# ---------------------------------------------------------------------------


@triton.jit
def _clip_rows(x, threshold):
    """Per-row L2 clip of a ``[1, N]`` fp32 row, rounded to bf16 like the reference."""
    norm = tl.sqrt(tl.sum(x * x, axis=1))[:, None]
    return (x * tl.minimum(threshold / (norm + 1e-6), 1.0)).to(tl.bfloat16)


@triton.jit
def _q_prep_kernel(
    qs_ptr,
    q_scale_ptr,
    rel_scale_ptr,
    pos_ptr,
    q_out_ptr,
    rel_out_ptr,
    sink_out_ptr,
    stride_qs,
    rel_off,
    sink_off,
    q_clip,
    rel_clip,
    n_floor,
    alpha,
    HQ: tl.constexpr,
    D: tl.constexpr,
    DR: tl.constexpr,
    LOG_SCALE: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    offs_d = tl.arange(0, D)[None, :]
    offs_r = tl.arange(0, DR)[None, :]
    row = qs_ptr + t * stride_qs
    q = tl.load(row + h * D + offs_d).to(tl.float32) * tl.load(q_scale_ptr + h * D + offs_d).to(tl.float32)
    rel = tl.load(row + rel_off + h * DR + offs_r).to(tl.float32) * tl.load(rel_scale_ptr + h * DR + offs_r).to(tl.float32)
    q = _clip_rows(q, q_clip)
    rel = _clip_rows(rel, rel_clip)
    if LOG_SCALE:
        tau = 1.0 + alpha * tl.log(tl.maximum((tl.load(pos_ptr + t).to(tl.float32) + 1.0) / n_floor, 1.0))
        q = (q.to(tl.float32) * tau).to(tl.bfloat16)
        rel = (rel.to(tl.float32) * tau).to(tl.bfloat16)
    tl.store(q_out_ptr + (t * HQ + h) * D + offs_d, q)
    tl.store(rel_out_ptr + (t * HQ + h) * DR + offs_r, rel)
    tl.store(sink_out_ptr + t * HQ + h, tl.load(row + sink_off + h).to(tl.float32))


def q_prep(
    q_side: torch.Tensor,
    q_scale: torch.Tensor,
    rel_scale: torch.Tensor,
    positions: torch.Tensor,
    *,
    rel_off: int,
    sink_off: int,
    q_clip: float,
    rel_clip: float,
    log_scaling: tuple[float, float] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(q [T, Hq, D] bf16, rel [T, Hq, DR] bf16, sinks [T, Hq] fp32)`` from the fused q-side
    projection: per-head ``l2clip(x * scale)``, then ``* log_scale(pos)`` when ``log_scaling``
    is ``(n_floor, alpha)``."""
    T = q_side.shape[0]
    Hq, D = q_scale.shape
    DR = rel_scale.shape[1]
    q = torch.empty((T, Hq, D), dtype=torch.bfloat16, device=q_side.device)
    rel = torch.empty((T, Hq, DR), dtype=torch.bfloat16, device=q_side.device)
    sinks = torch.empty((T, Hq), dtype=torch.float32, device=q_side.device)
    if T == 0:
        return q, rel, sinks
    n_floor, alpha = log_scaling if log_scaling is not None else (1.0, 0.0)
    _q_prep_kernel[(T, Hq)](
        q_side, q_scale, rel_scale, positions, q, rel, sinks, q_side.stride(0), rel_off, sink_off,
        q_clip, rel_clip, float(n_floor), float(alpha),
        HQ=Hq, D=D, DR=DR, LOG_SCALE=log_scaling is not None, num_warps=1,
    )
    return q, rel, sinks


# ---------------------------------------------------------------------------
# short convolutions
# ---------------------------------------------------------------------------

EPI_NONE = 0
EPI_L2_CLIP = 1  # per head
EPI_RMS_NORM = 2  # per head, no weight
EPI_RESIDUAL = 3  # out = round(r + round(conv)), r = res (+ res2)


@triton.jit
def _short_conv_kernel(
    x_ptr,
    out_ptr,
    w_ptr,
    dyn_ptr,
    rows_ptr,
    row_map_ptr,
    kv_indices_ptr,
    indptr_ptr,
    q_to_req_ptr,
    cu_q_ptr,
    prefix_lens_ptr,
    res_ptr,
    res2_ptr,
    T,
    C,
    scale,
    epi_arg,
    stride_xt,
    stride_ot,
    stride_dt,
    stride_row,
    stride_rt,
    stride_r2t,
    HEAD_DIM: tl.constexpr,
    K: tl.constexpr,
    DYNAMIC: tl.constexpr,
    DECODE: tl.constexpr,
    EPI: tl.constexpr,
    HAS_RES2: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    offs_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_t = offs_t < T
    mask_c = offs_c < C
    mask = mask_t[:, None] & mask_c[None, :]
    if DECODE:
        req = offs_t
        local = tl.zeros((BLOCK_T,), dtype=tl.int32)
    else:
        req = tl.load(q_to_req_ptr + offs_t, mask=mask_t, other=0)
        local = offs_t - tl.load(cu_q_ptr + req, mask=mask_t, other=0)
    pos = tl.load(prefix_lens_ptr + req, mask=mask_t, other=0) + local
    base = tl.load(indptr_ptr + req, mask=mask_t, other=0).to(tl.int64)

    x_raw = tl.load(x_ptr + offs_t[:, None] * stride_xt + offs_c[None, :], mask=mask, other=0.0)
    x = x_raw.to(tl.float32)
    acc = x * (tl.load(w_ptr + offs_c * K, mask=mask_c, other=0.0)[None, :] + 1.0)
    if DYNAMIC:
        acc += x * tl.load(dyn_ptr + offs_t[:, None] * stride_dt + (offs_c // HEAD_DIM)[None, :] * K, mask=mask, other=0.0).to(tl.float32)
    for lag in tl.static_range(1, K):
        in_chunk = local - lag >= 0
        prev = tl.load(x_ptr + (offs_t - lag)[:, None] * stride_xt + offs_c[None, :], mask=mask & in_chunk[:, None], other=0.0).to(tl.float32)
        # an input before the chunk is the row of the token ``lag`` back; none before the sequence starts
        before = mask_t & ~in_chunk & (pos - lag >= 0)
        slot = tl.load(kv_indices_ptr + base + pos - lag, mask=before, other=0).to(tl.int64)
        row = tl.load(row_map_ptr + slot, mask=before, other=0)
        from_rows = tl.load(rows_ptr + row[:, None] * stride_row + offs_c[None, :], mask=mask & before[:, None], other=0.0).to(tl.float32)
        prev = tl.where(in_chunk[:, None], prev, from_rows)
        w = tl.load(w_ptr + offs_c * K + lag, mask=mask_c, other=0.0)[None, :]
        if DYNAMIC:
            w = w + tl.load(dyn_ptr + offs_t[:, None] * stride_dt + (offs_c // HEAD_DIM)[None, :] * K + lag, mask=mask, other=0.0).to(tl.float32)
        acc += prev * w
    # keep this token's input for the tokens after it, if the cache gave it a row (row 0: none)
    slot = tl.load(kv_indices_ptr + base + pos, mask=mask_t, other=0).to(tl.int64)
    row = tl.load(row_map_ptr + slot, mask=mask_t, other=0)
    tl.store(rows_ptr + row[:, None] * stride_row + offs_c[None, :], x_raw.to(rows_ptr.dtype.element_ty), mask=mask & (row > 0)[:, None])

    # the reference rounds the convolution to bf16 before anything reads it
    y = (acc * scale).to(tl.bfloat16).to(tl.float32)
    if EPI == 1:
        heads = tl.reshape(y, (BLOCK_T, BLOCK_C // HEAD_DIM, HEAD_DIM))
        norm = tl.sqrt(tl.sum(heads * heads, axis=2))[:, :, None]
        y = tl.reshape(heads * tl.minimum(epi_arg / (norm + 1e-6), 1.0), (BLOCK_T, BLOCK_C))
    elif EPI == 2:
        heads = tl.reshape(y, (BLOCK_T, BLOCK_C // HEAD_DIM, HEAD_DIM))
        inv = 1.0 / tl.sqrt(tl.sum(heads * heads, axis=2) / HEAD_DIM + epi_arg)[:, :, None]
        y = tl.reshape(heads * inv, (BLOCK_T, BLOCK_C))
    elif EPI == 3:
        r = tl.load(res_ptr + offs_t[:, None] * stride_rt + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
        if HAS_RES2:
            r += tl.load(res2_ptr + offs_t[:, None] * stride_r2t + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
        y = r + y
    tl.store(out_ptr + offs_t[:, None] * stride_ot + offs_c[None, :], y.to(out_ptr.dtype.element_ty), mask=mask)


def short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    rows: torch.Tensor,
    row_map: torch.Tensor,
    kv_indices: torch.Tensor,
    indptr: torch.Tensor,
    prefix_lens: torch.Tensor,
    scale: float,
    *,
    decode: bool,
    q_to_req: torch.Tensor | None = None,
    cu_q: torch.Tensor | None = None,
    dynamic: torch.Tensor | None = None,
    head_dim: int = 128,
    epilogue: int = EPI_NONE,
    epi_arg: float = 0.0,
    residual: torch.Tensor | None = None,
    residual2: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``(sum_lag w[c, lag] * x[t - lag, c] + x[t, c]) * scale`` for each row.

    ``weight`` ``[C, K]`` fp32 indexed by backward lag (lag 0 = the current row). ``dynamic``
    ``[T, C // head_dim, K]`` adds per-row taps shared by the ``head_dim`` channels of a head.
    The inputs of the tokens before a request's chunk come from ``rows`` ``[num_rows, C]`` (row
    stride may exceed C): token ``p`` of request ``r`` is at row ``row_map[kv_indices[indptr[r] +
    p]]``; ``prefix_lens[r]`` is the chunk start, and the sequence starts from zeros. Every token
    whose row is not 0 writes its input there. Decode: one row per request, no q metadata.
    The bf16-rounded output then goes through ``epilogue``: a per-head L2 clip to ``epi_arg``,
    a per-head RMSNorm with eps ``epi_arg``, or ``residual (+ residual2)`` added in fp32.
    """
    T, C = x.shape
    K = weight.shape[1]
    assert weight.shape[0] == C and rows.shape[1] == C and rows.stride(-1) == 1 and x.stride(-1) == 1
    out = torch.empty((T, C), dtype=out_dtype or x.dtype, device=x.device)
    if T == 0:
        return out
    if dynamic is not None:
        assert dynamic.shape == (T, C // head_dim, K) and dynamic.stride(-1) == 1 and dynamic.stride(-2) == K
    block_t, block_c = 16, 256
    assert epilogue not in (EPI_L2_CLIP, EPI_RMS_NORM) or block_c % head_dim == 0
    _short_conv_kernel[(triton.cdiv(T, block_t), triton.cdiv(C, block_c))](
        x, out, weight, dynamic if dynamic is not None else x, rows, row_map, kv_indices, indptr,
        prefix_lens if decode else q_to_req, prefix_lens if decode else cu_q, prefix_lens,
        residual if residual is not None else x, residual2 if residual2 is not None else x,
        T, C, scale, epi_arg, x.stride(0), out.stride(0), dynamic.stride(0) if dynamic is not None else 0,
        rows.stride(0),
        residual.stride(0) if residual is not None else 0, residual2.stride(0) if residual2 is not None else 0,
        HEAD_DIM=head_dim, K=K, DYNAMIC=dynamic is not None, DECODE=decode, EPI=epilogue, HAS_RES2=residual2 is not None,
        BLOCK_T=block_t, BLOCK_C=block_c, num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# attention residual
# ---------------------------------------------------------------------------


@triton.jit
def _attn_res_kernel(
    blk_ptr,
    part_ptr,
    part2_ptr,
    q_ptr,
    w_ptr,
    out_ptr,
    H,
    num_blocks,
    stride_bt,
    stride_bs,
    stride_pt,
    stride_p2t,
    stride_oq,
    stride_ot,
    scale,
    eps,
    norm_eps,
    NB_PAD: tl.constexpr,
    HAS_PART2: tl.constexpr,
    NQ: tl.constexpr,
    NORM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_H)
    mask_h = offs < H
    offs_s = tl.arange(0, NB_PAD)
    mask_s = offs_s < num_blocks
    blk = tl.load(blk_ptr + t * stride_bt + offs_s[:, None] * stride_bs + offs[None, :], mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
    part = tl.load(part_ptr + t * stride_pt + offs, mask=mask_h, other=0.0).to(tl.float32)
    if HAS_PART2:
        part += tl.load(part2_ptr + t * stride_p2t + offs, mask=mask_h, other=0.0).to(tl.float32)
    blk = blk / tl.sqrt(tl.sum(blk * blk, axis=1) / H + eps)[:, None]
    part = part / tl.sqrt(tl.sum(part * part, axis=0) / H + eps)
    for q in tl.static_range(NQ):
        query = tl.load(q_ptr + q * H + offs, mask=mask_h, other=0.0).to(tl.float32)
        s_blk = tl.where(mask_s, tl.sum(blk * query[None, :], axis=1) * scale, -float("inf"))
        s_part = tl.sum(part * query, axis=0) * scale
        mx = tl.maximum(tl.max(s_blk, axis=0), s_part)
        e_blk = tl.exp(s_blk - mx)
        e_part = tl.exp(s_part - mx)
        y = (tl.sum(e_blk[:, None] * blk, axis=0) + e_part * part) / (tl.sum(e_blk, axis=0) + e_part)
        if NORM:
            y = y / tl.sqrt(tl.sum(y * y, axis=0) / H + norm_eps) * tl.load(w_ptr + q * H + offs, mask=mask_h, other=0.0).to(tl.float32)
        tl.store(out_ptr + q * stride_oq + t * stride_ot + offs, y.to(out_ptr.dtype.element_ty), mask=mask_h)


def attn_res(
    blocks: torch.Tensor | None,
    num_blocks: int,
    partial: torch.Tensor,
    queries: torch.Tensor,
    *,
    partial2: torch.Tensor | None = None,
    eps: float = 1e-6,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[NQ, T, H]``: per row, ``softmax_s(q . n_s / sqrt(H)) @ n_s`` over ``n_s = rmsnorm(src_s)``.

    The sources are the first ``num_blocks`` of ``blocks`` ``[T, S, H]`` and the partial row
    ``partial (+ partial2)``, summed in fp32. ``queries`` ``[NQ, H]`` fp32. With
    ``norm_weight`` ``[NQ, H]`` each mixed row also goes through a weighted RMSNorm
    (``norm_eps``) before it is stored in ``out_dtype``.
    """
    T, H = partial.shape
    NQ = queries.shape[0]
    assert 1 <= NQ <= 4 and 0 <= num_blocks <= 4 and partial.stride(-1) == 1
    out = torch.empty((NQ, T, H), dtype=out_dtype, device=partial.device)
    if T == 0:
        return out
    blk = blocks if blocks is not None and num_blocks > 0 else partial.view(T, 1, H)
    assert blk.stride(-1) == 1
    _attn_res_kernel[(T,)](
        blk, partial, partial2 if partial2 is not None else partial, queries,
        norm_weight if norm_weight is not None else queries, out,
        H, num_blocks, blk.stride(0), blk.stride(1), partial.stride(0),
        partial2.stride(0) if partial2 is not None else 0, out.stride(0), out.stride(1),
        H**-0.5, eps, norm_eps,
        NB_PAD=4, HAS_PART2=partial2 is not None, NQ=NQ, NORM=norm_weight is not None,
        BLOCK_H=triton.next_power_of_2(H), num_warps=16,
    )
    return out


# ---------------------------------------------------------------------------
# message gates
# ---------------------------------------------------------------------------


@triton.jit
def _gate_inputs_kernel(
    x_ptr,
    m_ptr,
    out_ptr,
    H,
    stride_xg,
    stride_xt,
    stride_mg,
    stride_mt,
    stride_og,
    stride_ot,
    eps,
    BLOCK_H: tl.constexpr,
):
    g = tl.program_id(0)
    t = tl.program_id(1).to(tl.int64)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    x = tl.load(x_ptr + g * stride_xg + t * stride_xt + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + g * stride_mg + t * stride_mt + offs, mask=mask, other=0.0).to(tl.float32)
    row = out_ptr + g * stride_og + t * stride_ot
    tl.store(row + offs, (x / tl.sqrt(tl.sum(x * x, axis=0) / H + eps)).to(tl.bfloat16), mask=mask)
    tl.store(row + H + offs, (m / tl.sqrt(tl.sum(m * m, axis=0) / H + eps)).to(tl.bfloat16), mask=mask)


def gate_inputs(x: torch.Tensor, m: torch.Tensor, eps: float) -> torch.Tensor:
    """``[G, T, 2H]`` bf16 ``[rmsnorm(x), rmsnorm(m)]`` (no weights) of ``x`` / ``m`` ``[G, T, H]``."""
    G, T, H = m.shape
    out = torch.empty((G, T, 2 * H), dtype=torch.bfloat16, device=m.device)
    if T == 0:
        return out
    _gate_inputs_kernel[(G, T)](
        x, m, out, H, x.stride(0), x.stride(1), m.stride(0), m.stride(1), out.stride(0), out.stride(1), eps,
        BLOCK_H=triton.next_power_of_2(H), num_warps=8,
    )
    return out


@triton.jit
def _gate_combine_kernel(
    fg_ptr,
    x_ptr,
    m_ptr,
    w_ptr,
    out_ptr,
    H,
    stride_fg,
    stride_ft,
    stride_xg,
    stride_xt,
    stride_mg,
    stride_mt,
    stride_wg,
    stride_og,
    stride_ot,
    norm_eps,
    NORM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    g = tl.program_id(0)
    t = tl.program_id(1).to(tl.int64)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    fg = fg_ptr + g * stride_fg + t * stride_ft
    forget = tl.load(fg + offs, mask=mask, other=0.0)
    gain = tl.load(fg + H + offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + g * stride_xg + t * stride_xt + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + g * stride_mg + t * stride_mt + offs, mask=mask, other=0.0).to(tl.float32)
    y = (1.0 - _tanh(forget)) * x + _tanh(gain) * m
    if NORM:
        y = y / tl.sqrt(tl.sum(y * y, axis=0) / H + norm_eps) * tl.load(w_ptr + g * stride_wg + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + g * stride_og + t * stride_ot + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _tanh(x):
    e = tl.exp(-2.0 * tl.abs(x))
    t = (1.0 - e) / (1.0 + e)
    return tl.where(x < 0, -t, t)


def gate_combine(
    forget_gain: torch.Tensor,
    x: torch.Tensor,
    m: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    out_dtype: torch.dtype = torch.float32,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(1 - tanh(F z)) * x + tanh(G z) * m`` per gate, ``forget_gain`` ``[G, T, 2H]`` holding
    ``[F z, G z]``; with ``norm_weight`` ``[G, H]`` the result is RMS-normalized and scaled.
    Written into ``out`` when given, else into a new ``out_dtype`` tensor."""
    G, T, H = m.shape
    if out is None:
        out = torch.empty((G, T, H), dtype=out_dtype, device=m.device)
    if T == 0:
        return out
    w = norm_weight if norm_weight is not None else m
    _gate_combine_kernel[(G, T)](
        forget_gain, x, m, w, out, H,
        forget_gain.stride(0), forget_gain.stride(1), x.stride(0), x.stride(1), m.stride(0), m.stride(1),
        w.stride(0) if norm_weight is not None else 0, out.stride(0), out.stride(1), norm_eps,
        NORM=norm_weight is not None, BLOCK_H=triton.next_power_of_2(H), num_warps=8,
    )
    return out


__all__ = [
    "EPI_L2_CLIP",
    "EPI_NONE",
    "EPI_RESIDUAL",
    "EPI_RMS_NORM",
    "attn_res",
    "decode_attention",
    "extend_attention",
    "gate_combine",
    "gate_inputs",
    "q_prep",
    "short_conv",
]
