"""Multi-CTA (split-vocab) Triton sampling ops.

Optional pure-triton drop-in for freetoken.kernel.sampling / flashinfer.sampling
(softmax / top-k / top-p / combined + draw), self-contained.

Design:
  * Every row is split across many CTAs (``_plan`` -> G column-chunks) so bs=1 uses
    the whole GPU, unlike a single-block-per-row kernel that is single-SM-bound.
  * softmax is a multi-CTA online softmax; the draw is a multi-CTA inverse-CDF.
  * top-k and top-p each use one Triton kernel: the row's CTAs bin their chunk over
    the fp32 bit pattern (order-preserving for x >= 0), meet at a per-row spin barrier, and all
    redo the refine so they share the bracket. Four rounds of 256 bins bring the 2**31 range
    down to one bit pattern, so the threshold is exactly the k-th largest prob (top-k, counts)
    or the value where the descending cumulative mass reaches p (top-p, exact per-bin mass).
    Every boundary tie is kept, matching flashinfer, then the same kernel renormalizes or
    draws. No candidate buffer, data-dependent shape, or host sync is needed. Results are
    exact up to fp32 atomic summation order.
  * If a cooperative launch is unavailable, the same exact kernel is retried with one CTA
    per row; only parallelism changes.
  * deterministic, generator and check_nan exist for flashinfer signature compatibility and are
    ignored; seed and offset are honored. Given a seed, top-k draws reproduce; top-p may pick a
    different token on rows whose cumulative mass sits within fp32 rounding of p.
"""

from __future__ import annotations

import logging
from functools import cache

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.autotune_cache import autotune_cache_kwargs

logger = logging.getLogger(__name__)

_MIN_CHUNK = 4096  # do not split a row finer than this


@cache
def _num_sm(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def _plan(B, V, device):
    """Return (G, CHUNK): split each row into G column-chunks of size CHUNK."""
    g_by_sm = max(1, _num_sm(device) // B)
    g_by_chunk = max(1, triton.cdiv(V, _MIN_CHUNK))
    G = min(g_by_sm, g_by_chunk)
    CHUNK = triton.cdiv(V, G)
    return G, CHUNK


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


# ---------------------------------------------------------------------------
# softmax(logits / temperature)  -- multi-CTA online softmax
# ---------------------------------------------------------------------------
_SM_CFGS = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
    for bs in (1024, 2048, 4096)
    for w in (4, 8)
    for s in (1, 2)
]


@triton.autotune(configs=_SM_CFGS, key=["CHUNK"], **autotune_cache_kwargs)
@triton.jit
def _sm_partial(
    logits_ptr, pm_ptr, pl_ptr, temp_ptr, temp_scalar, HAS_TEMP: tl.constexpr,
    V, G, CHUNK, row_stride, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // G
    sp = pid % G
    if HAS_TEMP:
        inv_t = 1.0 / tl.load(temp_ptr + row)
    else:
        inv_t = 1.0 / temp_scalar
    base = row * row_stride
    start = sp * CHUNK
    end = tl.minimum(start + CHUNK, V)

    m = -float("inf")
    d = 0.0
    for s0 in tl.range(start, end, BLOCK_SIZE):
        offs = s0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        x = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32) * inv_t
        blk_max = tl.max(x, 0)
        new_m = tl.maximum(m, blk_max)
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), 0)
        m = new_m
    tl.store(pm_ptr + pid, m)
    tl.store(pl_ptr + pid, d)


@triton.autotune(configs=_SM_CFGS, key=["CHUNK"], **autotune_cache_kwargs)
@triton.jit
def _sm_finalize(
    logits_ptr, probs_ptr, pm_ptr, pl_ptr, temp_ptr, temp_scalar, HAS_TEMP: tl.constexpr,
    V, G, CHUNK, row_stride, G_POW2: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // G
    sp = pid % G
    if HAS_TEMP:
        inv_t = 1.0 / tl.load(temp_ptr + row)
    else:
        inv_t = 1.0 / temp_scalar

    goff = tl.arange(0, G_POW2)
    gmask = goff < G
    pm = tl.load(pm_ptr + row * G + goff, mask=gmask, other=-float("inf"))
    pl = tl.load(pl_ptr + row * G + goff, mask=gmask, other=0.0)
    gm = tl.max(pm, 0)
    gl = tl.sum(pl * tl.exp(pm - gm), 0)
    inv_gl = 1.0 / gl

    base = row * row_stride
    start = sp * CHUNK
    end = tl.minimum(start + CHUNK, V)
    for s0 in tl.range(start, end, BLOCK_SIZE):
        offs = s0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        x = tl.load(logits_ptr + base + offs, mask=mask, other=0.0).to(tl.float32) * inv_t
        p = tl.exp(x - gm) * inv_gl
        tl.store(probs_ptr + base + offs, p, mask=mask)


def softmax(logits, temperature=None, enable_pdl=None):
    logits = logits.float()
    B, V = logits.shape
    if B == 0:
        return logits.clone()
    probs = torch.empty_like(logits)
    G, CHUNK = _plan(B, V, logits.device)
    if temperature is None:
        temperature = 1.0
    if isinstance(temperature, torch.Tensor):
        temp_arr = temperature.float().contiguous()
        has_temp, temp_scalar = True, 1.0
    else:
        temp_arr, has_temp, temp_scalar = None, False, float(temperature)
    pm = torch.empty(B * G, device=logits.device, dtype=torch.float32)
    pl = torch.empty(B * G, device=logits.device, dtype=torch.float32)
    grid = (B * G,)
    _sm_partial[grid](logits, pm, pl, temp_arr, temp_scalar, has_temp, V, G, CHUNK, logits.stride(0))
    _sm_finalize[grid](logits, probs, pm, pl, temp_arr, temp_scalar, has_temp, V, G, CHUNK,
                       logits.stride(0), _next_pow2(G))
    return probs


_SR_CFGS = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
    for bs in (1024, 2048, 4096)
    for w in (4, 8)
    for s in (1, 2)
]


def top_p_renorm_probs(probs, top_p):
    probs = probs.float()
    return _topp(probs, _topp_target(top_p, probs.size(0), probs.device), None, False)


# ---------------------------------------------------------------------------
# multi-CTA inverse-CDF draw
# ---------------------------------------------------------------------------
@triton.autotune(configs=_SR_CFGS, key=["CHUNK"], reset_to_zero=["psum_ptr"], **autotune_cache_kwargs)
@triton.jit
def _draw_part(probs_ptr, thr_ptr, psum_ptr, V, G, CHUNK, row_stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // G
    thr = tl.load(thr_ptr + row)
    base = row * row_stride
    start = (pid % G) * CHUNK
    end = tl.minimum(start + CHUNK, V)
    s = 0.0
    for s0 in tl.range(start, end, BLOCK_SIZE):
        offs = s0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        x = tl.load(probs_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(tl.where(x >= thr, x, 0.0), 0)
    tl.store(psum_ptr + pid, s)


@triton.jit
def _draw_scan(psum_ptr, choff_ptr, u_ptr, target_ptr, last_ptr, G, G_POW2: tl.constexpr):
    row = tl.program_id(0)
    goff = tl.arange(0, G_POW2)
    gmask = goff < G
    ps = tl.load(psum_ptr + row * G + goff, mask=gmask, other=0.0)
    tl.store(choff_ptr + row * G + goff, tl.cumsum(ps, 0) - ps, mask=gmask)
    tl.store(target_ptr + row, tl.load(u_ptr + row) * tl.sum(ps, 0))
    tl.store(last_ptr + row, tl.max(tl.where(gmask & (ps > 0), goff, -1), 0))


@triton.autotune(configs=_SR_CFGS, key=["CHUNK"], **autotune_cache_kwargs)
@triton.jit
def _draw_find(probs_ptr, thr_ptr, choff_ptr, target_ptr, psum_ptr, last_ptr, out_ptr, V, G, CHUNK, row_stride,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // G
    thr = tl.load(thr_ptr + row)
    target = tl.load(target_ptr + row)
    acc = tl.load(choff_ptr + pid)
    incl = acc + tl.load(psum_ptr + pid)
    base = row * row_stride
    start = (pid % G) * CHUNK
    end = tl.minimum(start + CHUNK, V)
    last_kept = start * 0 - 1
    for s0 in tl.range(start, end, BLOCK_SIZE):
        offs = s0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        x = tl.load(probs_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        kept = (x >= thr) & mask
        wv = tl.where(kept, x, 0.0)
        cval = acc + tl.cumsum(wv, 0)
        idx = tl.where(cval > target, offs, V)
        blk_min = tl.min(idx, 0)
        if (blk_min < V) and (acc <= target):
            tl.store(out_ptr + row, blk_min)
        acc += tl.sum(wv, 0)
        last_kept = tl.maximum(last_kept, tl.max(tl.where(kept, offs, -1), 0))
    # see _keep_tail: the CTA owning an fp rounding gap writes its last kept token
    is_last_mass = tl.load(last_ptr + row) == pid % G
    if (acc <= target) and (last_kept >= 0) and ((incl > target) or is_last_mass):
        tl.store(out_ptr + row, last_kept)


_UGEN = {}


def _gen_u(B, device, seed, offset):
    if torch.cuda.is_current_stream_capturing():
        return torch.rand(B, device=device, dtype=torch.float32)
    g = _UGEN.get(device)
    if g is None:
        g = torch.Generator(device=device)
        _UGEN[device] = g
    if seed is not None:
        s = int(seed if not isinstance(seed, torch.Tensor) else seed.view(-1)[0])
        o = 0 if offset is None else int(offset if not isinstance(offset, torch.Tensor) else offset.view(-1)[0])
        g.manual_seed((s * 0x9E3779B97F4A7C15 + o) & 0x7FFFFFFFFFFFFFFF)
    return torch.rand(B, device=device, generator=g, dtype=torch.float32)


def _draw(probs, thr, seed, offset):
    B, V = probs.shape
    dev = probs.device
    if B == 0:
        return torch.empty(0, device=dev, dtype=torch.int32)
    G, CHUNK = _plan(B, V, dev)
    grid = (B * G,)
    psum = torch.empty(B * G, device=dev, dtype=torch.float32)
    choff = torch.empty(B * G, device=dev, dtype=torch.float32)
    target = torch.empty(B, device=dev, dtype=torch.float32)
    last = torch.empty(B, device=dev, dtype=torch.int32)
    out = torch.zeros(B, device=dev, dtype=torch.int32)
    u = _gen_u(B, dev, seed, offset)
    _draw_part[grid](probs, thr, psum, V, G, CHUNK, probs.stride(0))
    _draw_scan[(B,)](psum, choff, u, target, last, G, _next_pow2(G))
    _draw_find[grid](probs, thr, choff, target, psum, last, out, V, G, CHUNK, probs.stride(0))
    return out


def _zeros_thr(B, dev):
    return torch.zeros(B, device=dev, dtype=torch.float32)


def sampling_from_probs(probs, indices=None, deterministic=True, generator=None,
                        check_nan=False, seed=None, offset=None, return_valid=False):
    probs = probs.float()
    src = probs if indices is None else probs[indices].contiguous()
    out = _draw(src, _zeros_thr(src.size(0), src.device), seed, offset)
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


def top_p_sampling_from_probs(probs, top_p, indices=None, deterministic=True, generator=None,
                              check_nan=False, seed=None, offset=None, return_valid=False):
    probs = probs.float()
    src = probs if indices is None else probs[indices].contiguous()
    out = _topp(src, _topp_target(top_p, src.size(0), src.device), None, True, seed, offset)
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


# ===========================================================================
# top-k: one cooperative kernel per call. Every CTA of a row histograms its column chunk
# over the fp32 bit pattern, the row's CTAs meet at a spin barrier, then each one redoes
# the tiny refine step so all of them hold the same bracket. Four rounds (exponent, then
# 8+8+7 mantissa bits) end on a single bit pattern, so thr is exactly the k-th largest
# prob. The same kernel then either renormalizes (DRAW=0) or draws a token (DRAW=1).
# ===========================================================================
_KBINS = 256
_INF_BITS = tl.constexpr(0x7F800000)
_FUSED_BLOCK = 2048


def _topk_target(top_k, B, dev):
    if isinstance(top_k, torch.Tensor):
        return top_k.to(device=dev, dtype=torch.int32).contiguous()
    return torch.full((B,), max(int(top_k), 1), device=dev, dtype=torch.int32)


@triton.jit
def _row_barrier(bar_ptr, need):
    # the row's CTAs must be co-resident (cooperative launch), or a lone CTA (G == 1) passes at once.
    # every warp's preceding atomics must be issued before thread 0 announces arrival
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1)
    n = tl.atomic_add(bar_ptr, 0)
    while n < need:
        n = tl.atomic_add(bar_ptr, 0)


@triton.jit
def _bits_round(
    probs_ptr, base, start, end, hist_ptr, bar_ptr, target, lo, above, need,
    S: tl.constexpr, WIDTH: tl.constexpr, BINS: tl.constexpr, BLOCK: tl.constexpr,
):
    jj = tl.arange(0, BINS)
    acc = tl.zeros([BINS], tl.int32)
    for s0 in tl.range(start, end, BLOCK):
        offs = s0 + tl.arange(0, BLOCK)
        mask = offs < end
        y = tl.load(probs_ptr + base + offs, mask=mask, other=-1.0).to(tl.float32).to(tl.int32, bitcast=True)
        d = y - lo
        if WIDTH == 0:
            inrange = mask & (y >= lo) & (y <= _INF_BITS)
        else:
            inrange = mask & (y >= lo) & (d < WIDTH) & (y <= _INF_BITS)
        # every out-of-bracket lane (padding included) lands in bin 0 and is subtracted back out
        b = tl.where(inrange, d >> S, 0)
        h = tl.histogram(b, BINS)
        acc += h - tl.where(jj == 0, tl.sum((~inrange).to(tl.int32)), 0)
    tl.atomic_add(hist_ptr + jj, acc)
    _row_barrier(bar_ptr, need)
    h = tl.load(hist_ptr + jj, cache_modifier=".cg")
    prefix = tl.cumsum(h, 0)
    total = tl.sum(h, 0)
    ok = above + total - prefix + h >= target
    j = tl.max(tl.where(ok, jj, -1))
    prefix_j = tl.sum(tl.where(jj <= j, h, 0))
    upd = j >= 0
    lo = tl.where(upd, lo + (j << S), lo)
    above = tl.where(upd, above + total - prefix_j, above)
    return lo, above


@triton.jit
def _topk_fused(
    probs_ptr, target_ptr, hist_ptr, bar_ptr, ksum_ptr, psum_ptr, u_ptr, out_ptr, tok_ptr,
    V, G, CHUNK, row_stride,
    DRAW: tl.constexpr, G_POW2: tl.constexpr, BINS: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // G
    cta = pid % G
    base = row * row_stride
    start = cta * CHUNK
    end = tl.minimum(start + CHUNK, V)
    target = tl.maximum(tl.load(target_ptr + row), 1)
    hrow = hist_ptr + row * 4 * BINS
    brow = bar_ptr + row
    lo = 0
    above = lo
    lo, above = _bits_round(probs_ptr, base, start, end, hrow, brow, target, lo, above, G, 23, 0, BINS, BLOCK)
    lo, above = _bits_round(probs_ptr, base, start, end, hrow + BINS, brow, target, lo, above, 2 * G, 15, 1 << 23, BINS, BLOCK)
    lo, above = _bits_round(probs_ptr, base, start, end, hrow + 2 * BINS, brow, target, lo, above, 3 * G, 7, 1 << 15, BINS, BLOCK)
    lo, above = _bits_round(probs_ptr, base, start, end, hrow + 3 * BINS, brow, target, lo, above, 4 * G, 0, 1 << 7, BINS, BLOCK)
    _keep_tail(probs_ptr, base, start, end, pid, row, cta, lo.to(tl.float32, bitcast=True), brow, 5 * G,
               ksum_ptr, psum_ptr, u_ptr, out_ptr, tok_ptr, V, G, DRAW, G_POW2, BLOCK)


@triton.jit
def _keep_tail(
    probs_ptr, base, start, end, pid, row, cta, thr, bar_ptr, need,
    ksum_ptr, psum_ptr, u_ptr, out_ptr, tok_ptr, V, G,
    DRAW: tl.constexpr, G_POW2: tl.constexpr, BLOCK: tl.constexpr,
):
    # Keep x >= thr over this chunk. This deliberately retains every boundary tie,
    # matching flashinfer's top-k and top-p filtering semantics.
    s = 0.0
    for s0 in tl.range(start, end, BLOCK):
        offs = s0 + tl.arange(0, BLOCK)
        mask = offs < end
        x = tl.load(probs_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(tl.where(mask & (x >= thr), x, 0.0), 0)
    if DRAW:
        tl.store(psum_ptr + pid, s)
        _row_barrier(bar_ptr, need)
        goff = tl.arange(0, G_POW2)
        gmask = goff < G
        ps = tl.load(psum_ptr + row * G + goff, mask=gmask, other=0.0, cache_modifier=".cg")
        acc = tl.sum(tl.where(goff < cta, ps, 0.0), 0)
        incl = tl.sum(tl.where(goff <= cta, ps, 0.0), 0)
        tgt = tl.load(u_ptr + row) * tl.sum(ps, 0)
        last_kept = start * 0 - 1
        for s0 in tl.range(start, end, BLOCK):
            offs = s0 + tl.arange(0, BLOCK)
            mask = offs < end
            x = tl.load(probs_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            kept = mask & (x >= thr)
            wv = tl.where(kept, x, 0.0)
            cval = acc + tl.cumsum(wv, 0)
            idx = tl.where(cval > tgt, offs, V)
            blk_min = tl.min(idx, 0)
            if (blk_min < V) and (acc <= tgt):
                tl.store(tok_ptr + row, blk_min)
            acc += tl.sum(wv, 0)
            last_kept = tl.maximum(last_kept, tl.max(tl.where(kept & (x > 0), offs, -1), 0))
        # fp rounding can leave tgt between this CTA's running sum and the next CTA's prefix, or past the total;
        # the CTA that owns that gap (or the last one holding mass) writes its last kept token instead
        is_last_mass = tl.sum(tl.where((goff > cta) & (ps > 0), 1, 0), 0) == 0
        if (acc <= tgt) and (last_kept >= 0) and ((incl > tgt) or is_last_mass):
            tl.store(tok_ptr + row, last_kept)
    else:
        tl.atomic_add(ksum_ptr + row, s)
        _row_barrier(bar_ptr, need)
        inv = 1.0 / tl.atomic_add(ksum_ptr + row, 0.0)
        for s0 in tl.range(start, end, BLOCK):
            offs = s0 + tl.arange(0, BLOCK)
            mask = offs < end
            x = tl.load(probs_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            tl.store(out_ptr + base + offs, tl.where(x >= thr, x * inv, 0.0), mask=mask)


_PMBINS = 256


@triton.jit
def _pmass_round(
    probs_ptr, base, start, end, priv_ptr, mass_ptr, bar_ptr, target, lo, above, need,
    S: tl.constexpr, WIDTH: tl.constexpr, BINS: tl.constexpr, BLOCK: tl.constexpr,
):
    # top-p round over the bit pattern: per-bin MASS (exact up to fp32 atomic order) via scatter-add into this
    # CTA's private buffer, then one reduction into the row buffer, so the bin holding the p crossing is known
    jj = tl.arange(0, BINS)
    for s0 in tl.range(start, end, BLOCK):
        offs = s0 + tl.arange(0, BLOCK)
        mask = offs < end
        x = tl.load(probs_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = x.to(tl.int32, bitcast=True)
        d = y - lo
        if WIDTH == 0:
            inrange = mask & (y >= lo) & (y <= _INF_BITS)
        else:
            inrange = mask & (y >= lo) & (d < WIDTH) & (y <= _INF_BITS)
        tl.atomic_add(priv_ptr + tl.where(inrange, d >> S, 0), x, mask=inrange)
    # every warp's scatter-adds must land before any thread reads the private bins back
    tl.debug_barrier()
    tl.atomic_add(mass_ptr + jj, tl.load(priv_ptr + jj))
    _row_barrier(bar_ptr, need)
    m = tl.load(mass_ptr + jj, cache_modifier=".cg")
    prefix = tl.cumsum(m, 0)
    total = tl.sum(m, 0)
    ok = above + total - prefix + m >= target
    # p above the total mass (fp rounding at p = 1): keep the whole bracket
    j = tl.maximum(tl.max(tl.where(ok, jj, -1)), 0)
    prefix_j = tl.sum(tl.where(jj <= j, m, 0.0))
    return lo + (j << S), above + total - prefix_j


@triton.jit
def _topp_fused(
    probs_ptr, tp_ptr, tk_ptr, hist_ptr, priv_ptr, mass_ptr, bar_ptr, ksumk_ptr, ksum_ptr, psum_ptr, u_ptr,
    out_ptr, tok_ptr, V, G, CHUNK, row_stride,
    TOPK: tl.constexpr, DRAW: tl.constexpr, G_POW2: tl.constexpr, KBINS: tl.constexpr, PBINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # top-p, optionally after an exact top-k stage: the top-k threshold becomes the lower edge of the top-p
    # bracket and the p target is scaled by the kept top-k mass, so no renormalized copy is ever written
    pid = tl.program_id(0)
    row = pid // G
    cta = pid % G
    base = row * row_stride
    start = cta * CHUNK
    end = tl.minimum(start + CHUNK, V)
    brow = bar_ptr + row
    lo = 0
    if TOPK:
        tk = tl.maximum(tl.load(tk_ptr + row), 1)
        hk = hist_ptr + row * 4 * KBINS
        above_i = lo
        lo, above_i = _bits_round(probs_ptr, base, start, end, hk, brow, tk, lo, above_i, G, 23, 0, KBINS, BLOCK)
        lo, above_i = _bits_round(probs_ptr, base, start, end, hk + KBINS, brow, tk, lo, above_i, 2 * G, 15, 1 << 23, KBINS, BLOCK)
        lo, above_i = _bits_round(probs_ptr, base, start, end, hk + 2 * KBINS, brow, tk, lo, above_i, 3 * G, 7, 1 << 15, KBINS, BLOCK)
        lo, above_i = _bits_round(probs_ptr, base, start, end, hk + 3 * KBINS, brow, tk, lo, above_i, 4 * G, 0, 1 << 7, KBINS, BLOCK)
        thr_k = lo.to(tl.float32, bitcast=True)
        s = 0.0
        for s0 in tl.range(start, end, BLOCK):
            offs = s0 + tl.arange(0, BLOCK)
            mask = offs < end
            x = tl.load(probs_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            s += tl.sum(tl.where(mask & (x >= thr_k), x, 0.0), 0)
        tl.atomic_add(ksumk_ptr + row, s)
        _row_barrier(brow, 5 * G)
        target = tl.load(tp_ptr + row) * tl.atomic_add(ksumk_ptr + row, 0.0)
        done = 5
    else:
        target = tl.load(tp_ptr + row)
        done = 0
    mp = mass_ptr + row * 4 * PBINS
    pp = priv_ptr + pid * 4 * PBINS
    above = 0.0
    lo, above = _pmass_round(probs_ptr, base, start, end, pp, mp, brow, target, lo, above, (done + 1) * G,
                            23, 0, PBINS, BLOCK)
    lo, above = _pmass_round(probs_ptr, base, start, end, pp + PBINS, mp + PBINS, brow, target, lo, above,
                            (done + 2) * G, 15, 1 << 23, PBINS, BLOCK)
    lo, above = _pmass_round(probs_ptr, base, start, end, pp + 2 * PBINS, mp + 2 * PBINS, brow, target, lo, above,
                            (done + 3) * G, 7, 1 << 15, PBINS, BLOCK)
    lo, above = _pmass_round(probs_ptr, base, start, end, pp + 3 * PBINS, mp + 3 * PBINS, brow, target, lo, above,
                            (done + 4) * G, 0, 1 << 7, PBINS, BLOCK)
    _keep_tail(probs_ptr, base, start, end, pid, row, cta, lo.to(tl.float32, bitcast=True), brow,
               (done + 5) * G, ksum_ptr, psum_ptr, u_ptr, out_ptr, tok_ptr, V, G, DRAW, G_POW2, BLOCK)


_COOPERATIVE_DISABLED = set()
_COOP_CTAS_PER_SM = 2  # the fused kernels use ~80 regs/thread at 8 warps; 4/SM fails the cooperative launch


def _fused_plan(B, V, device, force_single=False):
    if force_single:
        return 1, V
    # the cooperative launch needs the whole grid co-resident, so cap B*G by an occupancy budget instead of _plan's one CTA per SM
    g_by_sm = max(1, (_COOP_CTAS_PER_SM * _num_sm(device)) // B)
    g_by_chunk = max(1, triton.cdiv(V, _MIN_CHUNK))
    G = min(g_by_sm, g_by_chunk)
    return G, triton.cdiv(V, G)


def _fused_launch(probs, kernel, tk, tp, draw, seed, offset, force_single=False):
    B, V = probs.shape
    dev = probs.device
    G, CHUNK = _fused_plan(B, V, dev, force_single)
    n_hist = 4 * _KBINS if (kernel is _topk_fused or tk is not None) else 0
    n_mass = 4 * _PMBINS if kernel is _topp_fused else 0
    # hist[B, n_hist] | mass[B, n_mass] | priv[B * G, n_mass] | bar/ksum/ksum_k/tok[B]
    ws = torch.zeros(B * (n_hist + n_mass) + B * G * n_mass + 4 * B, device=dev, dtype=torch.int32)
    hist = ws[:B * n_hist]
    mass = ws[B * n_hist:B * (n_hist + n_mass)].view(torch.float32)
    priv = ws[B * (n_hist + n_mass):B * (n_hist + n_mass) + B * G * n_mass].view(torch.float32)
    tail = B * (n_hist + n_mass) + B * G * n_mass
    bar = ws[tail:tail + B]
    ksum = ws[tail + B:tail + 2 * B].view(torch.float32)
    ksum_k = ws[tail + 2 * B:tail + 3 * B].view(torch.float32)
    if draw:
        psum = torch.empty(B * G, device=dev, dtype=torch.float32)
        u = _gen_u(B, dev, seed, offset)
        res = ws[tail + 3 * B:]
        out, tok = probs, res
    else:
        psum, u = ksum, ksum
        res = torch.empty_like(probs)
        out, tok = res, bar
    # a lone CTA per row in a single wave streams faster with more warps; with G > 1 the co-residency budget caps warps
    wide = G == 1 and B <= _num_sm(dev)
    common = dict(DRAW=draw, G_POW2=_next_pow2(G), BLOCK=8192 if wide else _FUSED_BLOCK, num_warps=32 if wide else 8,
                  launch_cooperative_grid=G > 1)
    if kernel is _topk_fused:
        _topk_fused[(B * G,)](probs, tk, hist, bar, ksum, psum, u, out, tok, V, G, CHUNK, probs.stride(0),
                              BINS=_KBINS, **common)
    else:
        _topp_fused[(B * G,)](probs, tp, tk if tk is not None else tp, hist, priv, mass, bar, ksum_k, ksum, psum, u, out, tok,
                              V, G, CHUNK, probs.stride(0), TOPK=tk is not None, KBINS=_KBINS, PBINS=_PMBINS, **common)
    return res


def _cooperative_key(probs, kernel, tk, draw):
    kind = "topk" if kernel is _topk_fused else "topk_topp" if tk is not None else "topp"
    return probs.device, kind, draw


def _is_cooperative_launch_error(exc):
    message = str(exc).lower()
    return "cooperative" in message or "too many resources requested for launch" in message


def _exact_launch(probs, kernel, tk, tp, draw, seed, offset):
    key = _cooperative_key(probs, kernel, tk, draw)
    force_single = key in _COOPERATIVE_DISABLED
    G, _ = _fused_plan(*probs.shape, probs.device, force_single)
    try:
        return _fused_launch(probs, kernel, tk, tp, draw, seed, offset, force_single)
    except RuntimeError as exc:
        if force_single or G == 1 or not _is_cooperative_launch_error(exc):
            raise
        _COOPERATIVE_DISABLED.add(key)
        logger.warning("cooperative triton sampling unavailable on %s (%s); retrying with one CTA per row",
                       probs.device, exc)
        return _fused_launch(probs, kernel, tk, tp, draw, seed, offset, force_single=True)


def _topk(probs, target, draw, seed=None, offset=None):
    if probs.size(0) == 0:
        return torch.empty(0, device=probs.device, dtype=torch.int32) if draw else probs.clone()

    return _exact_launch(probs, _topk_fused, target, None, draw, seed, offset)


def _topp(probs, tp, tk, draw, seed=None, offset=None):
    if probs.size(0) == 0:
        return torch.empty(0, device=probs.device, dtype=torch.int32) if draw else probs.clone()

    return _exact_launch(probs, _topp_fused, tk, tp, draw, seed, offset)


def _topp_target(top_p, B, dev):
    if isinstance(top_p, torch.Tensor):
        return top_p.float().to(dev).contiguous()
    return torch.full((B,), float(top_p), device=dev, dtype=torch.float32)


def top_k_renorm_probs(probs, top_k):
    probs = probs.float()
    return _topk(probs, _topk_target(top_k, probs.size(0), probs.device), False)


def top_k_sampling_from_probs(probs, top_k, indices=None, deterministic=True, generator=None,
                              check_nan=False, seed=None, offset=None, return_valid=False):
    probs = probs.float()
    src = probs if indices is None else probs[indices].contiguous()
    out = _topk(src, _topk_target(top_k, src.size(0), src.device), True, seed, offset)
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


def top_k_top_p_sampling_from_probs(probs, top_k, top_p, indices=None,
                                    filter_apply_order="top_k_first", deterministic=True,
                                    generator=None, check_nan=False, seed=None, offset=None,
                                    return_valid=False):
    probs = probs.float()
    src = probs if indices is None else probs[indices].contiguous()
    B = src.size(0)
    out = _topp(src, _topp_target(top_p, B, src.device), _topk_target(top_k, B, src.device), True, seed, offset)
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


__all__ = [
    "softmax", "top_k_renorm_probs", "top_p_renorm_probs",
    "sampling_from_probs", "top_k_sampling_from_probs",
    "top_p_sampling_from_probs", "top_k_top_p_sampling_from_probs",
]
