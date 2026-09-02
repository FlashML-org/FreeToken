"""Multi-CTA (split-vocab) Triton sampling ops.

Optional pure-triton drop-in for freetoken.kernel.sampling / flashinfer.sampling
(softmax / top-k / top-p / combined + draw), self-contained.

Design:
  * Every row is split across many CTAs (``_plan`` -> G column-chunks) so bs=1 uses
    the whole GPU, unlike a single-block-per-row kernel that is single-SM-bound.
  * softmax is a multi-CTA online softmax; the draw is a multi-CTA inverse-CDF.
  * top-k and top-p are each one cooperative kernel: the row's CTAs each bin their chunk over
    the fp32 bit pattern (order-preserving for x >= 0), meet at a per-row spin barrier, and all
    redo the refine so they share the bracket. Four rounds of 256 bins bring the 2**31 range
    down to one bit pattern, so the threshold is exactly the k-th largest prob (top-k, counts)
    or the value where the descending cumulative mass reaches p (top-p, exact per-bin mass);
    the same kernel then renormalizes or draws. No candidate buffer, no data-dependent shape,
    no host sync. Results are exact up to fp32 atomic summation order.
  * If the cooperative launch is unavailable the module falls back to the multi-launch
    search below (exact for top-k; top-p there is the older bin-center estimate).
  * deterministic, generator and check_nan exist for flashinfer signature compatibility and are
    ignored; seed and offset are honored. Given a seed, top-k draws reproduce; top-p may pick a
    different token on rows whose cumulative mass sits within fp32 rounding of p.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.autotune_cache import autotune_cache_kwargs

logger = logging.getLogger(__name__)

_NUM_SM = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
_MIN_CHUNK = 4096  # do not split a row finer than this


def _plan(B, V):
    """Return (G, CHUNK): split each row into G column-chunks of size CHUNK."""
    g_by_sm = max(1, _NUM_SM // B)
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
    probs = torch.empty_like(logits)
    G, CHUNK = _plan(B, V)
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


# ===========================================================================
# multi-CTA top-p via histogram-bracket refinement.  Every full-vocab pass is
# split across all SMs; the sequential refine step is a tiny grid=(B,) kernel.
# ===========================================================================
_PBINS = 64        # fallback top-p only: count-hist + bin-center mass (64**4 ~ 1.7e7)
_PR = 4

_SR_CFGS = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
    for bs in (1024, 2048, 4096)
    for w in (4, 8)
    for s in (1, 2)
]


@triton.jit
def _rmax_pass(probs_ptr, rmax_ptr, V, G, CHUNK, row_stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // G
    base = row * row_stride
    start = (pid % G) * CHUNK
    end = tl.minimum(start + CHUNK, V)
    m = 0.0
    for s0 in tl.range(start, end, BLOCK_SIZE):
        offs = s0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        x = tl.load(probs_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        m = tl.maximum(m, tl.max(x, 0))
    tl.atomic_max(rmax_ptr + row, m)


@triton.autotune(configs=_SR_CFGS, key=["CHUNK", "BINS", "BITS"], reset_to_zero=["hist_ptr"], **autotune_cache_kwargs)
@triton.jit
def _count_hist_pass(
    probs_ptr, lo_ptr, hi_ptr, hist_ptr, V, G, CHUNK, row_stride,
    BINS: tl.constexpr, BITS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # BITS: lo/hi are int32 bit patterns and bins are exact integer ranges; else float value bins
    pid = tl.program_id(0)
    row = pid // G
    lo = tl.load(lo_ptr + row)
    hi = tl.load(hi_ptr + row)
    if BITS:
        w = (hi - lo + BINS - 1) // BINS
    else:
        invw = BINS / tl.maximum(hi - lo, 1e-30)
    base = row * row_stride
    start = (pid % G) * CHUNK
    end = tl.minimum(start + CHUNK, V)
    acc = tl.zeros([BINS], tl.int32)
    for s0 in tl.range(start, end, BLOCK_SIZE):
        offs = s0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        x = tl.load(probs_ptr + base + offs, mask=mask, other=-1.0).to(tl.float32)
        if BITS:
            y = x.to(tl.int32, bitcast=True)
            b = (y - lo) // w
        else:
            y = x
            b = ((x - lo) * invw).to(tl.int32)
        inrange = mask & (y >= lo) & (y < hi)
        # tl.histogram does NOT cleanly drop out-of-range indices; route every
        # out-of-bracket element to bin 0 and then subtract that count back out so
        # the histogram holds ONLY in-[lo,hi) counts (out-of-range is tracked via
        # `above`/excluded, exactly like the one-hot path).
        b = tl.where(inrange, tl.maximum(0, tl.minimum(b, BINS - 1)), 0)
        hcnt = tl.histogram(b, BINS)
        noor = tl.sum((mask & (~inrange)).to(tl.int32))
        hcnt = hcnt - tl.where(tl.arange(0, BINS) == 0, noor, 0)
        acc += hcnt
    tl.atomic_add(hist_ptr + row * BINS + tl.arange(0, BINS), acc.to(tl.float32))


@triton.jit
def _refine_pass(lo_ptr, hi_ptr, above_ptr, hist_ptr, target_ptr, BINS: tl.constexpr, BITS: tl.constexpr):
    row = tl.program_id(0)
    lo = tl.load(lo_ptr + row)
    hi = tl.load(hi_ptr + row)
    above = tl.load(above_ptr + row)
    target = tl.load(target_ptr + row)
    if BITS:
        w = (hi - lo + BINS - 1) // BINS
    else:
        w = (hi - lo) / BINS
    jj = tl.arange(0, BINS)
    h = tl.load(hist_ptr + row * BINS + jj)
    prefix = tl.cumsum(h, 0)
    total = tl.sum(h, 0)
    c_ge_bottom = above + total - prefix + h
    ok = c_ge_bottom >= target
    j = tl.max(tl.where(ok, jj, -1))
    prefix_j = tl.sum(tl.where(jj <= j, h, 0.0))
    upd = j >= 0
    new_hi = lo + (j + 1) * w
    if BITS:
        new_hi = tl.minimum(new_hi, hi)
    tl.store(lo_ptr + row, tl.where(upd, lo + j * w, lo))
    tl.store(hi_ptr + row, tl.where(upd, new_hi, hi))
    tl.store(above_ptr + row, tl.where(upd, above + total - prefix_j, above))
    # zero the row so the next iteration's atomic_add starts clean (reset_to_zero
    # only fires during autotuning, not on production calls)
    tl.store(hist_ptr + row * BINS + jj, 0.0)


@triton.jit
def _refine_mass_pass(lo_ptr, hi_ptr, above_ptr, hist_ptr, target_ptr, BINS: tl.constexpr):
    # top-p refine: hist holds COUNTS; approximate per-bin MASS as count*bin_center
    # (exact in the limit as the bracket narrows).  target is the p mass threshold.
    row = tl.program_id(0)
    lo = tl.load(lo_ptr + row)
    hi = tl.load(hi_ptr + row)
    above = tl.load(above_ptr + row)
    target = tl.load(target_ptr + row)
    w = (hi - lo) / BINS
    jj = tl.arange(0, BINS)
    h = tl.load(hist_ptr + row * BINS + jj)
    center = lo + (jj.to(tl.float32) + 0.5) * w
    massbin = h * center
    prefix = tl.cumsum(massbin, 0)
    total = tl.sum(massbin, 0)
    c_ge_bottom = above + total - prefix + massbin
    ok = c_ge_bottom >= target
    j = tl.max(tl.where(ok, jj, -1))
    prefix_j = tl.sum(tl.where(jj <= j, massbin, 0.0))
    upd = j >= 0
    tl.store(lo_ptr + row, tl.where(upd, lo + j * w, lo))
    tl.store(hi_ptr + row, tl.where(upd, lo + (j + 1) * w, hi))
    tl.store(above_ptr + row, tl.where(upd, above + total - prefix_j, above))
    tl.store(hist_ptr + row * BINS + jj, 0.0)


@triton.autotune(configs=_SR_CFGS, key=["CHUNK"], reset_to_zero=["ksum_ptr"], **autotune_cache_kwargs)
@triton.jit
def _ksum_pass(probs_ptr, thr_ptr, ksum_ptr, V, G, CHUNK, row_stride, BLOCK_SIZE: tl.constexpr):
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
    tl.atomic_add(ksum_ptr + row, s)


@triton.autotune(configs=_SR_CFGS, key=["CHUNK"], **autotune_cache_kwargs)
@triton.jit
def _write_pass(probs_ptr, out_ptr, thr_ptr, ksum_ptr, V, G, CHUNK, row_stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // G
    thr = tl.load(thr_ptr + row)
    inv_s = 1.0 / tl.load(ksum_ptr + row)
    base = row * row_stride
    start = (pid % G) * CHUNK
    end = tl.minimum(start + CHUNK, V)
    for s0 in tl.range(start, end, BLOCK_SIZE):
        offs = s0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        x = tl.load(probs_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + base + offs, tl.where(x >= thr, x * inv_s, 0.0), mask=mask)


def _search(probs, target, mass, R, BINS, bits=False):
    """Return per-row threshold: keep x >= thr, with count/mass(>=thr) ~ target."""
    B, V = probs.shape
    dev = probs.device
    G, CHUNK = _plan(B, V)
    grid = (B * G,)
    rmax = torch.zeros(B, device=dev, dtype=torch.float32)
    _rmax_pass[grid](probs, rmax, V, G, CHUNK, probs.stride(0), BLOCK_SIZE=2048, num_warps=8)
    if bits:
        lo = torch.zeros(B, device=dev, dtype=torch.int32)
        hi = (rmax.view(torch.int32) + 1).contiguous()
    else:
        lo = torch.zeros(B, device=dev, dtype=torch.float32)
        hi = (rmax * 1.0000001).contiguous()
    above = torch.zeros(B, device=dev, dtype=torch.float32)
    hist = torch.zeros(B * BINS, device=dev, dtype=torch.float32)
    for _ in range(R):
        _count_hist_pass[grid](probs, lo, hi, hist, V, G, CHUNK, probs.stride(0), BINS, bits)
        if mass:
            _refine_mass_pass[(B,)](lo, hi, above, hist, target, BINS)
        else:
            _refine_pass[(B,)](lo, hi, above, hist, target, BINS, bits)
    return lo.view(torch.float32) if bits else lo


def _renorm(probs, thr):
    B, V = probs.shape
    dev = probs.device
    G, CHUNK = _plan(B, V)
    grid = (B * G,)
    out = torch.empty_like(probs)
    ksum = torch.zeros(B, device=dev, dtype=torch.float32)
    _ksum_pass[grid](probs, thr, ksum, V, G, CHUNK, probs.stride(0))
    _write_pass[grid](probs, out, thr, ksum, V, G, CHUNK, probs.stride(0))
    return out


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
    G, CHUNK = _plan(B, V)
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
        target = top_k.float().to(dev).contiguous()
    else:
        target = torch.full((B,), float(int(top_k)), device=dev, dtype=torch.float32)
    # k <= 0 would satisfy every bin and push the bracket past the max; k > V needs no clamp, the search leaves lo at 0
    return torch.clamp(target, min=1.0)


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
    ok = (above + total - prefix + h).to(tl.float32) >= target
    j = tl.max(tl.where(ok, jj, -1))
    prefix_j = tl.sum(tl.where(jj <= j, h, 0))
    upd = j >= 0
    lo = tl.where(upd, lo + (j << S), lo)
    above = tl.where(upd, above + total - prefix_j, above)
    return lo, above


@triton.jit
def _topk_fused(
    probs_ptr, target_ptr, lo0_ptr, hist_ptr, bar_ptr, ksum_ptr, psum_ptr, u_ptr, out_ptr, tok_ptr,
    V, G, CHUNK, row_stride,
    DRAW: tl.constexpr, G_POW2: tl.constexpr, BINS: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // G
    cta = pid % G
    base = row * row_stride
    start = cta * CHUNK
    end = tl.minimum(start + CHUNK, V)
    target = tl.load(target_ptr + row)
    hrow = hist_ptr + row * 4 * BINS
    brow = bar_ptr + row
    lo = tl.load(lo0_ptr + row)
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
    # keep x >= thr over this chunk: DRAW picks one token per row, else renormalize the kept mass to 1
    s = 0.0
    for s0 in tl.range(start, end, BLOCK):
        offs = s0 + tl.arange(0, BLOCK)
        mask = offs < end
        x = tl.load(probs_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(tl.where(x >= thr, x, 0.0), 0)
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
            kept = (x >= thr) & mask
            wv = tl.where(kept, x, 0.0)
            cval = acc + tl.cumsum(wv, 0)
            idx = tl.where(cval > tgt, offs, V)
            blk_min = tl.min(idx, 0)
            if (blk_min < V) and (acc <= tgt):
                tl.store(tok_ptr + row, blk_min)
            acc += tl.sum(wv, 0)
            last_kept = tl.maximum(last_kept, tl.max(tl.where(kept, offs, -1), 0))
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
    probs_ptr, tp_ptr, tk_ptr, lo0_ptr, hist_ptr, priv_ptr, mass_ptr, bar_ptr, ksumk_ptr, ksum_ptr, psum_ptr, u_ptr,
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
    lo = tl.load(lo0_ptr + row)
    if TOPK:
        tk = tl.load(tk_ptr + row)
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
            s += tl.sum(tl.where(x >= thr_k, x, 0.0), 0)
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
    lo, above = _pmass_round(probs_ptr, base, start, end, pp, mp, brow, target, lo, above, (done + 1) * G, 23, 0, PBINS, BLOCK)
    lo, above = _pmass_round(probs_ptr, base, start, end, pp + PBINS, mp + PBINS, brow, target, lo, above, (done + 2) * G, 15, 1 << 23, PBINS, BLOCK)
    lo, above = _pmass_round(probs_ptr, base, start, end, pp + 2 * PBINS, mp + 2 * PBINS, brow, target, lo, above, (done + 3) * G, 7, 1 << 15, PBINS, BLOCK)
    lo, above = _pmass_round(probs_ptr, base, start, end, pp + 3 * PBINS, mp + 3 * PBINS, brow, target, lo, above, (done + 4) * G, 0, 1 << 7, PBINS, BLOCK)
    _keep_tail(probs_ptr, base, start, end, pid, row, cta, lo.to(tl.float32, bitcast=True), brow, (done + 5) * G,
               ksum_ptr, psum_ptr, u_ptr, out_ptr, tok_ptr, V, G, DRAW, G_POW2, BLOCK)


_fused_ok = True
_COOP_CTAS_PER_SM = 2  # the fused kernels use ~80 regs/thread at 8 warps; 4/SM fails the cooperative launch


def _fused_plan(B, V):
    # the cooperative launch needs the whole grid co-resident, so cap B*G by an occupancy budget instead of _plan's one CTA per SM
    g_by_sm = max(1, (_COOP_CTAS_PER_SM * _NUM_SM) // B)
    g_by_chunk = max(1, triton.cdiv(V, _MIN_CHUNK))
    G = min(g_by_sm, g_by_chunk)
    return G, triton.cdiv(V, G)


def _fused_launch(probs, kernel, tk, tp, draw, seed, offset):
    B, V = probs.shape
    dev = probs.device
    G, CHUNK = _fused_plan(B, V)
    n_hist = 4 * _KBINS if (kernel is _topk_fused or tk is not None) else 0
    n_mass = 4 * _PMBINS if kernel is _topp_fused else 0
    # hist[B, n_hist] | mass[B, n_mass] | priv[B * G, n_mass] | bar[B] | lo0[B] | ksum[B] | ksum_k[B] | tok[B]
    ws = torch.zeros(B * (n_hist + n_mass) + B * G * n_mass + 5 * B, device=dev, dtype=torch.int32)
    hist = ws[:B * n_hist]
    mass = ws[B * n_hist:B * (n_hist + n_mass)].view(torch.float32)
    priv = ws[B * (n_hist + n_mass):B * (n_hist + n_mass) + B * G * n_mass].view(torch.float32)
    tail = B * (n_hist + n_mass) + B * G * n_mass
    bar, lo0 = ws[tail:tail + B], ws[tail + B:tail + 2 * B]
    ksum = ws[tail + 2 * B:tail + 3 * B].view(torch.float32)
    ksum_k = ws[tail + 3 * B:tail + 4 * B].view(torch.float32)
    if draw:
        psum = torch.empty(B * G, device=dev, dtype=torch.float32)
        u = _gen_u(B, dev, seed, offset)
        res = ws[tail + 4 * B:]
        out, tok = probs, res
    else:
        psum, u = ksum, ksum
        res = torch.empty_like(probs)
        out, tok = res, lo0
    # a lone CTA per row in a single wave streams faster with more warps; with G > 1 the co-residency budget caps warps
    wide = G == 1 and B <= _NUM_SM
    common = dict(DRAW=draw, G_POW2=_next_pow2(G), BLOCK=8192 if wide else _FUSED_BLOCK, num_warps=32 if wide else 8,
                  launch_cooperative_grid=G > 1)
    if kernel is _topk_fused:
        _topk_fused[(B * G,)](probs, tk, lo0, hist, bar, ksum, psum, u, out, tok, V, G, CHUNK, probs.stride(0),
                              BINS=_KBINS, **common)
    else:
        _topp_fused[(B * G,)](probs, tp, tk if tk is not None else tp, lo0, hist, priv, mass, bar, ksum_k, ksum, psum, u, out, tok,
                              V, G, CHUNK, probs.stride(0), TOPK=tk is not None, KBINS=_KBINS, PBINS=_PMBINS, **common)
    return res


def _topk_fused_launch(probs, target, draw, seed=None, offset=None):
    return _fused_launch(probs, _topk_fused, target, None, draw, seed, offset)


def _topk_thr_search(probs, target):
    # fallback when the cooperative launch is unavailable: rmax + 4 x (count-hist + refine) launches over the bit pattern
    return _search(probs, target, False, 4, _KBINS, bits=True)


def _fused_or(fn_fused, fn_fallback):
    global _fused_ok
    if _fused_ok:
        try:
            return fn_fused()
        except RuntimeError as exc:
            # only a failed cooperative launch disqualifies this device; anything else (OOM, bad input) propagates
            if "cooperative" not in str(exc) and "Triton Error" not in str(exc):
                raise
            _fused_ok = False
            logger.warning("fused triton sampling unavailable (%s); using the multi-launch search", exc)
    return fn_fallback()


def _topk(probs, target, draw, seed=None, offset=None):
    if probs.size(0) == 0:
        return torch.empty(0, device=probs.device, dtype=torch.int32) if draw else probs.clone()

    def fallback():
        thr = _topk_thr_search(probs, target)
        return _draw(probs, thr, seed, offset) if draw else _renorm(probs, thr)
    return _fused_or(lambda: _topk_fused_launch(probs, target, draw, seed, offset), fallback)


def _topp(probs, tp, tk, draw, seed=None, offset=None):
    if probs.size(0) == 0:
        return torch.empty(0, device=probs.device, dtype=torch.int32) if draw else probs.clone()

    def fallback():
        src = probs if tk is None else _renorm(probs, _topk_thr_search(probs, tk))
        thr = _search(src, tp, True, _PR, _PBINS)
        return _draw(src, thr, seed, offset) if draw else _renorm(src, thr)
    return _fused_or(lambda: _fused_launch(probs, _topp_fused, tk, tp, draw, seed, offset), fallback)


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
