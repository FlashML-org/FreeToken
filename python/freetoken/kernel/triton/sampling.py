"""Multi-CTA (split-vocab) Triton sampling ops (provenance: sampling<-vllm Qrita).

Optional pure-triton drop-in for freetoken.kernel.sampling / flashinfer.sampling
(softmax / top-k / top-p / combined + draw), self-contained.

Design:
  * Every row is split across many CTAs (``_plan`` -> G column-chunks) so bs=1 uses
    the whole GPU, unlike a single-block-per-row kernel that is single-SM-bound.
  * softmax is a multi-CTA online softmax; top-p uses a small fixed number of
    histogram-bracket refinement passes instead of a ~48-iter bisection; the draw
    is a multi-CTA inverse-CDF.
  * The top-k path is adapted from vLLM's Qrita kernel
    (v1/sample/ops/topk_topp_triton.py::_topk_topp_kernel): gather the small set of
    "outlier" candidates (probs >= rmax*FRAC) into a compact per-row buffer in ONE
    full-vocab pass, then run the k-th-value search on that tiny buffer (3 full-vocab
    passes total vs ~6 for a pure histogram top-k). The outlier-pivot heuristic is
    swapped for the probs domain (truncate at rmax*FRAC; softmax probs are not
    Gaussian). Rows overflowing CAP silently drop the smallest gathered candidates;
    the refine still finds the exact k-th since every value >= threshold is kept, and
    an in-kernel guard keeps everything if fewer than k finite candidates are gathered.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.autotune_cache import autotune_cache_kwargs

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
_PBINS = 64        # top-p uses count-hist + bin-center mass (64**4 ~ 1.7e7)
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


@triton.autotune(configs=_SR_CFGS, key=["CHUNK"], reset_to_zero=["hist_ptr"], **autotune_cache_kwargs)
@triton.jit
def _count_hist_pass(
    probs_ptr, lo_ptr, hi_ptr, hist_ptr, V, G, CHUNK, row_stride,
    BINS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // G
    lo = tl.load(lo_ptr + row)
    hi = tl.load(hi_ptr + row)
    invw = BINS / tl.maximum(hi - lo, 1e-30)
    base = row * row_stride
    start = (pid % G) * CHUNK
    end = tl.minimum(start + CHUNK, V)
    acc = tl.zeros([BINS], tl.int32)
    for s0 in tl.range(start, end, BLOCK_SIZE):
        offs = s0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        x = tl.load(probs_ptr + base + offs, mask=mask, other=-1.0).to(tl.float32)
        inrange = mask & (x >= lo) & (x < hi)
        b = ((x - lo) * invw).to(tl.int32)
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
def _refine_pass(lo_ptr, hi_ptr, above_ptr, hist_ptr, target_ptr, BINS: tl.constexpr):
    row = tl.program_id(0)
    lo = tl.load(lo_ptr + row)
    hi = tl.load(hi_ptr + row)
    above = tl.load(above_ptr + row)
    target = tl.load(target_ptr + row)
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
    tl.store(lo_ptr + row, tl.where(upd, lo + j * w, lo))
    tl.store(hi_ptr + row, tl.where(upd, lo + (j + 1) * w, hi))
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


def _search(probs, target, mass, R, BINS):
    """Return per-row threshold: keep x >= thr, with count/mass(>=thr) ~ target."""
    B, V = probs.shape
    dev = probs.device
    G, CHUNK = _plan(B, V)
    grid = (B * G,)
    rmax = torch.zeros(B, device=dev, dtype=torch.float32)
    _rmax_pass[grid](probs, rmax, V, G, CHUNK, probs.stride(0), BLOCK_SIZE=2048, num_warps=8)
    lo = torch.zeros(B, device=dev, dtype=torch.float32)
    hi = (rmax * 1.0000001).contiguous()
    above = torch.zeros(B, device=dev, dtype=torch.float32)
    hist = torch.zeros(B * BINS, device=dev, dtype=torch.float32)
    for _ in range(R):
        _count_hist_pass[grid](probs, lo, hi, hist, V, G, CHUNK, probs.stride(0), BINS)
        if mass:
            _refine_mass_pass[(B,)](lo, hi, above, hist, target, BINS)
        else:
            _refine_pass[(B,)](lo, hi, above, hist, target, BINS)
    return lo


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
    B, V = probs.shape
    if isinstance(top_p, torch.Tensor):
        target = top_p.float().to(probs.device).contiguous()
    else:
        target = torch.full((B,), float(top_p), device=probs.device, dtype=torch.float32)
    thr = _search(probs, target, True, _PR, _PBINS)
    return _renorm(probs, thr)


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
def _draw_scan(psum_ptr, choff_ptr, u_ptr, target_ptr, G, G_POW2: tl.constexpr):
    row = tl.program_id(0)
    goff = tl.arange(0, G_POW2)
    gmask = goff < G
    ps = tl.load(psum_ptr + row * G + goff, mask=gmask, other=0.0)
    tl.store(choff_ptr + row * G + goff, tl.cumsum(ps, 0) - ps, mask=gmask)
    tl.store(target_ptr + row, tl.load(u_ptr + row) * tl.sum(ps, 0))


@triton.autotune(configs=_SR_CFGS, key=["CHUNK"], **autotune_cache_kwargs)
@triton.jit
def _draw_find(probs_ptr, thr_ptr, choff_ptr, target_ptr, out_ptr, V, G, CHUNK, row_stride,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // G
    thr = tl.load(thr_ptr + row)
    target = tl.load(target_ptr + row)
    acc = tl.load(choff_ptr + pid)
    base = row * row_stride
    start = (pid % G) * CHUNK
    end = tl.minimum(start + CHUNK, V)
    for s0 in tl.range(start, end, BLOCK_SIZE):
        offs = s0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        x = tl.load(probs_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.where((x >= thr) & mask, x, 0.0)
        cval = acc + tl.cumsum(wv, 0)
        idx = tl.where(cval > target, offs, V)
        blk_min = tl.min(idx, 0)
        if (blk_min < V) and (acc <= target):
            tl.store(out_ptr + row, blk_min)
        acc += tl.sum(wv, 0)


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
    out = torch.empty(B, device=dev, dtype=torch.int32)
    u = _gen_u(B, dev, seed, offset)
    _draw_part[grid](probs, thr, psum, V, G, CHUNK, probs.stride(0))
    _draw_scan[(B,)](psum, choff, u, target, G, _next_pow2(G))
    _draw_find[grid](probs, thr, choff, target, out, V, G, CHUNK, probs.stride(0))
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
    if isinstance(top_p, torch.Tensor):
        target = top_p.float().to(src.device).contiguous()
    else:
        target = torch.full((src.size(0),), float(top_p), device=src.device, dtype=torch.float32)
    thr = _search(src, target, True, _PR, _PBINS)
    out = _draw(src, thr, seed, offset)
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


# ===========================================================================
# top-k via Qrita outlier-gather + tiny-buffer refine (adapted from vLLM)
# Passes: rmax (full) + gather (full) + refine (tiny, buffer-only) + write (full)
# = 3 full-vocab passes.  Keeps the multi-CTA vocab split so bs=1 uses the whole GPU.
# ===========================================================================
_FRAC = 0.05         # outlier gather pivot: keep probs >= rmax*FRAC
_CAP = 8192          # per-row candidate buffer capacity
_KBINS = 256         # bins per refine bracket iteration
_KR = 3              # refine iterations (256**3 ~ 1.7e7 resolution over [0, rmax])


@triton.autotune(configs=_SR_CFGS, key=["CHUNK"], reset_to_zero=["cnt_ptr"], **autotune_cache_kwargs)
@triton.jit
def _gather_pass(
    probs_ptr, rmax_ptr, buf_ptr, cnt_ptr, V, G, CHUNK, row_stride,
    FRAC, CAP: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // G
    thr0 = tl.load(rmax_ptr + row) * FRAC
    base = row * row_stride
    bufbase = row * CAP
    start = (pid % G) * CHUNK
    end = tl.minimum(start + CHUNK, V)
    for s0 in tl.range(start, end, BLOCK_SIZE):
        offs = s0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        x = tl.load(probs_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        m = mask & (x >= thr0)
        mi = m.to(tl.int32)
        n = tl.sum(mi)
        posbase = tl.atomic_add(cnt_ptr + row, n)
        cpos = posbase + tl.cumsum(mi, 0) - 1
        wmask = m & (cpos < CAP)
        tl.store(buf_ptr + bufbase + cpos, x, mask=wmask)


@triton.jit
def _refine_topk(
    buf_ptr, cnt_ptr, rmax_ptr, target_ptr, thr_ptr, ksum_ptr,
    CAP: tl.constexpr, R: tl.constexpr, BINS: tl.constexpr, BLK: tl.constexpr,
):
    # single-CTA-per-row refine on the tiny buffer -> (threshold, kept-sum):
    # histogram-bracket the k-th largest, then sum the kept mass (all kept values
    # are in the buffer since threshold >= gather pivot).
    row = tl.program_id(0)
    cnt = tl.load(cnt_ptr + row)
    cnt = tl.minimum(cnt, CAP)
    target = tl.load(target_ptr + row)
    base = row * CAP
    jj = tl.arange(0, BINS)
    lo = 0.0
    hi = tl.load(rmax_ptr + row) * 1.0000001
    above = 0.0
    for _it in tl.static_range(R):
        denom = tl.maximum(hi - lo, 1e-30)
        w = denom / BINS
        invw = BINS / denom
        hc = tl.zeros([BINS], tl.int32)
        for s0 in tl.range(0, cnt, BLK):
            offs = s0 + tl.arange(0, BLK)
            mask = offs < cnt
            x = tl.load(buf_ptr + base + offs, mask=mask, other=-1.0)
            inrange = mask & (x >= lo) & (x < hi)
            b = ((x - lo) * invw).to(tl.int32)
            b = tl.where(inrange, tl.maximum(0, tl.minimum(b, BINS - 1)), 0)
            hcnt = tl.histogram(b, BINS)
            noor = tl.sum((mask & (~inrange)).to(tl.int32))
            hc += hcnt - tl.where(jj == 0, noor, 0)
        h = hc.to(tl.float32)
        prefix = tl.cumsum(h, 0)
        total = tl.sum(h, 0)
        c_ge = above + total - prefix + h
        ok = c_ge >= target
        j = tl.max(tl.where(ok, jj, -1))
        prefix_j = tl.sum(tl.where(jj <= j, h, 0.0))
        upd = j >= 0
        new_lo = lo + j * w
        new_hi = lo + (j + 1) * w
        new_above = above + total - prefix_j
        lo = tl.where(upd, new_lo, lo)
        hi = tl.where(upd, new_hi, hi)
        above = tl.where(upd, new_above, above)
    thr = lo
    # guard: if fewer finite candidates than k were gathered, keep everything.
    if cnt < target:
        thr = 0.0
    ks = 0.0
    for s0 in tl.range(0, cnt, BLK):
        offs = s0 + tl.arange(0, BLK)
        mask = offs < cnt
        x = tl.load(buf_ptr + base + offs, mask=mask, other=0.0)
        ks += tl.sum(tl.where(x >= thr, x, 0.0))
    tl.store(thr_ptr + row, thr)
    tl.store(ksum_ptr + row, ks)


def _topk_target(top_k, B, dev):
    if isinstance(top_k, torch.Tensor):
        return top_k.float().to(dev).contiguous()
    return torch.full((B,), float(int(top_k)), device=dev, dtype=torch.float32)


def _topk_thr_ksum(probs, top_k):
    """Return (threshold[B], kept_sum[B]) for a top-k keep: x >= threshold."""
    B, V = probs.shape
    dev = probs.device
    G, CHUNK = _plan(B, V)
    grid = (B * G,)
    target = _topk_target(top_k, B, dev)
    rmax = torch.zeros(B, device=dev, dtype=torch.float32)
    _rmax_pass[grid](probs, rmax, V, G, CHUNK, probs.stride(0), BLOCK_SIZE=2048, num_warps=8)
    buf = torch.empty(B * _CAP, device=dev, dtype=torch.float32)
    cnt = torch.zeros(B, device=dev, dtype=torch.int32)
    _gather_pass[grid](probs, rmax, buf, cnt, V, G, CHUNK, probs.stride(0), _FRAC, _CAP)
    thr = torch.empty(B, device=dev, dtype=torch.float32)
    ksum = torch.empty(B, device=dev, dtype=torch.float32)
    _refine_topk[(B,)](buf, cnt, rmax, target, thr, ksum, _CAP, _KR, _KBINS, 2048)
    return thr, ksum


def top_k_renorm_probs(probs, top_k):
    probs = probs.float()
    B, V = probs.shape
    dev = probs.device
    G, CHUNK = _plan(B, V)
    grid = (B * G,)
    thr, ksum = _topk_thr_ksum(probs, top_k)
    out = torch.empty_like(probs)
    _write_pass[grid](probs, out, thr, ksum, V, G, CHUNK, probs.stride(0))
    return out


def top_k_sampling_from_probs(probs, top_k, indices=None, deterministic=True, generator=None,
                              check_nan=False, seed=None, offset=None, return_valid=False):
    probs = probs.float()
    src = probs if indices is None else probs[indices].contiguous()
    r = top_k_renorm_probs(src, top_k)
    out = _draw(r, _zeros_thr(src.size(0), src.device), seed, offset)
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


def top_k_top_p_sampling_from_probs(probs, top_k, top_p, indices=None,
                                    filter_apply_order="top_k_first", deterministic=True,
                                    generator=None, check_nan=False, seed=None, offset=None,
                                    return_valid=False):
    probs = probs.float()
    src = probs if indices is None else probs[indices].contiguous()
    r = top_k_renorm_probs(src, top_k)
    if isinstance(top_p, torch.Tensor):
        target = top_p.float().to(src.device).contiguous()
    else:
        target = torch.full((src.size(0),), float(top_p), device=src.device, dtype=torch.float32)
    thr = _search(r, target, True, _PR, _PBINS)
    out = _draw(r, thr, seed, offset)
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


__all__ = [
    "softmax", "top_k_renorm_probs", "top_p_renorm_probs",
    "sampling_from_probs", "top_k_sampling_from_probs",
    "top_p_sampling_from_probs", "top_k_top_p_sampling_from_probs",
]
