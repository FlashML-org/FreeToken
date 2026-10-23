"""Triton moe_align_block_size (provenance: moe_align<-vllm/redesign).

Optional pure-triton drop-in producing the same buffers as the vendored
sgl/CUDA ``moe_align_block_size``:

    moe_align_block_size(topk_ids, block_size, num_experts)
        -> (sorted_token_ids, expert_ids, num_tokens_post_pad)

Semantics (large-batch path of the sgl CUDA kernel, which is the only branch the
required shapes exercise, since num_experts+1 > 64):

  * effective_E = num_experts + 1  (fused.py's call convention: one extra sentinel
    expert slot). Buffer sizes mirror fused.py exactly so shapes match the vendored op.
  * count[e]   = #tokens routed to expert e   (e in [0, effective_E))
  * cumsum[0]=0, cumsum[i] = cumsum[i-1] + ceil(count[i-1]/block)*block
  * num_tokens_post_pad = cumsum[effective_E]
  * expert_ids[cumsum[i]/block : cumsum[i+1]/block) = i
  * sorted_token_ids: tokens scattered into [cumsum[e], cumsum[e]+count[e]); the
    order *within* an expert region is nondeterministic (atomicAdd), exactly like
    the reference. Unwritten slots hold the sentinel value ``numel``.

Two paths, mirroring the sgl CUDA kernel's small/large split:

  * small (numel <= 1024, every decode shape): ONE fused single-CTA launch
    (_moe_align_small). Histogram/cumsum/expert_ids live in registers
    (tl.histogram + tl.cumsum); cumsum spills through global scratch across one
    tl.debug_barrier() so the scatter can gather per-token bases; rank via
    atomic_add. Single launch vs sgl's 2 -- launch overhead dominates here.
  * large (prefill): 4 parallel launches, data-dependent chain
    count -> cumsum -> {expert_ids, scatter}:
      1. _fill_and_count  (fixed BLOCK_SIZE=256, H100-tuned): sentinel-fill
         sorted_token_ids, zero fill_counter, atomic-histogram topk_ids -> counts.
      2. _cumsum_experts  (1 CTA): parallel padded prefix-sum (tl.cumsum). (A prior
         version scanned experts serially on one lane -> O(E) latency-bound,
         ~6x native at 256 experts.)
      3. _fill_expert_ids (fixed BLOCK_SIZE=256, H100-tuned): parallel binary search
         over cumsum.
      4. _scatter         (fixed BLOCK_SIZE=256, H100-tuned): pos = cumsum[e] +
         atomic_add(fill_counter[e]).

No triton.autotune anywhere in this module: all launch configs below are fixed,
chosen from a one-time H100 sweep (see comments at each launch site), matching
the upstream (vLLM/sglang) style of hardcoding/heuristics instead of autotuning.
"""

from __future__ import annotations

import functools
import os
from typing import Tuple

import torch
import triton
import triton.language as tl
from packaging import version

from freetoken.utils.arch import is_rocm
from freetoken.utils.logger import init_logger

logger = init_logger(__name__)

# Compared against Version.release (the numeric part only), so 3.8.0rc1 and
# 3.8.0.dev builds -- which sort *below* 3.8.0 under normal PEP 440 ordering --
# are still recognized as 3.8-era and gated.
_MIN_BROKEN_TRITON_RELEASE = (3, 8)


@functools.cache
def _warn_once(msg: str, value: str) -> None:
    """Log a warning the first time it is raised for a given value.

    This dispatcher runs per MoE layer per step, so warning per call would
    flood the log. Cached on the arguments, so a *different* bad value still
    gets reported rather than being swallowed by an earlier one's latch.
    """
    logger.warning(msg, value)


_SMALL_CAP = 1024  # fused single-CTA path for numel <= this (covers all decode shapes)


@triton.jit(do_not_specialize=["numel", "sentinel"])
def _moe_align_small(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    cumsum_ptr,        # scratch [effective_E+1]: spills cumsum across the barrier
    fill_counter_ptr,  # scratch [effective_E]: scatter rank counters
    numel,
    sentinel,
    effective_E: tl.constexpr,
    block_size: tl.constexpr,
    N_PAD: tl.constexpr,   # next_pow2(numel)
    HIST: tl.constexpr,    # next_pow2(effective_E+1) -> top bin is a spare for invalid ids
    FILL: tl.constexpr,
):
    tn = tl.arange(0, N_PAD)
    tmask = tn < numel
    e = tl.load(topk_ids_ptr + tn, mask=tmask, other=-1)
    valid = tmask & (e >= 0) & (e < effective_E)
    e_h = tl.where(valid, e, HIST - 1)  # invalid ids -> spare bin (>= effective_E)

    counts = tl.histogram(e_h, HIST)
    le = tl.arange(0, HIST)
    m_e = le < effective_E
    cnt = tl.where(m_e, counts, 0)
    nblk = (cnt + block_size - 1) // block_size
    excl_blk = tl.cumsum(nblk, 0) - nblk
    npp = tl.sum(nblk, 0) * block_size

    tl.store(cumsum_ptr + le, excl_blk * block_size, mask=m_e)
    tl.store(cumsum_ptr + effective_E, npp)
    tl.store(num_tokens_post_pad_ptr, npp)
    tl.store(fill_counter_ptr + le, 0, mask=m_e)

    # expert_ids: each expert lane writes its own padded block range (register-only)
    for j in tl.range(0, tl.max(nblk, 0)):
        tl.store(expert_ids_ptr + excl_blk + j, le, mask=m_e & (j < nblk))

    # sentinel-fill sorted[0:npp) (pre-barrier so the scatter stores win below)
    fo = tl.arange(0, FILL)
    for s in tl.range(0, npp, FILL):
        tl.store(sorted_token_ids_ptr + s + fo, sentinel, mask=s + fo < npp)

    tl.debug_barrier()  # cumsum/fill_counter stores visible; sentinel ordered before scatter

    base = tl.load(cumsum_ptr + e, mask=valid, other=0)
    rank = tl.atomic_add(fill_counter_ptr + e, 1, mask=valid)
    tl.store(sorted_token_ids_ptr + base + rank, tn, mask=valid)


@triton.jit(do_not_specialize=["numel", "sorted_numel", "sentinel"])
def _fill_and_count(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    counts_ptr,
    fill_counter_ptr,
    numel,             # #valid flattened tokens
    sorted_numel,      # len(sorted_token_ids) == max_num_tokens_padded
    sentinel,          # == numel
    effective_E: tl.constexpr,
    HIST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # (a) sentinel-fill sorted_token_ids (covers padding slots the scatter never touches)
    tl.store(sorted_token_ids_ptr + offs, sentinel, mask=offs < sorted_numel)
    # (b) zero the scatter fill-counter (read in the scatter kernel after a barrier)
    tl.store(fill_counter_ptr + offs, 0, mask=offs < effective_E)
    # (c) per-program register histogram, then one merged atomic per touched bin
    #     (vs one scattered atomic per element -- far fewer, conflict-free atomics)
    e = tl.load(topk_ids_ptr + offs, mask=offs < numel, other=-1)
    valid = (offs < numel) & (e >= 0) & (e < effective_E)
    e_h = tl.where(valid, e, HIST - 1)  # invalid -> spare top bin (>= effective_E)
    h = tl.histogram(e_h, HIST)
    le = tl.arange(0, HIST)
    tl.atomic_add(counts_ptr + le, h, mask=(le < effective_E) & (h > 0))


@triton.jit
def _cumsum_experts(
    counts_ptr,
    cumsum_ptr,
    num_tokens_post_pad_ptr,
    effective_E: tl.constexpr,
    block_size: tl.constexpr,
    E_PADDED: tl.constexpr,
):
    # Padded prefix-sum over all experts computed in parallel with tl.cumsum. (The old
    # kernel walked the experts serially on a single lane -> O(effective_E) dependent
    # loads, ~5x the native op at 256 experts / bs=1 decode.) cumsum[e] = token offset
    # where expert e's padded region starts; cumsum[E] = num_tokens_post_pad.
    lane = tl.arange(0, E_PADDED)
    m = lane < effective_E
    c = tl.load(counts_ptr + lane, mask=m, other=0)
    nblk = tl.where(m, (c + block_size - 1) // block_size, 0)   # padded blocks per expert
    excl = tl.cumsum(nblk, axis=0) - nblk                        # exclusive block offset
    tl.store(cumsum_ptr + lane, excl * block_size, mask=m)
    total_tok = tl.sum(nblk, axis=0) * block_size
    tl.store(cumsum_ptr + effective_E, total_tok)
    tl.store(num_tokens_post_pad_ptr, total_tok)


@triton.jit
def _fill_expert_ids(
    cumsum_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    block_size: tl.constexpr,
    effective_E: tl.constexpr,
    STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # expert_ids[b] = expert owning block b = largest e with block_offset[e] <= b, where
    # block_offset[e] = cumsum[e] / block_size. Resolved for every block index in
    # parallel by a binary search over the small monotone cumsum array (O(log E) vs the
    # old serial per-block fill).
    pid = tl.program_id(0)
    b = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_blk = tl.load(num_tokens_post_pad_ptr) // block_size
    mask = b < total_blk
    lo = b - b
    hi = lo + effective_E
    for _ in tl.static_range(STEPS):
        mid = (lo + hi) // 2
        off = tl.load(cumsum_ptr + mid, mask=mask, other=0) // block_size
        go = off <= b
        lo = tl.where(go, mid + 1, lo)
        hi = tl.where(go, hi, mid)
    tl.store(expert_ids_ptr + b, lo - 1, mask=mask)


@triton.jit(do_not_specialize=["numel"])
def _scatter(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    cumsum_ptr,
    fill_counter_ptr,
    numel,
    effective_E: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    e = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
    valid = mask & (e >= 0) & (e < effective_E)
    rank = tl.atomic_add(fill_counter_ptr + e, 1, mask=valid)
    base = tl.load(cumsum_ptr + e, mask=valid, other=0)
    pos = base + rank
    tl.store(sorted_token_ids_ptr + pos, offs, mask=valid)


def _div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def _moe_align_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The original Triton dispatch. Correct except on ROCm with Triton 3.8,
    where Triton's AMD backend miscompiles it -- see _needs_torch_fallback for
    the exact gate, the evidence behind it, and why the gate stays open-ended
    past 3.8 even though the fix is upstream.
    """
    device = topk_ids.device
    numel = topk_ids.numel()
    effective_E = num_experts + 1  # mirrors fused.py's num_experts+1 convention

    # Buffer sizes mirror freetoken.moe.fused.moe_align_block_size exactly.
    if numel < num_experts + 1:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_m_blocks = _div_ceil(max_num_tokens_padded, block_size)

    sorted_token_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)

    fill_counter = torch.empty((effective_E,), dtype=torch.int32, device=device)
    cumsum = torch.empty((effective_E + 1,), dtype=torch.int32, device=device)

    if 0 < numel <= _SMALL_CAP:
        # Fixed via H100 sweep (9-config grid; this num_warps ladder -- w2 at
        # numel<=64, w4@128, w8@256, w16@1024 -- won every decode shape, within 5%
        # of a live tuned search; forced-fixed beat live-tuned by 25-38% on the
        # atomic-heavy kernels because do_bench noise picks bad winners).
        num_warps = triton.next_power_of_2(min(16, max(2, numel // 32)))
        _moe_align_small[(1,)](
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum,
            fill_counter,
            numel,
            numel,          # sentinel
            effective_E,
            block_size,
            triton.next_power_of_2(numel),
            triton.next_power_of_2(effective_E + 1),
            1024,           # FILL
            num_warps=num_warps,
            num_stages=3,
        )
        return sorted_token_ids, expert_ids, num_tokens_post_pad

    counts = torch.zeros((effective_E,), dtype=torch.int32, device=device)
    sorted_numel = max_num_tokens_padded
    n_big = max(sorted_numel, numel, effective_E)
    def grid1(meta):
        return (triton.cdiv(n_big, meta["BLOCK_SIZE"]),)
    # Fixed via H100 sweep (9-config grid; BLOCK 256 won every kernel/shape; forced-
    # fixed beat live-tuned by 25-38% on the atomic-heavy kernels because do_bench
    # noise picks bad winners).
    _fill_and_count[grid1](
        topk_ids,
        sorted_token_ids,
        counts,
        fill_counter,
        numel,
        sorted_numel,
        numel,          # sentinel
        effective_E,
        triton.next_power_of_2(effective_E + 1),
        BLOCK_SIZE=256,
        num_warps=8,
        num_stages=3,
    )

    _cumsum_experts[(1,)](
        counts,
        cumsum,
        num_tokens_post_pad,
        effective_E,
        block_size,
        triton.next_power_of_2(effective_E),
    )

    def grid_eids(meta):
        return (triton.cdiv(max(max_num_m_blocks, 1), meta["BLOCK_SIZE"]),)
    # Same fixed-config rationale as _fill_and_count above.
    _fill_expert_ids[grid_eids](
        cumsum,
        expert_ids,
        num_tokens_post_pad,
        block_size,
        effective_E,
        effective_E.bit_length(),
        BLOCK_SIZE=256,
        num_warps=4,
        num_stages=3,
    )

    def grid3(meta):
        return (triton.cdiv(max(numel, 1), meta["BLOCK_SIZE"]),)
    # Same fixed-config rationale as _fill_and_count above.
    _scatter[grid3](
        topk_ids,
        sorted_token_ids,
        cumsum,
        fill_counter,
        numel,
        effective_E,
        BLOCK_SIZE=256,
        num_warps=8,
        num_stages=3,
    )

    return sorted_token_ids, expert_ids, num_tokens_post_pad


def _count_torch_experts(
    topk_ids: torch.Tensor, effective_E: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map invalid IDs to a spare bin and count every fixed-size bin."""
    flat = topk_ids.reshape(-1).to(torch.int64)
    valid = (flat >= 0) & (flat < effective_E)
    # invalid ids go to the spare bin -- SAME LENGTH as flat, no filtering
    e = torch.where(valid, flat, torch.full_like(flat, effective_E))

    # Fixed bins via scatter_add_ are graph-safe; bincount can size from contents.
    count_full = torch.zeros(effective_E + 1, dtype=torch.int64, device=topk_ids.device)
    count_full.scatter_add_(0, e, torch.ones_like(e))
    return e, count_full


def _build_torch_expert_ids(
    cum_blk: torch.Tensor, max_num_m_blocks: int
) -> torch.Tensor:
    """Return each padded block's expert, with an all-zero unused tail."""
    # repeat_interleave is banned because its output length depends on cum_blk.
    blk = torch.arange(max_num_m_blocks, device=cum_blk.device)
    eid = torch.searchsorted(cum_blk, blk, right=True)
    return torch.where(blk < cum_blk[-1], eid, torch.zeros_like(eid)).to(torch.int32)


def _build_torch_sorted_token_ids(
    e: torch.Tensor,
    count_full: torch.Tensor,
    start: torch.Tensor,
    max_num_tokens_padded: int,
) -> torch.Tensor:
    """Group token indices by expert into a fixed-size, sentinel-filled buffer."""
    numel = e.numel()

    # Sort the full input. The unique composite key gives deterministic ordering
    # without stable=True and leaves invalid entries in the spare trailing bin.
    token_ids = torch.arange(numel, device=e.device)
    order = torch.argsort(e * (numel + 1) + token_ids)
    sorted_e = e[order]

    group_start = torch.cumsum(count_full, 0) - count_full
    # token_ids is also arange(numel), so reuse it as sorted positions.
    rank = token_ids - group_start[sorted_e]

    # Steer invalid tokens to one trash slot after the real buffer, then drop it.
    start_full = torch.cat([start, start.new_full((1,), max_num_tokens_padded)])
    dest = start_full[sorted_e] + rank
    dest = torch.where(
        sorted_e < start.numel(), dest, torch.full_like(dest, max_num_tokens_padded)
    )

    buf = torch.full(
        (max_num_tokens_padded + 1,), numel, dtype=torch.int32, device=e.device
    )
    buf[dest] = order.to(torch.int32)
    return buf[:max_num_tokens_padded]


def _moe_align_torch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch fallback for the Triton kernels miscompiled on ROCm 3.8.

    Buffer shapes depend only on numel/block_size/num_experts, never tensor
    contents. The helpers therefore avoid boolean-mask indexing, bincount,
    data-dependent repeat_interleave, and host-sync operations so this remains
    safe during HIP/CUDA graph capture.
    """
    numel = topk_ids.numel()
    effective_E = num_experts + 1  # mirrors fused.py's num_experts+1 convention

    # Mirror freetoken.moe.fused.moe_align_block_size's buffer sizes exactly.
    if numel < effective_E:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + effective_E * (block_size - 1)
    max_num_m_blocks = _div_ceil(max_num_tokens_padded, block_size)

    e, count_full = _count_torch_experts(topk_ids, effective_E)
    count = count_full[:effective_E]

    nblk = (count + block_size - 1) // block_size  # padded blocks per expert
    cum_blk = torch.cumsum(nblk, 0)  # inclusive, in block units
    # cum_blk always has length effective_E >= 1 (num_experts >= 0), so cum_blk[-1]
    # is safe even when numel == 0 (nblk is then all zeros and cum_blk[-1] == 0).
    start = (cum_blk - nblk) * block_size  # each expert's first token slot
    num_tokens_post_pad = (cum_blk[-1] * block_size).to(torch.int32).reshape(1)

    expert_ids = _build_torch_expert_ids(cum_blk, max_num_m_blocks)
    sorted_token_ids = _build_torch_sorted_token_ids(
        e, count_full, start, max_num_tokens_padded
    )

    return sorted_token_ids, expert_ids, num_tokens_post_pad


def _needs_torch_fallback() -> bool:
    """True on any ROCm device running a 3.8-or-newer Triton.

    What breaks: comparisons against a ``tl.histogram`` result evaluate
    all-false (minimal repro: ``tl.histogram(x, H) > 0`` stored as data comes
    back all zeros). Both paths here build store masks from that histogram, so
    the large path reports ``num_tokens_post_pad == 0`` and the small path
    writes wrong ``expert_ids``.

    Root cause is a compile-time constant fold, not a runtime fault. The AMD
    backend gives ``tt.histogram`` a signed range of ``[0, getMaxValue]``,
    which is ``[0, -1]`` -- an empty set -- so any signed ordered comparison
    against it folds to false in ``tritonamdgpu-fold-true-cmpi`` and is deleted
    from the generated code. Fixed upstream by triton-lang/triton#11246 (merged
    2026-08-20), which missed v3.8.0; the backport onto ``release/3.8.x`` is
    pending as triton-lang/triton#11577.

    The gate is ROCm-wide rather than per-arch because the fold does not vary
    by target: compiling the repro for gfx1100/1101/1102/1200/1201/90a/942/950
    keeps the comparison on 3.6.0-3.7.1 and folds it on 3.8.0, on every one.
    Wrong results were confirmed on real silicon for gfx1101 only; the rest is
    compile evidence, which is the same defect but weaker proof.

    The gate has no upper bound, so a future Triton carrying the fix still
    lands here. That is deliberate -- guessing a bound risks routing callers
    onto genuinely broken kernels, and slow beats silently wrong -- and it is
    harmless only because pyproject.toml pins ``triton<3.9``. Whoever raises
    that pin adds the bound here in the same change, beside
    ``_MIN_BROKEN_TRITON_RELEASE`` rather than deleting the check, so 3.8
    itself stays covered. A release counts as fixed when
    ``test_dispatch_picks_correct_impl_for_real_device`` passes on a ROCm card
    with ``FREETOKEN_MOE_ALIGN_IMPL=triton``.
    """
    override = os.environ.get("FREETOKEN_MOE_ALIGN_IMPL", "").strip().lower()
    if override == "torch":
        return True
    if override == "triton":
        return False
    if override:  # empty/unset is not a bad value, it just means "auto-detect"
        _warn_once(
            "Unrecognized FREETOKEN_MOE_ALIGN_IMPL=%r (expected 'torch' or "
            "'triton'); falling back to auto-detection.",
            override,
        )

    if not is_rocm():
        return False
    try:
        release = version.parse(triton.__version__).release
    except version.InvalidVersion:
        # Unreadable version on the platform known to miscompile: take the
        # torch path, which is correct on every Triton version, rather than
        # gambling on the kernels or raising out of a dispatch call.
        _warn_once(
            "Could not parse triton.__version__=%r on ROCm; using the "
            "torch fallback, which is correct on any Triton version.",
            triton.__version__,
        )
        return True
    return release >= _MIN_BROKEN_TRITON_RELEASE


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch: torch on the affected ROCm config, the Triton kernels above
    everywhere else -- see ``_needs_torch_fallback`` for exactly which config.

    This module is reached whenever ``sgl_kernel`` is absent (see
    ``freetoken.moe.fused.moe_align_block_size``), which is always the case on
    ROCm but can also happen on CUDA. Auto-detection switches over only the
    evidence-backed broken config, leaving every other platform -- including
    CUDA-without-sgl_kernel and unaffected ROCm configs -- bit-for-bit as it
    was.

    ``FREETOKEN_MOE_ALIGN_IMPL`` overrides that detection on any platform:
    "torch" forces the fallback, "triton" forces the kernels above, and any
    other non-empty value logs one warning and falls through to auto-detection.
    Forcing "torch" is the one way a non-ROCm caller reaches the fallback.
    """
    assert topk_ids.dtype == torch.int32
    assert topk_ids.is_contiguous()
    if _needs_torch_fallback():
        return _moe_align_torch(topk_ids, block_size, num_experts)
    return _moe_align_triton(topk_ids, block_size, num_experts)


__all__ = ["moe_align_block_size"]
