"""Tests for freetoken.kernel.triton.moe_align.moe_align_block_size.

This is the no-sgl_kernel path that decides which expert handles which block of
tokens. It had no direct test coverage: every existing test exercised it only
through the MoE GEMMs that consume its output, so a routing bug surfaced as
wrong model output with nothing pointing at the cause. That is how Triton
3.8.0's gfx1101 miscompile stayed hidden. These tests check the module's own
contract instead.

Two properties are locked down: correctness against a pure-Python reference, and
graph-capture safety (a real torch.cuda.graph capture plus replay with mutated
inputs), since buffer shapes here must depend only on numel/block_size/
num_experts and never on tensor contents.

The correctness tests target _moe_align_torch directly. Both Triton paths leave
expert_ids past the used region uninitialized (torch.empty plus masked stores),
and the small-batch path leaves sorted_token_ids past num_tokens_post_pad
uninitialized too, so the strict fill assertions here apply only to the torch
implementation.
"""

from __future__ import annotations

import logging

import pytest
import torch

import freetoken.kernel.triton.moe_align as moe_align_mod
from freetoken.kernel.triton.moe_align import _moe_align_torch, moe_align_block_size

# _moe_align_torch is pure torch: identical code path on CPU, CUDA, and ROCm.
# Running the correctness suite on CPU keeps it green on every CI runner;
# the cuda parametrization adds real-GPU coverage when a GPU is present.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def _buffer_sizes(numel: int, block_size: int, num_experts: int) -> tuple[int, int]:
    """Mirrors the buffer-size formula in moe_align.py's module docstring exactly."""
    if numel < num_experts + 1:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_m_blocks = -(-max_num_tokens_padded // block_size)  # ceil div
    return max_num_tokens_padded, max_num_m_blocks


def _tokens_by_expert(topk_ids: torch.Tensor, num_experts: int) -> list[list[int]]:
    """buckets[e] = flat token indices routed to expert e; invalid entries dropped."""
    effective_E = num_experts + 1
    flat = topk_ids.reshape(-1).tolist()
    buckets: list[list[int]] = [[] for _ in range(effective_E)]
    for tok_idx, e in enumerate(flat):
        if 0 <= e < effective_E:
            buckets[e].append(tok_idx)
    return buckets


def _expert_layout(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """Per-expert token buckets, counts, padded-block counts, and start offsets
    (in token units). Shared by the reference builder and the output checker below
    so both use the exact same layout math."""
    buckets = _tokens_by_expert(topk_ids, num_experts)
    counts = [len(b) for b in buckets]
    nblk = [-(-c // block_size) for c in counts]  # ceil div
    starts: list[int] = []
    blocks_so_far = 0
    for n in nblk:
        starts.append(blocks_so_far * block_size)
        blocks_so_far += n
    return buckets, counts, nblk, starts


def _reference(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[list[int], int]:
    """Pure-Python ground truth for expert_ids and num_tokens_post_pad (see
    moe_align.py's module docstring for the authoritative semantics). Deliberately
    slow and obvious rather than clever: nested loops over blocks, not vectorized."""
    numel = topk_ids.numel()
    effective_E = num_experts + 1
    _, _, nblk, starts = _expert_layout(topk_ids, block_size, num_experts)
    num_tokens_post_pad = sum(nblk) * block_size
    _, max_num_m_blocks = _buffer_sizes(numel, block_size, num_experts)

    expert_ids = [0] * max_num_m_blocks
    for e in range(effective_E):
        s = starts[e]
        for b in range(nblk[e]):
            expert_ids[s // block_size + b] = e

    return expert_ids, num_tokens_post_pad


def _assert_outputs_match_reference(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    """Exact match on num_tokens_post_pad and expert_ids; per-expert token SETS
    (order within an expert is explicitly unspecified -- atomicAdd-ranked scatter);
    every slot not claimed by any expert holds the sentinel (== numel)."""
    numel = topk_ids.numel()
    effective_E = num_experts + 1
    ref_expert_ids, ref_npp = _reference(topk_ids, block_size, num_experts)

    assert int(num_tokens_post_pad.item()) == ref_npp
    assert expert_ids.tolist() == ref_expert_ids

    buckets, counts, _, starts = _expert_layout(topk_ids, block_size, num_experts)
    got = sorted_token_ids.tolist()
    claimed: set[int] = set()
    for e in range(effective_E):
        s, c = starts[e], counts[e]
        assert sorted(got[s : s + c]) == sorted(buckets[e]), f"expert {e} token set mismatch"
        claimed.update(range(s, s + c))

    sentinel = numel
    for i, v in enumerate(got):
        if i not in claimed:
            assert v == sentinel, f"slot {i} should hold sentinel {sentinel}, got {v}"


def _assert_matches_reference(topk_ids: torch.Tensor, block_size: int, num_experts: int) -> None:
    numel = topk_ids.numel()
    sorted_token_ids, expert_ids, num_tokens_post_pad = _moe_align_torch(
        topk_ids, block_size, num_experts
    )
    max_num_tokens_padded, max_num_m_blocks = _buffer_sizes(numel, block_size, num_experts)

    assert sorted_token_ids.dtype == torch.int32
    assert expert_ids.dtype == torch.int32
    assert num_tokens_post_pad.dtype == torch.int32
    assert sorted_token_ids.shape == (max_num_tokens_padded,)
    assert expert_ids.shape == (max_num_m_blocks,)
    assert num_tokens_post_pad.shape == (1,)

    _assert_outputs_match_reference(
        topk_ids, block_size, num_experts, sorted_token_ids, expert_ids, num_tokens_post_pad
    )


@pytest.mark.parametrize(
    "numel,num_experts,block_size",
    [
        (1, 8, 16),
        (7, 8, 16),
        (64, 8, 16),
        (256, 64, 32),
        (1536, 256, 64),
        (300, 4, 32),
    ],
)
def test_matches_reference(numel, num_experts, block_size, device):
    torch.manual_seed(0)
    effective_E = num_experts + 1
    topk_ids = torch.randint(0, effective_E, (numel,), device=device, dtype=torch.int32)
    _assert_matches_reference(topk_ids, block_size, num_experts)


def test_empty_input(device):
    num_experts, block_size = 8, 16
    topk_ids = torch.empty(0, dtype=torch.int32, device=device)

    _assert_matches_reference(topk_ids, block_size, num_experts)


def test_invalid_expert_ids_are_ignored(device):
    num_experts, block_size = 4, 16
    effective_E = num_experts + 1
    # mix of valid ids [0, num_experts] with invalid ones: negative and >= effective_E
    values = [0, 1, 2, 3, 4, -1, -2, 5, 100, 3, 0, -1, 2, 4, 5, 1000, 2, 3, 0, 1]
    topk_ids = torch.tensor(values, dtype=torch.int32, device=device)

    # full reference comparison: this also proves invalid entries don't shift where
    # the valid tokens land, since the reference is computed on this same input and
    # already drops invalid entries the same way the kernel must.
    _assert_matches_reference(topk_ids, block_size, num_experts)

    sorted_token_ids, _, _ = _moe_align_torch(topk_ids, block_size, num_experts)
    invalid_tok_idx = {i for i, e in enumerate(values) if not (0 <= e < effective_E)}
    present = set(sorted_token_ids.tolist())
    assert not (invalid_tok_idx & present), "an invalid entry's token index leaked into output"


def test_zero_experts(device):
    """num_experts == 0 still leaves effective_E == 1 (the sentinel slot), so an
    input of all-zero ids routes everything to that one slot."""
    num_experts, block_size = 0, 16
    topk_ids = torch.zeros(8, dtype=torch.int32, device=device)
    _assert_matches_reference(topk_ids, block_size, num_experts)


def test_all_invalid_ids(device):
    """Nonempty input where every id is invalid: nothing is routed, so the padded
    count is 0, expert_ids is all zeros, and every sorted slot holds the sentinel
    (all invalid tokens collide on the single trash slot past the real buffer)."""
    num_experts, block_size = 4, 16
    values = [-3, -1, 5, 6, 99, 1000, -7, 5]  # all outside [0, effective_E)
    topk_ids = torch.tensor(values, dtype=torch.int32, device=device)

    _assert_matches_reference(topk_ids, block_size, num_experts)

    sorted_token_ids, expert_ids, num_tokens_post_pad = _moe_align_torch(
        topk_ids, block_size, num_experts
    )
    assert int(num_tokens_post_pad.item()) == 0
    assert (expert_ids == 0).all()
    assert (sorted_token_ids == topk_ids.numel()).all()  # sentinel everywhere


def test_expert_ids_past_used_region_are_zero(device):
    """_moe_align_torch zero-fills expert_ids past the used region; this pins that.

    Today's consumers never read those slots -- fused_moe_kernel and friends all
    early-return on `pid_m * BLOCK_SIZE_M >= num_tokens_post_padded` before
    loading expert_ids[pid_m] -- so this is a stricter contract than the Triton
    paths offer (they leave the tail as torch.empty garbage), not a fix for a
    live fault. It is worth pinning because expert_ids[pid_m] is multiplied into
    a weight base pointer: any future consumer that reads the tail without
    checking the bound gets a valid slice from a 0 instead of a wild address."""
    torch.manual_seed(5)
    num_experts, block_size, numel = 64, 32, 256
    topk_ids = torch.randint(0, num_experts + 1, (numel,), device=device, dtype=torch.int32)

    _, expert_ids, num_tokens_post_pad = _moe_align_torch(topk_ids, block_size, num_experts)
    used_blocks = int(num_tokens_post_pad.item()) // block_size
    assert (expert_ids[used_blocks:] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="graph capture needs a GPU")
def test_is_graph_capturable():
    """Regression guard for the graph-capture property in this file's docstring.

    Any data-dependent shape (boolean-mask indexing, bincount, nonzero, .item(),
    repeat_interleave-by-tensor) forces a device->host sync, and torch.cuda.graph
    raises "operation not permitted when stream is capturing" the moment that
    happens during capture. moe_align_block_size's buffer shapes depend only on
    numel/block_size/num_experts -- never on topk_ids' contents -- so capture must
    succeed, and replaying with genuinely different routing must still produce
    correct output.

    Warmup/capture pattern copied from tests/moe/test_fused_moe.py's
    test_fused_experts_grouped_impl_is_cuda_graph_capturable: run the op into a
    stable pre-allocated output via .copy_() a few times on the default stream,
    synchronize, then capture that same run() under torch.cuda.graph.
    """
    device = torch.device("cuda")
    num_experts, block_size, numel = 8, 16, 32
    torch.manual_seed(11)

    topk_ids = torch.randint(0, num_experts + 1, (numel,), device=device, dtype=torch.int32)
    max_num_tokens_padded, max_num_m_blocks = _buffer_sizes(numel, block_size, num_experts)
    sorted_out = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    expert_out = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    npp_out = torch.empty(1, dtype=torch.int32, device=device)

    def run():
        st, ei, npp = _moe_align_torch(topk_ids, block_size, num_experts)
        sorted_out.copy_(st)
        expert_out.copy_(ei)
        npp_out.copy_(npp)

    for _ in range(3):
        run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    graph.replay()
    torch.cuda.synchronize()
    _assert_outputs_match_reference(
        topk_ids, block_size, num_experts, sorted_out, expert_out, npp_out
    )

    # replay with genuinely different routing -- only the captured input tensor's
    # CONTENTS change (via copy_), never its shape, since that's what capture allows.
    torch.manual_seed(23)
    topk_ids.copy_(
        torch.randint(0, num_experts + 1, (numel,), device=device, dtype=torch.int32)
    )
    graph.replay()
    torch.cuda.synchronize()
    _assert_outputs_match_reference(
        topk_ids, block_size, num_experts, sorted_out, expert_out, npp_out
    )


def _patch_dispatch_signals(monkeypatch, *, rocm, triton_version):
    """Patch the two module-local signals _needs_torch_fallback reads, without
    touching the real (functools.cache'd) platform detectors -- see the existing
    test_rocm_arch.py suite for real detection coverage."""
    monkeypatch.setattr(moe_align_mod, "is_rocm", lambda: rocm)
    monkeypatch.setattr(moe_align_mod.triton, "__version__", triton_version)


def _stub_impls(monkeypatch):
    calls: list[str] = []
    fake = (object(), object(), object())
    monkeypatch.setattr(
        moe_align_mod, "_moe_align_torch", lambda *a: (calls.append("torch"), fake)[1]
    )
    monkeypatch.setattr(
        moe_align_mod, "_moe_align_triton", lambda *a: (calls.append("triton"), fake)[1]
    )
    return calls, fake


@pytest.mark.parametrize(
    "rocm,triton_version,want",
    [
        # the broken combo this fallback exists for: ROCm + Triton >= 3.8
        (True, "3.8.0", "torch"),
        (True, "3.9.2", "torch"),  # still 3.8-or-newer
        (True, "3.10.0", "torch"),  # string compare would read "3.10" < "3.8"
        # real-world build strings for that same broken release
        (True, "3.8.0+gitabcdef", "torch"),  # AMD fork / local version
        (True, "3.8.0rc1", "torch"),  # prerelease sorts below 3.8.0 -- still gated
        (True, "3.8.0.dev20260801", "torch"),  # nightly, same reason
        # ROCm at a Triton version that compiles the comparison correctly
        (True, "3.7.1", "triton"),
        (True, "3.7.0", "triton"),
        (True, "3.6.0", "triton"),
        # not ROCm at all: never gated, and the version check must not run
        (False, "3.8.0", "triton"),
        (False, "3.8.0-not-a-version", "triton"),
    ],
)
def test_dispatch_auto_detects_affected_config(monkeypatch, rocm, triton_version, want):
    """moe_align_block_size routes to the torch fallback on any ROCm device
    running Triton >= 3.8.0 -- the miscompile is a constant fold in an
    AMD-backend MLIR pass that does not vary by gfx target, so it is not
    per-card. See _needs_torch_fallback's docstring for the compile evidence."""
    monkeypatch.delenv("FREETOKEN_MOE_ALIGN_IMPL", raising=False)
    _patch_dispatch_signals(monkeypatch, rocm=rocm, triton_version=triton_version)
    calls, fake = _stub_impls(monkeypatch)
    topk_ids = torch.zeros(4, dtype=torch.int32)

    assert moe_align_block_size(topk_ids, 16, 8) is fake
    assert calls == [want]


@pytest.mark.parametrize(
    "override,want",
    [
        ("torch", "torch"),
        ("triton", "triton"),
        # operators type these: casing and stray whitespace must not silently
        # demote a deliberate override into "unrecognized, auto-detect instead"
        ("  Torch  ", "torch"),
        ("TRITON", "triton"),
    ],
)
def test_dispatch_env_override_wins_both_directions(monkeypatch, override, want):
    """FREETOKEN_MOE_ALIGN_IMPL forces a path regardless of what auto-detection
    would otherwise choose -- checked against BOTH an affected and an unaffected
    auto-detect config to prove the override, not the config, decides."""
    auto_detect_config = (
        dict(rocm=True, triton_version="3.8.0")
        if want == "triton"
        else dict(rocm=True, triton_version="3.7.1")
    )
    monkeypatch.setenv("FREETOKEN_MOE_ALIGN_IMPL", override)
    _patch_dispatch_signals(monkeypatch, **auto_detect_config)
    calls, fake = _stub_impls(monkeypatch)
    topk_ids = torch.zeros(4, dtype=torch.int32)

    assert moe_align_block_size(topk_ids, 16, 8) is fake
    assert calls == [want]


@pytest.mark.parametrize(
    "override,want_warning",
    [
        ("bogus", True),
        # an empty value means "not configured", not "bad value": compose files
        # and shell wrappers export empty vars routinely, and warning about one
        # on every process start would train operators to ignore the log
        ("  ", False),
    ],
)
# Both directions of auto-detection, because "fell through" and "gave up and
# took the safe path" look identical on an affected config. A gate that treated
# any unrecognized value as "force torch" would pass the rocm=True row alone.
@pytest.mark.parametrize("rocm", [True, False])
def test_dispatch_env_override_falls_through_to_auto_detect(
    monkeypatch, caplog, override, want_warning, rocm
):
    """A FREETOKEN_MOE_ALIGN_IMPL value that names neither implementation falls
    through to auto-detection, which then decides on its own merits -- torch on
    the affected ROCm config, Triton off it. An unrecognized value also logs a
    warning naming it, exactly once per distinct value: silent-but-safe would
    violate this repo's no-silent-fallback convention, and warning per dispatch
    call would flood the log. Both halves are checked here, so dropping the
    once-guard fails this test rather than passing it quietly."""
    moe_align_mod._warn_once.cache_clear()  # stay independent of test order
    monkeypatch.setenv("FREETOKEN_MOE_ALIGN_IMPL", override)
    _patch_dispatch_signals(monkeypatch, rocm=rocm, triton_version="3.8.0")
    calls, fake = _stub_impls(monkeypatch)
    topk_ids = torch.zeros(4, dtype=torch.int32)
    want = "torch" if rocm else "triton"

    with caplog.at_level(logging.WARNING, logger=moe_align_mod.logger.name):
        assert moe_align_block_size(topk_ids, 16, 8) is fake
        assert moe_align_block_size(topk_ids, 16, 8) is fake  # latch: still once

    assert calls == [want, want]  # auto-detect ran, and ran the same way twice
    warnings = [
        r for r in caplog.records if "FREETOKEN_MOE_ALIGN_IMPL" in r.getMessage()
    ]
    assert len(warnings) == (1 if want_warning else 0)
    if want_warning:  # the warning has to name the value, or it can't be debugged
        assert override.strip().lower() in warnings[0].getMessage()


def test_dispatch_unparseable_triton_version_warns_and_takes_torch(monkeypatch, caplog):
    """A Triton version string that ``packaging`` cannot parse must not raise out
    of a dispatch call. On ROCm, where the miscompile lives, the safe answer is
    the torch path -- correct on every Triton version -- plus a warning naming
    the string, so this never becomes a silent fallback."""
    moe_align_mod._warn_once.cache_clear()  # stay independent of test order
    monkeypatch.delenv("FREETOKEN_MOE_ALIGN_IMPL", raising=False)
    _patch_dispatch_signals(monkeypatch, rocm=True, triton_version="3.8.0gitabcdef")
    calls, fake = _stub_impls(monkeypatch)
    topk_ids = torch.zeros(4, dtype=torch.int32)

    with caplog.at_level(logging.WARNING, logger=moe_align_mod.logger.name):
        assert moe_align_block_size(topk_ids, 16, 8) is fake
        assert moe_align_block_size(topk_ids, 16, 8) is fake  # latch: still once

    assert calls == ["torch", "torch"]
    warnings = [r for r in caplog.records if "3.8.0gitabcdef" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="exercises the real gate on a real device")
def test_dispatch_picks_correct_impl_for_real_device():
    """Every test above monkeypatches every signal _needs_torch_fallback reads --
    an off-by-one in the gate itself (e.g. comparing against (3, 9) instead of
    (3, 8)) would pass all of them while the real box stayed broken. This calls the PUBLIC,
    unpatched moe_align_block_size against the real detectors on whatever device
    is running the suite, so it's the actual end-to-end guarantee.

    Compares the used region only, not the full buffer: the Triton path leaves
    padding past the used region uninitialized by design (this file's module
    docstring), so a full-buffer comparison would spuriously fail on an
    unaffected device where the gate correctly picks the Triton path.
    """
    torch.manual_seed(7)
    for numel, num_experts, block_size in [(32, 8, 16), (1536, 256, 64)]:  # small + large path
        effective_E = num_experts + 1
        topk_ids = torch.randint(
            0, effective_E, (numel,), device="cuda", dtype=torch.int32
        )
        sorted_token_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
            topk_ids, block_size, num_experts
        )
        npp = int(num_tokens_post_pad.item())
        ref_expert_ids, ref_npp = _reference(topk_ids, block_size, num_experts)
        assert npp == ref_npp
        used_blocks = npp // block_size
        assert expert_ids[:used_blocks].tolist() == ref_expert_ids[:used_blocks]

        buckets, counts, _, starts = _expert_layout(topk_ids, block_size, num_experts)
        got = sorted_token_ids.tolist()
        for e in range(effective_E):
            s, c = starts[e], counts[e]
            assert sorted(got[s : s + c]) == sorted(buckets[e]), f"expert {e} mismatch"
