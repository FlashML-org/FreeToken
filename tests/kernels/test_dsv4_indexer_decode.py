"""Fused Lightning-Indexer decode scoring against a float32 reference.

The kernel replaces a torch chain that materialized the gathered keys and the per-head score
tensor, and it reads its column bound from device memory. The reference here is computed in
fp32 on purpose: the chain it replaces accumulated in bf16, so comparing against that chain
would measure the CHAIN's rounding, not the kernel's (the kernel is the more accurate of the
two -- same choice the prefill kernel in this file's module already made).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.dsv4.indexer import indexer_decode_logits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, D, RATIO = 64, 128, 4
POOL_ROWS = 1 << 14
TOL = dict(atol=5e-2, rtol=5e-2)


@pytest.fixture(scope="module")
def pool():
    g = torch.Generator(device="cuda").manual_seed(3)
    return torch.randn(POOL_ROWS, D, device="cuda", dtype=torch.bfloat16, generator=g)


def _build(positions, n_stage, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    b = len(positions)
    q = torch.randn(b, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(b, H, device="cuda", dtype=torch.bfloat16, generator=g)
    snap = torch.full((b, n_stage * RATIO), -1, dtype=torch.int64, device="cuda")
    for i, p in enumerate(positions):
        if p >= 0:
            snap[i, : p + 1] = torch.randint(0, POOL_ROWS * RATIO, (p + 1,), device="cuda",
                                             dtype=torch.int64, generator=g)
    valid = torch.tensor([(p + 1) // RATIO for p in positions], device="cuda", dtype=torch.int64)
    return q, w, snap, valid


def _reference(q, w, pool, snap, valid, n_stage, row_map=None):
    """fp32 ground truth: sum_h relu(q_h . k_t) * w_h over the live blocks, -inf past them."""
    b = q.shape[0]
    rows = torch.arange(b, device=q.device) if row_map is None else row_map
    blk = torch.arange(n_stage, device=q.device)
    live = blk[None, :] < valid[:, None]
    at = snap[rows[:, None], (blk * RATIO)[None, :]]
    kv = pool[torch.div(at, RATIO, rounding_mode="floor").clamp_min(0)].float()
    s = torch.einsum("bhd,btd->bht", q.float(), kv).relu_() * w.float().unsqueeze(-1)
    return torch.where(live, s.sum(dim=1), float("-inf"))


def _check(pool, positions, n_stage, seed=0):
    q, w, snap, valid = _build(positions, n_stage, seed)
    rows = torch.arange(q.shape[0], dtype=torch.int64, device=q.device)
    got = indexer_decode_logits(q, w, pool, snap, rows, valid, n_stage, RATIO).float()
    ref = _reference(q, w, pool, snap, valid, n_stage)
    assert torch.equal(torch.isfinite(got), torch.isfinite(ref)), "live/-inf column set differs"
    live = torch.isfinite(ref)
    if live.any():
        torch.testing.assert_close(got[live], ref[live], **TOL)
    return got, ref, valid


def test_basic(pool):
    _check(pool, [800], 2048)


def test_live_prefix_only(pool):
    """Everything past the per-row live count must be -inf, whatever the staged width."""
    got, _, valid = _check(pool, [800], 4096)
    assert torch.isinf(got[0, int(valid[0]):]).all() and (got[0, int(valid[0]):] < 0).all()
    assert torch.isfinite(got[0, : int(valid[0])]).all()


def test_valid_zero(pool):
    got, _, _ = _check(pool, [-1], 2048)
    assert torch.isinf(got).all()


def test_valid_equals_staged_width(pool):
    n_stage = 1024
    _check(pool, [n_stage * RATIO - 1], n_stage)


def test_unaligned_staged_width(pool):
    """n_stage not a multiple of the kernel's tile."""
    _check(pool, [300], 100)


def test_per_row_valid(pool):
    """Co-tenant decode: rows with different live counts stay isolated."""
    got, ref, valid = _check(pool, [-1, 40, 800, 8000], 2048)
    for i, v in enumerate(valid.tolist()):
        assert torch.isfinite(got[i, :v]).all()
        assert torch.isinf(got[i, v:]).all()


def test_multiple_queries_share_one_request_snapshot(pool):
    """DSpark verifies anchor+gamma query rows against one snapshot row per request."""
    n_stage = 256
    q, w, _, _ = _build([400] * 6, n_stage, seed=17)
    _, _, snap, _ = _build([400], n_stage, seed=19)
    rows = torch.zeros(6, dtype=torch.int64, device=q.device)
    valid = torch.tensor([1, 7, 19, 41, 73, 100], dtype=torch.int64, device=q.device)
    got = indexer_decode_logits(q, w, pool, snap, rows, valid, n_stage, RATIO).float()
    ref = _reference(q, w, pool, snap, valid, n_stage, row_map=rows)
    assert torch.equal(torch.isfinite(got), torch.isfinite(ref))
    live = torch.isfinite(ref)
    torch.testing.assert_close(got[live], ref[live], **TOL)


def test_cuda_graph_follows_valid(pool):
    """A captured graph must read the bound from device memory, not from capture time."""
    n_stage = 2048
    q, w, snap, valid = _build([800], n_stage)
    out = torch.empty((1, n_stage), dtype=pool.dtype, device="cuda")
    rows = torch.arange(q.shape[0], dtype=torch.int64, device=q.device)
    call = lambda: indexer_decode_logits(q, w, pool, snap, rows, valid, n_stage, RATIO)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(call())
    for live in (200, 2048, 0, 37):
        valid.fill_(live)
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference(q, w, pool, snap, valid, n_stage)
        got = out.float()
        assert torch.equal(torch.isfinite(got), torch.isfinite(ref)), f"live={live}"
        mask = torch.isfinite(ref)
        if mask.any():
            torch.testing.assert_close(got[mask], ref[mask], **TOL)
