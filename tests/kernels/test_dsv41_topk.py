"""dsv41_topk / dsv41_candidate_blocks against an independent torch reference with the documented tie
rule, over the contract's edge cases: empty and short histories, unequal live counts in one batch,
-inf holes, widths crossing the slice and level boundaries, exact ties, newest-block inclusion,
the sorted-valid-prefix / -1-tail output layout, and CUDA-graph replay across changing live lengths
(stale workspace reads would surface as wrong picks)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def ref_topk(scores: torch.Tensor, live: torch.Tensor, k: int) -> torch.Tensor:
    """(score desc, column asc) over the live, finite columns; ascending columns, -1 padded."""
    R, T = scores.shape
    out = torch.full((R, k), -1, dtype=torch.int32)
    for r in range(R):
        n = int(live[r])
        s = scores[r, :n].double().cpu()
        cols = torch.arange(n)
        alive = torch.isfinite(s) | (s == float("inf"))
        cand = [(float(-s[c]), int(c)) for c in cols[alive]]
        cand.sort()
        picked = sorted(c for _, c in cand[:k])
        out[r, : len(picked)] = torch.tensor(picked, dtype=torch.int32)
    return out.to(scores.device)


def ref_candidates(scores, live, topk_blocks, block_size):
    R, T = scores.shape
    nb = -(-T // block_size)
    out = torch.full((R, topk_blocks * block_size), -1, dtype=torch.int32)
    for r in range(R):
        n = int(live[r])
        if n == 0:
            continue
        nbl = -(-n // block_size)
        s = torch.nn.functional.pad(scores[r, :n].float().cpu(), (0, nbl * block_size - n), value=float("-inf")).view(nbl, block_size).amax(-1)
        s[nbl - 1] = float("inf")
        cand = sorted((float(-s[b]), b) for b in range(nbl) if s[b] > float("-inf"))
        keep = sorted(b for _, b in cand[:topk_blocks])
        pos = [p for b in keep for p in range(b * block_size, (b + 1) * block_size) if p < n]
        out[r, : len(pos)] = torch.tensor(pos, dtype=torch.int32)
    return out.to(scores.device)


def _scores(R, T, seed=0, ties=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randn(R, T, generator=g)
    if ties:
        s = torch.randint(0, 5, (R, T), generator=g).float()  # heavy exact ties
    return s.cuda()


@pytest.mark.parametrize("T,k", [(64, 8), (4096, 512), (4097, 512), (8192, 512), (100_000, 512), (262_144, 512), (1_048_576, 512), (131_072, 2048)])
def test_topk_matches_reference_across_widths(T, k):
    from freetoken.kernel.triton.dsv41.topk import dsv41_topk

    R = 3
    scores = _scores(R, T, seed=T)
    live = torch.tensor([0, min(T, 5000), T], dtype=torch.int32, device="cuda")[:R]
    # holes inside the live range
    scores[2, ::7] = float("-inf")
    got = dsv41_topk(scores, live, k)
    want = ref_topk(scores, live, k)
    assert torch.equal(got, want), (got[:, :8], want[:, :8])


def test_topk_short_and_unequal_rows_and_exact_ties():
    from freetoken.kernel.triton.dsv41.topk import dsv41_topk

    T, k = 20_000, 512
    scores = _scores(5, T, seed=3, ties=True)
    live = torch.tensor([0, 1, 300, 4096, 20_000], dtype=torch.int32, device="cuda")
    got = dsv41_topk(scores, live, k)
    want = ref_topk(scores, live, k)
    assert torch.equal(got, want)
    # the layout contract: a sorted valid prefix, then -1
    for r in range(5):
        row = got[r].tolist()
        n = sum(1 for v in row if v >= 0)
        assert row[n:] == [-1] * (k - n) and row[:n] == sorted(row[:n]) and n == min(int(live[r]), k)


def test_topk_all_dead_row_and_fewer_finite_than_k():
    from freetoken.kernel.triton.dsv41.topk import dsv41_topk

    scores = torch.full((2, 10_000), float("-inf"), device="cuda")
    scores[1, [10, 4095, 4096, 9000]] = torch.tensor([1.0, 2.0, 3.0, 0.5], device="cuda")
    live = torch.tensor([10_000, 9001], dtype=torch.int32, device="cuda")
    got = dsv41_topk(scores, live, 8)
    assert (got[0] == -1).all()
    assert got[1].tolist() == [10, 4095, 4096, 9000, -1, -1, -1, -1]


@pytest.mark.parametrize("T", [4000, 200_000])
def test_candidate_blocks_match_reference_and_pin_the_newest_block(T):
    from freetoken.kernel.triton.dsv41.topk import dsv41_candidate_blocks

    R, KB, BS = 4, 2048, 8
    scores = _scores(R, T, seed=T + 1)
    scores[0, :] = -1000.0  # the newest block is kept even when every score is poor
    live = torch.tensor([13, 8, min(T, 16_000), T], dtype=torch.int32, device="cuda")
    got = dsv41_candidate_blocks(scores, live, KB, BS)
    want = ref_candidates(scores, live, KB, BS)
    assert got.shape == (R, KB * BS)
    assert torch.equal(got, want)
    # short histories: every live position is a candidate (the implicit fast path)
    assert got[0].tolist()[:13] == list(range(13)) and (got[0, 13:] == -1).all()
    assert got[1].tolist()[:8] == list(range(8)) and (got[1, 8:] == -1).all()


def test_topk_under_graph_replay_across_live_lengths():
    """Capture once at the staged width, replay with live lengths that move up and down: every replay
    must equal the reference (a stale candidate slot or live count from a previous replay would not)."""
    from freetoken.kernel.triton.dsv41.topk import dsv41_candidate_blocks, dsv41_topk

    T, k, R = 300_000, 512, 2
    scores = torch.empty((R, T), device="cuda")
    live = torch.empty((R,), dtype=torch.int32, device="cuda")
    out = torch.empty((R, k), dtype=torch.int32, device="cuda")
    cand = torch.empty((R, 2048 * 8), dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        scores.copy_(_scores(R, T, seed=9))
        live.fill_(T)
        dsv41_topk(scores, live, k, out)  # warm-up compiles
        cand.copy_(dsv41_candidate_blocks(scores, live, 2048, 8))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            dsv41_topk(scores, live, k, out)
            cand.copy_(dsv41_candidate_blocks(scores, live, 2048, 8))
    torch.cuda.synchronize()
    for step, (l0, l1) in enumerate([(T, 7), (4096, 4097), (0, 1), (250_000, 12_000), (3, T), (16_384, 16_385)]):
        scores.copy_(_scores(R, T, seed=100 + step))
        live.copy_(torch.tensor([l0, l1], dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref_topk(scores, live, k)), (step, l0, l1)
        assert torch.equal(cand, ref_candidates(scores, live, 2048, 8)), (step, l0, l1)


def test_topk_plan_terminates_or_rejects():
    """Every accepted k converges (each level at least halves the row); k past CHUNK // 2 -- which
    would stall the planner (k == CHUNK never shrinks, 3072 stalls at 6144) -- is rejected up front."""
    from freetoken.kernel.triton.dsv41.topk import CHUNK, FINAL_MAX, MAX_K, topk_plan

    for k in (1, 8, 512, 2048, MAX_K):
        for columns in (1, 4096, 4097, 1 << 20, (1 << 20) + 3):
            levels = topk_plan(columns, k)
            widths = [w for w, _ in levels] + [(levels[-1][1] * k) if levels else columns]
            assert all(a > b for a, b in zip(widths, widths[1:])) and widths[-1] <= FINAL_MAX
            assert len(levels) <= 12
    for k in (MAX_K + 1, 2049, 3072, CHUNK, 0, -1):
        with pytest.raises(ValueError):
            topk_plan(1 << 20, k)
