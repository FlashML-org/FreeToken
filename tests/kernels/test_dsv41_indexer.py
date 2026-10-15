"""DSV41 indexer logits over packed fp4 index keys: full-range (Full mode) and candidate-restricted
(Reindex mode) against a torch reference on the dequantized keys, with causality from ``live``. The
row of a compressed position comes from the request's full-token locs (``locs[t * ratio] // ratio``);
in Full mode only the live columns are written."""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.dsv4.v41_row_format import FP4_E8M0_B32

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

D, H = 128, 32
ROWS = 4200


RATIO = 2


def _setup(b, s, t, seed):
    from freetoken.kernel.triton.dsv41.pack import pack_rows, unpack_rows

    torch.manual_seed(seed)
    keys = torch.randn(ROWS, D, device="cuda", dtype=torch.bfloat16)
    pool = pack_rows(keys, FP4_E8M0_B32)
    deq = unpack_rows(pool, FP4_E8M0_B32, D)
    q = torch.randn(b, s, H, D, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(b, s, H, device="cuda") * 0.3).bfloat16()
    # each request owns a random permutation of pool rows for its compressed positions; a few holes.
    # The kernel derives rows from full-token locs: compressed position p -> loc = row * RATIO (+ any
    # in-group offset) at locs[p * RATIO]; a hole is a negative loc.
    k_rows = torch.stack([torch.randperm(ROWS, device="cuda")[:t] for _ in range(b)]).to(torch.int32)
    k_rows[:, 5::11] = -1
    locs = torch.full((b, t * RATIO), -1, device="cuda", dtype=torch.int32)
    for i in range(b):
        for p in range(t):
            r = int(k_rows[i, p])
            if r >= 0:
                locs[i, p * RATIO : (p + 1) * RATIO] = torch.arange(r * RATIO, (r + 1) * RATIO, device="cuda", dtype=torch.int32)
    return pool, deq, q, w, k_rows, locs


def _reference(q, w, deq, k_rows, live, positions):
    """positions: [B, S, N] compressed positions (-1 empty) -> logits [B, S, N]."""
    b, s, n = positions.shape
    out = torch.full((b, s, n), float("-inf"), device="cuda")
    for i in range(b):
        for j in range(s):
            for c in range(n):
                p = int(positions[i, j, c])
                if p < 0 or p >= int(live[i, j]):
                    continue
                r = int(k_rows[i, p])
                if r < 0:
                    continue
                score = (q[i, j] @ deq[r]).relu() * w[i, j]
                out[i, j, c] = score.sum()
    return out


def test_full_range_matches_reference_with_causal_live():
    from freetoken.kernel.triton.dsv41.indexer import indexer_logits_packed

    b, s, t = 2, 5, 300
    pool, deq, q, w, k_rows, locs = _setup(b, s, t, seed=1)
    live = torch.tensor([[10, 64, 65, 200, 300], [0, 1, 128, 129, 250]], device="cuda", dtype=torch.int32)
    out = torch.full((b, s, t), 12345.0, device="cuda")  # a sentinel: the dead area must stay untouched
    got = indexer_logits_packed(q, w, pool, FP4_E8M0_B32, locs, RATIO, live, out=out)
    positions = torch.arange(t, device="cuda").view(1, 1, t).expand(b, s, t)
    want = _reference(q, w, deq, k_rows, live, positions)
    assert got.shape == (b, s, t)
    written = positions < live.unsqueeze(-1)
    touched = positions < (torch.div(live + 63, 64, rounding_mode="floor") * 64).unsqueeze(-1)  # whole live tiles
    assert (got[~touched] == 12345.0).all() and torch.isneginf(got[touched & ~written]).all()
    assert torch.equal(torch.isinf(got[written]), torch.isinf(want[written]))
    finite = written & ~torch.isinf(want)
    torch.testing.assert_close(got[finite], want[finite], atol=2e-2, rtol=2e-2)


def test_quantized_queries_preserve_reference_score_rounding():
    from freetoken.kernel.triton.dsv41.indexer import indexer_logits_packed
    from freetoken.kernel.triton.dsv41.pack import pack_rows, unpack_rows

    torch.manual_seed(23)
    t, s, topk = 4096, 32, 512
    pool = pack_rows(torch.randn(t, D, device="cuda", dtype=torch.bfloat16), FP4_E8M0_B32)
    keys = unpack_rows(pool, FP4_E8M0_B32, D)
    q = torch.randn(s * H, D, device="cuda", dtype=torch.bfloat16)
    q = unpack_rows(pack_rows(q, FP4_E8M0_B32), FP4_E8M0_B32, D).view(1, s, H, D)
    w = torch.randn(1, s, H, device="cuda", dtype=torch.bfloat16) * (D * H) ** -0.5
    locs = torch.arange(t, device="cuda", dtype=torch.int32).view(1, t)
    live = torch.full((1, s), t, device="cuda", dtype=torch.int32)
    got = indexer_logits_packed(q, w, pool, FP4_E8M0_B32, locs, 1, live)
    reference = torch.einsum("bshd,btd->bsht", q, keys.unsqueeze(0))
    reference = (reference.relu() * w.unsqueeze(-1)).sum(dim=2).float()
    torch.testing.assert_close(got, reference, atol=0, rtol=0)
    # BF16 scores can tie at the cutoff; compare selected scores, not arbitrary tie order.
    torch.testing.assert_close(got.topk(topk).values, reference.topk(topk).values, atol=0, rtol=0)


def test_full_range_worker_grid_covers_wide_histories(monkeypatch):
    """More live tiles than worker programs: each worker strides over its share."""
    from freetoken.kernel.triton.dsv41 import indexer as mod

    b, s, t = 1, 2, 4096  # 64 tiles
    pool, deq, q, w, k_rows, locs = _setup(b, s, t, seed=4)
    live = torch.tensor([[4096, 4000]], device="cuda", dtype=torch.int32)
    from types import SimpleNamespace

    properties = torch.cuda.get_device_properties(q.device)
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(
            multi_processor_count=5, major=properties.major, minor=properties.minor,
        ))
        got = mod.indexer_logits_packed(q, w, pool, FP4_E8M0_B32, locs, RATIO, live)
    positions = torch.arange(t, device="cuda").view(1, 1, t).expand(b, s, t)
    want = _reference(q, w, deq, k_rows, live, positions)
    written = positions < live.unsqueeze(-1)
    assert torch.equal(torch.isinf(got[written]), torch.isinf(want[written]))
    finite = written & ~torch.isinf(want)
    torch.testing.assert_close(got[finite], want[finite], atol=2e-2, rtol=2e-2)


def test_candidate_pool_matches_reference():
    from freetoken.kernel.triton.dsv41.indexer import indexer_logits_packed

    b, s, t, nc = 2, 3, 400, 96
    pool, deq, q, w, k_rows, locs = _setup(b, s, t, seed=2)
    live = torch.tensor([[400, 350, 33], [1, 400, 200]], device="cuda", dtype=torch.int32)
    cand = torch.randint(0, t, (b, s, nc), device="cuda", dtype=torch.int32)
    cand[..., -10:] = -1  # empty tail of the pool
    got = indexer_logits_packed(q, w, pool, FP4_E8M0_B32, locs, RATIO, live, candidates=cand)
    want = _reference(q, w, deq, k_rows, live, cand)
    assert got.shape == (b, s, nc)
    assert torch.equal(torch.isinf(got), torch.isinf(want))
    finite = ~torch.isinf(want)
    torch.testing.assert_close(got[finite], want[finite], atol=2e-2, rtol=2e-2)
