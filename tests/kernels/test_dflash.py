"""The DFlash 2 ring attention against a gathered torch reference, and the selector walk on a cdf that ends below 1; the other kernels are checked through the draft forward in tests/models/dflash/test_draft.py."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")
HEADS, KV_HEADS, HEAD_DIM, WINDOW = 8, 2, 128, 2048
# the serving geometry: the ring is longer than window + block (the GDN track slack), so slots past the window must be masked by position
RING = WINDOW + 64


def _ref_ring_attention(q, ring_k, ring_v, layer, rows, starts, block, scale):
    """Gather every position a block query may see, then an explicit-mask softmax: non-causal over the block, a window behind it, nothing before position 0."""
    group, n = HEADS // KV_HEADS, WINDOW - 1 + block
    t = torch.arange(n, device=DEV)
    j = torch.arange(block, device=DEV)
    outs = []
    for b in range(starts.shape[0]):
        pos = starts[b] - (WINDOW - 1) + t
        k = ring_k[layer, rows[b], pos % RING].float().repeat_interleave(group, dim=1)
        v = ring_v[layer, rows[b], pos % RING].float().repeat_interleave(group, dim=1)
        qb = q[b * block : (b + 1) * block].float().view(block, HEADS, HEAD_DIM)
        scores = torch.einsum("jhd,nhd->jhn", qb, k) * scale
        keep = (t[None, :] >= j[:, None]) & (pos >= 0)[None, :]
        probs = torch.softmax(scores.masked_fill(~keep[:, None, :], float("-inf")), dim=-1)
        outs.append(torch.einsum("jhn,nhd->jhd", probs, v).reshape(block, HEADS * HEAD_DIM))
    return torch.cat(outs)


@pytest.mark.parametrize("block", [4, 8])
@pytest.mark.parametrize("starts, rows", [([5000, 3], [2, 0]), ([10, 2500], [1, 2]), ([70000], [0])])
def test_ring_attention_matches_gathered_reference(block, starts, rows):
    """Sequences shorter than the window, wrapped rings and a block at the very start all see the same keys as the reference."""
    from freetoken.kernel.triton.dflash import ring_attention, ring_attention_workspace

    torch.manual_seed(0)
    layers, num_rows = 2, 3
    ring_k = torch.randn(layers, num_rows, RING, KV_HEADS, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    ring_v = torch.randn_like(ring_k)
    # one key's share of a 2048-wide softmax is below the bf16 noise, so the values just outside and just inside the first query's window (or at positions -1 and 0) are made large enough for an off-by-one mask to show
    for b, start in enumerate(starts):
        outside = max(start - WINDOW, -1)
        ring_v[1, rows[b], outside % RING] = 1e3
        ring_v[1, rows[b], (outside + 1) % RING] = -1e3
    workspace = ring_attention_workspace(num_rows, block, HEADS, KV_HEADS, HEAD_DIM, RING, DEV)
    starts_t, rows_t = torch.tensor(starts, device=DEV), torch.tensor(rows, device=DEV)
    q = torch.randn(len(starts) * block, HEADS * HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    scale = HEAD_DIM**-0.5

    got = ring_attention(q, ring_k, ring_v, 1, rows_t, starts_t, WINDOW, scale, workspace)
    want = _ref_ring_attention(q, ring_k, ring_v, 1, rows_t, starts_t, block, scale)

    assert got.dtype == q.dtype
    torch.testing.assert_close(got.float(), want, atol=1e-2, rtol=1e-2)


def test_selector_walk_never_draws_a_zero_q_candidate():
    """A draw just below 1 past an fp32 cdf total that ends below 1 still lands on a candidate with q > 0, not on the last one whose q underflowed to 0."""
    from freetoken.kernel.triton.dflash import selector_walk

    torch.manual_seed(0)
    bs, top_k, rank, vocab = 512, 16, 32, 1000
    pre = torch.randn(vocab, rank, device=DEV, dtype=torch.bfloat16)
    suc = torch.randn(vocab, rank, device=DEV, dtype=torch.bfloat16)
    ids = torch.arange(top_k, device=DEV).expand(bs, 1, top_k).contiguous()
    unary = torch.full((bs, 1, top_k), -1e4, device=DEV)
    unary[..., :2] = torch.rand(bs, 1, 2, device=DEV)
    state = torch.zeros(bs, 1, rank, device=DEV, dtype=torch.bfloat16)
    anchor = torch.zeros(bs, dtype=torch.int64, device=DEV)
    temperature = torch.ones(bs, device=DEV)
    greedy = torch.zeros(bs, dtype=torch.bool, device=DEV)
    uniforms = torch.full((bs, 1), 1 - 2**-24, device=DEV)

    tokens, q = selector_walk(ids, unary, state, anchor, pre, suc, temperature, greedy, uniforms)

    # two nonzero terms sum the same in any order, so this is the kernel's own cdf total: the draw must sit past it on some rows
    assert (q.sum(-1) <= uniforms).any()
    assert (q.gather(-1, tokens[..., None]) > 0).all()
