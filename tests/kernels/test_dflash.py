"""The DFlash 2 draft kernels against torch references: block attention over the wrapped KV ring, the grouped 2-tap conv, the fused norm + rope ring store, and the selector walk."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")
HEADS, KV_HEADS, HEAD_DIM, WINDOW = 8, 2, 128, 2048
# the serving geometry: the ring is longer than window + block (the GDN track slack), so slots past the window must be masked by position
RING = WINDOW + 64
EPS = 1e-6


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
    from freetoken.kernel.triton.dflash import RingAttention

    torch.manual_seed(0)
    layers, num_rows = 2, 3
    ring_k = torch.randn(layers, num_rows, RING, KV_HEADS, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    ring_v = torch.randn_like(ring_k)
    # one key's share of a 2048-wide softmax is below the bf16 noise, so the values just outside and just inside the first query's window (or at positions -1 and 0) are made large enough for an off-by-one mask to show
    for b, start in enumerate(starts):
        outside = max(start - WINDOW, -1)
        ring_v[1, rows[b], outside % RING] = 1e3
        ring_v[1, rows[b], (outside + 1) % RING] = -1e3
    attention = RingAttention(
        max_bs=num_rows, block=block, num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM, ring=RING, window=WINDOW, device=DEV,
    )
    starts_t, rows_t = torch.tensor(starts, device=DEV), torch.tensor(rows, device=DEV)
    q = torch.randn(len(starts) * block, HEADS * HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    scale = HEAD_DIM**-0.5

    got = attention(q, ring_k, ring_v, 1, rows_t, starts_t, scale)
    want = _ref_ring_attention(q, ring_k, ring_v, 1, rows_t, starts_t, block, scale)

    assert got.dtype == q.dtype
    torch.testing.assert_close(got.float(), want, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("block", [4, 8])
def test_grouped_conv_matches_reference(block):
    """``out[n] = sum_t (base[t] + coef[n, t, group]) * x[n - t]``, with the tap into the previous row dropped on a block's first row."""
    from freetoken.kernel.triton.dflash import grouped_conv

    torch.manual_seed(0)
    n, hidden, group_size, taps = 2 * block, 512, 16, 2
    groups = hidden // group_size
    x = torch.randn(n, hidden, device=DEV, dtype=torch.bfloat16)
    proj = torch.randn(n, 2 * taps * groups, device=DEV, dtype=torch.bfloat16)
    base = torch.randn(2, taps, hidden, device=DEV, dtype=torch.bfloat16)
    # the model hands the kernel one strided half of the projection
    coef = proj.view(n, 2, taps, groups)
    for side in (0, 1):
        got = grouped_conv(x, coef[:, side], base[side], block, group_size)
        blocks = x.float().view(n, groups, group_size)
        c = base[side].float().view(1, taps, groups, group_size) + coef[:, side].float().unsqueeze(-1)
        first = (torch.arange(n, device=DEV) % block).view(-1, 1, 1)
        want = c[:, 0] * blocks + c[:, 1] * F.pad(blocks[:-1], (0, 0, 0, 0, 1, 0)) * (first >= 1)
        assert torch.equal(got, want.flatten(-2).to(x.dtype))


def _ref_norm_rope(x, weight, cos_sin, pos):
    """Per-head RMSNorm then NeoX rope of ``[n, heads*HEAD_DIM]`` rows in fp32."""
    n = x.shape[0]
    x = x.float().view(n, -1, HEAD_DIM)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()
    half = HEAD_DIM // 2
    cos, sin = cos_sin[pos.long()][:, None, :half], cos_sin[pos.long()][:, None, half:]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).view(n, -1)


@pytest.mark.parametrize("with_q", [True, False])
def test_norm_rope_store_matches_reference_and_lands_in_the_ring(with_q):
    """The block path (q out, one layer's k/v into the ring) and the context path (every layer's k/v at once): q and k against an fp32 norm + rope, v stored as is, each at slot ``pos % ring`` of its row and layer."""
    from freetoken.kernel.triton.dflash import norm_rope_store

    torch.manual_seed(0)
    n, layers, layer0 = 4, 3, 1
    num_layers, q_heads = (1, HEADS) if with_q else (layers - layer0, 0)
    width = num_layers * (q_heads + 2 * KV_HEADS) * HEAD_DIM
    src = (torch.randn(n, width, device=DEV) * 3).to(torch.bfloat16)
    q_norm = (1 + 0.3 * torch.randn(HEAD_DIM, device=DEV)).to(torch.bfloat16)
    k_norms = (1 + 0.3 * torch.randn(layers, HEAD_DIM, device=DEV)).to(torch.bfloat16)
    pos = torch.tensor([5047, 5048, 5049, 5050], device=DEV, dtype=torch.int32)
    inv = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, device=DEV, dtype=torch.float32) / HEAD_DIM))
    freqs = torch.outer(torch.arange(6000, device=DEV, dtype=torch.float32), inv)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], -1)
    ring_k = torch.zeros(layers, 2, RING, KV_HEADS, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    ring_v = torch.zeros_like(ring_k)
    q = torch.empty(n, HEADS * HEAD_DIM, device=DEV, dtype=torch.bfloat16) if with_q else None
    rows = torch.ones(n, dtype=torch.int64, device=DEV)

    norm_rope_store(
        src, num_layers=num_layers, num_q_heads=q_heads, q_out=q, q_norm=q_norm if with_q else None, k_norms=k_norms,
        cos_sin=cos_sin, positions=pos, rows=rows, ring_k=ring_k, ring_v=ring_v, layer0=layer0, eps=EPS,
    )

    slot = pos.long() % RING
    per_layer = (q_heads + 2 * KV_HEADS) * HEAD_DIM
    kv = KV_HEADS * HEAD_DIM
    if with_q:
        torch.testing.assert_close(q.float(), _ref_norm_rope(src[:, : q_heads * HEAD_DIM], q_norm, cos_sin, pos), atol=0, rtol=2**-7)
    for i in range(num_layers):
        layer = layer0 + i
        k_src = src[:, i * per_layer + q_heads * HEAD_DIM : i * per_layer + q_heads * HEAD_DIM + kv]
        v_src = src[:, (i + 1) * per_layer - kv : (i + 1) * per_layer]
        k_got = ring_k[layer, 1, slot].reshape(n, -1)
        torch.testing.assert_close(k_got.float(), _ref_norm_rope(k_src, k_norms[layer], cos_sin, pos), atol=0, rtol=2**-7)
        assert torch.equal(ring_v[layer, 1, slot].reshape(n, -1), v_src)
    # nothing else was touched: row 0, the untouched layers and every other slot stay zero
    written = ring_k.abs().sum() + ring_v.abs().sum()
    expected = ring_k[layer0 : layer0 + num_layers, 1, slot].abs().sum() + ring_v[layer0 : layer0 + num_layers, 1, slot].abs().sum()
    assert written == expected


@pytest.mark.parametrize("greedy", [True, False])
def test_selector_walk_matches_torch_walk(greedy):
    """Each slot scores ``unary + <pre[prev] * state, suc[cand]>`` for the path's predecessor only; argmax takes the first of tied candidates, sampling draws with the given uniforms."""
    from freetoken.kernel.triton.dflash import selector_walk

    torch.manual_seed(0)
    bs, slots, top_k, rank, vocab = 3, 3, 16, 32, 1000
    pre = torch.randn(vocab, rank, device=DEV, dtype=torch.bfloat16)
    suc = torch.randn(vocab, rank, device=DEV, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(vocab, device=DEV)[:top_k] for _ in range(bs * slots)]).view(bs, slots, top_k)
    unary = torch.randn(bs, slots, top_k, device=DEV)
    state = torch.randn(bs, slots, rank, device=DEV, dtype=torch.bfloat16)
    # request 2: a flat score row, where the first candidate must win the tie
    state[2] = 0
    unary[2] = 0
    anchor = torch.tensor([77, 999, 5], device=DEV)
    temperature = torch.tensor([1.0, 0.7, 1.0], device=DEV)
    greedy_t = torch.full((bs,), greedy, device=DEV)
    uniforms = torch.rand(bs, slots, device=DEV)

    tokens, q = selector_walk(ids, unary, state, anchor, pre, suc, temperature, greedy_t, uniforms)

    prev = anchor
    want_tokens, want_q = [], []
    for s in range(slots):
        w = pre[prev].float() * state[:, s].float()
        scores = unary[:, s] + (suc[ids[:, s]].float() * w[:, None]).sum(-1)
        if greedy:
            index = scores.argmax(-1)
            probs = F.one_hot(index, top_k).float()
        else:
            probs = torch.softmax(scores / temperature[:, None], -1)
            index = (uniforms[:, s : s + 1] >= probs.cumsum(-1)).sum(-1).clamp(max=top_k - 1)
        prev = ids[:, s].gather(1, index[:, None])[:, 0]
        want_tokens.append(prev)
        want_q.append(probs)
    assert torch.equal(tokens, torch.stack(want_tokens, 1))
    torch.testing.assert_close(q, torch.stack(want_q, 1), atol=1e-5, rtol=0)
    if greedy:
        assert torch.equal(tokens[2], ids[2, :, 0])


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
