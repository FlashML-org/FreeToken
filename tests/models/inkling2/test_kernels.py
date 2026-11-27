"""Inkling 2 kernels against eager references written from the Hugging Face model
(transformers PR #51050, ``modeling_inkling2_loop.py``): the attention with its distance bias,
per-token sinks and output gate; the short convolutions with their inputs carried across prefill
chunks and decode steps; the attention residual."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from freetoken.kernel.triton.inkling2 import (  # noqa: E402
    EPI_L2_CLIP,
    EPI_RESIDUAL,
    EPI_RMS_NORM,
    attn_res,
    decode_attention,
    extend_attention,
    short_conv,
)

DEV = "cuda"
HQ, HKV, D, E = 24, 8, 128, 1024


def _ref_attention(q, k, v, gate, bias, sinks, q_pos, window):
    """One request: q [T, Hq, D] at positions q_pos, keys/values [S, Hkv, D] at positions 0..S-1."""
    S = k.shape[0]
    kk = k.float().repeat_interleave(HQ // HKV, dim=1)
    vv = v.float().repeat_interleave(HQ // HKV, dim=1)
    scores = torch.einsum("thd,shd->hts", q.float(), kk) * D**-0.5
    dist = q_pos[:, None] - torch.arange(S, device=DEV)[None, :]
    inside = (dist >= 0) & (dist < E)
    rel = torch.gather(bias.float(), 2, dist.clamp(0, E - 1)[:, None, :].expand(-1, HQ, -1))
    scores = scores + torch.where(inside[:, None, :], rel, 0.0).transpose(0, 1)
    allowed = dist >= 0
    if window is not None:
        allowed &= dist < window
    scores = scores.masked_fill(~allowed[None], float("-inf"))
    logits = torch.cat([scores, sinks.float().t()[:, :, None]], dim=-1)
    probs = logits.softmax(-1)[..., :-1]
    out = torch.einsum("hts,shd->thd", probs, vv).reshape(q.shape[0], -1)
    return (out.bfloat16().float() * torch.sigmoid(gate.float())).bfloat16()


def _paged(S):
    """K/V rows scattered over a larger pool through a shuffled slot list."""
    pool = S + 37
    slots = torch.randperm(pool, device=DEV)[:S].int()
    k_cache = torch.randn(pool, HKV, D, device=DEV).bfloat16()
    v_cache = torch.randn(pool, HKV, D, device=DEV).bfloat16()
    return slots, k_cache, v_cache


@pytest.mark.parametrize("window", [None, 300])
def test_extend_attention_over_a_cached_prefix(window):
    torch.manual_seed(0)
    prefix, T = 1100, 333
    S = prefix + T
    slots, k_cache, v_cache = _paged(S)
    q = torch.randn(T, HQ, D, device=DEV).bfloat16()
    gate = torch.randn(T, HQ * D, device=DEV).bfloat16()
    bias = torch.randn(T, HQ, E, device=DEV).bfloat16()
    sinks = torch.randn(T, HQ, device=DEV) + 6
    out = torch.empty(T, HQ * D, device=DEV, dtype=torch.bfloat16)
    i32 = dict(dtype=torch.int32, device=DEV)
    # two bias windows: the second launch must only write its own rows
    for lo, hi in ((0, 200), (200, T)):
        extend_attention(
            q, k_cache, v_cache, gate, bias[lo:hi].contiguous(), sinks, torch.tensor([0, T], **i32), torch.tensor([0, S], **i32),
            slots, torch.tensor([prefix], **i32), T, D**-0.5, window, E, out, row_lo=lo, row_hi=hi,
        )
    want = _ref_attention(q, k_cache[slots.long()], v_cache[slots.long()], gate, bias, sinks, prefix + torch.arange(T, device=DEV), window)
    torch.testing.assert_close(out.float(), want.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("window", [None, 300])
def test_decode_attention_splits_match_the_reference(window):
    torch.manual_seed(1)
    lens = [1, 290, 1500, 4000]
    B = len(lens)
    slots, k_cache, v_cache = _paged(sum(lens))
    indptr = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=DEV)
    q = torch.randn(B, HQ, D, device=DEV).bfloat16()
    gate = torch.randn(B, HQ * D, device=DEV).bfloat16()
    bias = torch.randn(B, HQ, E, device=DEV).bfloat16()
    sinks = torch.randn(B, HQ, device=DEV) + 6
    out = torch.empty(B, HQ * D, device=DEV, dtype=torch.bfloat16)
    decode_attention(q, k_cache, v_cache, gate, bias, sinks, indptr, slots, D**-0.5, window, E, out)
    for b, n in enumerate(lens):
        rows = slots[indptr[b] : indptr[b + 1]].long()
        want = _ref_attention(q[b : b + 1], k_cache[rows], v_cache[rows], gate[b : b + 1], bias[b : b + 1], sinks[b : b + 1],
                              torch.tensor([n - 1], device=DEV), window)
        torch.testing.assert_close(out[b : b + 1].float(), want.float(), atol=2e-2, rtol=2e-2)


def _ref_conv(x, weight, scale, dyn=None, head_dim=128):
    """``(sum_lag w[c, lag] x[t - lag] + x) * scale`` over a whole sequence starting from zeros, bf16-rounded."""
    xf = x.float()
    T, C = xf.shape
    padded = F.pad(xf.t(), (weight.shape[1] - 1, 0))
    windows = padded.unfold(-1, weight.shape[1], 1).flip(-1)  # [C, T, lag]
    taps = weight[:, None, :].expand(C, T, -1)
    if dyn is not None:
        taps = taps + dyn.float().repeat_interleave(head_dim, dim=1).transpose(0, 1)
    return ((windows * taps).sum(-1).t() + xf).mul(scale).bfloat16()


class _Rows:
    """Per-token conv rows behind full slots, as the KV pool keeps them: a request gets rows only
    where its cache manager gives them, every other slot maps to row 0, filled with NaN so a read
    the cache did not plan shows up in the output."""

    def __init__(self, C, num_slots=128, num_rows=64):
        self.rows = torch.full((num_rows, C), float("nan"), device=DEV).bfloat16()
        self.row_map = torch.zeros(num_slots + 1, dtype=torch.int64, device=DEV)
        self.next = 1

    def give(self, slots):
        n = slots.numel()
        self.row_map[slots.long()] = torch.arange(self.next, self.next + n, device=DEV)
        self.next += n


@pytest.mark.parametrize("dynamic", [False, True])
def test_short_conv_carries_its_inputs_across_chunks_and_decode(dynamic):
    torch.manual_seed(2)
    T, C, K = 40, 1024, 4
    scale = 1 / math.sqrt((2 * K if dynamic else K) + 1)
    x = torch.randn(T, C, device=DEV).bfloat16()
    w = torch.randn(C, K, device=DEV) * 0.4
    dyn = torch.randn(T, C // 128, K, device=DEV).bfloat16() if dynamic else None
    want = _ref_conv(x, w, scale, dyn)
    # two requests in one batch: request 0 prefills x in two chunks then decodes, request 1 is a fresh short prompt
    i32 = dict(dtype=torch.int32, device=DEV)
    rows = _Rows(C)
    perm = torch.randperm(128, device=DEV).int()
    slots0, slots1 = perm[:T], perm[T : T + 5]
    got = []
    for lo, hi in ((0, 2), (2, 30)):
        n = hi - lo
        rows.give(slots0[max(lo, hi - (K - 1)) : hi])  # only the chunk's tail, which the next chunk reads
        xs = torch.cat([x[lo:hi], x[:5]])
        ds = torch.cat([dyn[lo:hi], dyn[:5]]) if dynamic else None
        y = short_conv(xs, w, rows.rows, rows.row_map, torch.cat([slots0[:hi], slots1]), torch.tensor([0, hi, hi + 5], **i32),
                       torch.tensor([lo, 0], **i32), scale, decode=False, q_to_req=torch.tensor([0] * n + [1] * 5, **i32),
                       cu_q=torch.tensor([0, n, n + 5], **i32), dynamic=ds)
        got.append(y[:n])
        torch.testing.assert_close(y[n:].float(), want[:5].float(), atol=1e-2, rtol=1e-2)
    for t in range(30, T):
        rows.give(slots0[t : t + 1])
        got.append(short_conv(x[t : t + 1], w, rows.rows, rows.row_map, slots0[: t + 1], torch.tensor([0, t + 1], **i32),
                              torch.tensor([t], **i32), scale, decode=True, dynamic=dyn[t : t + 1] if dynamic else None))
    torch.testing.assert_close(torch.cat(got).float(), want.float(), atol=1e-2, rtol=1e-2)


def test_short_conv_epilogues():
    torch.manual_seed(3)
    T, C, K = 24, 1024, 4
    x = torch.randn(T, C, device=DEV).bfloat16() * 4
    w = torch.randn(C, K, device=DEV) * 0.4
    i32 = dict(dtype=torch.int32, device=DEV)
    rows = _Rows(C)
    meta = dict(decode=False, q_to_req=torch.zeros(T, **i32), cu_q=torch.tensor([0, T], **i32))
    args = (rows.rows, rows.row_map, torch.arange(T, **i32), torch.tensor([0, T], **i32), torch.tensor([0], **i32))
    conv = _ref_conv(x, w, 0.5).float().view(T, -1, 128)
    norm = conv.norm(dim=-1, keepdim=True)
    clipped = short_conv(x, w, *args, 0.5, epilogue=EPI_L2_CLIP, epi_arg=3.0, **meta)
    torch.testing.assert_close(clipped.float().view(T, -1, 128), conv * (3.0 / (norm + 1e-6)).clamp(max=1.0), atol=2e-2, rtol=2e-2)
    normed = short_conv(x, w, *args, 0.5, epilogue=EPI_RMS_NORM, epi_arg=1e-6, **meta)
    torch.testing.assert_close(normed.float().view(T, -1, 128), conv * torch.rsqrt(conv.square().mean(-1, keepdim=True) + 1e-6), atol=2e-2, rtol=2e-2)
    r1, r2 = torch.randn(T, C, device=DEV).bfloat16(), torch.randn(T, C, device=DEV).bfloat16()
    added = short_conv(x, w, *args, 0.5, epilogue=EPI_RESIDUAL, residual=r1, residual2=r2, **meta)
    torch.testing.assert_close(added.float(), (r1.float() + r2.float() + conv.view(T, C)).bfloat16().float(), atol=0, rtol=0)


def _ref_attn_res(sources, queries):
    stacked = torch.stack([s.float() for s in sources], dim=1)
    n = stacked * torch.rsqrt(stacked.square().mean(-1, keepdim=True) + 1e-6)
    p = torch.einsum("tsh,qh->tqs", n, queries).mul(n.shape[-1] ** -0.5).softmax(-1)
    return torch.einsum("tqs,tsh->qth", p, n)


@pytest.mark.parametrize("num_blocks", [0, 1, 4])
def test_attn_res_mixes_normalized_sources(num_blocks):
    torch.manual_seed(4)
    T, H = 37, 2304
    blocks = torch.randn(T, 4, H, device=DEV).bfloat16() * 3
    x, h = torch.randn(T, H, device=DEV).bfloat16(), torch.randn(T, H, device=DEV).bfloat16()
    queries = torch.randn(3, H, device=DEV) * 0.05
    weight = torch.randn(3, H, device=DEV).bfloat16()
    sources = [blocks[:, i] for i in range(num_blocks)] + [x.float() + h.float()]
    want = _ref_attn_res(sources, queries)
    got = attn_res(blocks, num_blocks, x, queries, partial2=h)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
    normed = attn_res(blocks, num_blocks, x, queries, partial2=h, norm_weight=weight, out_dtype=torch.bfloat16)
    want_n = want * torch.rsqrt(want.square().mean(-1, keepdim=True) + 1e-6) * weight.float()[:, None]
    torch.testing.assert_close(normed.float(), want_n.bfloat16().float(), atol=2e-2, rtol=2e-2)
