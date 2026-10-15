"""DSV41 sparse attention over packed pools against a torch reference on the DEQUANTIZED pools.

``m == 1`` takes the split-k decode path, ``m > 1`` the single-program prefill one. Tolerance follows
the dsv4 sparse-attention tests (5e-2 on fp32-cast bf16): the online-softmax accumulation order differs
from the reference and split-k merges through log-sum-exp on top of that."""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.dsv4.v41_row_format import BF16, FP4_E4M3_B16, FP8_E8M0_B32

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

D, H = 128, 8
N_WINDOW = 128  # a multiple of BLOCK_T
N_WIN_SLOTS, N_CMP = 512, 1024
TOL = dict(atol=5e-2, rtol=5e-2)


def _reference(q, win, cmp, sink, idx, n_window, scale, counts):
    """Per (request, query, head) softmax over the live columns; ``sink`` is a null key."""
    b, m, h, d = q.shape
    out = torch.zeros_like(q, dtype=torch.float32)
    for i in range(b):
        for j in range(m):
            width = idx.shape[-1] if counts is None else n_window + int(counts[i, j])
            kv = []
            for c in range(min(width, idx.shape[-1])):
                slot = int(idx[i, j, c])
                if slot >= 0:
                    kv.append((win if c < n_window else cmp)[slot].float())
            if not kv:
                continue
            kv_t = torch.stack(kv)
            logits = q[i, j].float() @ kv_t.T * scale
            mx = torch.maximum(logits.max(dim=-1).values, sink.float())
            probs = torch.exp(logits - mx[:, None])
            denom = probs.sum(dim=-1) + torch.exp(sink.float() - mx)
            out[i, j] = (probs @ kv_t) / denom[:, None]
    return out


def _pools(win_fmt, cmp_fmt, seed):
    from freetoken.kernel.triton.dsv41.pack import pack_rows, unpack_rows

    torch.manual_seed(seed)
    win_vals = torch.randn(N_WIN_SLOTS, D, device="cuda", dtype=torch.bfloat16)
    cmp_vals = torch.randn(N_CMP, D, device="cuda", dtype=torch.bfloat16)
    win_pool = pack_rows(win_vals, win_fmt)
    cmp_pool = pack_rows(cmp_vals, cmp_fmt)
    return win_pool, cmp_pool, unpack_rows(win_pool, win_fmt, D), unpack_rows(cmp_pool, cmp_fmt, D)


def _topk(b, m, n_cmp_cols, seed, ring_fill=None):
    torch.manual_seed(seed)
    win = torch.randint(0, N_WIN_SLOTS, (b, m, N_WINDOW), device="cuda", dtype=torch.int32)
    if ring_fill is not None:  # a ring still filling: trailing window columns empty
        win[..., ring_fill:] = -1
    cmp = torch.randint(0, N_CMP, (b, m, n_cmp_cols), device="cuda", dtype=torch.int32)
    cmp[..., ::7] = -1  # scattered unreachable picks
    return torch.cat([win, cmp], dim=-1)


@pytest.mark.parametrize("m", [1, 3])
@pytest.mark.parametrize("fmts", [(FP8_E8M0_B32, FP4_E4M3_B16), (BF16, BF16)], ids=["fp8+fp4", "bf16"])
def test_matches_reference_on_dequantized_pools(m, fmts):
    from freetoken.kernel.triton.dsv41.sparse_attn import sparse_attn_packed

    win_fmt, cmp_fmt = fmts
    b, n_cmp_cols = 2, 256
    win_pool, cmp_pool, win_deq, cmp_deq = _pools(win_fmt, cmp_fmt, seed=1)
    q = torch.randn(b, m, H, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda") * 0.5
    idx = _topk(b, m, n_cmp_cols, seed=2, ring_fill=100)
    scale = D**-0.5

    got = sparse_attn_packed(q, win_pool, win_fmt, cmp_pool, cmp_fmt, sink, idx, N_WINDOW, scale)
    want = _reference(q, win_deq, cmp_deq, sink, idx, N_WINDOW, scale, None)
    torch.testing.assert_close(got.float(), want, **TOL)


def test_cmp_counts_bound_the_visited_columns():
    from freetoken.kernel.triton.dsv41.sparse_attn import sparse_attn_packed

    b, m, n_cmp_cols = 3, 1, 512
    win_pool, cmp_pool, win_deq, cmp_deq = _pools(FP8_E8M0_B32, FP4_E4M3_B16, seed=3)
    q = torch.randn(b, m, H, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(H, device="cuda")
    idx = _topk(b, m, n_cmp_cols, seed=4)
    counts = torch.tensor([[0], [37], [512]], device="cuda", dtype=torch.int32)
    scale = D**-0.5

    got = sparse_attn_packed(q, win_pool, FP8_E8M0_B32, cmp_pool, FP4_E4M3_B16, sink, idx, N_WINDOW, scale, cmp_counts=counts)
    want = _reference(q, win_deq, cmp_deq, sink, idx, N_WINDOW, scale, counts)
    torch.testing.assert_close(got.float(), want, **TOL)


def test_all_masked_query_yields_zeros():
    from freetoken.kernel.triton.dsv41.sparse_attn import sparse_attn_packed

    win_pool, cmp_pool, _, _ = _pools(FP8_E8M0_B32, FP4_E4M3_B16, seed=5)
    q = torch.randn(1, 2, H, D, device="cuda", dtype=torch.bfloat16)
    idx = torch.full((1, 2, N_WINDOW + 64), -1, device="cuda", dtype=torch.int32)
    out = sparse_attn_packed(q, win_pool, FP8_E8M0_B32, cmp_pool, FP4_E4M3_B16, torch.zeros(H, device="cuda"), idx, N_WINDOW, 0.1)
    assert torch.equal(out, torch.zeros_like(out))
