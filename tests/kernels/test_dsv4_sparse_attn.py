"""DSV4 paged sparse-attention kernel against a torch reference.

Which kernel runs follows the launch shape, so the cases drive both through ``m``: ``m == 1``
(decode) takes the split-k path, ``m > 1`` (prefill) the single-program one. Tolerance follows
sglang's equivalent sparse-MLA kernel tests (atol/rtol 5e-2 on fp32-cast bf16 outputs): the
online-softmax accumulation order differs from the reference, and split-k merges through
log-sum-exp on top of that.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.dsv4.pack import pack_rows, unpack_rows
from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged, split_count
from freetoken.kvcache.dsv4.row_format import FP8_E8M0_B64_ROPE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# m == 1 is a decode launch (split-k), m == 2 a prefill one (single program per head block).
MS = (1, 2)
D, H, N_WINDOW = 128, 8, 32
FMT = FP8_E8M0_B64_ROPE
N_WIN_SLOTS, N_CMP = 256, 512
# wide enough that a decode launch really splits (see test_split_count_follows_shape)
N_CMP_COLS = 256
TOL = dict(atol=5e-2, rtol=5e-2)


def _reference(q, win_pool, cmp_pool, sink, idx, n_window, scale, counts):
    """Per (request, query, head) softmax over the live candidate columns.

    A column contributes when it is inside the live width (``n_window + counts``, or the whole
    buffer when counts is None) AND its slot is >= 0. ``sink`` is a null key: it adds
    ``exp(sink_h - m)`` to the denominator and nothing to the numerator.
    """
    b, m, h, d = q.shape
    out = torch.zeros_like(q, dtype=torch.float32)
    for i in range(b):
        for j in range(m):
            width = idx.shape[-1] if counts is None else n_window + int(counts[i, j])
            cols, kv = [], []
            for c in range(min(width, idx.shape[-1])):
                slot = int(idx[i, j, c])
                if slot < 0:
                    continue
                cols.append(c)
                kv.append((win_pool if c < n_window else cmp_pool)[slot].float())
            logits = torch.full((h, len(cols)), float("-inf"), device=q.device)
            if cols:
                kv_t = torch.stack(kv)  # [n, d]
                logits = q[i, j].float() @ kv_t.T * scale  # [h, n]
            mx = torch.maximum(logits.max(dim=-1).values, sink.float()) if cols else sink.float()
            probs = torch.exp(logits - mx[:, None]) if cols else logits
            denom = (probs.sum(dim=-1) if cols else torch.zeros(h, device=q.device)) + torch.exp(
                sink.float() - mx
            )
            if cols:
                out[i, j] = (probs @ kv_t) / denom[:, None]
    return out


def _build(b=2, m=1, n_cmp_cols=N_CMP_COLS, cmp_valid=None, win_valid=N_WINDOW, seed=0, holes=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, m, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    idx = torch.full((b, m, N_WINDOW + n_cmp_cols), -1, dtype=torch.int32, device="cuda")
    cmp_valid = [n_cmp_cols] * b if cmp_valid is None else cmp_valid
    for i in range(b):
        wv = win_valid if isinstance(win_valid, int) else win_valid[i]
        idx[i, :, :wv] = torch.randint(0, N_WIN_SLOTS, (m, wv), device="cuda",
                                       dtype=torch.int32, generator=g)
        cv = cmp_valid[i]
        idx[i, :, N_WINDOW:N_WINDOW + cv] = torch.randint(
            0, N_CMP, (m, cv), device="cuda", dtype=torch.int32, generator=g)
        if holes:  # scattered invalid slots INSIDE the live range
            idx[i, :, N_WINDOW + 1:N_WINDOW + cv:3] = -1
    counts = torch.tensor(cmp_valid, dtype=torch.int32, device="cuda").view(b, 1).expand(b, m)
    return q, idx, counts.contiguous()


@pytest.fixture(scope="module")
def pools():
    """Packed window / compressed pools, the sink, and the dequantized pools for the reference."""
    g = torch.Generator(device="cuda").manual_seed(7)
    win = pack_rows(torch.randn(N_WIN_SLOTS, D, device="cuda", dtype=torch.bfloat16, generator=g), FMT)
    cmp = pack_rows(torch.randn(N_CMP, D, device="cuda", dtype=torch.bfloat16, generator=g), FMT)
    sink = torch.randn(H, device="cuda", dtype=torch.float32, generator=g)
    return win, cmp, sink, unpack_rows(win, FMT, D), unpack_rows(cmp, FMT, D)


def _check(pools, q, idx, counts):
    win, cmp, sink, win_ref, cmp_ref = pools
    scale = D ** -0.5
    got = sparse_attn_paged(q, win, cmp, sink, idx, N_WINDOW, scale, cmp_counts=counts)
    ref = _reference(q, win_ref, cmp_ref, sink, idx, N_WINDOW, scale, counts)
    torch.testing.assert_close(got.float(), ref, **TOL)
    return got


def test_split_count_follows_shape():
    """No knob picks the kernel: prefill never splits, a decode launch with a real candidate
    list does, and a candidate list too short to slice falls back."""
    dev = torch.device("cuda")
    assert split_count(b=2, m=2, h=H, topk=N_WINDOW + N_CMP_COLS, device=dev) == 0
    assert split_count(b=2, m=1, h=H, topk=N_WINDOW + N_CMP_COLS, device=dev) > 1
    assert split_count(b=1, m=1, h=H, topk=N_WINDOW + 8, device=dev) == 0


@pytest.mark.parametrize("m", MS)
def test_model_width(m):
    """DeepSeek-V4's 512-wide rows (448 e4m3 + 64 rope channels): the width the kernel's shared memory must fit."""
    d = 512
    g = torch.Generator(device="cuda").manual_seed(11)
    win = pack_rows(torch.randn(N_WIN_SLOTS, d, device="cuda", dtype=torch.bfloat16, generator=g), FMT)
    cmp = pack_rows(torch.randn(N_CMP, d, device="cuda", dtype=torch.bfloat16, generator=g), FMT)
    sink = torch.randn(H, device="cuda", dtype=torch.float32, generator=g)
    q = torch.randn(2, m, H, d, device="cuda", dtype=torch.bfloat16, generator=g)
    _, idx, counts = _build(m=m, cmp_valid=[N_CMP_COLS, 40], holes=True)
    got = sparse_attn_paged(q, win, cmp, sink, idx, N_WINDOW, d ** -0.5, cmp_counts=counts)
    ref = _reference(q, unpack_rows(win, FMT, d), unpack_rows(cmp, FMT, d), sink, idx, N_WINDOW, d ** -0.5, counts)
    torch.testing.assert_close(got.float(), ref, **TOL)


@pytest.mark.parametrize("m", MS)
def test_no_counts(pools, m):
    """Without counts the kernel walks the whole buffer; -1 columns contribute nothing."""
    q, idx, _ = _build(m=m, cmp_valid=[N_CMP_COLS, 40])
    _check(pools, q, idx, None)


@pytest.mark.parametrize("m", MS)
def test_with_counts(pools, m):
    q, idx, counts = _build(m=m, cmp_valid=[N_CMP_COLS, 40])
    _check(pools, q, idx, counts)


@pytest.mark.parametrize("m", MS)
def test_small_topk(pools, m):
    q, idx, counts = _build(b=1, m=m, n_cmp_cols=8, cmp_valid=[8])
    _check(pools, q, idx, counts)


@pytest.mark.parametrize("m", MS)
def test_negative_indices_are_masked(pools, m):
    """Slots < 0 scattered INSIDE the live range must contribute zero."""
    q, idx, counts = _build(m=m, cmp_valid=[N_CMP_COLS, 61], holes=True)
    _check(pools, q, idx, counts)


@pytest.mark.parametrize("m", MS)
def test_per_row_counts(pools, m):
    """Rows with different live widths stay isolated (co-tenant decode)."""
    q, idx, counts = _build(b=4, m=m, cmp_valid=[0, 7, 64, N_CMP_COLS], win_valid=[1, 9, 32, 32])
    _check(pools, q, idx, counts)


@pytest.mark.parametrize("m", MS)
def test_empty_row_is_finite(pools, m):
    """A fully padded row (dummy/decode padding) attends to nothing: sink only, no NaN."""
    q, idx, counts = _build(b=1, m=m, cmp_valid=[0], win_valid=0)
    got = _check(pools, q, idx, counts)
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got) == 0


@pytest.mark.parametrize("m", MS)
def test_counts_bound_the_loop(pools, m):
    """counts -- not the -1 padding -- is what stops the walk: leave VALID slots past the live
    width and they must still be ignored. This is the property a captured graph relies on."""
    win, cmp, sink, win_ref, cmp_ref = pools
    scale = D ** -0.5
    q, idx, counts = _build(b=1, m=m, cmp_valid=[N_CMP_COLS])
    counts = torch.full_like(counts, 24)
    got = sparse_attn_paged(q, win, cmp, sink, idx, N_WINDOW, scale, cmp_counts=counts)
    ref = _reference(q, win_ref, cmp_ref, sink, idx, N_WINDOW, scale, counts)
    torch.testing.assert_close(got.float(), ref, **TOL)
    # ... and it really differs from walking the full buffer
    full = sparse_attn_paged(q, win, cmp, sink, idx, N_WINDOW, scale)
    assert not torch.allclose(got.float(), full.float(), **TOL)


@pytest.mark.parametrize("m", MS)
def test_cuda_graph_follows_counts(pools, m):
    """The captured graph must read its loop bound from device memory, not from capture time."""
    win, cmp, sink, win_ref, cmp_ref = pools
    scale = D ** -0.5
    q, idx, counts = _build(b=1, m=m, cmp_valid=[N_CMP_COLS])
    call = lambda: sparse_attn_paged(q, win, cmp, sink, idx, N_WINDOW, scale,
                                     cmp_counts=counts)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    out = torch.empty_like(q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(call())
    for live in (N_CMP_COLS, 12, 0, 64):
        counts.fill_(live)
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference(q, win_ref, cmp_ref, sink, idx, N_WINDOW, scale, counts)
        torch.testing.assert_close(out.float(), ref, **TOL)
