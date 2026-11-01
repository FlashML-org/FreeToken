from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _reference_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    q_to_req: torch.Tensor,
    q_positions: torch.Tensor,
    sm_scale: float,
    sliding_window: int | None,
    sinks: torch.Tensor | None = None,
    block_ends: torch.Tensor | None = None,
) -> torch.Tensor:
    outs = []
    group = q.shape[1] // k_cache.shape[1]
    for tok in range(q.shape[0]):
        req = int(q_to_req[tok].item())
        start = int(indptr[req].item())
        end = int(indptr[req + 1].item())
        slots = indices[start:end].to(torch.long)
        k = k_cache[slots].repeat_interleave(group, dim=1).transpose(0, 1)
        v = v_cache[slots].repeat_interleave(group, dim=1).transpose(0, 1)
        scores = torch.einsum("hd,hkd->hk", q[tok].float(), k.float()) * sm_scale
        key_pos = torch.arange(end - start, device=q.device)
        mask = key_pos <= q_positions[tok]
        if block_ends is not None:
            mask = mask | (key_pos < block_ends[tok])
        if sliding_window is not None:
            mask = mask & (key_pos + sliding_window > q_positions[tok])
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        if sinks is None:
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hk,hkd->hd", probs, v.float())
        else:
            sink_logits = sinks.to(device=q.device, dtype=torch.float32)
            m = torch.maximum(scores.max(dim=-1).values, sink_logits)
            probs = torch.exp(scores - m[:, None])
            denom = probs.sum(dim=-1) + torch.exp(sink_logits - m)
            out = torch.einsum("hk,hkd->hd", probs, v.float()) / denom[:, None]
        outs.append(out.to(q.dtype))
    return torch.stack(outs, dim=0)


def test_triton_backend_passes_attention_sinks_to_paged_kernel(monkeypatch):
    from freetoken.attention import AttentionSpec
    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata

    class FakeKVCache:
        def __init__(self):
            self.device = torch.device("cpu")
            self.k = torch.zeros(4, 1, 4)
            self.v = torch.zeros(4, 1, 4)

        def store_kv(self, k, v, out_loc, layer_id):
            _ = layer_id
            self.k[out_loc.to(torch.long)] = k.view(k.shape[0], 1, -1)
            self.v[out_loc.to(torch.long)] = v.view(v.shape[0], 1, -1)

        def k_cache(self, layer_id):
            return self.k

        def v_cache(self, layer_id):
            return self.v

    kv_cache = FakeKVCache()
    monkeypatch.setattr(
        "freetoken.attention.triton.get_global_ctx",
        lambda: SimpleNamespace(kv_cache=kv_cache),
    )

    captured = {}

    def fake_paged_attention(*args, **kwargs):
        captured["sinks"] = kwargs["sinks"]
        return torch.zeros_like(kwargs["q"])

    monkeypatch.setattr("freetoken.kernel.triton.attention.paged_attention", fake_paged_attention)

    backend = TritonAttentionBackend(SimpleNamespace())
    batch = SimpleNamespace(
        attn_metadata=TritonMetadata(
            cu_seqlens_q_gpu=torch.tensor([0, 1, 2], dtype=torch.int32),
            indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
            indices=torch.tensor([0, 1], dtype=torch.int32),
            q_to_req=torch.tensor([0, 1], dtype=torch.int32),
            q_positions=torch.tensor([0, 0], dtype=torch.int64),
            is_decode=False,
            prefix_lens=torch.tensor([0, 0], dtype=torch.int32),
            max_q_len=1,
        ),
        out_loc=torch.tensor([0, 1], dtype=torch.int32),
    )
    q = torch.randn(2, 2, 4)
    k = torch.randn(2, 4)
    v = torch.randn(2, 4)
    sinks = torch.tensor([0.25, -0.5])

    out = backend.forward(q, k, v, 0, batch, attn_spec=AttentionSpec(sinks=sinks))

    assert out.shape == q.shape
    assert captured["sinks"] is sinks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("sliding_window", [None, 3])
def test_paged_triton_attention_matches_reference(head_dim: int, sliding_window: int | None):
    from freetoken.kernel.triton.attention import paged_attention

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_q_heads = 2
    num_kv_heads = 1
    q = torch.randn(4, num_q_heads, head_dim, device=device)
    k_cache = torch.randn(8, num_kv_heads, head_dim, device=device)
    v_cache = torch.randn(8, num_kv_heads, head_dim, device=device)
    indptr = torch.tensor([0, 5, 8], dtype=torch.int32, device=device)
    indices = torch.arange(8, dtype=torch.int32, device=device)
    q_to_req = torch.tensor([0, 0, 0, 1], dtype=torch.int32, device=device)
    q_positions = torch.tensor([2, 3, 4, 2], dtype=torch.int64, device=device)
    sm_scale = head_dim**-0.5

    actual = paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sliding_window=sliding_window,
        block_n=4,
    )
    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sliding_window,
    )

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
def test_paged_triton_attention_with_sinks_matches_reference():
    from freetoken.kernel.triton.attention import paged_attention

    torch.manual_seed(10)
    device = torch.device("cuda")
    head_dim = 256
    num_q_heads = 4
    num_kv_heads = 1
    q = torch.randn(3, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(6, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(6, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, 4, 6], dtype=torch.int32, device=device)
    indices = torch.arange(6, dtype=torch.int32, device=device)
    q_to_req = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    q_positions = torch.tensor([1, 3, 1], dtype=torch.int64, device=device)
    sinks = torch.tensor([1.2, -0.4, 0.7, -1.0], dtype=torch.float32, device=device)
    sm_scale = head_dim**-0.5

    actual = paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sinks=sinks,
        block_n=4,
    )
    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        None,
        sinks=sinks,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
def test_paged_triton_attention_skips_all_masked_sliding_blocks():
    from freetoken.kernel.triton.attention import paged_attention

    torch.manual_seed(0)
    device = torch.device("cuda")
    head_dim = 256
    q = torch.randn(1, 2, head_dim, device=device)
    k_cache = torch.randn(8, 1, head_dim, device=device)
    v_cache = torch.randn(8, 1, head_dim, device=device)
    indptr = torch.tensor([0, 8], dtype=torch.int32, device=device)
    indices = torch.arange(8, dtype=torch.int32, device=device)
    q_to_req = torch.tensor([0], dtype=torch.int32, device=device)
    q_positions = torch.tensor([7], dtype=torch.int64, device=device)
    sliding_window = 3
    sm_scale = head_dim**-0.5

    actual = paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sliding_window=sliding_window,
        block_n=4,
    )
    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sliding_window,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize(
    ("head_dim", "num_kv_heads", "sliding_window"),
    [
        (256, 8, 3),
        (512, 2, None),
    ],
)
def test_decode_triton_attention_matches_reference(
    head_dim: int,
    num_kv_heads: int,
    sliding_window: int | None,
):
    from freetoken.kernel.triton.attention import decode_paged_attention

    torch.manual_seed(1)
    device = torch.device("cuda")
    batch = 2
    num_q_heads = 16
    max_kv_splits = 8
    seq_lens = [5, 7]
    total_kv = sum(seq_lens)
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32, device=device)
    indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    q_positions = torch.tensor([seq_lens[0] - 1, seq_lens[1] - 1], dtype=torch.int64, device=device)
    q_to_req = torch.arange(batch, dtype=torch.int32, device=device)
    sm_scale = head_dim**-0.5
    attn_logits = torch.empty(
        batch,
        num_q_heads,
        max_kv_splits,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    attn_lse = torch.empty(batch, num_q_heads, max_kv_splits, dtype=torch.float32, device=device)

    actual = decode_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_positions,
        attn_logits,
        attn_lse,
        sm_scale,
        sliding_window=sliding_window,
    )
    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sliding_window,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize(("num_q_heads", "num_kv_heads"), [(24, 4), (20, 4), (28, 4)])
def test_decode_triton_attention_non_pow2_group(num_q_heads: int, num_kv_heads: int):
    """GQA groups that are not a power of two (e.g. Qwen3.6-27B's 24/4 == 6). The grouped
    decode tiles the head axis to a power of two (tl.arange constraint) and masks the extra
    lanes; the result must still match the reference."""
    from freetoken.kernel.triton.attention import decode_paged_attention

    torch.manual_seed(3)
    device = torch.device("cuda")
    batch = 2
    head_dim = 256
    max_kv_splits = 8
    seq_lens = [5, 7]
    total_kv = sum(seq_lens)
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32, device=device)
    indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    q_positions = torch.tensor([seq_lens[0] - 1, seq_lens[1] - 1], dtype=torch.int64, device=device)
    q_to_req = torch.arange(batch, dtype=torch.int32, device=device)
    sm_scale = head_dim**-0.5
    attn_logits = torch.empty(
        batch, num_q_heads, max_kv_splits, head_dim, dtype=torch.float32, device=device
    )
    attn_lse = torch.empty(batch, num_q_heads, max_kv_splits, dtype=torch.float32, device=device)

    actual = decode_paged_attention(
        q, k_cache, v_cache, indptr, indices, q_positions,
        attn_logits, attn_lse, sm_scale,
    )
    expected = _reference_paged_attention(
        q, k_cache, v_cache, indptr, indices, q_to_req, q_positions, sm_scale, None,
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
def test_decode_triton_attention_with_sinks_matches_reference():
    from freetoken.kernel.triton.attention import decode_paged_attention

    torch.manual_seed(11)
    device = torch.device("cuda")
    batch = 2
    num_q_heads = 4
    num_kv_heads = 1
    head_dim = 256
    max_kv_splits = 8
    seq_lens = [5, 7]
    total_kv = sum(seq_lens)
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32, device=device)
    indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    q_positions = torch.tensor(
        [seq_lens[0] - 1, seq_lens[1] - 1],
        dtype=torch.int64,
        device=device,
    )
    q_to_req = torch.arange(batch, dtype=torch.int32, device=device)
    sinks = torch.tensor([1.4, -0.6, 0.25, -1.2], dtype=torch.float32, device=device)
    sm_scale = head_dim**-0.5
    attn_logits = torch.empty(
        batch,
        num_q_heads,
        max_kv_splits,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    attn_lse = torch.empty(batch, num_q_heads, max_kv_splits, dtype=torch.float32, device=device)

    actual = decode_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_positions,
        attn_logits,
        attn_lse,
        sm_scale,
        sinks=sinks,
    )
    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        None,
        sinks=sinks,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize(
    ("seq_lens", "num_kv_heads", "sliding_window", "scratch_splits", "use_sinks"),
    [
        ([1, 129, 1000, 8193], 2, None, 64, False),
        ([1, 129, 1000, 8193], 2, None, 8, False),
        ([300, 2000, 70], 8, 300, 64, True),
    ],
)
def test_decode_triton_attention_splits_long_requests(
    seq_lens: list[int],
    num_kv_heads: int,
    sliding_window: int | None,
    scratch_splits: int,
    use_sinks: bool,
):
    """Requests of many splits next to short ones, a scratch smaller than the split cap, and windows that span several splits."""
    from freetoken.kernel.triton.attention import decode_paged_attention

    torch.manual_seed(5)
    device = torch.device("cuda")
    batch = len(seq_lens)
    num_q_heads = 16
    head_dim = 256
    total_kv = sum(seq_lens)
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, *seq_lens], dtype=torch.int32, device=device).cumsum(0).to(torch.int32)
    indices = torch.randperm(total_kv, device=device).to(torch.int32)
    q_positions = torch.tensor([n - 1 for n in seq_lens], dtype=torch.int64, device=device)
    q_to_req = torch.arange(batch, dtype=torch.int32, device=device)
    sinks = torch.randn(num_q_heads, dtype=torch.float32, device=device) if use_sinks else None
    sm_scale = head_dim**-0.5
    attn_logits = torch.empty(batch, num_q_heads, scratch_splits, head_dim, dtype=torch.float32, device=device)
    attn_lse = torch.empty(batch, num_q_heads, scratch_splits, dtype=torch.float32, device=device)

    actual = decode_paged_attention(
        q, k_cache, v_cache, indptr, indices, q_positions,
        attn_logits, attn_lse, sm_scale,
        sliding_window=sliding_window, sinks=sinks,
    )
    expected = _reference_paged_attention(
        q, k_cache, v_cache, indptr, indices, q_to_req, q_positions, sm_scale, sliding_window, sinks=sinks,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize("use_split_inputs", [False, True])
@pytest.mark.parametrize(
    ("head_dim", "num_kv_heads", "sliding_window", "cached_lens", "extend_lens"),
    [
        (256, 8, None, [0, 0], [5, 3]),
        (256, 8, 4, [3, 2], [4, 3]),
        (512, 2, None, [2, 4], [3, 2]),
    ],
)
def test_extend_triton_attention_matches_reference(
    use_split_inputs: bool,
    head_dim: int,
    num_kv_heads: int,
    sliding_window: int | None,
    cached_lens: list[int],
    extend_lens: list[int],
):
    from freetoken.kernel.triton.attention import extend_paged_attention

    torch.manual_seed(2)
    device = torch.device("cuda")
    num_q_heads = 16
    seq_lens = [c + e for c, e in zip(cached_lens, extend_lens)]
    total_q = sum(extend_lens)
    total_kv = sum(seq_lens)
    q = torch.randn(total_q, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_extend = torch.randn(total_q, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_extend = torch.randn(total_q, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    qo_indptr = torch.tensor(
        [0] + extend_lens,
        dtype=torch.int32,
        device=device,
    ).cumsum_(0)
    kv_indptr = torch.tensor([0] + seq_lens, dtype=torch.int32, device=device).cumsum_(0)
    indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    prefix_lens = torch.tensor(cached_lens, dtype=torch.int32, device=device)
    q_to_req = torch.empty(total_q, dtype=torch.int32, device=device)
    q_positions = torch.empty(total_q, dtype=torch.int64, device=device)
    offset = 0
    kv_offset = 0
    for req_idx, (cached_len, extend_len) in enumerate(zip(cached_lens, extend_lens)):
        q_to_req[offset : offset + extend_len].fill_(req_idx)
        q_positions[offset : offset + extend_len] = torch.arange(
            cached_len,
            cached_len + extend_len,
            dtype=torch.int64,
            device=device,
        )
        k_cache[kv_offset + cached_len : kv_offset + cached_len + extend_len] = k_extend[
            offset : offset + extend_len
        ]
        v_cache[kv_offset + cached_len : kv_offset + cached_len + extend_len] = v_extend[
            offset : offset + extend_len
        ]
        offset += extend_len
        kv_offset += cached_len + extend_len
    sm_scale = head_dim**-0.5

    actual = extend_paged_attention(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        indices,
        prefix_lens,
        max(extend_lens),
        sm_scale,
        sliding_window=sliding_window,
        k_extend=k_extend if use_split_inputs else None,
        v_extend=v_extend if use_split_inputs else None,
    )
    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        kv_indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        sliding_window,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize("use_split_inputs", [False, True])
@pytest.mark.parametrize("sliding_window", [None, 48])
def test_extend_triton_attention_with_bidirectional_blocks_matches_reference(use_split_inputs: bool, sliding_window: int | None):
    """Rows of an image span see the span's later keys across q tiles, text rows stay causal, the window still bounds the past."""
    from freetoken.kernel.triton.attention import extend_paged_attention

    torch.manual_seed(3)
    device = torch.device("cuda")
    head_dim, num_q_heads, num_kv_heads = 256, 16, 8
    cached_lens, extend_lens = [2, 0], [300, 40]
    spans = [(70, 260), (30, 60)]  # req 0: straddles the q tiles; req 1: cut off by the chunk end
    seq_lens = [c + e for c, e in zip(cached_lens, extend_lens)]
    total_q, total_kv = sum(extend_lens), sum(seq_lens)
    q = torch.randn(total_q, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    qo_indptr = torch.tensor([0] + extend_lens, dtype=torch.int32, device=device).cumsum_(0)
    kv_indptr = torch.tensor([0] + seq_lens, dtype=torch.int32, device=device).cumsum_(0)
    indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    prefix_lens = torch.tensor(cached_lens, dtype=torch.int32, device=device)
    q_to_req = torch.empty(total_q, dtype=torch.int32, device=device)
    q_positions = torch.empty(total_q, dtype=torch.int64, device=device)
    block_ends = torch.zeros(total_q, dtype=torch.int32, device=device)
    offset = kv_offset = 0
    for req_idx, (cached_len, extend_len, (lo, hi)) in enumerate(zip(cached_lens, extend_lens, spans)):
        q_to_req[offset : offset + extend_len].fill_(req_idx)
        q_positions[offset : offset + extend_len] = torch.arange(cached_len, cached_len + extend_len, device=device)
        block_ends[offset + max(lo, cached_len) - cached_len : offset + min(hi, cached_len + extend_len) - cached_len] = hi
        offset += extend_len
        kv_offset += cached_len + extend_len
    k_extend = torch.cat([k_cache[kv_indptr[i] + cached_lens[i] : kv_indptr[i + 1]] for i in range(2)])
    v_extend = torch.cat([v_cache[kv_indptr[i] + cached_lens[i] : kv_indptr[i + 1]] for i in range(2)])
    sm_scale = head_dim**-0.5

    actual = extend_paged_attention(
        q, k_cache, v_cache, qo_indptr, kv_indptr, indices, prefix_lens, max(extend_lens), sm_scale,
        sliding_window=sliding_window, block_ends=block_ends,
        k_extend=k_extend if use_split_inputs else None, v_extend=v_extend if use_split_inputs else None,
    )
    expected = _reference_paged_attention(
        q, k_cache, v_cache, kv_indptr, indices, q_to_req, q_positions, sm_scale, sliding_window, block_ends=block_ends
    )
    causal_only = _reference_paged_attention(q, k_cache, v_cache, kv_indptr, indices, q_to_req, q_positions, sm_scale, sliding_window)

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
    assert not torch.allclose(expected.float(), causal_only.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize("use_split_inputs", [False, True])
def test_extend_triton_attention_with_sinks_matches_reference(use_split_inputs: bool):
    from freetoken.kernel.triton.attention import extend_paged_attention

    torch.manual_seed(12)
    device = torch.device("cuda")
    head_dim = 256
    num_q_heads = 4
    num_kv_heads = 1
    cached_lens = [1, 2]
    extend_lens = [2, 1]
    seq_lens = [c + e for c, e in zip(cached_lens, extend_lens)]
    total_q = sum(extend_lens)
    total_kv = sum(seq_lens)
    q = torch.randn(total_q, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_extend = torch.randn(total_q, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_extend = torch.randn(total_q, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    qo_indptr = torch.tensor([0] + extend_lens, dtype=torch.int32, device=device).cumsum_(0)
    kv_indptr = torch.tensor([0] + seq_lens, dtype=torch.int32, device=device).cumsum_(0)
    indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    prefix_lens = torch.tensor(cached_lens, dtype=torch.int32, device=device)
    q_to_req = torch.empty(total_q, dtype=torch.int32, device=device)
    q_positions = torch.empty(total_q, dtype=torch.int64, device=device)
    offset = 0
    kv_offset = 0
    for req_idx, (cached_len, extend_len) in enumerate(zip(cached_lens, extend_lens)):
        q_to_req[offset : offset + extend_len].fill_(req_idx)
        q_positions[offset : offset + extend_len] = torch.arange(
            cached_len,
            cached_len + extend_len,
            dtype=torch.int64,
            device=device,
        )
        k_cache[kv_offset + cached_len : kv_offset + cached_len + extend_len] = k_extend[
            offset : offset + extend_len
        ]
        v_cache[kv_offset + cached_len : kv_offset + cached_len + extend_len] = v_extend[
            offset : offset + extend_len
        ]
        offset += extend_len
        kv_offset += cached_len + extend_len
    sinks = torch.tensor([1.0, -0.25, 0.5, -1.1], dtype=torch.float32, device=device)
    sm_scale = head_dim**-0.5

    actual = extend_paged_attention(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        indices,
        prefix_lens,
        max(extend_lens),
        sm_scale,
        sinks=sinks,
        k_extend=k_extend if use_split_inputs else None,
        v_extend=v_extend if use_split_inputs else None,
    )
    expected = _reference_paged_attention(
        q,
        k_cache,
        v_cache,
        kv_indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        None,
        sinks=sinks,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("head_dim", "smem_optin", "expected"),
    [
        # datacenter opt-in smem (A100 ~164KB / H100 ~227KB): keep the fast tiles where they fit
        (128, 232448, (128, 64)),
        (256, 232448, (128, 64)),
        (512, 232448, (32, 64)),
        (256, 167936, (128, 64)),  # A100: hd256 fast tile fits
        (512, 167936, (16, 16)),  # A100: hd512 fast tile does not fit -> shrink (no smem overflow)
        # consumer opt-in smem (sm_89 ~99KB): shrink once head_dim >= 256
        (256, 101376, (64, 32)),
        (512, 101376, (16, 16)),
        # unknown budget -> conservative small tiles (prior consumer-safe behavior)
        (256, 0, (64, 32)),
        (512, 0, (16, 16)),
    ],
)
def test_select_extend_tile_is_shared_memory_aware(head_dim, smem_optin, expected):
    import triton

    from freetoken.kernel.triton.attention import _select_extend_tile

    block_d = triton.next_power_of_2(head_dim)
    assert _select_extend_tile(head_dim, block_d, smem_optin) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
def test_triton_backend_stores_kv_and_matches_reference(monkeypatch):
    from freetoken.attention import AttentionSpec
    from freetoken.attention.triton import TritonAttentionBackend

    class FakeKVCache:
        def __init__(self, device: torch.device, head_dim: int):
            self.device = device
            self.dtype = torch.float32
            self.k = torch.zeros(4, 1, head_dim, device=device)
            self.v = torch.zeros(4, 1, head_dim, device=device)

        def store_kv(self, k, v, out_loc, layer_id):
            self.k[out_loc.to(torch.long)] = k.view(k.shape[0], 1, -1)
            self.v[out_loc.to(torch.long)] = v.view(v.shape[0], 1, -1)

        def k_cache(self, layer_id):
            return self.k

        def v_cache(self, layer_id):
            return self.v

    device = torch.device("cuda")
    head_dim = 256
    page_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    kv_cache = FakeKVCache(device, head_dim)
    ctx = SimpleNamespace(kv_cache=kv_cache, page_table=page_table)
    monkeypatch.setattr("freetoken.attention.triton.get_global_ctx", lambda: ctx)

    backend = TritonAttentionBackend(SimpleNamespace())
    batch = SimpleNamespace(
        padded_reqs=[
            SimpleNamespace(extend_len=1, device_len=2, cached_len=1, table_idx=0),
            SimpleNamespace(extend_len=1, device_len=2, cached_len=1, table_idx=1),
        ],
        positions=torch.tensor([1, 1], dtype=torch.int64, device=device),
        out_loc=torch.tensor([1, 3], dtype=torch.int32, device=device),
    )
    kv_cache.k[0] = torch.randn(1, head_dim, device=device)
    kv_cache.v[0] = torch.randn(1, head_dim, device=device)
    kv_cache.k[2] = torch.randn(1, head_dim, device=device)
    kv_cache.v[2] = torch.randn(1, head_dim, device=device)
    q = torch.randn(2, 2, head_dim, device=device)
    k = torch.randn(2, head_dim, device=device)
    v = torch.randn(2, head_dim, device=device)

    backend.prepare_metadata(batch)
    actual = backend.forward(
        q,
        k,
        v,
        layer_id=0,
        batch=batch,
        attn_spec=AttentionSpec(sliding_window=None, sm_scale=head_dim**-0.5),
    )
    expected = _reference_paged_attention(
        q,
        kv_cache.k,
        kv_cache.v,
        batch.attn_metadata.indptr,
        batch.attn_metadata.indices,
        batch.attn_metadata.q_to_req,
        batch.attn_metadata.q_positions,
        head_dim**-0.5,
        sliding_window=None,
    )

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
def test_triton_backend_applies_the_batch_block_ends_only_when_the_spec_asks(monkeypatch):
    from freetoken.attention import AttentionSpec
    from freetoken.attention.triton import TritonAttentionBackend

    class FakeKVCache:
        def __init__(self, device: torch.device, head_dim: int):
            self.device = device
            self.dtype = torch.bfloat16
            self.k = torch.randn(3, 1, head_dim, device=device, dtype=torch.bfloat16)
            self.v = torch.randn(3, 1, head_dim, device=device, dtype=torch.bfloat16)

        def store_kv(self, k, v, out_loc, layer_id):
            self.k[out_loc.to(torch.long)] = k.view(k.shape[0], 1, -1)
            self.v[out_loc.to(torch.long)] = v.view(v.shape[0], 1, -1)

        def k_cache(self, layer_id):
            return self.k

        def v_cache(self, layer_id):
            return self.v

    device = torch.device("cuda")
    head_dim = 256
    kv_cache = FakeKVCache(device, head_dim)
    ctx = SimpleNamespace(kv_cache=kv_cache, page_table=torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device))
    monkeypatch.setattr("freetoken.attention.triton.get_global_ctx", lambda: ctx)
    backend = TritonAttentionBackend(SimpleNamespace())
    # one request: cached token 0, chunk tokens 1 and 2 forming one image span [1, 3)
    batch = SimpleNamespace(
        padded_reqs=[SimpleNamespace(extend_len=2, device_len=3, cached_len=1, table_idx=0)],
        positions=torch.tensor([1, 2], dtype=torch.int64, device=device),
        out_loc=torch.tensor([1, 2], dtype=torch.int32, device=device),
        mm_block_ends=torch.tensor([3, 3], dtype=torch.int32, device=device),
    )
    # bf16 takes the extend kernel path; fp32 would fall back to the paged kernel, which has no block mask
    q = torch.randn(2, 2, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(2, head_dim, device=device, dtype=torch.bfloat16)
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    ref = lambda ends: _reference_paged_attention(
        q, kv_cache.k, kv_cache.v, md.indptr, md.indices, md.q_to_req, md.q_positions, head_dim**-0.5, None, block_ends=ends
    )
    causal = backend.forward(q, k, v, layer_id=0, batch=batch, attn_spec=AttentionSpec(sm_scale=head_dim**-0.5))
    torch.testing.assert_close(causal.float(), ref(None).float(), atol=2e-2, rtol=2e-2)
    blocks = backend.forward(q, k, v, layer_id=0, batch=batch, attn_spec=AttentionSpec(sm_scale=head_dim**-0.5, bidirectional_mm_blocks=True))
    torch.testing.assert_close(blocks.float(), ref(batch.mm_block_ends).float(), atol=2e-2, rtol=2e-2)
    assert not torch.allclose(blocks[0].float(), causal[0].float(), atol=2e-2, rtol=2e-2)  # token 1 now sees token 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
def test_triton_backend_replay_metadata_uses_capture_buffers(monkeypatch):
    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata

    class FakeKVCache:
        def __init__(self, device: torch.device):
            self.device = device

    device = torch.device("cuda")
    page_table = torch.arange(16, dtype=torch.int32, device=device).view(2, 8)
    ctx = SimpleNamespace(kv_cache=FakeKVCache(device), page_table=page_table)
    monkeypatch.setattr("freetoken.attention.triton.get_global_ctx", lambda: ctx)

    backend = TritonAttentionBackend(SimpleNamespace())
    backend.init_capture_graph(max_seq_len=8, bs_list=[2])
    assert backend.capture is not None

    capture_batch = SimpleNamespace(size=2)
    backend.prepare_for_capture(capture_batch)
    capture_metadata = capture_batch.attn_metadata
    assert isinstance(capture_metadata, TritonMetadata)
    assert capture_metadata.cu_seqlens_q_gpu.data_ptr() == backend.capture.cu_seqlens_q.data_ptr()
    assert capture_metadata.indptr.data_ptr() == backend.capture.cu_seqlens_k.data_ptr()
    assert capture_metadata.indices.data_ptr() == backend.capture.page_table.view(-1).data_ptr()
    assert capture_metadata.q_positions.data_ptr() == backend.capture.positions.data_ptr()

    runtime_batch = SimpleNamespace(
        padded_size=2,
        padded_reqs=[
            SimpleNamespace(extend_len=1, device_len=3, cached_len=2, table_idx=0),
            SimpleNamespace(extend_len=1, device_len=5, cached_len=4, table_idx=1),
        ],
        positions=torch.tensor([2, 4], dtype=torch.int64, device=device),
    )
    backend.prepare_metadata(runtime_batch)
    runtime_metadata = runtime_batch.attn_metadata
    assert isinstance(runtime_metadata, TritonMetadata)
    expected_indptr = runtime_metadata.indptr.clone()
    expected_indices = runtime_metadata.indices.clone()
    expected_positions = runtime_metadata.q_positions.clone()

    backend.prepare_for_replay(runtime_batch)
    replay_metadata = runtime_batch.attn_metadata
    assert isinstance(replay_metadata, TritonMetadata)
    assert replay_metadata.cu_seqlens_q_gpu.data_ptr() == backend.capture.cu_seqlens_q.data_ptr()
    assert replay_metadata.indptr.data_ptr() == backend.capture.cu_seqlens_k.data_ptr()
    assert replay_metadata.indices.data_ptr() == backend.capture.page_table.view(-1).data_ptr()
    assert replay_metadata.q_positions.data_ptr() == backend.capture.positions.data_ptr()
    assert replay_metadata.attn_logits is not None
    assert replay_metadata.attn_lse is not None
    assert replay_metadata.attn_logits.data_ptr() == backend.capture.attn_logits.data_ptr()
    assert replay_metadata.attn_lse.data_ptr() == backend.capture.attn_lse.data_ptr()

    torch.testing.assert_close(backend.capture.cu_seqlens_k[:3], expected_indptr)
    torch.testing.assert_close(backend.capture.page_table.view(-1)[:8], expected_indices)
    torch.testing.assert_close(backend.capture.positions[:2].to(torch.int64), expected_positions)


def test_triton_metadata_keeps_full_indices_and_optional_swa_indices(monkeypatch):
    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata

    page_table = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        dtype=torch.int32,
    )
    # Global-paged SWA: swa_indices = translate(full page-table indices) via the full->swa map.
    ctx = SimpleNamespace(
        kv_cache=SimpleNamespace(
            device=torch.device("cpu"),
            swa_paged=True,
            translate_loc_from_full_to_swa=lambda idx: idx + 100,
        ),
        page_table=page_table,
    )
    monkeypatch.setattr("freetoken.attention.triton.get_global_ctx", lambda: ctx)

    backend = TritonAttentionBackend(SimpleNamespace())
    batch = SimpleNamespace(
        padded_reqs=[
            SimpleNamespace(extend_len=1, device_len=2, cached_len=1, table_idx=0),
            SimpleNamespace(extend_len=2, device_len=3, cached_len=1, table_idx=1),
        ],
        positions=torch.tensor([1, 1, 2], dtype=torch.int64),
    )

    backend.prepare_metadata(batch)
    metadata = batch.attn_metadata

    assert isinstance(metadata, TritonMetadata)
    assert metadata.indices.tolist() == [10, 11, 20, 21, 22]
    assert metadata.swa_indices is not None
    assert metadata.swa_indices.tolist() == [110, 111, 120, 121, 122]


def _capture_output_gate_call(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = call()
    return graph, output


def test_output_gate_fallback_and_hybrid_apply_gate_once():
    from freetoken.attention.base import AttentionSpec, BaseAttnBackend, HybridBackend

    class DenseBackend(BaseAttnBackend):
        def __init__(self, value):
            self.value = value
            self.calls = []

        def forward(self, q, k, v, layer_id, batch, attn_spec=None):
            self.calls.append(attn_spec)
            return self.value

        def prepare_metadata(self, batch): pass
        def init_capture_graph(self, max_seq_len, bs_list): pass
        def prepare_for_capture(self, batch): pass
        def prepare_for_replay(self, batch): pass

    torch.manual_seed(29)
    out = torch.randn(2, 3, 8)
    gate = torch.randn(2, 24)
    prefill, decode = DenseBackend(out), DenseBackend(out + 1)
    backend = HybridBackend(prefill, decode)
    spec = AttentionSpec(sm_scale=0.25)
    for is_prefill, selected in [(True, prefill), (False, decode)]:
        batch = SimpleNamespace(is_prefill=is_prefill)
        actual = backend.forward_gated(out, out, out, 0, batch, gate, attn_spec=spec)
        expected = (selected.value.reshape(2, 24) * gate.sigmoid()).reshape_as(out)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert selected.calls == [spec]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize(
    ("lens", "heads", "kv_heads", "dim", "window", "use_sinks"),
    [([512], 16, 2, 256, None, False),
     ([512, 495, 478, 461], 16, 2, 256, None, False),
     ([512], 16, 4, 256, None, False),
     ([7, 129], 24, 4, 256, None, False),
     ([1, 1000], 8, 2, 128, 300, True),
     ([0, 65], 4, 1, 256, None, False)],
)
def test_decode_output_gate_matches_unfused_and_reference(lens, heads, kv_heads, dim, window, use_sinks):
    from freetoken.kernel.triton.attention import decode_paged_attention, _decode_gate_sigmoid_table

    torch.manual_seed(37)
    batch, total = len(lens), sum(lens)
    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.2
    k = torch.randn(total, kv_heads, dim, device="cuda", dtype=torch.bfloat16) * 0.2
    v = torch.randn_like(k) * 0.2
    gate = torch.randn_like(q) * 8
    gate.flatten()[:3] = torch.tensor([0, float("inf"), -float("inf")], device="cuda")
    ip = torch.tensor([0, *lens], device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    ix = torch.randperm(total, device="cuda").to(torch.int32)
    pos = torch.tensor([n - 1 for n in lens], device="cuda", dtype=torch.int64)
    partial = torch.empty(batch, heads, 8, dim, device="cuda", dtype=torch.float32)
    lse = torch.empty(batch, heads, 8, device="cuda", dtype=torch.float32)
    sinks = torch.randn(heads, device="cuda") if use_sinks else None
    args = (q, k, v, ip, ix, pos, partial, lse, dim ** -0.5)
    kwargs = dict(sliding_window=window, sinks=sinks)
    _decode_gate_sigmoid_table(q.device.index)
    expected = decode_paged_attention(*args, **kwargs) * gate.sigmoid()
    output = torch.empty_like(q)
    actual = decode_paged_attention(*args, **kwargs, out=output, output_gate=gate)
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    if all(lens):
        oracle = _reference_paged_attention(
            q, k, v, ip, ix, torch.arange(batch, device="cuda"), pos,
            dim ** -0.5, window, sinks=sinks,
        ) * gate.sigmoid()
        torch.testing.assert_close(actual.float(), oracle.float(), rtol=2e-2, atol=2e-2)

    graph, replay_output = _capture_output_gate_call(
        lambda: decode_paged_attention(*args, **kwargs, output_gate=gate)
    )
    # Change the inputs and effective CSR lengths after capture.
    q.mul_(0.5)
    gate.neg_()
    pos.sub_(1)
    expected = decode_paged_attention(*args, **kwargs) * gate.sigmoid()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replay_output, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize(
    ("capacity", "batch_size", "enabled"),
    [(512, 1, True), (4096, 1, True), (512, 1, False), (512, 4, True)],
)
def test_output_gate_backend_graph_and_eager_fallback(monkeypatch, capacity, batch_size, enabled):
    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata
    from freetoken.kernel.triton import attention as kernel

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(41)
    q = torch.randn(batch_size, 16, 256, device=device, dtype=torch.bfloat16) * 0.2
    gate = torch.randn(batch_size, 4096, device=device, dtype=torch.bfloat16)
    k = torch.randn(512, 2, 256, device=device, dtype=torch.bfloat16) * 0.2
    v = torch.randn_like(k)
    stores, calls = [], []
    cache = SimpleNamespace(device=device, k_cache=lambda _: k, v_cache=lambda _: v,
                            store_kv=lambda *args: stores.append(True))
    monkeypatch.setattr("freetoken.attention.triton.get_global_ctx", lambda: SimpleNamespace(kv_cache=cache))
    if enabled:
        monkeypatch.delenv("FREETOKEN_TRITON_FUSE_OUTPUT_GATE", raising=False)
    else:
        monkeypatch.setenv("FREETOKEN_TRITON_FUSE_OUTPUT_GATE", "0")
    config = SimpleNamespace(num_qo_heads=16, num_kv_heads=2, head_dim=256)
    backend = TritonAttentionBackend(config)
    cu_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32)
    ip = cu_q * 512
    ix = torch.cat((torch.arange(512, device=device, dtype=torch.int32),
                    torch.zeros(capacity - 512, device=device, dtype=torch.int32))).repeat(batch_size)
    pos = torch.full((batch_size,), 511, device=device, dtype=torch.int64)
    metadata = TritonMetadata(cu_q, ip, ix, torch.arange(batch_size, device=device), pos, True, torch.zeros_like(pos), 1)
    batch = SimpleNamespace(attn_metadata=metadata, out_loc=pos, mm_block_ends=None)
    original = kernel.decode_paged_attention

    def observed(*args, **kwargs):
        calls.append(kwargs.get("output_gate") is not None)
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "decode_paged_attention", observed)
    expected = backend.forward(q, k[-1:], v[-1:], 0, batch) * gate.sigmoid().view_as(q)
    eager = backend.forward_gated(q, k[-1:], v[-1:], 0, batch, gate)
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    assert len(stores) == 2 and calls == [False, False]
    expected_fusion = enabled and capacity == 512 and batch_size == 1 and bool(torch.version.hip) and getattr(
        torch.cuda.get_device_properties(device), "gcnArchName", ""
    ).split(":")[0] == "gfx1201"
    if expected_fusion:
        original(q, k, v, ip, ix, pos, metadata.attn_logits, metadata.attn_lse,
                 256 ** -0.5, output_gate=gate.view_as(q))
    graph, actual = _capture_output_gate_call(
        lambda: backend.forward_gated(q, k[-1:], v[-1:], 0, batch, gate)
    )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(stores) == 3 and calls == [False, False, expected_fusion]
