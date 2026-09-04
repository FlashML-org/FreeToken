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


@pytest.mark.parametrize("quant_name", ["q8_0", "q4_0"])
@pytest.mark.parametrize("dispatch", ["decode", "extend", "paged"])
def test_triton_backend_forwards_quantized_payload_scales_and_layout(
    monkeypatch, quant_name: str, dispatch: str
):
    """The backend-to-kernel handoff is part of the quantized cache data path.

    Q8 payloads retain the logical shape, so omitting their scales does not fail a
    shape check: it silently makes attention consume raw int8 codes. Sub-byte payloads
    additionally need their packed physical dimension preserved while K/V extension
    tensors retain the logical dimension.
    """
    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata
    from freetoken.kvcache.quant import resolve_kv_quant
    from freetoken.kernel.triton import attention as attention_kernels

    quant = resolve_kv_quant(quant_name)
    logical_d = 32
    physical_d = quant.physical_head_dim(logical_d)
    num_slots, num_kv_heads = 4, 1

    class FakeQuantizedKVCache:
        def __init__(self):
            self.device = torch.device("cpu")
            self.quant = quant
            # Keep a page axis so the test also verifies that payload and scale slabs
            # are flattened consistently by the backend.
            self.k = torch.empty(2, 2, num_kv_heads, physical_d, dtype=quant.storage_dtype)
            self.v = torch.empty_like(self.k)
            self.ks = torch.empty(2, 2, num_kv_heads, 1, dtype=torch.float16)
            self.vs = torch.empty_like(self.ks)
            self.stored = False

        def store_kv(self, k, v, out_loc, layer_id):
            self.stored = True

        def k_cache(self, layer_id):
            return self.k

        def v_cache(self, layer_id):
            return self.v

        def k_scale(self, layer_id):
            return self.ks

        def v_scale(self, layer_id):
            return self.vs

    kv_cache = FakeQuantizedKVCache()
    monkeypatch.setattr(
        "freetoken.attention.triton.get_global_ctx",
        lambda: SimpleNamespace(kv_cache=kv_cache),
    )

    captured = {}

    def fake_attention(*args, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(kwargs["q"])

    monkeypatch.setattr(attention_kernels, "decode_paged_attention", fake_attention)
    monkeypatch.setattr(attention_kernels, "extend_paged_attention", fake_attention)
    monkeypatch.setattr(attention_kernels, "paged_attention", fake_attention)

    is_decode = dispatch == "decode"
    q_dtype = torch.float32 if dispatch == "paged" else torch.bfloat16
    metadata = TritonMetadata(
        cu_seqlens_q_gpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        indices=torch.tensor([0, 1], dtype=torch.int32),
        q_to_req=torch.tensor([0, 1], dtype=torch.int32),
        q_positions=torch.tensor([0, 0], dtype=torch.int64),
        is_decode=is_decode,
        prefix_lens=torch.tensor([0, 0], dtype=torch.int32),
        max_q_len=1,
    )
    batch = SimpleNamespace(
        attn_metadata=metadata,
        out_loc=torch.tensor([0, 1], dtype=torch.int32),
    )
    q = torch.randn(2, 2, logical_d, dtype=q_dtype)
    k = torch.randn(2, logical_d, dtype=q_dtype)
    v = torch.randn(2, logical_d, dtype=q_dtype)

    backend = TritonAttentionBackend(SimpleNamespace())
    out = backend.forward(q, k, v, 0, batch)

    assert kv_cache.stored
    assert out.shape == q.shape
    assert captured["k_cache"].shape == (num_slots, num_kv_heads, physical_d)
    assert captured["v_cache"].shape == (num_slots, num_kv_heads, physical_d)
    assert captured["k_scale"].shape == (num_slots, num_kv_heads, 1)
    assert captured["v_scale"].shape == (num_slots, num_kv_heads, 1)
    assert captured["layout"] == quant.layout
    if dispatch == "extend":
        assert captured["k_extend"].shape == (2, num_kv_heads, logical_d)
        assert captured["v_extend"].shape == (2, num_kv_heads, logical_d)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="quantized Triton KV needs CUDA")
def test_q8_store_and_decode_attention_match_oracles():
    """GPU regression for the complete generation-critical Q8 path.

    This checks the store payload/scales independently, including a V tensor whose
    strides differ from K, then checks decode attention against those stored values
    dequantized by the PyTorch oracle.
    """
    from freetoken.kernel.triton.attention import decode_paged_attention
    from freetoken.kernel.triton.kv_quant import store_kv_quant
    from freetoken.kvcache.quant import Q8_0

    torch.manual_seed(11)
    device = torch.device("cuda")
    num_tokens, num_slots = 12, 19
    num_kv_heads, num_q_heads, head_dim = 2, 4, 128
    k = torch.randn(
        num_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    # Deliberately use distinct token/head/D strides to guard the V-specific source
    # addressing in the fused K+V store kernel.
    v_backing = torch.randn(
        num_tokens, num_kv_heads, head_dim * 2, device=device, dtype=torch.bfloat16
    )
    v = v_backing[..., ::2]
    indices = torch.randperm(num_slots, device=device)[:num_tokens].to(torch.int32)
    k_cache = torch.empty(
        num_slots, num_kv_heads, head_dim, device=device, dtype=Q8_0.storage_dtype
    )
    v_cache = torch.empty_like(k_cache)
    k_scale = torch.empty(
        num_slots, num_kv_heads, head_dim // 32, device=device, dtype=torch.float16
    )
    v_scale = torch.empty_like(k_scale)

    store_kv_quant(k_cache, k_scale, v_cache, v_scale, indices, k, v, Q8_0)

    expected_k_payload, expected_k_scale = Q8_0.quantize(k)
    expected_v_payload, expected_v_scale = Q8_0.quantize(v)
    dst = indices.to(torch.long)
    torch.testing.assert_close(k_cache[dst], expected_k_payload, atol=0, rtol=0)
    torch.testing.assert_close(v_cache[dst], expected_v_payload, atol=0, rtol=0)
    torch.testing.assert_close(k_scale[dst], expected_k_scale, atol=0, rtol=0)
    torch.testing.assert_close(v_scale[dst], expected_v_scale, atol=0, rtol=0)

    seq_lens = [5, 7]
    batch = len(seq_lens)
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, 5, 12], dtype=torch.int32, device=device)
    q_positions = torch.tensor([4, 6], dtype=torch.int64, device=device)
    q_to_req = torch.arange(batch, dtype=torch.int32, device=device)
    max_kv_splits = 8
    attn_logits = torch.empty(
        batch, num_q_heads, max_kv_splits, head_dim, device=device, dtype=torch.float32
    )
    attn_lse = torch.empty(
        batch, num_q_heads, max_kv_splits, device=device, dtype=torch.float32
    )
    num_kv_splits = torch.full(
        (batch,), max_kv_splits, device=device, dtype=torch.int32
    )
    sm_scale = head_dim**-0.5

    actual = decode_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_positions,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        layout=Q8_0.layout,
    )
    k_dequant = torch.zeros(
        num_slots, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v_dequant = torch.zeros_like(k_dequant)
    k_dequant[dst] = Q8_0.dequantize(k_cache[dst], k_scale[dst]).to(torch.bfloat16)
    v_dequant[dst] = Q8_0.dequantize(v_cache[dst], v_scale[dst]).to(torch.bfloat16)
    expected = _reference_paged_attention(
        q,
        k_dequant,
        v_dequant,
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        None,
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="quantized Triton KV needs CUDA")
@pytest.mark.parametrize("quant_name", ["q8_0", "q4_0", "q6_0"])
def test_quantized_paged_attention_matches_dequantized_oracle(quant_name: str):
    """Every packed layout must interpret _load_kv offsets exactly once."""
    from freetoken.kernel.triton.attention import paged_attention
    from freetoken.kvcache.quant import resolve_kv_quant

    quant = resolve_kv_quant(quant_name)
    torch.manual_seed(12)
    device = torch.device("cuda")
    num_tokens, num_q_heads, num_kv_heads, head_dim = 12, 4, 2, 128
    q = torch.randn(2, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(
        num_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    k_cache, k_scale = quant.quantize(k)
    v_cache, v_scale = quant.quantize(v)
    indptr = torch.tensor([0, 5, 12], dtype=torch.int32, device=device)
    indices = torch.arange(num_tokens, dtype=torch.int32, device=device)
    q_to_req = torch.tensor([0, 1], dtype=torch.int32, device=device)
    q_positions = torch.tensor([4, 6], dtype=torch.int64, device=device)
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
        k_scale=k_scale,
        v_scale=v_scale,
        layout=quant.layout,
    )
    expected = _reference_paged_attention(
        q,
        quant.dequantize(k_cache, k_scale).to(torch.bfloat16),
        quant.dequantize(v_cache, v_scale).to(torch.bfloat16),
        indptr,
        indices,
        q_to_req,
        q_positions,
        sm_scale,
        None,
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


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
    num_kv_splits = torch.full((batch,), max_kv_splits, dtype=torch.int32, device=device)

    actual = decode_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_positions,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
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
    num_kv_splits = torch.full((batch,), max_kv_splits, dtype=torch.int32, device=device)

    actual = decode_paged_attention(
        q, k_cache, v_cache, indptr, indices, q_positions,
        attn_logits, attn_lse, num_kv_splits, max_kv_splits, sm_scale,
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
    num_kv_splits = torch.full((batch,), max_kv_splits, dtype=torch.int32, device=device)

    actual = decode_paged_attention(
        q,
        k_cache,
        v_cache,
        indptr,
        indices,
        q_positions,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
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
    assert replay_metadata.num_kv_splits is not None
    assert replay_metadata.attn_logits.data_ptr() == backend.capture.attn_logits.data_ptr()
    assert replay_metadata.attn_lse.data_ptr() == backend.capture.attn_lse.data_ptr()
    assert replay_metadata.num_kv_splits.data_ptr() == backend.capture.num_kv_splits.data_ptr()

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
