from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.format import KV_FORMATS, NONE

KV_QUANT_FORMATS = ["fp8_per_token_head", "int8_per_token_head"]


class _PlainKVStorage:
    kv_format = NONE

    def k_scale(self, layer_id):
        return None

    def v_scale(self, layer_id):
        return None


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs a GPU")
@pytest.mark.parametrize("path", ["paged", "decode", "extend", "split_extend"])
@pytest.mark.parametrize("offset_bits", [31, 32])
def test_triton_attention_high_physical_slots(path: str, offset_bits: int):
    from freetoken.kernel.triton.attention import (
        decode_paged_attention,
        extend_paged_attention,
        paged_attention,
    )

    torch.manual_seed(1347)
    device = torch.device("cuda")
    head_dim, num_kv_heads, num_q_heads = 64, 2, 4
    stride = num_kv_heads * head_dim
    low, count = 128, 66
    high = (1 << 32) // stride + low
    required = (high + count) * stride * 2
    if torch.cuda.mem_get_info()[0] < required + (1 << 30):
        # Release inactive cached blocks before deciding this case cannot fit.
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] < required + (1 << 30):
            pytest.skip("High-offset attention regression needs 9 GiB of free GPU memory")

    # Share K/V storage to keep the real allocation near 8 GiB. Int32 overflow
    # aliases initialized low rows, so a regression fails numerically, not by OOB.
    cache = torch.empty(high + count, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    cache[low : low + count].fill_(-2)
    cache[high : high + count].normal_(mean=3, std=2)
    if offset_bits == 31:
        # Keep the old negative offset inside allocated guard storage too.
        guard = (1 << 31) // stride
        cache = cache[guard:]
        high -= guard
    q = torch.randn(1, num_q_heads, head_dim, device=device, dtype=cache.dtype)
    indices = torch.arange(high, high + count, device=device, dtype=torch.int32)
    indptr = torch.tensor([0, count], device=device, dtype=torch.int32)
    q_to_req = torch.zeros(1, device=device, dtype=torch.int32)
    q_positions = torch.tensor([count - 1], device=device, dtype=torch.int64)
    sm_scale = head_dim**-0.5
    expected = _reference_paged_attention(
        q, cache, cache, indptr, indices, q_to_req, q_positions, sm_scale, None,
    )

    if path == "paged":
        actual = paged_attention(q, cache, cache, indptr, indices, q_to_req, q_positions, sm_scale)
    elif path == "decode":
        mid = torch.empty(1, num_q_heads, 64, head_dim, device=device, dtype=torch.float32)
        lse = torch.empty(1, num_q_heads, 64, device=device, dtype=torch.float32)
        actual = decode_paged_attention(q, cache, cache, indptr, indices, q_positions, mid, lse, sm_scale)
    else:
        qo_indptr = torch.tensor([0, 1], device=device, dtype=torch.int32)
        prefix_lens = torch.tensor([count - 1], device=device, dtype=torch.int32)
        # 65 cached tokens cover both the full-tile and masked split-prefix loads.
        extend = cache[high + count - 1 : high + count] if path == "split_extend" else None
        actual = extend_paged_attention(
            q, cache, cache, qo_indptr, indptr, indices, prefix_lens, 1, sm_scale,
            k_extend=extend, v_extend=extend,
        )

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


def _dequantize(fmt_name: str, codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if fmt_name == "fp8_per_token_head":
        values = codes.cpu().view(torch.float8_e4m3fn).float().to(codes.device)
    else:
        values = codes.float()
    return values * scale


def _quantized_pool(fmt_name: str, k, v, slots, capacity: int, num_kv_heads: int, head_dim: int):
    from freetoken.kernel.triton.kv_rows import store_kv_rows

    fmt = KV_FORMATS[fmt_name]
    device = k.device
    k_cache = torch.zeros(capacity, num_kv_heads, head_dim, device=device, dtype=fmt.elem_store_dtype)
    v_cache = torch.zeros_like(k_cache)
    k_scale = torch.zeros(capacity, num_kv_heads, 1, device=device, dtype=fmt.scale_store_dtype)
    v_scale = torch.zeros_like(k_scale)
    store_kv_rows(fmt, k, v, slots, k_cache, v_cache, k_scale, v_scale)
    return k_cache, v_cache, k_scale, v_scale


def _attend(path: str, q, k_cache, v_cache, indices, kv_args: dict, k_extend=None, v_extend=None):
    from freetoken.kernel.triton.attention import (
        decode_paged_attention,
        extend_paged_attention,
        paged_attention,
    )

    device = q.device
    q_len, num_q_heads, head_dim = q.shape
    count = indices.numel()
    sm_scale = head_dim**-0.5
    indptr = torch.tensor([0, count], device=device, dtype=torch.int32)
    q_to_req = torch.zeros(q_len, device=device, dtype=torch.int32)
    q_positions = torch.arange(count - q_len, count, device=device, dtype=torch.int64)
    if path == "paged":
        return paged_attention(q, k_cache, v_cache, indptr, indices, q_to_req, q_positions, sm_scale, **kv_args)
    if path == "decode":
        mid = torch.empty(1, num_q_heads, 64, head_dim, device=device, dtype=torch.float32)
        lse = torch.empty(1, num_q_heads, 64, device=device, dtype=torch.float32)
        return decode_paged_attention(q, k_cache, v_cache, indptr, indices, q_positions, mid, lse, sm_scale, **kv_args)
    qo_indptr = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    prefix_lens = torch.tensor([count - q_len], device=device, dtype=torch.int32)
    return extend_paged_attention(
        q, k_cache, v_cache, qo_indptr, indptr, indices, prefix_lens, q_len, sm_scale,
        k_extend=k_extend, v_extend=v_extend, **kv_args,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The KV store kernel needs CUDA")
@pytest.mark.parametrize("fmt_name", KV_QUANT_FORMATS)
def test_store_kv_rows_round_trips_within_the_format_error_bound(fmt_name: str):
    torch.manual_seed(5)
    device = torch.device("cuda")
    tokens, heads, head_dim = 9, 4, 96
    row = heads * head_dim
    # K and V rows may come with different source strides
    k = torch.randn(tokens, row + 7, device=device, dtype=torch.bfloat16)[:, :row]
    v = torch.randn(tokens, row, device=device, dtype=torch.bfloat16) * 300
    k[3, head_dim : 2 * head_dim] = 0
    slots = torch.randperm(2 * tokens, device=device)[:tokens].to(torch.int32)

    k_cache, v_cache, k_scale, v_scale = _quantized_pool(fmt_name, k, v, slots, 2 * tokens, heads, head_dim)

    rows = slots.long()
    for raw, codes, scale in ((k, k_cache, k_scale), (v, v_cache, v_scale)):
        expected = raw.float().view(tokens, heads, head_dim)
        stored = _dequantize(fmt_name, codes[rows], scale[rows])
        amax = expected.abs().amax(dim=-1)
        # the largest element of a row is exactly representable after scaling
        torch.testing.assert_close(stored.abs().amax(dim=-1), amax, rtol=1e-5, atol=0.0)
        if fmt_name == "fp8_per_token_head":
            # 3 mantissa bits: half a step is 1/16 of the value, or half the 2^-9 subnormal step near zero
            bound = torch.maximum(expected.abs() / 16, scale[rows] * 2.0**-10)
        else:
            bound = scale[rows] * 0.5
        assert ((stored - expected).abs() <= bound * (1 + 1e-5)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize("fmt_name", KV_QUANT_FORMATS)
@pytest.mark.parametrize("path", ["paged", "decode", "extend", "split_extend"])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
def test_quantized_kv_attention_matches_dequantized_reference(fmt_name: str, path: str, head_dim: int):
    torch.manual_seed(11)
    device = torch.device("cuda")
    num_kv_heads, num_q_heads = 2, 4
    count, capacity = 66, 200
    q_len = 2 if "extend" in path else 1
    row = num_kv_heads * head_dim
    # rows and heads of different magnitude, so a scale read from the wrong slot or head changes the result
    head_gain = torch.ones(row, device=device)
    head_gain[head_dim:] = 3.0
    k = (torch.randn(count, row, device=device) * torch.logspace(-1, 1, count, device=device)[:, None] * head_gain).to(torch.bfloat16)
    v = (torch.randn(count, row, device=device) * torch.logspace(-1, 1, count, device=device)[:, None] * head_gain).to(torch.bfloat16)
    slots = torch.randperm(capacity, device=device)[:count].to(torch.int32)
    k_cache, v_cache, k_scale, v_scale = _quantized_pool(fmt_name, k, v, slots, capacity, num_kv_heads, head_dim)
    q = torch.randn(q_len, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)

    k_ref = _dequantize(fmt_name, k_cache, k_scale)
    v_ref = _dequantize(fmt_name, v_cache, v_scale)
    k_extend = v_extend = None
    if path == "split_extend":
        k_extend = k[-q_len:].view(q_len, num_kv_heads, head_dim)
        v_extend = v[-q_len:].view(q_len, num_kv_heads, head_dim)
        current = slots[-q_len:].long()
        k_ref[current] = k_extend.float()
        v_ref[current] = v_extend.float()

    actual = _attend(
        path, q, k_cache, v_cache, slots,
        {"kv_format": KV_FORMATS[fmt_name], "k_scale": k_scale, "v_scale": v_scale},
        k_extend, v_extend,
    )
    expected = _reference_paged_attention(
        q.float(), k_ref, v_ref,
        torch.tensor([0, count], device=device, dtype=torch.int32),
        slots,
        torch.zeros(q_len, device=device, dtype=torch.int32),
        torch.arange(count - q_len, count, device=device, dtype=torch.int64),
        head_dim**-0.5,
        None,
    )

    # the bf16 output rounds large elements by more than a fixed tolerance, and elements near zero are cancellations
    torch.testing.assert_close(actual.float(), expected, atol=2e-2 * expected.abs().max().item(), rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton attention needs CUDA")
@pytest.mark.parametrize("path", ["paged", "decode", "extend", "split_extend"])
def test_quantized_kv_attention_high_physical_slots(path: str):
    from freetoken.kernel.triton.kv_rows import store_kv_rows

    torch.manual_seed(1347)
    device = torch.device("cuda")
    fmt = KV_FORMATS["fp8_per_token_head"]
    head_dim, num_kv_heads, num_q_heads = 64, 2, 4
    stride = num_kv_heads * head_dim
    count = 66
    high = (1 << 32) // stride + 128
    required = (high + count) * (stride + num_kv_heads * fmt.scale_store_dtype.itemsize)
    if torch.cuda.mem_get_info()[0] < required + (1 << 30):
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] < required + (1 << 30):
            pytest.skip("High-offset attention regression needs 6 GiB of free GPU memory")

    # K and V share storage to keep the allocation small; an int32 offset overflow reads zeroed low rows and fails numerically
    cache = torch.zeros(high + count, num_kv_heads, head_dim, device=device, dtype=fmt.elem_store_dtype)
    scale = torch.zeros(high + count, num_kv_heads, 1, device=device, dtype=fmt.scale_store_dtype)
    rows = torch.randn(count, stride, device=device, dtype=torch.bfloat16)
    slots = torch.arange(high, high + count, device=device, dtype=torch.int32)
    store_kv_rows(fmt, rows, rows, slots, cache, cache, scale, scale)
    q = torch.randn(1, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    ref = _dequantize(fmt.name, cache[high:], scale[high:])
    extend = None
    if path == "split_extend":
        extend = rows[-1:].view(1, num_kv_heads, head_dim)
        ref[-1:] = extend.float()

    actual = _attend(path, q, cache, cache, slots, {"kv_format": fmt, "k_scale": scale, "v_scale": scale}, extend, extend)
    expected = _reference_paged_attention(
        q, ref, ref,
        torch.tensor([0, count], device=device, dtype=torch.int32),
        torch.arange(count, device=device, dtype=torch.int32),
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.tensor([count - 1], device=device, dtype=torch.int64),
        head_dim**-0.5,
        None,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


def test_triton_backend_passes_attention_sinks_to_paged_kernel(monkeypatch):
    from freetoken.attention import AttentionSpec
    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata

    class FakeKVCache(_PlainKVStorage):
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
@pytest.mark.parametrize("fmt_name", [None, *KV_QUANT_FORMATS])
def test_extend_triton_attention_with_bidirectional_blocks_matches_reference(
    use_split_inputs: bool, sliding_window: int | None, fmt_name: str | None
):
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
    current = torch.cat([torch.arange(kv_indptr[i] + cached_lens[i], kv_indptr[i + 1], device=device) for i in range(2)])
    k_extend, v_extend = k_cache[current], v_cache[current]
    sm_scale = head_dim**-0.5
    pool_k, pool_v, k_ref, v_ref, kv_args = k_cache, v_cache, k_cache, v_cache, {}
    if fmt_name is not None:
        pool_k, pool_v, k_scale, v_scale = _quantized_pool(
            fmt_name, k_cache.view(total_kv, -1), v_cache.view(total_kv, -1), indices, total_kv, num_kv_heads, head_dim
        )
        kv_args = {"kv_format": KV_FORMATS[fmt_name], "k_scale": k_scale, "v_scale": v_scale}
        k_ref, v_ref = _dequantize(fmt_name, pool_k, k_scale), _dequantize(fmt_name, pool_v, v_scale)
        if use_split_inputs:
            k_ref[current], v_ref[current] = k_extend.float(), v_extend.float()

    actual = extend_paged_attention(
        q, pool_k, pool_v, qo_indptr, kv_indptr, indices, prefix_lens, max(extend_lens), sm_scale,
        sliding_window=sliding_window, block_ends=block_ends,
        k_extend=k_extend if use_split_inputs else None, v_extend=v_extend if use_split_inputs else None, **kv_args,
    )
    expected = _reference_paged_attention(
        q, k_ref, v_ref, kv_indptr, indices, q_to_req, q_positions, sm_scale, sliding_window, block_ends=block_ends
    )
    causal_only = _reference_paged_attention(q, k_ref, v_ref, kv_indptr, indices, q_to_req, q_positions, sm_scale, sliding_window)

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

    class FakeKVCache(_PlainKVStorage):
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

    class FakeKVCache(_PlainKVStorage):
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
