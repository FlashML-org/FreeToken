from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.attention import (
    decode_paged_attention,
    extend_paged_attention,
    paged_attention,
)
from freetoken.kernel.triton.kv_quant import quantize_kv_to_cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _data(tokens, heads=4, dim=128):
    torch.manual_seed(73)
    k = torch.randn(tokens, heads * dim * 3, device="cuda", dtype=torch.bfloat16)[:, heads * dim:2 * heads * dim]
    v = torch.randn(tokens, heads * dim, device="cuda", dtype=torch.bfloat16) * 0.13
    slots = torch.randperm(tokens + 64, device="cuda")[:tokens].to(torch.int32)
    kc = torch.zeros(tokens + 64, heads, dim, device="cuda", dtype=torch.uint8)
    vc = torch.zeros_like(kc)
    ks = torch.zeros(tokens + 64, heads, device="cuda", dtype=torch.float32)
    vs = torch.zeros_like(ks)
    quantize_kv_to_cache(k, v, slots, kc, vc, ks, vs)
    return k.view(tokens, heads, dim), v.view(tokens, heads, dim), slots, kc, vc, ks, vs


def _read(codes, scales, slots):
    return codes.view(torch.float8_e4m3fn).float()[slots.long()] * scales[slots.long(), :, None]


def _oracle(q, k, v, prefix):
    group = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1).transpose(0, 1)
    v = v.repeat_interleave(group, dim=1).transpose(0, 1)
    score = torch.einsum("thd,hnd->htn", q.float(), k.float()) * q.shape[-1]**-0.5
    mask = torch.arange(k.shape[1], device=q.device)[None, :] <= prefix + torch.arange(q.shape[0], device=q.device)[:, None]
    return torch.einsum("htn,hnd->thd", score.masked_fill(~mask[None], -torch.inf).softmax(-1), v.float())


@pytest.mark.parametrize("tokens", [17, 512, 8193, 32768])
@pytest.mark.parametrize("splits", [1, 8, 64])
def test_fp8_decode_keeps_adaptive_split_range(tokens, splits):
    _, _, slots, kc, vc, ks, vs = _data(tokens)
    q = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16)
    out = decode_paged_attention(
        q, kc, vc, torch.tensor([0, tokens], device="cuda", dtype=torch.int32), slots,
        torch.tensor([tokens-1], device="cuda", dtype=torch.int32),
        torch.empty(1, 32, 64, 128, device="cuda"), torch.empty(1, 32, 64, device="cuda"),
        128**-0.5, max_kv_splits=splits, k_scale=ks, v_scale=vs,
    )
    reference = _oracle(q, _read(kc, ks, slots), _read(vc, vs, slots), tokens-1)
    torch.testing.assert_close(out.float(), reference, atol=0.002, rtol=0.02)


@pytest.mark.parametrize("path", ["paged", "extend", "split-extend"])
def test_fp8_prefill_reads_scales_and_keeps_live_chunk(path):
    prefix, query = 65, 129
    k, v, slots, kc, vc, ks, vs = _data(prefix + query, heads=2)
    q = torch.randn(query, 8, 128, device="cuda", dtype=torch.bfloat16)
    qptr = torch.tensor([0, query], device="cuda", dtype=torch.int32)
    kptr = torch.tensor([0, prefix+query], device="cuda", dtype=torch.int32)
    prefix_t = torch.tensor([prefix], device="cuda", dtype=torch.int32)
    k_ref, v_ref = _read(kc, ks, slots), _read(vc, vs, slots)
    if path == "paged":
        out = paged_attention(q, kc, vc, kptr, slots,
                              torch.zeros(query, device="cuda", dtype=torch.int32),
                              prefix+torch.arange(query, device="cuda", dtype=torch.int32),
                              128**-0.5, k_scale=ks, v_scale=vs)
    else:
        live = path == "split-extend"
        out = extend_paged_attention(q, kc, vc, qptr, kptr, slots, prefix_t, query, 128**-0.5,
                                     k_extend=k[prefix:] if live else None,
                                     v_extend=v[prefix:] if live else None,
                                     k_scale=ks, v_scale=vs)
        if live:
            k_ref[prefix:], v_ref[prefix:] = k[prefix:].float(), v[prefix:].float()
    torch.testing.assert_close(out.float(), _oracle(q, k_ref, v_ref, prefix), atol=0.003, rtol=0.03)


def test_fp8_multi_request_prefill_reads_independent_row_scales():
    prefixes = (65, 37)
    queries = (4, 7)
    lengths = tuple(p + n for p, n in zip(prefixes, queries))
    k, v, slots, kc, vc, ks, vs = _data(sum(lengths), heads=2)
    q = torch.randn(sum(queries), 8, 128, device="cuda", dtype=torch.bfloat16)
    qptr = torch.tensor([0, queries[0], sum(queries)], device="cuda", dtype=torch.int32)
    kptr = torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32)
    prefix_t = torch.tensor(prefixes, device="cuda", dtype=torch.int32)
    live_k = torch.cat((k[prefixes[0]:lengths[0]], k[lengths[0]+prefixes[1]:]))
    live_v = torch.cat((v[prefixes[0]:lengths[0]], v[lengths[0]+prefixes[1]:]))
    out = extend_paged_attention(q, kc, vc, qptr, kptr, slots, prefix_t, max(queries),
                                 128**-0.5, k_extend=live_k, v_extend=live_v,
                                 k_scale=ks, v_scale=vs)
    decoded_k = _read(kc, ks, slots)
    decoded_v = _read(vc, vs, slots)
    refs = []
    q_start = k_start = live_start = 0
    for prefix, query, length in zip(prefixes, queries, lengths):
        ref_k = decoded_k[k_start:k_start+length].clone()
        ref_v = decoded_v[k_start:k_start+length].clone()
        ref_k[prefix:] = live_k[live_start:live_start+query]
        ref_v[prefix:] = live_v[live_start:live_start+query]
        refs.append(_oracle(q[q_start:q_start+query], ref_k, ref_v, prefix))
        q_start += query
        k_start += length
        live_start += query
    torch.testing.assert_close(out.float(), torch.cat(refs), atol=0.003, rtol=0.03)


def test_fp8_multi_request_decode_keeps_rows_separate():
    lengths = (65, 37)
    _, _, slots, kc, vc, ks, vs = _data(sum(lengths), heads=2)
    q = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    kptr = torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32)
    positions = torch.tensor([n - 1 for n in lengths], device="cuda", dtype=torch.int32)
    mid = torch.empty(2, 8, 64, 128, device="cuda")
    lse = torch.empty(2, 8, 64, device="cuda")
    out = decode_paged_attention(q, kc, vc, kptr, slots, positions, mid, lse,
                                 128**-0.5, k_scale=ks, v_scale=vs)
    decoded_k = _read(kc, ks, slots)
    decoded_v = _read(vc, vs, slots)
    refs = [_oracle(q[:1], decoded_k[:lengths[0]], decoded_v[:lengths[0]], lengths[0] - 1),
            _oracle(q[1:], decoded_k[lengths[0]:], decoded_v[lengths[0]:], lengths[1] - 1)]
    torch.testing.assert_close(out.float(), torch.cat(refs), atol=0.003, rtol=0.03)


def test_fp8_graph_replays_changed_inputs_and_slot_mapping():
    k, v, slots, kc, vc, ks, vs = _data(8, heads=2)
    k, v = k.reshape(8, -1).contiguous(), v.reshape(8, -1).contiguous()
    q = torch.randn(1, 8, 128, device="cuda", dtype=torch.bfloat16)
    kptr = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    pos = torch.tensor([7], device="cuda", dtype=torch.int32)
    mid = torch.empty(1, 8, 64, 128, device="cuda")
    lse = torch.empty(1, 8, 64, device="cuda")
    out = torch.empty_like(q)

    def forward():
        quantize_kv_to_cache(k, v, slots, kc, vc, ks, vs)
        return decode_paged_attention(q, kc, vc, kptr, slots, pos, mid, lse, 128**-0.5,
                                      out=out, k_scale=ks, v_scale=vs)

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    for step in range(5):
        k.normal_(std=step+1)
        v.normal_(std=0.1*(step+1))
        q.normal_()
        slots.copy_(torch.randperm(72, device="cuda")[:8].int())
        graph.replay()
        reference = _oracle(q, _read(kc, ks, slots), _read(vc, vs, slots), 7)
        torch.testing.assert_close(out.float(), reference, atol=0.005, rtol=0.03)


def test_fp8_first_prefill_preserves_bf16_attention_and_populates_codes():
    from types import SimpleNamespace

    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.mha_pool import MHAKVCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    torch.manual_seed(104)
    tokens, heads, dim = 129, 2, 128
    q = torch.randn(tokens, 8, dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(tokens, heads * dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(tokens, heads * dim, device='cuda', dtype=torch.bfloat16)
    slots = torch.randperm(tokens + 16, device='cuda')[:tokens].int()
    indptr = torch.tensor([0, tokens], device='cuda', dtype=torch.int32)
    metadata = TritonMetadata(
        cu_seqlens_q_gpu=indptr, indptr=indptr, indices=slots,
        q_to_req=torch.zeros(tokens, device='cuda', dtype=torch.int32),
        q_positions=torch.arange(tokens, device='cuda', dtype=torch.int32),
        is_decode=False, prefix_lens=torch.zeros(1, device='cuda', dtype=torch.int32),
        max_q_len=tokens, has_prefix=False,
    )
    batch = SimpleNamespace(attn_metadata=metadata, out_loc=slots, mm_block_ends=None)
    outputs = []
    for mode in ('none', 'fp8'):
        pool = MHAKVCache(heads, 1, dim, tokens+16, 1, torch.bfloat16, torch.device('cuda'), kv_quant=mode)
        backend = object.__new__(TritonAttentionBackend)
        backend.kvcache = pool
        backend.prefill_tile_min_q = 128
        outputs.append(backend.forward(q, k, v, 0, batch))
        if mode == 'fp8':
            assert pool.k_cache(0).dtype == torch.uint8
            assert (pool.k_scale(0)[slots.long()] > 0).all()
            assert (pool.v_scale(0)[slots.long()] > 0).all()
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize("page_size", [1, 8])
@pytest.mark.parametrize("query", [4, 129])
def test_fp8_workspace_reuse_rebuild_and_graph(page_size, query):
    from types import SimpleNamespace
    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.mha_pool import MHAKVCache

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    torch.manual_seed(2203)
    prefix, heads, dim = 65, 2, 128
    pool = MHAKVCache(heads, 2, dim, 256 // page_size, page_size,
                      torch.bfloat16, torch.device("cuda"), kv_quant="fp8")
    backend = object.__new__(TritonAttentionBackend)
    backend.kvcache, backend.device = pool, pool.device
    backend.prefill_tile_min_q = 128
    backend.prefill_workspace = None
    backend.init_workspace()
    original_pointer = backend.prefill_workspace.data_ptr()

    for pages in (256 // page_size, 512 // page_size, 224 // page_size):
        if pool._kv_buffer.shape[2] != pages:
            torch.cuda.synchronize()
            backend.reset_workspace()
            assert backend.prefill_workspace is None
            pool.rebuild(pages)
            backend.init_workspace()
        capacity = pages * page_size
        assert backend.prefill_workspace.shape == (2, capacity, heads, dim)
        pointer = backend.prefill_workspace.data_ptr()
        q = torch.randn(query, 8, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(prefix + query, heads * dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k) * .1
        slots = torch.randperm(capacity, device="cuda")[:prefix + query].int()
        metadata = TritonMetadata(
            cu_seqlens_q_gpu=torch.tensor([0, query], device="cuda", dtype=torch.int32),
            indptr=torch.tensor([0, prefix + query], device="cuda", dtype=torch.int32),
            indices=slots, q_to_req=torch.zeros(query, device="cuda", dtype=torch.int32),
            q_positions=torch.arange(prefix, prefix + query, device="cuda", dtype=torch.int32),
            is_decode=False, prefix_lens=torch.tensor([prefix], device="cuda", dtype=torch.int32),
            max_q_len=query, has_prefix=True,
        )
        batch = SimpleNamespace(attn_metadata=metadata, out_loc=slots[prefix:], mm_block_ends=None)

        def forward():
            pool.store_kv(k[:prefix], v[:prefix], slots[:prefix], 0)
            return backend.forward(q, k[prefix:], v[prefix:], 0, batch)

        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = forward()
        for step in range(3):
            k.normal_(std=step + 1)
            v.normal_(std=.1 * (step + 1))
            q.normal_()
            slots.copy_(torch.randperm(capacity, device="cuda")[:prefix + query].int())
            graph.replay()
            kc = pool.k_cache(0).view(capacity, heads, dim)
            vc = pool.v_cache(0).view(capacity, heads, dim)
            kr = torch.cat([_read(kc, pool.k_scale(0), slots[:prefix]).bfloat16(), k[prefix:].view(query, heads, dim)])
            vr = torch.cat([_read(vc, pool.v_scale(0), slots[:prefix]).bfloat16(), v[prefix:].view(query, heads, dim)])
            torch.testing.assert_close(out.float(), _oracle(q, kr, vr, prefix), atol=.003, rtol=.03)
            # A different layer overwrites the same workspace; the next replay must refresh it.
            pool.store_kv(k[:prefix] * 2, v[:prefix] * 3, slots[:prefix], 1)
            backend.forward(q, k[prefix:], v[prefix:], 1, batch)
            assert backend.prefill_workspace.data_ptr() == pointer
        torch.cuda.synchronize()
        del graph, out
        if capacity == 256:
            assert pointer == original_pointer
