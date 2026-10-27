"""Attention graphs captured for `rows_per_req` query rows per request (1 = decode, more = a uniform extend such as a speculative verify) replay what the eager path computes."""

from types import SimpleNamespace

import pytest
import torch

import freetoken.core as core
from freetoken.attention import attention_backend_info
from freetoken.core import Batch, Context, Req
from freetoken.distributed import set_tp_info, try_get_tp_info

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HEADS, KV_HEADS, HEAD_DIM = 8, 2, 128
BS, WIDTH = 2, 64
CACHED = (5, 17)


def _backend(name: str):
    config = SimpleNamespace(num_qo_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM)
    if name == "fa":
        pytest.importorskip("sgl_kernel")
        from freetoken.attention.fa import FlashAttentionBackend

        return FlashAttentionBackend(config)
    if name == "fi":
        pytest.importorskip("flashinfer")
        from freetoken.attention.fi import FlashInferBackend

        return FlashInferBackend(config)
    from freetoken.attention.triton import TritonAttentionBackend

    return TritonAttentionBackend(config)


def _context(monkeypatch) -> Context:
    from freetoken.kvcache.mha_pool import MHAKVCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    device = torch.device("cuda")
    ctx = Context(page_size=1)
    # rows 0..BS-1 belong to the requests, row BS is the dummy request's (padded rows write there)
    ctx.page_table = torch.arange((BS + 1) * WIDTH, dtype=torch.int32, device=device).view(BS + 1, WIDTH)
    ctx.kv_cache = MHAKVCache(KV_HEADS, 1, HEAD_DIM, num_pages=(BS + 1) * WIDTH, page_size=1, dtype=torch.bfloat16, device=device)
    n = (BS + 1) * WIDTH
    ctx.kv_cache.store_kv(
        torch.randn(n, KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device),
        torch.randn(n, KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=device),
        torch.arange(n, dtype=torch.int32, device=device),
        0,
    )
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    return ctx


def _batch(reqs, rows_per_req: int, ctx: Context) -> Batch:
    batch = Batch(reqs=reqs, phase="decode" if rows_per_req == 1 else "verify", rows_per_req=rows_per_req)
    batch.padded_reqs = reqs
    batch.positions = torch.cat([torch.arange(r.cached_len, r.device_len, dtype=torch.int32) for r in reqs]).cuda()
    batch.out_loc = torch.cat([ctx.page_table[r.table_idx, r.cached_len : r.device_len] for r in reqs])
    return batch


def _req(table_idx: int, cached: int, rows_per_req: int) -> Req:
    return Req(input_ids=torch.zeros(cached + rows_per_req, dtype=torch.int32), table_idx=table_idx, cached_len=cached,
               output_len=1, uid=table_idx, sampling_params=None, cache_handle=None)


@pytest.mark.parametrize("rows_per_req", [1, 4])
@pytest.mark.parametrize("name", ["fa", "triton", "fi"])
def test_graph_replay_matches_eager_attention(monkeypatch, name, rows_per_req):
    assert attention_backend_info(name).uniform_extend_graph != "none"
    ctx = _context(monkeypatch)
    backend = _backend(name)
    rows = BS * rows_per_req
    q = torch.randn(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(rows, KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(rows, KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    eager = _batch([_req(i, CACHED[i], rows_per_req) for i in range(BS)], rows_per_req, ctx)
    backend.prepare_metadata(eager)
    expected = backend.forward(q, k, v, 0, eager)

    backend.init_capture_graph(max_seq_len=WIDTH, bs_list=[BS], rows_per_req=rows_per_req)
    capture = _batch([_req(BS, 0, rows_per_req)] * BS, rows_per_req, ctx)
    backend.prepare_for_capture(capture)
    static_q, static_k, static_v = q.clone(), k.clone(), v.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        backend.forward(static_q, static_k, static_v, 0, capture)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = backend.forward(static_q, static_k, static_v, 0, capture)
    torch.cuda.current_stream().wait_stream(stream)

    replay = _batch([_req(i, CACHED[i], rows_per_req) for i in range(BS)], rows_per_req, ctx)
    backend.prepare_metadata(replay)
    backend.prepare_for_replay(replay)
    capture.out_loc.copy_(replay.out_loc)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
