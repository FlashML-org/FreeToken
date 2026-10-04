"""--embed-device cpu: the pinned-host UVA embedding lookup matches the on-device table."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _tp(monkeypatch, rank: int, size: int):
    import freetoken.layers.embedding as emb_mod
    from freetoken.distributed import DistributedInfo

    tp_info = DistributedInfo(rank=rank, size=size)
    monkeypatch.setattr(emb_mod, "get_tp_info", lambda: tp_info)


def _embedding(vocab, dim, dtype, *, embed_scale=None):
    from freetoken.layers.embedding import VocabParallelEmbedding

    emb = VocabParallelEmbedding(vocab, dim, embed_scale=embed_scale)
    emb.weight = torch.randn(emb.num_embeddings_tp, dim, dtype=dtype, device="cuda")
    return emb


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dim", [64, 2048, 5120, 1000])
def test_gather_matches_index_select(dtype, dim):
    from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr
    from freetoken.kernel.triton.host_embedding import host_embedding_gather

    rows = 517
    table = torch.randn(rows, dim).to(dtype)
    host = alloc_pinned_tensor(rows, dim, dtype=dtype)
    host.copy_(table)
    ids = torch.tensor([0, 5, rows - 1, 5, 300, -1, rows, rows + 9], device="cuda")
    out = torch.full((ids.numel(), dim), 7.0, dtype=dtype, device="cuda")
    host_embedding_gather(device_ptr(host), rows, dim, ids, out)
    ref = table.cuda()[ids[:5]]
    assert torch.equal(out[:5], ref)
    # out-of-table ids store zeros
    assert torch.count_nonzero(out[5:]) == 0


def test_gather_row_start_shard():
    from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr
    from freetoken.kernel.triton.host_embedding import host_embedding_gather

    rows, dim, start = 100, 256, 1000
    host = alloc_pinned_tensor(rows, dim, dtype=torch.bfloat16)
    host.copy_(torch.randn(rows, dim).bfloat16())
    ids = torch.tensor([999, 1000, 1050, 1099, 1100], device="cuda")
    out = torch.empty(ids.numel(), dim, dtype=torch.bfloat16, device="cuda")
    host_embedding_gather(device_ptr(host), rows, dim, ids, out, start)
    assert torch.count_nonzero(out[0]) == 0 and torch.count_nonzero(out[4]) == 0
    assert torch.equal(out[1:4].cpu(), host[[0, 50, 99]])


@pytest.mark.parametrize("embed_scale", [None, 45.25])
def test_layer_host_matches_device(monkeypatch, embed_scale):
    _tp(monkeypatch, 0, 1)
    emb = _embedding(4099, 640, torch.bfloat16, embed_scale=embed_scale)
    ids = torch.randint(0, 4099, (33,), device="cuda")
    ref = emb.forward(ids)
    emb._embed_scale_t = None
    nbytes = emb.move_to_host()
    assert nbytes == 4099 * 640 * 2
    assert not emb.weight.is_cuda and emb.weight.is_pinned()
    assert torch.equal(emb.forward(ids), ref)


def test_layer_host_tp_shard(monkeypatch):
    # rank 1 of 2 holds rows [2050, 4099); other ids contribute zeros before the all-reduce
    _tp(monkeypatch, 1, 2)
    emb = _embedding(4099, 128, torch.bfloat16)
    full = emb.weight.clone()
    emb.move_to_host()
    ids = torch.tensor([0, 2049, 2050, 3000, 4098], device="cuda")
    out = emb._lookup(ids)
    assert torch.count_nonzero(out[:2]) == 0
    assert torch.equal(out[2:], full[ids[2:] - 2050])


def test_lookup_replays_in_cuda_graph(monkeypatch):
    _tp(monkeypatch, 0, 1)
    emb = _embedding(1000, 512, torch.bfloat16)
    ref_table = emb.weight.clone()
    emb.move_to_host()
    ids = torch.zeros(8, dtype=torch.long, device="cuda")
    emb.forward(ids)  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = emb.forward(ids)
    ids.copy_(torch.arange(100, 108))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref_table[100:108])


def test_move_skips_tied_tables(monkeypatch):
    from freetoken.layers.base import BaseOP, OPList
    from freetoken.layers.embedding import ParallelLMHead, move_input_embeddings_to_host

    _tp(monkeypatch, 0, 1)

    # a bare head carrying only the tie: the quant method is not under test
    monkeypatch.setattr(ParallelLMHead, "__init__", lambda self, tied: setattr(self, "tied_embedding", tied))

    class Inner(BaseOP):
        def __init__(self):
            self.embed_tokens = _embedding(64, 32, torch.bfloat16)
            self.per_layer = OPList([_embedding(16, 32, torch.bfloat16)])

    class Model(BaseOP):
        def __init__(self, tie):
            self.model = Inner()
            self.lm_head = ParallelLMHead(self.model.embed_tokens if tie else None)

    untied = Model(tie=False)
    nbytes, moved = move_input_embeddings_to_host(untied)
    assert moved == ["model.embed_tokens", "model.per_layer.0"]
    assert nbytes == (64 + 16) * 32 * 2
    assert not untied.model.embed_tokens.weight.is_cuda

    tied = Model(tie=True)
    nbytes, moved = move_input_embeddings_to_host(tied)
    assert moved == ["model.per_layer.0"]
    assert tied.model.embed_tokens.weight.is_cuda
