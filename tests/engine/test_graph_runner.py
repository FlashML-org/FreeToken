"""GraphRunner over `rows_per_req` rows per request: it picks graphs by phase and row count, and replay returns every output of `forward` for the real rows only."""

from types import SimpleNamespace

import pytest
import torch

import freetoken.core as core
from freetoken.core import Batch, Context, Req
from freetoken.distributed import set_tp_info, try_get_tp_info

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _req(table_idx: int, cached: int, rows_per_req: int) -> Req:
    return Req(input_ids=torch.zeros(cached + rows_per_req, dtype=torch.int32), table_idx=table_idx, cached_len=cached,
               output_len=1, uid=table_idx, sampling_params=None, cache_handle=None)


def test_graph_runner_returns_each_output_for_the_real_rows(monkeypatch):
    from freetoken.engine.graph import GraphRunner

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", Context(page_size=1))
    rows_per_req = 3
    attn = SimpleNamespace(init_capture_graph=lambda **kwargs: None, prepare_for_capture=lambda batch: None,
                           prepare_for_replay=lambda batch: None)

    def forward():
        batch = core.get_global_ctx().batch
        ids = batch.input_ids.float()
        return ids.unsqueeze(1) * 2, ids.view(-1, rows_per_req).sum(dim=1)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner = GraphRunner(stream=stream, device=torch.device("cuda"), forward=forward, attn_backend=attn,
                             cuda_graph_bs=[2, 4], cuda_graph_max_bs=None, free_memory=1024**3, max_seq_len=16,
                             dummy_req=_req(0, 0, rows_per_req), rows_per_req=rows_per_req, phase="verify")
        reqs = [_req(i, 1, rows_per_req) for i in range(3)]
        assert not runner.can_use_cuda_graph(Batch(reqs=reqs, phase="decode"))
        batch = Batch(reqs=reqs, phase="verify", rows_per_req=rows_per_req)
        assert runner.can_use_cuda_graph(batch)
        runner.pad_batch(batch)
        assert batch.padded_size == 4
        batch.input_ids = torch.arange(4 * rows_per_req, dtype=torch.int32, device="cuda")
        batch.out_loc = torch.zeros(4 * rows_per_req, dtype=torch.int32, device="cuda")
        batch.positions = torch.zeros(4 * rows_per_req, dtype=torch.int32, device="cuda")
        per_row, per_req = runner.replay(batch)
    torch.cuda.synchronize()
    ids = torch.arange(3 * rows_per_req, dtype=torch.float32, device="cuda")
    assert torch.equal(per_row, ids.unsqueeze(1) * 2)
    assert torch.equal(per_req, ids.view(3, rows_per_req).sum(dim=1))
    runner.destroy_cuda_graphs()
