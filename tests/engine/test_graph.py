"""GraphRunner capture: the forwards run on the dummy request's page, and the MoE offload cache stays warm from the eager run into the capture."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch

import freetoken.core as core
from freetoken.core import Context, Req, get_global_ctx
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.engine.graph import GraphRunner


def test_capture_runs_on_the_dummy_page_with_a_warm_offload_cache(monkeypatch):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    ctx = Context(page_size=1)
    # row 0 is a live request holding slot 0, row 1 is the dummy request on the dummy page
    ctx.page_table = torch.tensor([[0], [9]], dtype=torch.int32)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    events = []

    @contextmanager
    def cuda_graph(graph, pool=None, stream=None):
        events.append("capture")
        yield
        events.append("captured")

    def forward():
        batch = get_global_ctx().batch
        events.append(batch.out_loc.tolist())
        return torch.zeros(batch.size, 3)

    def noop(*args, **kwargs):
        return None

    monkeypatch.setattr("torch.cuda.CUDAGraph", lambda: SimpleNamespace(pool=lambda: "pool"))
    monkeypatch.setattr("torch.cuda.graph", cuda_graph)
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(f"torch.cuda.{name}", noop)
    monkeypatch.setattr("freetoken.engine.graph.get_free_memory", lambda device: 0)
    monkeypatch.setattr("freetoken.engine.graph._upload_graph", noop)

    dummy = Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=1, cached_len=0, output_len=1, uid=-1,
                sampling_params=None, cache_handle=None)
    GraphRunner(stream=None, device=torch.device("cpu"), forward=forward,
                attn_backend=SimpleNamespace(init_capture_graph=noop, prepare_for_capture=noop),
                cuda_graph_bs=[2], cuda_graph_max_bs=None, free_memory=0, max_seq_len=1, dummy_req=dummy,
                moe_offload_cache=SimpleNamespace(reset=lambda: events.append("reset")))

    assert events == ["reset", [9, 9], "capture", [9, 9], "captured", "reset", "reset"]
