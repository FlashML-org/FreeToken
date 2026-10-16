"""CPU contracts for opt-in Qwen3.8 layer-major long prefill."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Context, set_global_ctx
from freetoken.engine.engine import Engine


class _Pool:
    conv_states = torch.zeros(1, 8, 1, 3)


class _AttnBackend:
    def __init__(self):
        self.prepared = []

    def prepare_metadata(self, batch):
        self.prepared.append(batch)
        batch.attn_metadata = object()


@pytest.fixture(autouse=True)
def _restore_global_context():
    import freetoken.core as core

    previous = core._GLOBAL_CTX
    try:
        yield
    finally:
        core._GLOBAL_CTX = previous


def _ctx(pool=None):
    import freetoken.core as core

    core._GLOBAL_CTX = None
    set_global_ctx(Context(page_size=64, linear_state_pool=pool or _Pool()))


def _batch(*, cached_len=0, extend_len=8193, ping_pong=(5, 6), next_track=0):
    req = SimpleNamespace(
        table_idx=1, uid=1, linear_slot_idx=2, cached_len=cached_len,
        device_len=cached_len + extend_len, extend_len=extend_len,
        max_device_len=cached_len + extend_len, mamba_ping_pong=ping_pong,
        mamba_next_track_idx=next_track, mamba_last_track_seqlen=None,
        mamba_restore_src=None, swa_evicted_seqlen=0, sampling_params=None,
        mrope_delta=0, mm_items=None, mrope_positions_full=None,
    )
    return SimpleNamespace(
        reqs=[req], padded_reqs=[req], input_ids=torch.arange(extend_len, dtype=torch.int32),
        positions=torch.arange(cached_len, cached_len + extend_len),
        out_loc=torch.arange(extend_len), mrope_positions=None,
        linear_table_idx=torch.tensor([2], dtype=torch.int32), fla_metadata=None,
        is_decode=False, is_prefill=True, size=1,
    )


def _engine(chunk):
    engine = object.__new__(Engine)
    engine.config = SimpleNamespace(max_extend_tokens=chunk)
    engine.device = torch.device("cpu")
    engine.attn_backend = _AttnBackend()
    return engine


def test_layer_major_flag_defaults_off():
    from freetoken.server.args import ServerArgs

    flag = next(field for field in fields(ServerArgs) if field.name == "prefill_layer_major")
    assert flag.default is False


@pytest.mark.parametrize(
    ("enabled", "model_type", "tp_size", "strategy", "expert_quant", "ple_backend", "expected"),
    [
        (True, "qwen4_exp", 1, "offload", "nvfp4", "pinned", True),
        (False, "qwen4_exp", 1, "offload", "nvfp4", "pinned", False),
        (True, "qwen3_5", 1, "offload", "nvfp4", "pinned", False),
        (True, "qwen4_exp", 2, "offload", "nvfp4", "pinned", False),
        (True, "qwen4_exp", 1, "none", "nvfp4", "pinned", False),
        (True, "qwen4_exp", 1, "offload", "fp8", "pinned", False),
        (True, "qwen4_exp", 1, "offload", "nvfp4", "disk", False),
        (True, "qwen4_exp", 1, "offload", "nvfp4", "auto", False),
    ],
    ids=["eligible", "flag-off", "other-model", "tp", "other-strategy", "other-format", "disk-ple", "auto-ple"],
)
def test_layer_major_runtime_gate_rejects_every_unsupported_mode(
    enabled, model_type, tp_size, strategy, expert_quant, ple_backend, expected,
):
    from freetoken.scheduler.scheduler import _supports_layer_major_prefill

    config = SimpleNamespace(
        prefill_layer_major=enabled,
        model_config=SimpleNamespace(model_type=model_type, expert_quant=expert_quant),
        tp_info=SimpleNamespace(size=tp_size),
        moe_strategy=strategy,
        ple_backend=ple_backend,
    )

    assert _supports_layer_major_prefill(config) is expected


def test_layer_major_runtime_gate_rejects_missing_ple_backend():
    from freetoken.scheduler.scheduler import _supports_layer_major_prefill

    config = SimpleNamespace(
        prefill_layer_major=True,
        model_config=SimpleNamespace(model_type="qwen4_exp", expert_quant="nvfp4"),
        tp_info=SimpleNamespace(size=1),
        moe_strategy="offload",
    )

    assert _supports_layer_major_prefill(config) is False


def test_layer_major_prefill_admits_one_bounded_text_request_whole():
    """A qualifying whole request reaches the engine, while ordinary admission stays chunked."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    _ctx()
    page_table = torch.zeros((5, 64), dtype=torch.int32)
    manager = CacheManager(num_pages=64, page_size=1, page_table=page_table, type="radix")
    prefill = PrefillManager(manager, TableManager(max_running_reqs=4, page_table=page_table), DecodeManager(1))
    prefill.layer_major = True
    prefill.pending_list = [
        PendingReq(1, torch.arange(24, dtype=torch.int32), SamplingParams(max_tokens=2))
    ]

    batch = prefill.schedule_next_batch(8)

    assert batch is not None
    assert batch.reqs[0].extend_len == 24
    assert not isinstance(batch.reqs[0], ChunkedReq)


def test_layer_major_prefill_falls_back_for_multi_request_pass():
    from freetoken.core import SamplingParams
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    _ctx()
    page_table = torch.zeros((5, 64), dtype=torch.int32)
    manager = CacheManager(num_pages=64, page_size=1, page_table=page_table, type="radix")
    prefill = PrefillManager(manager, TableManager(max_running_reqs=4, page_table=page_table), DecodeManager(1))
    prefill.layer_major = True
    prefill.pending_list = [
        PendingReq(1, torch.arange(24, dtype=torch.int32), SamplingParams(max_tokens=2)),
        PendingReq(2, torch.arange(24, dtype=torch.int32), SamplingParams(max_tokens=2)),
    ]

    batch = prefill.schedule_next_batch(8)

    assert batch is not None
    assert isinstance(batch.reqs[0], ChunkedReq)
    assert batch.reqs[0].extend_len == 8


def test_layer_major_prefill_falls_back_for_multimodal_request():
    from freetoken.core import SamplingParams
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.utils import PendingReq

    table = SimpleNamespace(token_pool=torch.zeros((1, 25), dtype=torch.int32))
    cache = SimpleNamespace(swa_paged=False, prefill_chunk_align=1, page_size=1)
    adder = PrefillAdder(
        token_budget=8, reserved_size=0, cache_manager=cache, table_manager=table,
        layer_major=True,
    )
    pending = PendingReq(
        1, torch.arange(24, dtype=torch.int32), SamplingParams(max_tokens=2),
        mm_items=[object()],
    )

    req = adder._add_one_req(pending, SimpleNamespace(), 0, 0)

    assert isinstance(req, ChunkedReq)
    assert req.extend_len == 8


def test_layer_major_prefill_falls_back_for_empty_multimodal_marker():
    """An empty multimodal list still requires the regular PLE staging path."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.utils import PendingReq

    table = SimpleNamespace(token_pool=torch.zeros((1, 25), dtype=torch.int32))
    cache = SimpleNamespace(swa_paged=False, prefill_chunk_align=1, page_size=1)
    adder = PrefillAdder(
        token_budget=8, reserved_size=0, cache_manager=cache, table_manager=table,
        layer_major=True,
    )
    pending = PendingReq(
        1, torch.arange(24, dtype=torch.int32), SamplingParams(max_tokens=2),
        mm_items=[],
    )

    req = adder._add_one_req(pending, SimpleNamespace(), 0, 0)

    assert isinstance(req, ChunkedReq)
    assert req.extend_len == 8


@pytest.mark.parametrize(
    ("prompt_len", "expected_len"), [(32_768, 32_768), (32_769, 4096)]
)
def test_layer_major_prefill_has_a_hard_request_size_bound(prompt_len, expected_len):
    """The scheduler's admission bound controls retained activation memory."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.utils import PendingReq

    table = SimpleNamespace(token_pool=torch.zeros((1, prompt_len + 1), dtype=torch.int32))
    cache = SimpleNamespace(swa_paged=False, prefill_chunk_align=1, page_size=1)
    adder = PrefillAdder(
        token_budget=4096, reserved_size=0, cache_manager=cache, table_manager=table,
        layer_major=True,
    )
    pending = PendingReq(1, torch.arange(prompt_len, dtype=torch.int32), SamplingParams(max_tokens=2))

    req = adder._add_one_req(pending, SimpleNamespace(), 0, 0)

    assert req is not None and req.extend_len == expected_len
    assert isinstance(req, ChunkedReq) is (expected_len != prompt_len)


@pytest.mark.parametrize("cached_len", [0, 17, 64])
def test_layer_major_children_keep_absolute_lengths_and_one_checkpoint(cached_len):
    _ctx()
    batch = _batch(cached_len=cached_len)
    batch.fla_metadata = build_fla_metadata(batch, torch.device("cpu"))
    chunks = _engine(4096)._build_layer_major_chunks(batch)

    assert [(c.reqs[0].cached_len, c.reqs[0].device_len) for c in chunks] == [
        (cached_len, cached_len + 4096), (cached_len + 4096, cached_len + 8192),
        (cached_len + 8192, cached_len + 8193),
    ]
    assert batch.reqs[0].mamba_next_track_idx == 1
    assert [chunk.fla_metadata.track_dst is not None for chunk in chunks] == [False, False, False]
    assert sum(chunk.track_initial_dst is not None for chunk in chunks) == 1
    assert chunks[-1].track_initial_src == 2
    assert chunks[-1].track_initial_dst == 5
    assert all(chunk.skip_begin_prefill for chunk in chunks)


def test_layer_major_projects_an_interior_checkpoint_once():
    _ctx()
    batch = _batch(extend_len=7000)
    batch.fla_metadata = build_fla_metadata(batch, torch.device("cpu"))
    chunks = _engine(4096)._build_layer_major_chunks(batch)

    assert len(chunks) == 2
    tracked = [chunk for chunk in chunks if chunk.fla_metadata.track_dst is not None]
    assert len(tracked) == 1
    assert tracked[0].fla_metadata.track_dst.tolist() == [5]
    assert batch.reqs[0].mamba_next_track_idx == 1


def test_layer_major_calls_offload_begin_once_before_model(monkeypatch):
    engine = _engine(8)
    calls = []
    chunks = [object(), object()]
    engine.moe_offload_cache = SimpleNamespace(begin_prefill=lambda: calls.append("begin"))
    engine.model = SimpleNamespace(forward_layer_major=lambda value: calls.append(value) or torch.tensor([[1.0]]))
    monkeypatch.setattr(engine, "_build_layer_major_chunks", lambda batch: chunks)

    result = engine._forward_layer_major(object())

    assert result.tolist() == [[1.0]]
    assert calls == ["begin", chunks]


def test_moe_child_does_not_restart_prefill_stream():
    from freetoken.layers.moe import OffloadMoELayer

    events = []
    cache = SimpleNamespace(
        begin_prefill=lambda: events.append("begin"),
        prefetch_prefill_layer=lambda layer: events.append(("prefetch", layer)),
        wait_prefill_layer=lambda layer: (layer,),
    )
    _ctx()
    import freetoken.core as core
    core.get_global_ctx()._batch = SimpleNamespace(skip_begin_prefill=True)
    layer = SimpleNamespace(layer_id=0)

    assert OffloadMoELayer._wait_prefill_overlap(layer, cache) == (0,)
    assert events == [("prefetch", 0), ("prefetch", 1)]


def test_ple_checkpoint_copies_incoming_ngram_context_before_chunk_commit():
    from freetoken.models.qwen4_exp.model import _copy_layer_major_track_state
    from freetoken.models.qwen4_exp.ple import build_ple_metadata, commit_ngram_context

    pool = SimpleNamespace(
        recurrent_states=torch.zeros(1, 2, 1),
        conv_states=torch.zeros(1, 2, 1, 1),
        slot_states={
            "ple_conv": torch.tensor([[[[3.0]], [[30.0]]]]),
            "ple_ngram_ctx": torch.tensor([[1469, 17], [0, 0]]),
        },
    )
    pool.local_index = lambda layer_id: layer_id
    pool.slot_state = lambda name, layer_id=None: (
        pool.slot_states[name][0] if layer_id is not None else pool.slot_states[name]
    )
    req = SimpleNamespace(extend_len=1, cached_len=8, table_idx=0, linear_slot_idx=0)
    batch = SimpleNamespace(
        padded_reqs=[req], input_ids=torch.tensor([23]), is_decode=False,
        fla_metadata=SimpleNamespace(
            has_initial_state=torch.tensor([True]), cu_seqlens=torch.tensor([0, 1]),
            cache_indices=torch.tensor([0]),
        ),
        track_initial_src=0, track_initial_dst=1,
    )
    args = SimpleNamespace(ngram_size=3, ngram_boundary_token_id=0, image_token_id=None)
    batch.ple_meta = build_ple_metadata(batch, args, torch.device("cpu"), context_pool=pool.slot_states["ple_ngram_ctx"])
    layer = SimpleNamespace(_is_linear=False, ple=SimpleNamespace(layer_id=0))

    _copy_layer_major_track_state(pool, layer, batch)
    commit_ngram_context(batch.ple_meta, None, context_pool=pool.slot_states["ple_ngram_ctx"])

    assert pool.slot_states["ple_ngram_ctx"].tolist() == [[17, 23], [1469, 17]]


def test_layer_major_model_restores_outer_batch_after_failure(monkeypatch):
    import freetoken.core as core
    from freetoken.models.qwen4_exp.model import Qwen4ExpModel

    pool = SimpleNamespace()
    _ctx(pool)
    outer = SimpleNamespace(name="outer")
    core.get_global_ctx()._batch = outer
    model = object.__new__(Qwen4ExpModel)
    model.embed_tokens = object()
    model.hc_count = 1
    model._ple = ()
    model.hyper_connection_mixer = SimpleNamespace(mix=lambda hidden: (hidden,))
    model.layers = SimpleNamespace(op_list=[SimpleNamespace(ple=None, _is_linear=False,
        forward=lambda hidden, batch: (_ for _ in ()).throw(RuntimeError("layer failed")))])
    monkeypatch.setattr("freetoken.models.qwen4_exp.model.embed_input_ids", lambda *_: torch.ones(1, 2))

    with pytest.raises(RuntimeError, match="layer failed"):
        model.forward_layer_major([SimpleNamespace(input_ids=torch.tensor([1]))])

    assert core.get_global_ctx().batch is outer
