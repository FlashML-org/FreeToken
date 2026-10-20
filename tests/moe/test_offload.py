from contextlib import contextmanager

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.quantization import QuantKind


def _init_tp():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _bf16_offload_layer(layer_id: int, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int):
    """A bf16 offload layer on the fused kernel."""
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig

    return OffloadMoELayer(
        layer_id, num_experts, top_k, hidden_size, intermediate_size,
        quant_config=NoQuantConfig(), prefix=f"model.layers.{layer_id}.mlp.experts",
    )


def _make_layer_and_cache():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    layer = _bf16_offload_layer(0, 4, 2, 8, 16)
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
    )
    cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]})
    layer.offload_cache = cache
    return layer, cache


def test_dummy_expert_banks_follow_the_kernel_layout(monkeypatch):

    from freetoken.kernel import backend
    from freetoken.layers.quantization import QuantBackend, QuantConfig, set_quant_backend
    from freetoken.moe.expert_banks import build_expert_banks

    _init_tp()
    L, E, H, I = 3, 4, 64, 32
    monkeypatch.setattr(backend, "device_capability", lambda: (0, 0))
    monkeypatch.setattr(backend, "is_vllm_installed", lambda: False)
    monkeypatch.setattr(backend, "is_flashinfer_installed", lambda: False)

    def _bound(quant):
        layer = _bf16_offload_layer(0, E, 2, H, I) if quant is None else None
        if layer is None:
            from freetoken.layers.moe import OffloadMoELayer

            layer = OffloadMoELayer(0, E, 2, H, I, quant_config=quant, prefix="model.layers.0.mlp.experts")
        return layer

    bf16 = _bound(None)
    banks = build_expert_banks(bf16.quant_method, L, None, device=torch.device("cpu"), dummy=True)
    assert banks.kind is QuantKind.NONE and set(banks.sources) == {"gate_up", "down"}
    assert len(banks.sources["gate_up"]) == L and all(t.shape == (E, 2 * I, H) for t in banks.sources["gate_up"])
    assert all(t.shape == (E, H, I) for t in banks.sources["down"])

    set_quant_backend(QuantBackend.parse("moe.nvfp4=triton"))
    quant = QuantConfig.from_hf({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"]}})
    nvfp4 = _bound(quant)
    banks = build_expert_banks(nvfp4.quant_method, L, None, device=torch.device("cpu"), dummy=True)
    assert banks.kind is QuantKind.NVFP4 and banks.kernel == "triton"
    assert {len(layers) for layers in banks.sources.values()} == {L}
    assert {t.shape[0] for layers in banks.sources.values() for t in layers} == {E}
    assert torch.all(banks.sources["gate_up_scale"][0].float() == 1.0)
    assert torch.all(banks.sources["gate_up_global"][0].float() > 0)


def test_offload_moe_layer_prefill_forward_uses_single_layer_cache_view(monkeypatch):
    layer, cache = _make_layer_and_cache()
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    router_logits = torch.randn(1, 4)
    calls = {}

    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (topk_weights, topk_ids),
    )
    monkeypatch.setattr(cache, "materialize_layer", lambda layer_id: calls.setdefault("layer_id", layer_id))
    monkeypatch.setattr(cache, "copy_missing", lambda: calls.setdefault("copied", True))

    def fake_fused(
        hidden_states,
        w1,
        w2,
        got_topk_weights,
        got_topk_ids,
        activation,
        apply_router_weight_on_input,
        act_alpha=1.0,
        act_limit=float("inf"),
    ):
        calls["w1"] = w1
        calls["w2"] = w2
        calls["topk_weights"] = got_topk_weights
        calls["topk_ids"] = got_topk_ids.clone()
        return hidden_states

    monkeypatch.setattr("freetoken.moe.fused.fused_experts_impl", fake_fused)

    out = layer.prefill_forward(hidden_states, router_logits)

    assert out is hidden_states
    assert calls["layer_id"] == 0
    assert calls["copied"] is True
    assert calls["w1"].shape[0] == layer.num_experts
    assert calls["w2"].shape[0] == layer.num_experts
    assert calls["w1"].data_ptr() == cache.bank_caches["gate_up"].data_ptr()
    assert calls["w2"].data_ptr() == cache.bank_caches["down"].data_ptr()
    assert calls["topk_weights"] is topk_weights
    assert calls["topk_ids"].dtype == torch.int32
    # slot == expert id after materialize, so the routing ids pass through unmapped
    assert calls["topk_ids"].tolist() == [[2, 1]]


def test_offload_moe_layer_prefill_overlap_prefetches_layers_into_two_buffers(monkeypatch):
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    num_layers = 3
    num_experts = 4
    layers = [_bf16_offload_layer(layer_id, num_experts, 2, 8, 16) for layer_id in range(num_layers)]
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=True,
    )
    gate_up_source = list(torch.arange(num_layers * num_experts * 32 * 8, dtype=torch.float32).reshape(
        num_layers * num_experts, 32, 8
    ).split(num_experts))
    down_source = list(torch.arange(num_layers * num_experts * 8 * 16, dtype=torch.float32).reshape(
        num_layers * num_experts, 8, 16
    ).split(num_experts))
    cache.set_bank_sources({"gate_up": gate_up_source, "down": down_source})
    for layer in layers:
        layer.offload_cache = cache

    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    router_logits = torch.randn(1, num_experts)
    fused_calls = []

    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (
            topk_weights,
            topk_ids.clone(),
        ),
    )

    def unexpected_fast_index_copy(*args, **kwargs):
        raise AssertionError("prefill overlap should use direct async copy")

    monkeypatch.setattr("freetoken.kernel.fast_index_copy_jit", unexpected_fast_index_copy)

    def fake_fused(
        hidden_states,
        w1,
        w2,
        got_topk_weights,
        got_topk_ids,
        activation,
        apply_router_weight_on_input,
        act_alpha=1.0,
        act_limit=float("inf"),
    ):
        layer_id = len(fused_calls)
        fused_calls.append(
            {
                "w1_ptr": w1.data_ptr(),
                "w2_ptr": w2.data_ptr(),
                "w1": w1.clone(),
                "w2": w2.clone(),
                "topk_weights": got_topk_weights,
                "topk_ids": got_topk_ids.clone(),
            }
        )
        return hidden_states + layer_id

    monkeypatch.setattr("freetoken.moe.fused.fused_experts_impl", fake_fused)

    out = hidden_states
    for layer in layers:
        out = layer.prefill_forward(out, router_logits)

    assert torch.allclose(out, hidden_states + 3)
    for layer_id in range(num_layers):
        assert fused_calls[layer_id]["topk_weights"] is topk_weights
        assert fused_calls[layer_id]["topk_ids"].tolist() == [[2, 1]]
        assert torch.equal(fused_calls[layer_id]["w1"], gate_up_source[layer_id])
        assert torch.equal(fused_calls[layer_id]["w2"], down_source[layer_id])

    assert fused_calls[0]["w1_ptr"] == fused_calls[2]["w1_ptr"]
    assert fused_calls[0]["w2_ptr"] == fused_calls[2]["w2_ptr"]
    assert fused_calls[0]["w1_ptr"] != fused_calls[1]["w1_ptr"]
    assert fused_calls[0]["w2_ptr"] != fused_calls[1]["w2_ptr"]
    prefill_gate_up_buffer, prefill_down_buffer = cache.stage_views(0, 0)
    assert prefill_gate_up_buffer.data_ptr() == cache.bank_caches["gate_up"].data_ptr()
    assert prefill_down_buffer.data_ptr() == cache.bank_caches["down"].data_ptr()


def test_offload_moe_cache_prefill_overlap_requires_two_layer_slots():
    from freetoken.moe.offload_cache import OffloadMoeCache

    with pytest.raises(AssertionError):
        OffloadMoeCache(
            num_layers=3,
            num_experts=4,
            cache_size=7,
            device=torch.device("cpu"),
            prefill_overlap=True,
        )


def test_offload_moe_cache_marlin_rejects_slot_count_beyond_kernel_limit():
    from freetoken.moe.offload_cache import OffloadMoeCache

    with pytest.raises(ValueError, match="992"):
        OffloadMoeCache(
            num_layers=2,
            num_experts=8,
            cache_size=1024,
            device=torch.device("cpu"),
            quant_format="nvfp4_marlin",
        )


def test_prefill_overlap_prefetch_invalidates_borrowed_unified_cache_slots():
    from freetoken.moe.offload_cache import OffloadMoeCache

    num_layers = 3
    num_experts = 4
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=True,
    )
    gate_up_source = list(torch.arange(num_layers * num_experts * 32 * 8, dtype=torch.float32).reshape(
        num_layers * num_experts, 32, 8
    ).split(num_experts))
    down_source = list(torch.arange(num_layers * num_experts * 8 * 16, dtype=torch.float32).reshape(
        num_layers * num_experts, 8, 16
    ).split(num_experts))
    cache.set_bank_sources({"gate_up": gate_up_source, "down": down_source})

    old_layers = torch.tensor([2, 2, 1, 1], dtype=torch.int32)
    old_experts = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    cache.id_of_slot[:num_experts] = old_layers * num_experts + old_experts
    cache.usage[:num_experts] = torch.arange(1, num_experts + 1, dtype=torch.int64)
    for slot, (layer_id, expert_id) in enumerate(zip(old_layers.tolist(), old_experts.tolist())):
        cache.slot_for_id[layer_id, expert_id] = slot

    cache.prefetch_prefill_layer(0)

    assert cache.id_of_slot[:num_experts].tolist() == [-1] * num_experts
    assert cache.usage[:num_experts].tolist() == [0] * num_experts
    for layer_id, expert_id in zip(old_layers.tolist(), old_experts.tolist()):
        assert int(cache.slot_for_id[layer_id, expert_id].item()) == -1
    assert torch.equal(cache.bank_caches["gate_up"][:num_experts], gate_up_source[0])
    assert torch.equal(cache.bank_caches["down"][:num_experts], down_source[0])


def test_prefill_overlap_waits_for_previous_prefill_release_after_begin(monkeypatch):
    from freetoken.moe.offload_cache import OffloadMoeCache

    num_layers = 2
    num_experts = 4
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=True,
    )
    gate_up_source = list(torch.zeros(num_layers * num_experts, 32, 8).split(num_experts))
    down_source = list(torch.zeros(num_layers * num_experts, 8, 16).split(num_experts))
    cache.set_bank_sources({"gate_up": gate_up_source, "down": down_source})

    class FakeStream:
        def __init__(self):
            self.waited = []

        def wait_event(self, event):
            self.waited.append(event.name)

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream=None):
            pass

    @contextmanager
    def fake_cuda_stream(stream):
        yield

    copy_stream = FakeStream()
    cache.prefill_copy_stream = copy_stream
    cache.prefill_begin_event = FakeEvent("begin")
    cache.prefill_ready_events = [FakeEvent("ready0"), FakeEvent("ready1")]
    cache.prefill_release_events = [FakeEvent("release0"), FakeEvent("release1")]
    monkeypatch.setattr("torch.cuda.stream", fake_cuda_stream)
    monkeypatch.setattr("torch.cuda.current_stream", lambda device=None: object())

    cache.prefetch_prefill_layer(0)
    cache.release_prefill_layer(0)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)

    # begin_prefill fences the copy stream behind the compute stream (so a prefetch
    # cannot race the preceding decode batch), then the buffer reuse waits on the
    # previous prefill's release event.
    assert copy_stream.waited == ["begin", "release0"]


def test_offload_moe_layer_decode_forward_uses_remapped_slot_ids(monkeypatch):
    layer, cache = _make_layer_and_cache()
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    router_logits = torch.randn(1, 4)
    calls = {}

    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (topk_weights, topk_ids),
    )

    def fake_ensure(layer_id, expert_ids):
        calls["ensure_layer_id"] = layer_id
        calls["ensure_expert_ids"] = expert_ids.clone()
        expert_ids.copy_(torch.tensor([[5, 0]], dtype=torch.int32))

    monkeypatch.setattr(cache, "ensure_experts", fake_ensure)
    monkeypatch.setattr(cache, "copy_missing", lambda: calls.setdefault("copied", True))

    def fake_fused_decode(
        hidden_states,
        w1,
        w2,
        got_topk_weights,
        got_topk_ids,
        activation,
        apply_router_weight_on_input,
        act_alpha=1.0,
        act_limit=float("inf"),
    ):
        calls["w1"] = w1
        calls["w2"] = w2
        calls["topk_weights"] = got_topk_weights
        calls["topk_ids"] = got_topk_ids.clone()
        return hidden_states

    monkeypatch.setattr("freetoken.moe.fused.fused_experts_decode_impl", fake_fused_decode)

    out = layer.decode_forward(hidden_states, router_logits)

    assert out is hidden_states
    assert calls["ensure_layer_id"] == 0
    assert calls["ensure_expert_ids"].tolist() == [[2, 1]]
    assert calls["copied"] is True
    assert calls["w1"] is cache.bank_caches["gate_up"]
    assert calls["w2"] is cache.bank_caches["down"]
    assert calls["topk_weights"] is topk_weights
    assert calls["topk_ids"].dtype == torch.int32
    assert calls["topk_ids"].tolist() == [[5, 0]]



def test_lru_gpu_cache_assigns_unique_slots_for_large_miss_batch():
    import pytest
    from freetoken.moe.offload_cache import OffloadMoeCache

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU offload cache kernel")

    cache = OffloadMoeCache(
        num_layers=40,
        num_experts=256,
        cache_size=1664,
        device=torch.device("cuda"),
    )
    expert_ids = torch.arange(256, dtype=torch.int32, device="cuda").view(32, 8)

    cache.ensure_experts(0, expert_ids)
    torch.cuda.synchronize()

    assert int(cache.num_indices.item()) == 256
    assert expert_ids.min().item() >= 0
    assert expert_ids.max().item() < cache.cache_size
    evict_slots = cache.evict_slots[:256]
    assert evict_slots.min().item() >= 0
    assert evict_slots.max().item() < cache.cache_size
    assert torch.unique(evict_slots).numel() == evict_slots.numel()
    assert cache.src_indices[:256].tolist() == list(range(256))


def test_adjust_config_converts_moe_cache_rate_to_cache_size(monkeypatch):
    from types import SimpleNamespace

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    import freetoken.engine.engine as engine_module
    from freetoken.engine.engine import _adjust_config

    # This test exercises the discrete-GPU offload path regardless of the host
    # running the suite (GB10 reports cudaDevAttrIntegrated=1).
    monkeypatch.setattr(engine_module, "_is_unified_memory_gpu", lambda index=None: False)

    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.float16,
        attention_backend="fi",
        moe_cache_rate=0.3,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=True,
            num_layers=10,
            num_moe_layers=10,
            num_experts=8,
            expert_quant="none",
            moe_strategy="auto",
        ),
    )

    _adjust_config(config)

    from freetoken.moe import is_offload_moe_strategy

    assert config.moe_cache_size == 24
    # Family, not member: which offload-family backend auto picks is not this test's claim.
    assert is_offload_moe_strategy(config.moe_strategy)


def test_graph_capture_reuses_warm_offload_cache_before_capture(monkeypatch):
    import freetoken.core as core
    from freetoken.core import Context, Req, get_global_ctx
    from freetoken.engine.graph import GraphRunner

    events = []
    _init_tp()
    monkeypatch.setattr(core, "_GLOBAL_CTX", Context(page_size=1))

    class FakeGraph:
        def pool(self):
            return "pool"

    @contextmanager
    def fake_cuda_graph(graph, pool=None, stream=None):
        events.append("graph_enter")
        yield
        events.append("graph_exit")

    class FakeAttnBackend:
        def init_capture_graph(self, max_seq_len, bs_list):
            pass

        def prepare_for_capture(self, batch):
            pass

    class FakeModel:
        def forward(self):
            events.append("forward")
            batch = get_global_ctx().batch
            return torch.zeros(batch.size, 3)

    class FakeOffloadCache:
        def reset(self):
            events.append("reset")

    monkeypatch.setattr("torch.cuda.CUDAGraph", FakeGraph)
    monkeypatch.setattr("torch.cuda.graph", fake_cuda_graph)
    monkeypatch.setattr("torch.cuda.synchronize", lambda device=None: None)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr("freetoken.engine.graph.get_free_memory", lambda device: 1024)

    dummy_req = Req(
        input_ids=torch.tensor([0], dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=-1,
        sampling_params=None,
        cache_handle=None,
    )
    GraphRunner(
        stream=None,
        device=torch.device("cpu"),
        model=FakeModel(),
        attn_backend=FakeAttnBackend(),
        cuda_graph_bs=[1],
        cuda_graph_max_bs=None,
        free_memory=1024,
        max_seq_len=1,
        vocab_size=3,
        dummy_req=dummy_req,
        moe_offload_cache=FakeOffloadCache(),
    )

    assert events == [
        "reset",
        "forward",
        "graph_enter",
        "forward",
        "graph_exit",
        "reset",
        "reset",
    ]


def test_nvfp4_materialize_keeps_bookkeeping_consistent_across_requests():
    """Regression: a full-layer prefill loads the layer's experts into slots [0, E).
    If that overwrite does not invalidate the previous owners' mappings, a later
    decode "hits" a stale slot_for_id entry and silently reads another expert's
    weights. materialize_layer must keep bookkeeping == slot contents."""
    import pytest
    from freetoken.moe.offload_cache import OffloadMoeCache

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU offload cache kernel")

    L, E, S = 2, 8, 8
    OUT, IN = 64, 512  # keep rows >= 128B so the fast_index_copy JIT has a kernel
    dev = torch.device("cuda")

    def bank(out, inner, dtype):
        # one independently allocated [E, out, inner] tensor per layer (the per-layer host
        # bank contract); row idx within layer l keeps the old flat fingerprint l*E+idx.
        layers = []
        for l in range(L):
            t = torch.zeros(E, out, inner, dtype=dtype)
            for e in range(E):
                t[e].view(torch.uint8).fill_(l * E + e)
            layers.append(t)
        return layers

    def pinned(layers):
        return [t.pin_memory() for t in layers]

    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=S, device=dev, quant_format="nvfp4"
    )
    cache.set_bank_sources(
        {
            "gate_up_packed": pinned(bank(OUT, IN // 2, torch.uint8)),
            "gate_up_scale": pinned(bank(OUT, IN // 16, torch.float8_e4m3fn)),
            "gate_up_global": pinned([t.squeeze(-1).contiguous() for t in bank(OUT, 1, torch.float16)]),
            "down_packed": pinned(bank(OUT, IN // 2, torch.uint8)),
            "down_scale": pinned(bank(OUT, IN // 16, torch.float8_e4m3fn)),
            "down_global": pinned([t.squeeze(-1).contiguous() for t in bank(OUT, 1, torch.float16)]),
        }
    )
    cache.reset()

    def fingerprint(slot):  # which source row's bytes live in this slot?
        return int(cache.bank_caches["gate_up_packed"][slot].view(torch.uint8).flatten()[0].item())

    # Request A, decode: layer 0 loads experts 3 and 5 somewhere in the cache.
    ids = torch.tensor([3, 5], dtype=torch.int32, device=dev)
    cache.ensure_experts(0, ids)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert [fingerprint(s) for s in ids.tolist()] == [3, 5]

    # Request B, prefill: layer 1 is materialized into slots [0, E), overwriting
    # every slot (S == E), including the ones decode A used.
    cache.materialize_layer(1)
    cache.copy_missing()
    torch.cuda.synchronize()
    # The layer's experts fill slots [0, E) bijectively and the bookkeeping agrees.
    assert [fingerprint(s) for s in range(E)] == [E + e for e in range(E)]
    assert cache.slot_for_id[1].tolist() == list(range(E))

    # Request B, decode: layer 0 routes to experts 3/5 again. Their old slots were
    # overwritten, so this must be a miss + reload -- never a stale hit serving
    # layer-1 bytes.
    ids2 = torch.tensor([3, 5], dtype=torch.int32, device=dev)
    cache.ensure_experts(0, ids2)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert [fingerprint(s) for s in ids2.tolist()] == [3, 5]

    # The prefilled layer's own experts still resolve to correct bytes (S == E, so
    # the layer-0 reload above evicted two layer-1 slots -- hit or miss, the
    # bookkeeping must never serve another expert's bytes).
    ids3 = torch.tensor([1, 2], dtype=torch.int32, device=dev)
    cache.ensure_experts(1, ids3)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert [fingerprint(s) for s in ids3.tolist()] == [E + 1, E + 2]


def test_offload_cache_rebuild_resizes_and_preserves_sources():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(num_layers=1, num_experts=4, cache_size=6, device=torch.device("cpu"))
    gate_up = torch.randn(4, 32, 8)
    down = torch.randn(4, 8, 16)
    cache.set_bank_sources({"gate_up": [gate_up], "down": [down]})

    cache.rebuild(10)

    assert cache.cache_size == 10
    # host sources preserved (same objects, not reloaded)
    assert cache.bank_sources["gate_up"][0] is gate_up
    assert cache.bank_sources["down"][0] is down
    # GPU slot caches resized to the new cache_size, row shape unchanged
    assert cache.bank_caches["gate_up"].shape == (10, 32, 8)
    assert cache.bank_caches["down"].shape == (10, 8, 16)
    # bookkeeping resized + reset
    assert cache.id_of_slot.shape == (10,)
    assert cache.usage.shape == (10,)
    assert torch.all(cache.slot_for_id == -1)
    assert torch.all(cache.id_of_slot == -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_slot_pools_of_two_bank_layouts_keep_their_rows_through_prefill():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    torch.manual_seed(0)
    L, E, top_k = 4, 8, 4
    # layers 1 and 3 keep wider down rows (uint8, like Q6_K next to Q4_K); their pools come first
    wide = (1, 3)
    sources = {
        "gate_up": [torch.randn(E, 1024, 256, dtype=torch.bfloat16).pin_memory() for _ in range(L)],
        "down": [
            torch.randint(0, 255, (E, 512, 768), dtype=torch.uint8).pin_memory() if layer in wide
            else torch.randn(E, 512, 256, dtype=torch.bfloat16).pin_memory()
            for layer in range(L)
        ],
    }
    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=33, device=torch.device("cuda"),
        prefill_overlap=True, prefill_hit_d2d=True, pool_of_layer=[2, 0, 3, 1], pool_sizes=[6, 5, 12, 10],
    )
    cache.set_bank_sources(sources)
    assert [v[1].dtype for v in map(cache.pool_bank_views, range(L))] == [torch.bfloat16, torch.uint8] * 2
    # staging buffers take 2 x 8 rows of every bank: all of the wide pools, the first 8 slots of layer 0's pool
    assert cache._pool_stable_from == [6, 5, 8, 0]

    def check_resident() -> None:
        torch.cuda.synchronize()
        for layer in range(L):
            for e, s in enumerate(cache.slot_for_id[layer].tolist()):
                if s >= 0:
                    for (src, _), view in zip(cache.banks, cache.pool_bank_views(layer)):
                        assert torch.equal(view[s].cpu(), src[layer][e]), (layer, e, s)

    def decode(steps: int) -> None:
        for _ in range(steps):
            for layer in range(L):
                cache.ensure_experts(layer, torch.randperm(E)[:top_k].to(torch.int32).view(1, top_k).cuda())
                cache.copy_missing()
        check_resident()

    decode(20)
    cache.begin_prefill()
    for layer in range(L):
        cache.prefetch_prefill_layer(layer)
        cache.prefetch_prefill_layer(layer + 1)
        views = cache.wait_prefill_layer(layer)
        torch.cuda.synchronize()
        for (src, _), view in zip(cache.banks, views):
            assert torch.equal(view.cpu(), src[layer])
        cache.release_prefill_layer(layer)
    assert cache.prefill_hit_rows > 0
    for p, stable in enumerate(cache._pool_stable_from):
        if stable:
            assert not any(view[0].any() for view in cache.pool_bank_views(cache.pool_of_layer.index(p)))
    check_resident()
    decode(20)

    cache.materialize_layer(3)
    cache.copy_missing()
    for (src, _), view in zip(cache.banks, cache.stage_views(3, 0)):
        assert torch.equal(view.cpu(), src[3])
    check_resident()

    cache.rebuild(32, [5, 5, 10, 12])
    decode(20)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_slot_pools_split_layers_of_one_bank_layout_by_quant_tag():
    from freetoken.moe.offload_cache import OffloadMoeCache, bank_layouts
    from freetoken.moe.pool_policy import group_layers

    _init_tp()
    L, E = 4, 8
    # uint8 rows of one width, as GGUF Q4_K and IQ4_NL rows are
    sources = {
        "gate_up": [torch.randint(1, 255, (E, 64, 144), dtype=torch.uint8).pin_memory() for _ in range(L)],
        "down": [torch.randint(1, 255, (E, 32, 144), dtype=torch.uint8).pin_memory() for _ in range(L)],
    }
    tags = [("ggml", "Q4_K", "Q4_K"), ("ggml", "IQ4_NL", "IQ4_NL")] * 2
    assert group_layers("unified", bank_layouts(sources, tags), [1] * L) == [[0, 2], [1, 3]]

    def make(pool_of_layer, pool_sizes):
        return OffloadMoeCache(
            num_layers=L, num_experts=E, cache_size=sum(pool_sizes), device=torch.device("cuda"),
            prefill_overlap=True, pool_of_layer=pool_of_layer, pool_sizes=pool_sizes,
        )

    with pytest.raises(AssertionError, match="layouts"):
        make([0] * L, [20]).set_bank_sources(sources, layer_tags=tags)
    cache = make([0, 1, 0, 1], [10, 10])
    cache.set_bank_sources(sources, layer_tags=tags)
    assert cache.bank_caches["down"].shape == (20, 32, 144)
    # the staging buffers cover pool 0 and slots [0, 6) of pool 1, so both slot 0 rows end up with another quant's bytes
    cache.begin_prefill()
    for layer in range(L):
        cache.prefetch_prefill_layer(layer)
        cache.prefetch_prefill_layer(layer + 1)
        cache.wait_prefill_layer(layer)
        cache.release_prefill_layer(layer)
    torch.cuda.synchronize()
    for layer in (0, 1):
        for view in cache.pool_bank_views(layer):
            assert not view[0].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("wide", [(), (1, 3)], ids=["cpu-mode", "two-layouts"])
def test_slot_pools_and_staging_buffers_end_in_zero_mmq_tails(wide):
    from freetoken.layers.quantization.ggml_types import mmq_tail_bytes, row_bytes
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.third_party.gguf import GGMLQuantizationType

    _init_tp()
    torch.manual_seed(0)
    L, E, top_k = 4, 8, 4

    def layout(ggml_type):
        """(rows, row bytes, tail bytes) of gate_up and down for Gemma-4 26B-A4B's K = 2816 and 704, with fewer rows."""
        return tuple((rows, row_bytes(ggml_type, k), mmq_tail_bytes(ggml_type, k)) for rows, k in ((32, 2816), (64, 704)))

    q4_0, q8_0 = layout(GGMLQuantizationType.Q4_0), layout(GGMLQuantizationType.Q8_0)
    shapes = [q8_0 if layer in wide else q4_0 for layer in range(L)]
    roles = ("gate_up", "down")
    sources = {role: [torch.randint(1, 255, (E, *shapes[l][k][:2]), dtype=torch.uint8).pin_memory() for l in range(L)] for k, role in enumerate(roles)}
    tails = {role: [shapes[l][k][2] for l in range(L)] for k, role in enumerate(roles)}
    # cpu mode: the slots are exactly the two staging buffers
    pools = dict(pool_of_layer=[2, 0, 3, 1], pool_sizes=[6, 5, 12, 10], prefill_hit_d2d=True) if wide else dict(pool_of_layer=[0] * L, pool_sizes=[2 * E])
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=sum(pools["pool_sizes"]), device=torch.device("cuda"), prefill_overlap=True, **pools)
    cache.set_bank_sources(sources, row_tail_bytes=tails)
    assert cache.prefill_overlap

    def assert_zero_tails(layer: int, views) -> None:
        torch.cuda.synchronize()
        for k, ((_, bank), view) in enumerate(zip(cache.banks, views)):
            end, tail = view.storage_offset() + view.numel(), shapes[layer][k][2]
            assert end + tail <= bank.numel() and not bank[end : end + tail].any(), (layer, k)

    for _ in range(20):
        for layer in range(L):
            cache.ensure_experts(layer, torch.randperm(E)[:top_k].to(torch.int32).view(1, top_k).cuda())
            cache.copy_missing()
    cache.begin_prefill()
    for layer in range(L):
        cache.prefetch_prefill_layer(layer)
        cache.prefetch_prefill_layer(layer + 1)
        assert_zero_tails(layer, cache.wait_prefill_layer(layer))
        cache.release_prefill_layer(layer)
    for layer in range(L):
        assert_zero_tails(layer, cache.pool_bank_views(layer))
    if wide:
        cache.materialize_layer(0)
        cache.copy_missing()
        assert_zero_tails(0, cache.stage_views(0, 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_offload_cache_rebuild_frees_the_old_slots_first():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=32, device=torch.device("cuda"),
        pool_of_layer=[0, 1], pool_sizes=[16, 16],
    )
    cache.set_bank_sources({
        "gate_up": [torch.zeros(4, 1024, 1024, dtype=torch.bfloat16).pin_memory() for _ in range(2)],
        "down": [torch.zeros(4, 512, 1024, dtype=torch.bfloat16).pin_memory() for _ in range(2)],
    })
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    cache.rebuild(32, [8, 24])
    # 96 MiB of slots: holding the old ones while allocating the new ones would double it
    assert torch.cuda.max_memory_allocated() - before < 8 << 20


def test_offload_cache_rebuild_disables_prefill_overlap_when_too_small():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=8, device=torch.device("cpu"),
        prefill_overlap=True,
    )
    cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]})
    assert cache.prefill_overlap is True

    cache.rebuild(5)  # 5 < 2*num_experts (8) -> overlap must auto-disable

    assert cache.cache_size == 5
    assert cache.prefill_overlap is False
    assert cache._stage_views[0][1] is None  # no room for the second staging buffer


def test_offload_cache_rebuild_keeps_overlap_at_boundary():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=8, device=torch.device("cpu"),
        prefill_overlap=True,
    )
    cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]})
    cache.rebuild(8)  # exactly 2*num_experts -> overlap stays on
    assert cache.prefill_overlap is True
    assert cache.cache_size == 8


def test_offload_cache_validate_rebuild_enforces_marlin_cap_and_floor():
    # The constructor caps nvfp4_marlin slots at 992; a runtime rebuild must enforce the
    # same upper cap (and the num_experts floor), else marlin decode kernels later break.
    from freetoken.moe.offload_cache import MARLIN_MAX_CACHE_SIZE, OffloadMoeCache

    _init_tp()
    marlin = OffloadMoeCache(
        num_layers=1, num_experts=8, cache_size=16,
        device=torch.device("cpu"), quant_format="nvfp4_marlin",
    )
    with pytest.raises(ValueError, match="992"):
        marlin.validate_rebuild(MARLIN_MAX_CACHE_SIZE + 1)
    marlin.validate_rebuild(MARLIN_MAX_CACHE_SIZE)  # exactly at the cap: allowed

    bf16 = OffloadMoeCache(num_layers=1, num_experts=4, cache_size=6, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="num_experts"):
        bf16.validate_rebuild(3)  # below the num_experts floor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("target", ["gpu", "hybrid"])
def test_slot_pools_keep_each_layer_in_its_own_slots(target):
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    torch.manual_seed(0)
    L, E, sizes = 3, 16, [5, 7, 20]
    # hybrid routes more experts than the smallest pool holds; GPU decode must fit them
    top_k = 2 if target == "gpu" else 8
    sources = {
        "gate_up": [torch.randn(E, 64, 32, dtype=torch.bfloat16).pin_memory() for _ in range(L)],
        "down": [torch.randn(E, 32, 32, dtype=torch.bfloat16).pin_memory() for _ in range(L)],
    }

    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=sum(sizes), device=torch.device("cuda"),
        decode_target=target, hybrid_max_fetch=E, pool_of_layer=[0, 1, 2], pool_sizes=sizes,
    )
    cache.set_bank_sources(sources)
    for _ in range(50):
        for layer in range(L):
            experts = torch.randperm(E)[:top_k].to(torch.int32)
            slots = experts.cuda()
            if target == "gpu":
                cache.ensure_experts(layer, slots)
            else:
                cache.ensure_experts_hybrid(layer, slots, torch.rand(top_k, device="cuda"))
            cache.copy_missing()
            resident = [(e, s) for e, s in zip(experts.tolist(), slots.tolist()) if s >= 0]
            assert len({s for _, s in resident}) == len(resident)
            for e, s in resident:
                assert s < sizes[layer]
                for (src, _), view in zip(cache.banks, cache.pool_bank_views(layer)):
                    assert torch.equal(view[s].cpu(), src[layer][e])

    # with several pools a whole-layer prefill stages slots [0, E) without registering them in any pool
    cache.materialize_layer(1)
    cache.copy_missing()
    for (src, _), view in zip(cache.banks, cache.stage_views(1, 0)):
        assert torch.equal(view.cpu(), src[1])
    slot_for_id, id_of_slot = cache.slot_for_id.cpu(), cache.id_of_slot.cpu()
    for layer in range(L):
        for e, s in enumerate(slot_for_id[layer].tolist()):
            if s >= 0:
                assert cache.pool_bases[layer] + s >= E
                assert id_of_slot[cache.pool_bases[layer] + s] == layer * E + e

    cache.rebuild(40, [8, 8, 24])
    assert [v[0].shape[0] for v in map(cache.pool_bank_views, range(L))] == [8, 8, 24]


def _make_split_cache(num_layers=2, locked=(1,), prefill_overlap=False, device="cpu"):
    """A [gate_up, down] bf16 cache with the given layers LOCKED (rest pinned)."""
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    dev = torch.device(device)
    cache = OffloadMoeCache(
        num_layers=num_layers, num_experts=4, cache_size=8,
        device=dev, prefill_overlap=prefill_overlap,
    )
    cache.cpu_layer_ids = frozenset(locked)
    src_dev = dev if dev.type == "cuda" else torch.device("cpu")
    sources = {
        # CUDA-resident pinned-layer sources keep _build_copy_plan's device_ptr happy in the CUDA variant; locked layers stay host tensors (never translated)
        "gate_up": [
            torch.randn(4, 32, 8, device=torch.device("cpu") if i in locked else src_dev)
            for i in range(num_layers)
        ],
        "down": [
            torch.randn(4, 8, 16, device=torch.device("cpu") if i in locked else src_dev)
            for i in range(num_layers)
        ],
    }
    residency = [
        HostResidency.LOCKED.value if i in locked else HostResidency.PINNED.value
        for i in range(num_layers)
    ]
    cache.set_bank_sources(sources, layer_residency=residency)
    return cache, sources


def test_set_bank_sources_locked_layer_requires_cpu_layer_ids():
    # a layer without a device address can only decode on the CPU executor; labeling it LOCKED outside cpu_layer_ids is a wiring bug and must fail loudly
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=8, device=torch.device("cpu"),
    )
    sources = {
        "gate_up": [torch.randn(4, 32, 8) for _ in range(2)],
        "down": [torch.randn(4, 8, 16) for _ in range(2)],
    }
    with pytest.raises(ValueError, match="cpu_layer_ids"):
        cache.set_bank_sources(
            sources,
            layer_residency=[HostResidency.PINNED.value, HostResidency.LOCKED.value],
        )


def test_set_bank_sources_locked_layer_rejects_prefill_overlap():
    # prefill overlap DMAs from registered banks; a LOCKED layer cannot feed it
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=8, device=torch.device("cpu"),
        prefill_overlap=True,
    )
    cache.cpu_layer_ids = frozenset({1})
    sources = {
        "gate_up": [torch.randn(4, 32, 8) for _ in range(2)],
        "down": [torch.randn(4, 8, 16) for _ in range(2)],
    }
    with pytest.raises(ValueError, match="[Pp]refill overlap"):
        cache.set_bank_sources(
            sources,
            layer_residency=[HostResidency.PINNED.value, HostResidency.LOCKED.value],
        )


def test_locked_layer_prefill_materialize_copies_whole_layer_pageable():
    # the only movement a LOCKED layer needs: copy_missing's pageable branch copies the whole layer into slots [0, E) with position == expert id
    # stage the state materialize_layer would (its kernel is CUDA-only; the fixture cache lives on the CPU)
    cache, sources = _make_split_cache(num_layers=2, locked=(1,))

    cache._pending_src_layer = 1
    cache._pending_whole_layer = True
    cache.copy_missing()

    gate_up_cache, down_cache = (c for _, c in cache.banks)
    assert torch.equal(gate_up_cache[:4], sources["gate_up"][1])
    assert torch.equal(down_cache[:4], sources["down"][1])
    # (The pinned layers' staged JIT path is covered by the mocked tests above.)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_copy_plan_skips_locked_layers_and_keeps_fused_path():
    # _build_copy_plan must not resolve a device alias for a LOCKED layer; its descriptor row stays a 0 placeholder while the pinned layers keep the fused path
    cache, _ = _make_split_cache(num_layers=2, locked=(1,), device="cuda")

    assert cache._copy_fused_ok
    assert (cache._copy_src_ptrs[1] == 0).all(), "locked layer row must stay 0"
    assert (cache._copy_src_ptrs[0] != 0).all(), "pinned layer rows must resolve"


def test_locked_layer_copy_missing_rejects_ensure_experts_staging():
    # the pageable branch presumes materialize_layer's position == expert id; staging via ensure_experts (LRU slot remap) on a locked layer must fail loudly, not gather other experts' weights
    # stage the state ensure_experts would (its kernel is CUDA-only; the fixture cache lives on the CPU)
    cache, _ = _make_split_cache(num_layers=2, locked=(1,))

    cache._pending_src_layer = 1
    cache._pending_whole_layer = False
    with pytest.raises(RuntimeError, match="unpinned"):
        cache.copy_missing()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("memops", [True, False])
def test_locked_hybrid_layer_fetches_through_staging_in_a_graph(monkeypatch, memops):
    # a locked layer's hybrid decode fetches at most max_fetch misses per step into its slots through the pinned staging; the rest go to the CPU
    # memops: the GPU waits on the doorbell with a stream memop, else the gather kernel spins on it
    from types import SimpleNamespace

    from freetoken.moe.locked_fetch import LockedLayerFetch

    monkeypatch.setenv("FREETOKEN_LOCKED_FETCH_THREADS", "2")
    torch.manual_seed(0)
    cache, sources = _make_split_cache(num_layers=2, locked=(1,), device="cuda")
    cache.decode_target = "hybrid"
    cache.locked_hybrid = frozenset({1})
    cache.locked_fetch = LockedLayerFetch(cache, [1], plan={1: (0, 1.0)}, max_fetch=2, executor=SimpleNamespace(core_ids=[], _coord_core=-1, _flag_sync=memops))

    top_k = 3
    ids = torch.zeros(top_k, dtype=torch.int32, device="cuda")
    weights = torch.rand(top_k, device="cuda")
    out = {}

    def step():
        slots, _, cpu_ids = cache.ensure_experts_hybrid(1, ids, weights)
        cache.copy_missing()
        out["slots"], out["cpu"] = slots, cpu_ids

    cache.collect_stats = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    cache.reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    def replay(graph, steps):
        resident = set()
        for _ in range(steps):
            experts = torch.randperm(4)[:top_k].to(torch.int32)
            ids.copy_(experts)
            graph.replay()
            torch.cuda.synchronize()
            slots, cpu_ids = out["slots"].tolist(), out["cpu"].tolist()
            fetched = [e for e, s in zip(experts.tolist(), slots) if s >= 0 and e not in resident]
            assert len(fetched) <= 2
            for e, s, c in zip(experts.tolist(), slots, cpu_ids):
                assert (s >= 0) != (c >= 0)
                if s >= 0:
                    for (src, _), view in zip(cache.banks, cache.pool_bank_views(1)):
                        assert torch.equal(view[s].cpu(), src[1][e])
            resident = {e for e, s in enumerate(cache.slot_for_id[1].tolist()) if s >= 0}

    replay(graph, 12)
    # --moe-collect-stats: the locked layer reports as its own group, and each line clears the counters
    line = cache.decode_stats_line()
    assert "locked 1 layer: hit" in line and "pinned" not in line, line
    assert cache.decode_stats_line() is None
    # a runtime rebuild reallocates evict_slots and the slot caches; the fetch follows and a new capture still copies right
    del graph
    cache.rebuild(12)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    replay(graph, 6)


def test_pin_probe_pins_middle_layers_until_refused(monkeypatch):
    # --moe-cpu-layers auto: deferred banks pin middle-out until the (simulated) quota refuses one; the rest lock, and the margin gives back the last pinned layer
    import freetoken.kernel.pinned as pinned
    import freetoken.moe.host_banks as hb

    monkeypatch.setattr(pinned, "host_register", lambda addr, nbytes: None)
    monkeypatch.setattr(pinned, "host_unregister", lambda addr: None)
    monkeypatch.setattr(hb, "_os_lock", lambda addr, nbytes: None)
    monkeypatch.setattr(hb, "_os_lock_failed", False)
    monkeypatch.setattr(hb, "_sim_pinned", 0)
    monkeypatch.setenv("FREETOKEN_SIMULATE_PIN_CAP_GB", str(3 * hb._BLK / 2**30))
    deferred = [hb.HostResidency.DEFERRED.value] * 6

    def probe(margin):
        banks = {"gate_up": [hb.HostBank((4,), torch.uint8) for _ in range(6)]}
        with hb.requested_residency(deferred) as plan:
            hb.pin_banks(banks)
        labels = hb.probe_pin(plan, 6, margin_bytes=margin)
        for bank in banks["gate_up"]:
            bank.unpin()
        return [i for i, r in enumerate(labels) if r == hb.HostResidency.LOCKED.value]

    # order 2, 3, 1, 4, 0, 5: the quota takes three, layer 4 is refused
    assert probe(0) == [0, 4, 5]
    assert probe(hb._BLK) == [0, 1, 4, 5]


def test_requested_residency_routes_layer_settles(monkeypatch):
    # the ambient plan installed by load_expert_banks must route each layer's banks by label at both slow-path settle points (PinPipeline layer sink, list-valued pin_banks) and record that it was consulted
    # without a plan everything pins
    import freetoken.moe.host_banks as hb

    settled = []
    monkeypatch.setattr(hb.HostBank, "pin", lambda self: settled.append("pin"))
    monkeypatch.setattr(hb.HostBank, "lock", lambda self: settled.append("lock"))
    banks = {
        "gate_up": [hb.HostBank((4,), torch.uint8) for _ in range(3)],
        "down": [hb.HostBank((4,), torch.uint8) for _ in range(3)],
    }
    labels = [
        hb.HostResidency.PINNED.value,
        hb.HostResidency.LOCKED.value,
        hb.HostResidency.PAGEABLE.value,
    ]

    with hb.requested_residency(labels) as plan:
        with hb.PinPipeline() as pins:
            for layer_id in range(3):
                pins(layer_id, {name: per[layer_id] for name, per in banks.items()})
    # the single drain thread settles FIFO: layer 0 pins, layer 1 locks, layer 2 passes
    assert settled == ["pin", "pin", "lock", "lock"]
    assert plan.applied

    settled.clear()
    with hb.requested_residency(labels) as plan:
        hb.pin_banks(banks)
    assert settled == ["pin", "lock", "pin", "lock"]  # per name: layer 0 pin, 1 lock, 2 skip
    assert plan.applied

    settled.clear()
    hb.pin_banks(banks)  # no ambient plan -> every layer pins
    assert settled == ["pin"] * 6


def test_echo_residency_stamps_honored_requests_only():
    # load_expert_banks stamps the request onto the provider's ExpertBanks only when a settle point consulted the plan; an unconsulted plan keeps None (the engine's degrade signal)
    from freetoken.moe.expert_banks import ExpertBanks, _echo_residency
    from freetoken.moe.host_banks import HostResidency, _ResidencyPlan

    labels = [HostResidency.PINNED.value, HostResidency.LOCKED.value]
    banks = ExpertBanks("bf16", {"gate_up": [], "down": []})

    plan = _ResidencyPlan(labels)
    plan.residency_for(1)  # a settle point consulted the plan
    assert _echo_residency(banks, labels, plan).layer_residency == labels

    stale = _ResidencyPlan(labels)  # never consulted -> keep None + warn
    assert _echo_residency(banks, labels, stale).layer_residency is None
    assert _echo_residency(banks, None, None) is banks


def test_lock_failure_downgrades_echoed_residency(monkeypatch):
    # a failed mlock leaves the bank pageable; the plan and the echoed labels must report that instead of the requested LOCKED
    import freetoken.moe.host_banks as hb
    from freetoken.moe.expert_banks import ExpertBanks, _echo_residency

    def boom(addr, nbytes):
        raise OSError(12, "mlock denied")

    monkeypatch.setattr(hb, "_os_lock", boom)
    monkeypatch.setattr(hb, "_os_lock_failed", False)
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")  # keep the pinned layer off CUDA
    labels = [hb.HostResidency.PINNED.value, hb.HostResidency.LOCKED.value]

    banks = {"gate_up": [hb.HostBank((4,), torch.uint8) for _ in range(2)]}
    with hb.requested_residency(labels) as plan:
        hb.pin_banks(banks)
    assert plan.actual == {1: hb.HostResidency.PAGEABLE.value}
    echoed = _echo_residency(ExpertBanks("bf16", {}), labels, plan)
    assert echoed.layer_residency == [
        hb.HostResidency.PINNED.value, hb.HostResidency.PAGEABLE.value,
    ]

    monkeypatch.setattr(hb, "_os_lock_failed", False)
    with hb.requested_residency(labels) as plan2:
        with hb.PinPipeline() as pins:
            pins(1, {"gate_up": hb.HostBank((4,), torch.uint8)})
    assert plan2.actual == {1: hb.HostResidency.PAGEABLE.value}
