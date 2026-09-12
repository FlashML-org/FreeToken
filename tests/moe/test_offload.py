from contextlib import contextmanager
from types import SimpleNamespace

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
    prefill_gate_up_buffer, prefill_down_buffer = cache.prefill_bank_buffers
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


def test_adjust_config_converts_moe_cache_rate_to_cache_size():
    from types import SimpleNamespace

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _adjust_config

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
    # Family, not member: a box with a benchbw profile resolves bf16 experts to hybrid.
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
    assert cache.prefill_bank_buffers == []


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


def test_copy_missing_consumes_the_staged_layer_exactly_once():
    # Regression: _pending_src_layer was never cleared, so a later copy_missing() with
    # nothing freshly staged replayed the PREVIOUS layer's src_indices/evict_slots and
    # overwrote slots that had since been reassigned to another layer. The owner adapter's
    # "nothing staged" guard (inner pending is None) depends on one-shot consumption.
    cache, _ = _make_split_cache(num_layers=2, locked=(1,))

    cache._pending_src_layer = 1
    cache._pending_whole_layer = True
    cache.copy_missing()

    assert cache._pending_src_layer is None
    assert cache._pending_whole_layer is False
    # a second call is an explicit "nothing staged", never a silent replay of layer 1
    with pytest.raises(AssertionError, match="no staged misses"):
        cache.copy_missing()


def test_owner_cache_rebuild_keeps_the_geometry_in_step():
    # Regression: __getattr__ forwarded rebuild() to the inner cache, whose implementation
    # disables prefill_overlap when the new size cannot hold two complete local layers.
    # The frozen geometry kept the old values, so materialize_layer() still took the
    # overlap path and waited on buffers the inner cache no longer had.
    from freetoken.moe.offload_cache import OwnerOffloadMoeCache
    from freetoken.moe.ownership import OwnerCacheGeometry

    _init_tp()
    geometry = OwnerCacheGeometry(
        global_num_experts=8, world_size=2, rank=0, num_layers=1,
        cache_size=8, prefill_overlap=True,
    )
    owner = OwnerOffloadMoeCache(geometry, torch.device("cpu"))
    owner.set_bank_sources(
        {"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]}
    )
    assert owner.geometry.prefill_overlap is True

    owner.rebuild(5)  # 5 < 2*local_num_experts (8) -> the inner cache drops overlap

    assert owner._cache.prefill_overlap is False
    assert owner.geometry.prefill_overlap is False, "geometry must follow the inner cache"
    assert owner.geometry.cache_size == 5
    assert owner._cache.cache_size == 5


def test_owner_cache_rebuild_keeps_overlap_when_the_new_size_still_fits():
    from freetoken.moe.offload_cache import OwnerOffloadMoeCache
    from freetoken.moe.ownership import OwnerCacheGeometry

    _init_tp()
    geometry = OwnerCacheGeometry(
        global_num_experts=8, world_size=2, rank=0, num_layers=1,
        cache_size=8, prefill_overlap=True,
    )
    owner = OwnerOffloadMoeCache(geometry, torch.device("cpu"))
    owner.set_bank_sources(
        {"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]}
    )

    owner.rebuild(8)  # exactly 2*local_num_experts -> overlap survives

    assert owner.geometry.prefill_overlap is True
    assert owner.geometry.cache_size == 8


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


def _owner_nvfp4_banks(num_layers, local_experts, out, inner, base):
    """Owner-local nvfp4 banks; row `e` of layer `l` carries fingerprint base+l*E+e."""

    def bank(o, i, dtype):
        layers = []
        for l in range(num_layers):
            t = torch.zeros(local_experts, o, i, dtype=dtype)
            for e in range(local_experts):
                t[e].view(torch.uint8).fill_(base + l * local_experts + e)
            layers.append(t)
        return layers

    def pinned(layers):
        return [t.pin_memory() for t in layers]

    return {
        "gate_up_packed": pinned(bank(out, inner // 2, torch.uint8)),
        "gate_up_scale": pinned(bank(out, inner // 16, torch.float8_e4m3fn)),
        "gate_up_global": pinned(
            [t.squeeze(-1).contiguous() for t in bank(out, 1, torch.float16)]
        ),
        "down_packed": pinned(bank(out, inner // 2, torch.uint8)),
        "down_scale": pinned(bank(out, inner // 16, torch.float8_e4m3fn)),
        "down_global": pinned(
            [t.squeeze(-1).contiguous() for t in bank(out, 1, torch.float16)]
        ),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_owner_offload_cache_cuda_route_copies_local_rows_through_real_kernels():
    """P3 namespace smoke on real GPU kernels: global route -> owner-local bank row ->
    legacy slot cache. Remote entries must never read a bank row, and a cache hit must
    not re-copy. No model is loaded; the banks are synthetic."""
    from freetoken.moe.offload_cache import OwnerOffloadMoeCache
    from freetoken.moe.ownership import OwnerCacheGeometry

    L, E_GLOBAL, E_LOCAL, S = 2, 8, 4, 4
    OUT, IN = 64, 512  # rows >= 128B so the fast_index_copy JIT has a kernel
    dev = torch.device("cuda")
    geometry = OwnerCacheGeometry(
        global_num_experts=E_GLOBAL, world_size=2, rank=1, num_layers=L, cache_size=S
    )
    cache = OwnerOffloadMoeCache(geometry, dev, quant_format="nvfp4")
    cache.set_bank_sources(_owner_nvfp4_banks(L, E_LOCAL, OUT, IN, base=100))
    cache.reset()

    def fingerprint(slot):
        packed = cache.bank_caches["gate_up_packed"]
        return int(packed[slot].view(torch.uint8).flatten()[0].item())

    # rank 1 owns global [4, 8) -> local rows [0, 4).
    ids = torch.tensor([[0, 4, 7, 5, 4]], dtype=torch.int32, device=dev)
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.15, 0.25]], device=dev)
    update = cache.ensure_route(0, weights, ids)
    cache.copy_missing()
    torch.cuda.synchronize()

    assert update.owned_mask.tolist() == [[False, True, True, True, True]]
    assert update.local_ids.tolist() == [[0, 0, 3, 1, 0]]
    # remote (global 0) stays a safe placeholder: slot 0, zero weight, never read.
    assert int(update.slot_ids[0, 0].item()) == 0
    assert float(update.weights[0, 0].item()) == 0.0
    # flashlib emits the miss set in ascending local-row order (the CPU reference
    # adapter emits route order) -- only the SET is part of the contract.
    assert sorted(update.missing_local_ids.tolist()) == [0, 1, 3]

    # each owned position's slot holds ITS OWN local row's bytes (layer 0 -> base 100).
    owned_local = update.local_ids[update.owned_mask].tolist()
    owned_slots = update.slot_ids[update.owned_mask].tolist()
    assert [fingerprint(s) for s in owned_slots] == [100 + e for e in owned_local]
    cache.validate_invariants()

    # A repeated route is a pure hit: no miss, no eviction, identical bytes.
    again = cache.ensure_route(0, weights, ids)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert again.missing_local_ids.numel() == 0
    assert again.evicted_flat_ids.numel() == 0
    assert again.slot_ids.tolist() == update.slot_ids.tolist()
    assert [fingerprint(s) for s in owned_slots] == [100 + e for e in owned_local]

    # The pool is unified: layer 1 must evict layer-0 entries and serve layer-1 bytes
    # (base 100 + 4) without ever mixing the two layers' rows.
    ids1 = torch.tensor([[4, 6]], dtype=torch.int32, device=dev)
    update1 = cache.ensure_route(1, torch.ones(1, 2, device=dev), ids1)
    cache.copy_missing()
    torch.cuda.synchronize()
    slots1 = update1.slot_ids.reshape(-1).tolist()
    assert [fingerprint(s) for s in slots1] == [104 + e for e in [0, 2]]
    assert int(update1.evicted_flat_ids.numel()) > 0  # S=6 < 2 layers * 4 local rows
    cache.validate_invariants()

    # Remote-only route: no admission, no copy, all placeholders.
    remote_only = cache.ensure_route(
        1, torch.full((1, 2), 0.5, device=dev),
        torch.tensor([[0, 1]], dtype=torch.int32, device=dev),
    )
    assert remote_only.missing_local_ids.numel() == 0
    assert remote_only.slot_ids.tolist() == [[0, 0]]
    assert remote_only.weights.tolist() == [[0.0, 0.0]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_owner_route_graph_admission_matches_eager_slots_and_zeroes_remote():
    """The graph-safe admission must place every OWNED route entry on the same local row as
    the eager compacting path, keep remote entries at zero weight, and keep the route shape
    statically known (that is what makes it capturable).  Remote entries point at the
    sentinel row instead of being dropped, so they must be resident and finite -- the zero
    weighting must never rely on ``0 * NaN``."""
    from freetoken.moe.offload_cache import OwnerOffloadMoeCache
    from freetoken.moe.ownership import OwnerCacheGeometry

    L, E_GLOBAL, E_LOCAL, S = 2, 8, 4, 8
    OUT, IN = 64, 512
    dev = torch.device("cuda")
    geometry = OwnerCacheGeometry(
        global_num_experts=E_GLOBAL, world_size=2, rank=1, num_layers=L, cache_size=S
    )

    def build(graph_safe):
        c = OwnerOffloadMoeCache(geometry, dev, quant_format="nvfp4", graph_safe=graph_safe)
        c.set_bank_sources(_owner_nvfp4_banks(L, E_LOCAL, OUT, IN, base=100))
        c.reset()
        return c

    def fingerprint(cache, slot):
        packed = cache.bank_caches["gate_up_packed"]
        return int(packed[slot].view(torch.uint8).flatten()[0].item())

    # rank 1 owns global [4, 8) -> local rows [0, 4); global 0/1 are remote.
    # Route chosen so the FIRST owned position is local row 3, not row 0: that makes the
    # remote fallback distinguishable from a fixed row-zero sentinel.
    ids = torch.tensor([[0, 7, 5, 6, 4]], dtype=torch.int32, device=dev)
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.15, 0.25]], device=dev)

    eager = build(graph_safe=False)
    up_eager = eager.ensure_route(0, weights, ids)
    eager.copy_missing()
    torch.cuda.synchronize()

    graph = build(graph_safe=True)
    assert graph.graph_safe is True
    up_graph = graph.ensure_route_graph(0, weights, ids)
    graph.copy_missing()
    torch.cuda.synchronize()

    # Weights agree exactly: remote positions are zero in both paths.
    assert up_graph.weights.tolist() == up_eager.weights.tolist()
    assert up_graph.weights[0, 0].item() == 0.0
    assert up_graph.owned_mask.tolist() == up_eager.owned_mask.tolist()

    # Shape/dtype are static, and the diagnostics contract is "empty, never a host read".
    assert up_graph.slot_ids.shape == ids.shape
    assert up_graph.slot_ids.dtype == torch.int32
    assert up_graph.missing_local_ids.numel() == 0
    assert up_graph.evicted_flat_ids.numel() == 0

    # The remote entry borrows the first owned row (row 3), NOT a fixed row-zero sentinel.
    local_row = up_graph.local_ids[0].tolist()
    assert local_row == [3, 3, 1, 2, 0]
    owned = up_graph.owned_mask[0].tolist()
    slots = up_graph.slot_ids[0].tolist()
    # The two admission paths must place every owned entry in the same slot: sharing the row
    # set is what keeps the graph path from changing cache behaviour (extra misses).
    assert [s for s, o in zip(slots, owned) if o] == [
        s for s, o in zip(up_eager.slot_ids[0].tolist(), owned) if o
    ]
    # Every owned position carries ITS OWN row's bytes; the remote position carries row 3's.
    assert [fingerprint(graph, s) for s, o in zip(slots, owned) if o] == [
        100 + r for r, o in zip(local_row, owned) if o
    ]
    assert fingerprint(graph, slots[0]) == 103
    assert torch.isfinite(graph.bank_caches["gate_up_packed"][slots[0]].float()).all()
    graph.validate_invariants()

    # All-remote route (no owned row to borrow) falls back to row zero and contributes
    # nothing -- the only case that admits an extra row.
    only_remote = graph.ensure_route_graph(
        1, torch.full((1, 2), 0.5, device=dev),
        torch.tensor([[0, 1]], dtype=torch.int32, device=dev),
    )
    graph.copy_missing()
    torch.cuda.synchronize()
    assert only_remote.weights.tolist() == [[0.0, 0.0]]
    assert set(only_remote.local_ids[0].tolist()) == {0}
    assert fingerprint(graph, only_remote.slot_ids[0, 0].item()) == 104  # layer 1, row 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_owner_route_graph_admission_is_capturable_and_replays():
    """The point of the graph-safe path: capture admission + copy into a real CUDA graph and
    replay it.  The eager path's ``nonzero``/``num_indices.item()`` make this fail, which is
    why owner EP used to be restricted to ``--cuda-graph-max-bs 0``."""
    from freetoken.moe.offload_cache import OwnerOffloadMoeCache
    from freetoken.moe.ownership import OwnerCacheGeometry

    L, E_GLOBAL, E_LOCAL, S = 2, 8, 4, 8
    OUT, IN = 64, 512
    dev = torch.device("cuda")
    geometry = OwnerCacheGeometry(
        global_num_experts=E_GLOBAL, world_size=2, rank=1, num_layers=L, cache_size=S
    )
    cache = OwnerOffloadMoeCache(geometry, dev, quant_format="nvfp4", graph_safe=True)
    cache.set_bank_sources(_owner_nvfp4_banks(L, E_LOCAL, OUT, IN, base=100))
    cache.reset()

    ids = torch.tensor([[0, 4, 7, 5, 4]], dtype=torch.int32, device=dev)
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.15, 0.25]], device=dev)

    def step(ids_buf, weights_buf):
        update = cache.ensure_route_graph(0, weights_buf, ids_buf)
        cache.copy_missing()
        return update

    # Warm the kernels on a side stream so capture starts from a steady state.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step(ids.clone(), weights.clone())
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    ids_buf = ids.clone()
    weights_buf = weights.clone()
    with torch.cuda.graph(graph):
        captured = step(ids_buf, weights_buf)

    graph.replay()
    torch.cuda.synchronize()

    packed = cache.bank_caches["gate_up_packed"]
    slots = captured.slot_ids[0].tolist()
    rows = captured.local_ids[0].tolist()
    owned = captured.owned_mask[0].tolist()
    assert captured.weights[0, 0].item() == 0.0
    assert [int(packed[s].view(torch.uint8).flatten()[0].item())
            for s, o in zip(slots, owned) if o] == [
        100 + r for r, o in zip(rows, owned) if o
    ]
    cache.validate_invariants()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_owner_prefill_materialize_copies_local_rows_and_does_not_leak_layers():
    """Owner prefill must move bytes, not just remap slots: materialize layer 0, check
    every local row's fingerprint, then materialize layer 1 and prove no layer-0 bytes
    remain in the materialized view."""
    from freetoken.moe.offload_cache import OwnerOffloadMoeCache
    from freetoken.moe.ownership import OwnerCacheGeometry

    L, E_GLOBAL, E_LOCAL, S = 2, 8, 4, 4
    OUT, IN = 64, 512
    dev = torch.device("cuda")
    geometry = OwnerCacheGeometry(
        global_num_experts=E_GLOBAL, world_size=2, rank=1, num_layers=L, cache_size=S
    )
    cache = OwnerOffloadMoeCache(geometry, dev, quant_format="nvfp4")
    cache.set_bank_sources(_owner_nvfp4_banks(L, E_LOCAL, OUT, IN, base=100))
    cache.reset()

    def fingerprint(slot):
        packed = cache.bank_caches["gate_up_packed"]
        return int(packed[slot].view(torch.uint8).flatten()[0].item())

    cache.materialize_layer(0)
    torch.cuda.synchronize()
    assert [fingerprint(s) for s in range(E_LOCAL)] == [100 + e for e in range(E_LOCAL)]

    cache.materialize_layer(1)
    torch.cuda.synchronize()
    assert [fingerprint(s) for s in range(E_LOCAL)] == [104 + e for e in range(E_LOCAL)]
    cache.validate_invariants()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_owner_prefill_overlap_keeps_two_layers_resident_at_once():
    """Owner prefill overlap must stream layer L+1 while layer L computes.  The tell is
    that the two borrowed buffers hold DIFFERENT layers SIMULTANEOUSLY after
    ``prefetch(0) -> prefetch(1)`` and before any release -- a choreography that only
    prefetched the current layer would leave buffer 1 untouched here."""
    from freetoken.moe.offload_cache import OwnerOffloadMoeCache
    from freetoken.moe.ownership import OwnerCacheGeometry

    L, E_GLOBAL, E_LOCAL = 2, 8, 4
    S = 2 * E_LOCAL  # the overlap floor: two full local layers
    OUT, IN = 64, 512
    dev = torch.device("cuda")
    geometry = OwnerCacheGeometry(
        global_num_experts=E_GLOBAL, world_size=2, rank=1, num_layers=L,
        cache_size=S, prefill_overlap=True,
    )
    cache = OwnerOffloadMoeCache(geometry, dev, quant_format="nvfp4")
    cache.set_bank_sources(_owner_nvfp4_banks(L, E_LOCAL, OUT, IN, base=100))
    cache.reset()
    assert cache.geometry.prefill_overlap is True
    assert cache.prefill_overlap is True  # forwarded to the wrapped cache

    def fingerprint(buf, row):
        return int(buf[row].view(torch.uint8).flatten()[0].item())

    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)  # -> buffer 0
    cache.prefetch_prefill_layer(1)  # -> buffer 1, issued while layer 0 will compute
    torch.cuda.synchronize()

    buffers = cache.prefill_bank_buffers[0]  # bank 0, [2, E_LOCAL, ...]
    assert [fingerprint(buffers[0], r) for r in range(E_LOCAL)] == [
        100 + r for r in range(E_LOCAL)
    ]
    # Layer 1 is already staged in the OTHER buffer: that is the overlap.
    assert [fingerprint(buffers[1], r) for r in range(E_LOCAL)] == [
        104 + r for r in range(E_LOCAL)
    ]

    # The hand-off the layer performs: wait(cur) returns cur's buffer, release frees it.
    views0 = cache.wait_prefill_layer(0)
    assert [fingerprint(views0[0], r) for r in range(E_LOCAL)] == [
        100 + r for r in range(E_LOCAL)
    ]
    cache.release_prefill_layer(0)
    views1 = cache.wait_prefill_layer(1)
    assert [fingerprint(views1[0], r) for r in range(E_LOCAL)] == [
        104 + r for r in range(E_LOCAL)
    ]
    cache.release_prefill_layer(1)

    # A second prefill over the same buffers must not leak the previous layer's bytes.
    cache.begin_prefill()
    cache.prefetch_prefill_layer(1)
    torch.cuda.synchronize()
    assert [fingerprint(cache.prefill_bank_buffers[0][1], r) for r in range(E_LOCAL)] == [
        104 + r for r in range(E_LOCAL)
    ]
    cache.validate_invariants()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "route_ids",
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 7],  # 5-5 split across the two owners
        [0] * 10,  # rank 0 owns every entry
        [4] * 10,  # rank 1 owns every entry
    ],
)
def test_owner_prefill_gemm_partials_sum_to_tp1(route_ids):
    """Two owner-local prefill GEMMs must sum to the TP1 GEMM for the same route.

    Each rank materializes its own local layer, remaps the global route to local rows
    with remote entries zero-weighted, and runs the real Triton NVFP4 prefill kernel.
    The TP1 reference runs the same kernel over the full 8-expert layer."""
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4
    from freetoken.moe.offload_cache import OffloadMoeCache, OwnerOffloadMoeCache
    from freetoken.moe.ownership import OwnerCacheGeometry

    L, E_GLOBAL, E_LOCAL = 1, 8, 4
    OUT, IN = 64, 512
    dev = torch.device("cuda")
    hidden = torch.randn(1, OUT, dtype=torch.bfloat16, device=dev) / 4
    weights = torch.arange(1, 11, dtype=torch.float32, device=dev).reshape(1, 10) / 55
    ids = torch.tensor([route_ids], dtype=torch.int32, device=dev)

    def random_sources(num_experts, seed):
        g = torch.Generator().manual_seed(seed)
        total = L * num_experts

        def rand_u8(*shape):
            return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

        def rand_scale(*shape):
            return (torch.rand(*shape, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)

        flat = {
            "gate_up_packed": rand_u8(total, 2 * IN, OUT // 2),
            "gate_up_scale": rand_scale(total, 2 * IN, OUT // 16),
            "gate_up_global": torch.full((total, 2 * IN), 1.0, dtype=torch.float16),
            "down_packed": rand_u8(total, OUT, IN // 2),
            "down_scale": rand_scale(total, OUT, IN // 16),
            "down_global": torch.full((total, OUT), 0.75, dtype=torch.float16),
        }
        return {name: list(t.pin_memory().split(num_experts)) for name, t in flat.items()}

    full_sources = random_sources(E_GLOBAL, seed=7)
    tp1 = OffloadMoeCache(
        num_layers=L, num_experts=E_GLOBAL, cache_size=E_GLOBAL, device=dev,
        quant_format="nvfp4",
    )
    tp1.set_bank_sources(full_sources)
    tp1.reset()
    tp1.materialize_layer(0)
    tp1.copy_missing()
    want = fused_experts_nvfp4(
        hidden, *tp1.bank_views(E_GLOBAL), weights, ids, E_GLOBAL, "silu", False,
    )

    partials = []
    for rank in range(2):
        geometry = OwnerCacheGeometry(
            global_num_experts=E_GLOBAL, world_size=2, rank=rank, num_layers=L,
            cache_size=E_LOCAL,
        )
        owner = OwnerOffloadMoeCache(geometry, dev, quant_format="nvfp4")
        local_sources = {
            name: [
                full_sources[name][0][rank * E_LOCAL:(rank + 1) * E_LOCAL]
                .clone()
                .pin_memory()
            ]
            for name in full_sources
        }
        owner.set_bank_sources(local_sources)
        owner.reset()
        owner.materialize_layer(0)
        torch.cuda.synchronize()
        local_ids, owned = geometry.global_to_local(ids)
        safe_ids = torch.where(owned, local_ids, torch.zeros_like(local_ids)).contiguous()
        safe_weights = torch.where(
            owned, weights, torch.zeros_like(weights)
        ).contiguous()
        partials.append(
            fused_experts_nvfp4(
                hidden, *owner.bank_views(E_LOCAL), safe_weights, safe_ids, E_LOCAL,
                "silu", False,
            )
        )
    got = (partials[0] + partials[1]).float()
    ref = want.float()
    tol = 0.03 * float(ref.abs().max())
    torch.testing.assert_close(got, ref, rtol=3e-2, atol=max(tol, 3e-2))


class _RecordingMoEMethod:
    """Stand-in for the layer's MoE quant method: records what ``_expert_gemm`` hands the
    kernel (bank views + routing ids) and returns the hidden states unchanged."""

    def __init__(self):
        self.calls = []

    def apply(self, hidden_states, topk_weights, topk_ids, view, *, layer, is_prefill):
        self.calls.append(
            SimpleNamespace(
                weights=topk_weights,
                ids=topk_ids,
                views=view.tensors,
                n=view.n,
                is_prefill=is_prefill,
            )
        )
        return hidden_states


def _make_owner_layer(quant_format="bf16", prefill_overlap=False):
    """OffloadMoELayer wired to an OwnerOffloadMoeCache with tiny local banks."""
    from freetoken.moe.offload_cache import OwnerOffloadMoeCache
    from freetoken.moe.ownership import OwnerCacheGeometry

    _init_tp()
    layer = _bf16_offload_layer(0, 8, 2, 8, 16)
    geometry = OwnerCacheGeometry(
        global_num_experts=8, world_size=2, rank=1, num_layers=1,
        cache_size=8, prefill_overlap=prefill_overlap,
    )
    owner = OwnerOffloadMoeCache(geometry, torch.device("cpu"), quant_format=quant_format)
    if quant_format == "bf16":
        owner.set_bank_sources({
            "gate_up": [torch.randn(4, 32, 8)],
            "down": [torch.randn(4, 8, 16)],
        })
    else:
        owner.set_bank_sources(_owner_nvfp4_banks(1, 4, 64, 512, base=100))
    layer.owner_cache = owner
    layer.offload_cache = owner._cache
    # the owner path is exercised through the layer's kernel seam, so record there
    layer.quant_method = _RecordingMoEMethod()
    return layer, owner


def test_owner_layer_decode_uses_owner_route_and_never_the_global_ids(monkeypatch):
    """P3 wiring: an attached owner cache must route decode through ``ensure_route`` and
    feed the kernel the LOCAL slot ids + masked weights, never the raw global ids."""
    layer, owner = _make_owner_layer()
    # rank 1 owns global [4, 8) -> local rows [0, 4).
    topk_weights = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[0, 4, 7]], dtype=torch.int32)
    hidden_states = torch.randn(1, 8)
    calls = {}

    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (topk_weights, topk_ids),
    )
    # The global-ID entry point must NOT be reached on the owner path.
    monkeypatch.setattr(
        owner._cache, "ensure_experts",
        lambda *a, **k: pytest.fail("owner path called the global-ID ensure_experts"),
    )

    def fake_update(layer_id, weights, expert_ids):
        owned = expert_ids >= 4
        slots = torch.where(owned, expert_ids - 4, torch.zeros_like(expert_ids))
        return SimpleNamespace(
            weights=torch.where(owned, weights, torch.zeros_like(weights)),
            slot_ids=slots,
        )

    monkeypatch.setattr(owner, "ensure_route", fake_update)
    monkeypatch.setattr(owner, "copy_missing", lambda: None)

    original_route = owner.ensure_route

    def fake_route(layer_id, weights, expert_ids):
        calls["route_args"] = (layer_id, weights.clone(), expert_ids.clone())
        return original_route(layer_id, weights, expert_ids)

    monkeypatch.setattr(owner, "ensure_route", fake_route)

    out = layer.decode_forward(hidden_states, torch.randn(1, 8))

    call = layer.quant_method.calls[-1]
    assert out is hidden_states
    assert calls["route_args"][0] == 0
    assert calls["route_args"][2].tolist() == [[0, 4, 7]]  # raw global ids reached the adapter
    # remote (global 0) is zero-weighted; owned entries keep their global weights.
    assert torch.allclose(call.weights, torch.tensor([[0.0, 0.2, 0.3]]))
    # ids handed to the kernel are LOCAL slots, strictly inside the local pool.
    assert call.ids.dtype == torch.int32
    assert int(call.ids.min()) >= 0
    assert int(call.ids.max()) < owner.cache_size
    # the banks the kernel reads are the owner-local ones (4 rows), not the global 8.
    assert call.views["gate_up"].shape[0] == owner.cache_size
    assert call.views["down"].shape[0] == owner.cache_size


def test_owner_layer_remote_only_route_zeroes_the_contribution(monkeypatch):
    """A rank that owns none of the routed experts must emit an all-zero, in-range route."""
    layer, owner = _make_owner_layer()
    topk_weights = torch.full((1, 2), 0.5, dtype=torch.float32)
    topk_ids = torch.tensor([[0, 2]], dtype=torch.int32)  # both owned by rank 0
    calls = {}

    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (topk_weights, topk_ids),
    )
    monkeypatch.setattr(
        owner,
        "ensure_route",
        lambda layer_id, weights, expert_ids: SimpleNamespace(
            weights=torch.zeros_like(weights),
            slot_ids=torch.zeros_like(expert_ids),
        ),
    )
    monkeypatch.setattr(owner, "copy_missing", lambda: None)

    layer.decode_forward(torch.randn(1, 8), torch.randn(1, 8))

    call = layer.quant_method.calls[-1]
    assert torch.equal(call.weights, torch.zeros_like(topk_weights))
    assert call.ids.tolist() == [[0, 0]]
    assert owner.resident == 0  # nothing was admitted


def test_owner_layer_prefill_remaps_global_ids_to_local_rows(monkeypatch):
    """P3 prefill: bank row ids must be LOCAL rows with remote entries zero-weighted."""
    layer, owner = _make_owner_layer(prefill_overlap=False)
    topk_weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    topk_ids = torch.tensor([[1, 6]], dtype=torch.int32)  # 1 = rank0, 6 = local row 2
    calls = {}
    monkeypatch.setattr(owner, "materialize_layer", lambda layer_id, buffer_id=0: None)
    monkeypatch.setattr(owner, "bank_views", lambda n=None: (torch.empty(8, 32, 8), torch.empty(8, 8, 16)))

    layer._prefill_routed(torch.randn(1, 8), topk_weights, topk_ids)

    call = layer.quant_method.calls[-1]
    assert torch.allclose(call.weights, torch.tensor([[0.0, 0.75]]))
    assert call.ids.tolist() == [[0, 2]]  # global 6 -> local row 2
    assert int(call.ids.max()) < owner.num_experts


def test_owner_layer_prefill_overlap_waits_and_releases_borrowed_buffer(monkeypatch):
    """The owner prefill path must use the same borrowed-buffer lifecycle as the global
    cache, INCLUDING the one-layer lookahead: prefetch(cur) stages this layer, prefetch(next)
    starts the following layer's H2D on the copy stream so it runs while this layer's GEMMs
    run on the compute stream.  Without the lookahead the copy is issued and immediately
    waited on, which serializes the two and defeats the overlap."""
    layer, owner = _make_owner_layer(prefill_overlap=True)
    topk_weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    topk_ids = torch.tensor([[1, 6]], dtype=torch.int32)
    calls = {}
    lifecycle = []
    monkeypatch.setattr(owner, "begin_prefill", lambda: lifecycle.append("begin"))
    monkeypatch.setattr(owner, "prefetch_prefill_layer", lambda layer_id: lifecycle.append(("prefetch", layer_id)))
    monkeypatch.setattr(
        owner,
        "wait_prefill_layer",
        lambda layer_id: (torch.empty(8, 32, 8), torch.empty(8, 8, 16)),
    )
    monkeypatch.setattr(owner, "release_prefill_layer", lambda layer_id: lifecycle.append("release"))

    layer._prefill_routed(torch.randn(1, 8), topk_weights, topk_ids)

    call = layer.quant_method.calls[-1]
    assert torch.allclose(call.weights, torch.tensor([[0.0, 0.75]]))
    assert call.ids.tolist() == [[0, 2]]
    assert call.n == owner.num_experts
    # layer 0 -> the lookahead asks for layer 1 (a no-op past the last layer).
    assert lifecycle == ["begin", ("prefetch", 0), ("prefetch", 1), "release"]


def test_owner_wrapper_forwards_engine_assigned_flags_to_the_inner_cache():
    """Engine sets these by assignment; without the properties they would land on the
    wrapper and silently disable stats / the route trace / CPU-layer routing."""
    layer, owner = _make_owner_layer()
    owner.collect_stats = True
    owner.collect_decode_freq = True
    owner.route_recorder = object()
    owner.cpu_layer_ids = frozenset({1})
    assert owner._cache.collect_stats is True
    assert owner._cache.collect_decode_freq is True
    assert owner._cache.route_recorder is not None
    assert owner._cache.cpu_layer_ids == frozenset({1})
    assert owner.collect_stats is True and owner.cpu_layer_ids == frozenset({1})


def test_owner_cache_bank_slots_start_zero_and_finite():
    """Slot-zero remote placeholders must not read torch.empty/NaN data."""
    _layer, owner = _make_owner_layer()
    for cache in owner._cache.bank_caches.values():
        assert torch.count_nonzero(cache).item() == 0
        assert torch.isfinite(cache).all().item()


def test_attach_owner_moe_cache_wires_layers_and_keeps_global_banks():
    from freetoken.layers import BaseOP
    from freetoken.moe.offload_cache import attach_owner_moe_cache

    layer, owner = _make_owner_layer()
    model = BaseOP()
    model.block = layer
    layers = attach_owner_moe_cache(model, owner)

    assert layers == [layer]
    assert layer.owner_cache is owner
    assert layer.offload_cache is owner
