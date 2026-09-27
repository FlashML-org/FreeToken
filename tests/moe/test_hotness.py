import json

import numpy as np
import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info


def _init_tp():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _make_hotness(tmp_path, num_layers=2, num_experts=4, flush_interval_s=60.0, device="cpu"):
    from freetoken.moe.hotness import ExpertHotness

    return ExpertHotness(
        num_layers=num_layers,
        num_experts=num_experts,
        device=torch.device(device),
        out_path=str(tmp_path / "hotstats.json"),
        flush_interval_s=flush_interval_s,
        meta={"model_path": "/tmp/model", "moe_strategy": "hybrid", "quant_format": "nvfp4", "top_k": 2},
    )


def test_record_accumulates_per_layer_expert_counts(tmp_path):
    """随机 topk_ids 的累计与 numpy 对照一致（含跨层平坦 id 空间映射）。"""
    hot = _make_hotness(tmp_path)
    rng = np.random.default_rng(7)
    expected = np.zeros((hot.num_layers, hot.num_experts), dtype=np.int64)
    for layer_id in range(hot.num_layers):
        for _ in range(5):
            topk_ids = torch.from_numpy(rng.integers(0, hot.num_experts, size=(3, 2)).astype(np.int32))
            hot.record(layer_id, topk_ids)
            expected[layer_id] += np.bincount(
                topk_ids.numpy().reshape(-1), minlength=hot.num_experts
            )
    got = hot.counts.view(hot.num_layers, hot.num_experts).numpy()
    assert np.array_equal(got, expected)
    assert hot.total_tokens == 5 * 3 * hot.num_layers


def test_save_load_roundtrip_schema(tmp_path):
    """save 写出的 JSON 带 schema_version=1 与完整 meta，load 校验后原样读回。"""
    hot = _make_hotness(tmp_path)
    hot.record(0, torch.tensor([[0, 2], [2, 2]], dtype=torch.int32))
    hot.record(1, torch.tensor([[3, 1]], dtype=torch.int32))
    hot._host_counts += hot.counts.numpy()  # 模拟一次排空，save 只读宿主累计
    hot.save()

    assert not (tmp_path / "hotstats.json.tmp").exists()  # 原子替换后无残留
    payload = json.loads((tmp_path / "hotstats.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    meta = payload["meta"]
    assert meta["num_layers"] == 2 and meta["num_experts"] == 4
    assert meta["total_tokens"] == hot.total_tokens == 3
    assert meta["moe_strategy"] == "hybrid" and meta["quant_format"] == "nvfp4"
    assert meta["model_path"] == "/tmp/model" and meta["top_k"] == 2
    assert isinstance(meta["created_at"], str) and isinstance(meta["duration_s"], (int, float))
    assert meta["created_at"].startswith("20")  # ISO8601 本地时间

    loaded = hot.load(str(tmp_path / "hotstats.json"))
    assert loaded["schema_version"] == 1
    assert loaded["meta"] == meta
    expected = hot._host_counts.reshape(2, 4).tolist()
    assert loaded["counts"] == expected

    # save 前自动排空 device 计数器：重复 save 不丢数据也不重复计数
    hot.save()
    again = hot.load(str(tmp_path / "hotstats.json"))
    assert again["counts"] == expected


@pytest.mark.parametrize("bad", ["version", "dims", "shape"])
def test_load_rejects_malformed_stats(tmp_path, bad):
    """load 对版本不符与维度不一致的文件抛 ValueError。"""
    from freetoken.moe.hotness import ExpertHotness

    payload = {
        "schema_version": 1,
        "meta": {"num_layers": 2, "num_experts": 4},
        "counts": [[0, 0, 0, 0], [0, 0, 0, 0]],
    }
    if bad == "version":
        payload["schema_version"] = 99
    elif bad == "dims":
        payload["meta"]["num_layers"] = 3
    else:
        payload["counts"][1] = [0, 0, 0]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        ExpertHotness.load(str(path))


def test_maybe_flush_is_time_gated(monkeypatch, tmp_path):
    """间隔未到返回 False 且不动数据；到达后排空累加、清零 device，且刷新门控。"""
    hot = _make_hotness(tmp_path, flush_interval_s=60.0)
    clock = {"t": hot._last_flush}
    monkeypatch.setattr("freetoken.moe.hotness.time.monotonic", lambda: clock["t"])

    hot.record(0, torch.tensor([[0, 1], [1, 3]], dtype=torch.int32))
    assert hot.maybe_flush() is False
    assert hot._host_counts.sum() == 0 and hot.counts.sum() != 0

    clock["t"] += 61.0
    assert hot.maybe_flush() is True
    assert hot.counts.sum() == 0  # device 侧已清零
    assert np.array_equal(hot._host_counts.reshape(2, 4)[0], np.array([1, 2, 0, 1]))

    # 刚排空过：门控重新计时
    hot.record(1, torch.tensor([[2, 2]], dtype=torch.int32))
    clock["t"] += 10.0
    assert hot.maybe_flush() is False


def test_ones_buffer_grows_and_reuses(tmp_path):
    """_ones 按需扩展缓存 buffer，同长度请求复用同一存储。"""
    hot = _make_hotness(tmp_path)
    ones = hot._ones(4)
    assert hot._ones(4).data_ptr() == ones.data_ptr() and hot._ones(2).numel() == 2
    grown = hot._ones(9)
    assert grown.numel() == 9 and bool(torch.all(grown == 1))


def test_offload_cache_hotness_defaults_none_and_record_path(tmp_path):
    """OffloadMoeCache.hotness 默认 None；挂上后 record 计入 cache 上的收集器。"""
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    cache = OffloadMoeCache(num_layers=2, num_experts=4, cache_size=6, device=torch.device("cpu"))
    assert cache.hotness is None

    hot = _make_hotness(tmp_path, num_layers=2, num_experts=4)
    cache.hotness = hot
    cache.hotness.record(1, torch.tensor([[0, 3]], dtype=torch.int32))
    assert int(hot.counts[1 * 4 + 0]) == 1 and int(hot.counts[1 * 4 + 3]) == 1


def test_prefill_forward_records_hotness(monkeypatch, tmp_path):
    """prefill_forward 在 fused_topk 之后、路由消费之前调用 hotness.record（CPU 全路径）。"""
    _init_tp()
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig
    from freetoken.moe.offload_cache import OffloadMoeCache

    layer = OffloadMoELayer(
        1, 4, 2, 8, 16, quant_config=NoQuantConfig(), prefix="model.layers.1.mlp.experts"
    )
    cache = OffloadMoeCache(num_layers=2, num_experts=4, cache_size=6, device=torch.device("cpu"))
    cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8) for _ in range(2)], "down": [torch.randn(4, 8, 16) for _ in range(2)]})
    layer.offload_cache = cache
    hot = _make_hotness(tmp_path, num_layers=2, num_experts=4)
    cache.hotness = hot

    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (topk_weights, topk_ids),
    )
    monkeypatch.setattr(cache, "materialize_layer", lambda layer_id: None)
    monkeypatch.setattr(cache, "copy_missing", lambda: None)
    monkeypatch.setattr("freetoken.moe.fused.fused_experts_impl", lambda *a, **k: a[0])

    out = layer.prefill_forward(torch.randn(1, 8), torch.randn(1, 4))
    assert out is not None
    # 第 1 层的专家 1 和 2 各被路由一次；fused_topk 产出的原始 id 未被改写前已计入
    assert int(hot.counts[1 * 4 + 1]) == 1 and int(hot.counts[1 * 4 + 2]) == 1
    assert hot.total_tokens == 1


def test_exit_flush_is_reentrancy_safe(monkeypatch, tmp_path):
    """install_exit_flush 幂等注册；_exit_flush 只落盘一次，失败吞异常。"""
    hot = _make_hotness(tmp_path)
    hot.record(0, torch.tensor([[0, 0]], dtype=torch.int32))
    hot.install_exit_flush()
    assert hot._exit_installed
    calls = []
    monkeypatch.setattr(hot, "save", lambda: calls.append(1))
    hot._exit_flush()
    hot._exit_flush()
    assert calls == [1] and hot._exit_done


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_record_inside_cuda_graph_accumulates_per_replay(tmp_path):
    """graph 冒烟：record 可被 CUDA graph 捕获，replay N 次按真实路由累加 N 次。"""
    hot = _make_hotness(tmp_path, num_layers=2, num_experts=8, device="cuda")
    dev = torch.device("cuda")
    # warmup：预分配 ones buffer（capture 期间不允许再走扩展分支）
    hot.record(0, torch.zeros((4, 2), dtype=torch.int32, device=dev))
    torch.cuda.synchronize()
    baseline = hot.counts.clone()

    topk = torch.full((4, 2), 3, dtype=torch.int32, device=dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hot.record(0, topk)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    delta = hot.counts - baseline
    assert int(delta.sum()) == 40  # 5 次 replay × 8 个路由
    assert int(delta[3]) == 40  # 全部落在专家 3（第 0 层平坦 id 空间）
