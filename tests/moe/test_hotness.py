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


def test_record_uses_fresh_ones_no_shared_buffer(tmp_path):
    """record 的增量不再共享可扩容缓存（多 batch-size graph 捕获下的 UAF 根因）：
    counts 上不暴露 _ones_buf，重复 record 的累加语义不变。"""
    hot = _make_hotness(tmp_path)
    assert not hasattr(hot, "_ones_buf")
    for _ in range(3):
        hot.record(0, torch.tensor([[0, 1]], dtype=torch.int32))
    assert int(hot.counts[0]) == 3 and int(hot.counts[1]) == 3


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


def test_routed_forward_records_hotness_before_slot_rewrite(monkeypatch, tmp_path):
    """外部路由只走 routed_forward。热度在 topk_ids 被改成槽位号之前记下。"""
    _init_tp()
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig
    from freetoken.moe.offload_cache import OffloadMoeCache

    layer = OffloadMoELayer(
        1, 4, 2, 8, 16, quant_config=NoQuantConfig(), prefix="model.layers.1.mlp.experts"
    )
    cache = OffloadMoeCache(num_layers=2, num_experts=4, cache_size=6, device=torch.device("cpu"))
    cache.set_bank_sources(
        {
            "gate_up": [torch.randn(4, 32, 8) for _ in range(2)],
            "down": [torch.randn(4, 8, 16) for _ in range(2)],
        }
    )
    layer.offload_cache = cache
    hot = _make_hotness(tmp_path, num_layers=2, num_experts=4)
    cache.hotness = hot

    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    seen: dict[str, torch.Tensor] = {}

    def _capture(_hidden, _weights, ids):
        seen["ids"] = ids.detach().clone()
        return _hidden

    monkeypatch.setattr(layer, "_decode_routed", _capture)
    monkeypatch.setattr(
        "freetoken.layers.moe.get_global_ctx",
        lambda: type("Ctx", (), {"batch": type("Batch", (), {"is_prefill": False})()})(),
    )
    out = layer.routed_forward(torch.randn(1, 8), topk_weights, topk_ids)
    assert out is not None
    assert int(hot.counts[1 * 4 + 1]) == 1 and int(hot.counts[1 * 4 + 2]) == 1
    assert hot.total_tokens == 1
    assert seen["ids"].tolist() == [[2, 1]]


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
    # CUDA 上 token 数留在设备标量。重放不跑 Python 加法，宿主计数在排空前仍是 0
    assert hot.total_tokens == 0
    hot.flush_interval_s = 0
    assert hot.maybe_flush()
    # warmup 1 次 + 5 次重放，每次 4 token。捕获那一次不留副作用，与 counts 一致
    assert hot.total_tokens == 4 + 5 * 4
    meta = json.loads((tmp_path / "hotstats.json").read_text(encoding="utf-8"))["meta"]
    assert meta["total_tokens"] == hot.total_tokens


# ---------------------------------------------------------------------------
# 动态重钉：滑动窗口 + EMA（设计 §10）
# ---------------------------------------------------------------------------


def _make_window_hotness(num_layers=1, num_experts=4, window_interval_s=10.0, flush_interval_s=3600.0):
    from freetoken.moe.hotness import ExpertHotness

    return ExpertHotness(
        num_layers=num_layers,
        num_experts=num_experts,
        device=torch.device("cpu"),
        out_path=None,
        flush_interval_s=flush_interval_s,
        window_interval_s=window_interval_s,
    )


def test_window_ema_semantics_and_full_count_coexistence(monkeypatch, tmp_path):
    """窗口封口：完整窗口 = 各次排空增量的并集（排空间隔 < 窗口间隔时跨多次排空）；
    EMA 首窗直取、其后 0.5*ema + 0.5*窗口；全量累计（_host_counts/save）语义不受
    窗口消费影响。"""
    from freetoken.moe.hotness import ExpertHotness

    # 排空每 5s 一次、窗口 10s：一个窗口横跨 ≥2 次排空
    hot = _make_window_hotness(window_interval_s=10.0, flush_interval_s=5.0)
    hot.out_path = str(tmp_path / "stats.json")  # 同时验证与 --hot-stats-out 共存
    clock = {"t": 1000.0}
    monkeypatch.setattr("freetoken.moe.hotness.time.monotonic", lambda: clock["t"])
    hot._last_flush = clock["t"]  # __init__ 取真实时钟，构造后对齐到受控时钟

    # 窗口 #1：三次排空的增量并集
    hot.record(0, torch.tensor([[0, 1], [0, 2]], dtype=torch.int32))  # 0:2, 1:1, 2:1
    clock["t"] = 1006.0
    assert hot.maybe_flush() is True  # 首次排空：开窗（started=1006），不封口
    assert hot.has_window is False
    assert hot._host_counts.tolist() == [2, 1, 1, 0]  # 全量累计照常

    hot.record(0, torch.tensor([[1, 1]], dtype=torch.int32))  # 1:+2
    clock["t"] = 1011.0
    assert hot.maybe_flush() is True  # 排空但窗口未满（5s < 10s）

    clock["t"] = 1017.0  # 无新记录的排空：封口窗口 #1 = 前两次增量并集
    assert hot.maybe_flush() is True
    assert hot.has_window
    np.testing.assert_array_equal(hot.ema_counts()[0], np.array([2.0, 3.0, 1.0, 0.0]))
    np.testing.assert_array_equal(hot._window.reshape(1, 4)[0], np.array([2, 3, 1, 0]))
    assert hot._window_acc.tolist() == [0, 0, 0, 0]

    # 窗口 #2：单次排空即封口；EMA = 0.5*W1 + 0.5*W2
    hot.record(0, torch.tensor([[3, 3]], dtype=torch.int32))  # 3:2
    clock["t"] = 1028.0
    assert hot.maybe_flush() is True
    np.testing.assert_array_equal(hot._window.reshape(1, 4)[0], np.array([0, 0, 0, 2]))
    np.testing.assert_array_equal(hot.ema_counts()[0], np.array([1.0, 1.5, 0.5, 1.0]))
    # 全量累计 = 两窗之和，save 落盘与窗口/EMA 完全解耦
    assert hot._host_counts.tolist() == [2, 3, 1, 2]
    hot.save()
    payload = ExpertHotness.load(str(tmp_path / "stats.json"))
    assert payload["counts"] == [[2, 3, 1, 2]]


def test_window_topk_and_ema_kth(monkeypatch):
    """window_topk：EMA 降序、平局取小 id；ema_kth：按名次取值、越界抛错；
    无完整窗口时查询拒绝。"""
    hot = _make_window_hotness(num_experts=4)
    assert hot.has_window is False
    with pytest.raises(RuntimeError, match="窗口"):
        hot.ema_counts()
    with pytest.raises(RuntimeError, match="窗口"):
        hot.window_topk(0, 2)

    hot._ema = np.array([5.0, 7.0, 5.0, 1.0])  # 直接注入：降序 1,0/2(平局),3
    assert hot.window_topk(0, 2) == [1, 0]  # 平局 5.0 取小 id 0
    assert hot.window_topk(0, 99) == [1, 0, 2, 3]  # k 截断到专家数
    assert hot.ema_kth(0, 1) == 7.0
    assert hot.ema_kth(0, 2) == 5.0
    assert hot.ema_kth(0, 4) == 1.0
    with pytest.raises(ValueError, match="名次"):
        hot.ema_kth(0, 0)
    with pytest.raises(ValueError, match="名次"):
        hot.ema_kth(0, 5)


def test_save_pinned_field_and_select_compat(tmp_path):
    """pinned_provider 给定时 save 附带 pin-list 风格 pinned 键（schema_version 仍 1），
    选点工具 load_stats 照常读取；out_path=None 时 save 是 no-op。"""
    import json

    from freetoken.hotness.select import load_stats
    from freetoken.moe.hotness import ExpertHotness

    hot = _make_hotness(tmp_path)
    hot.pinned_provider = lambda: [[3, 0], [1]]
    hot.record(0, torch.tensor([[0, 3]], dtype=torch.int32))
    hot._host_counts += hot.counts.numpy()
    hot.save()
    payload = json.loads((tmp_path / "hotstats.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["pinned"] == [{"layer": 0, "experts": [3, 0]}, {"layer": 1, "experts": [1]}]
    # 选点工具读取兼容：可选键不破坏既有校验
    stats = load_stats(str(tmp_path / "hotstats.json"))
    assert stats["meta"]["num_layers"] == 2

    # 无 provider：不写 pinned 键
    hot2 = _make_hotness(tmp_path)
    hot2.save()
    payload2 = json.loads((tmp_path / "hotstats.json").read_text(encoding="utf-8"))
    assert "pinned" not in payload2

    # out_path=None（纯窗口模式）：save no-op，不建文件
    hot3 = _make_window_hotness()
    hot3.record(0, torch.tensor([[0, 1]], dtype=torch.int32))
    (tmp_path / "hotstats.json").unlink()  # 清掉上一子用例的产物
    hot3.save()
    assert not (tmp_path / "hotstats.json").exists()



class _DeadCudaCounts:
    """模拟退出阶段 CUDA 上下文已失效：任何 .cpu() 读取都抛运行时错误。"""

    def cpu(self):
        raise RuntimeError("CUDA error: context is destroyed")


def test_save_survives_dead_cuda_drain(monkeypatch, tmp_path):
    """退出阶段 device 排空失败（CUDA 上下文已失效）时 save 仍用宿主累计落盘，
    已排空间隔的统计数据不丢；失败只告警不外抛。"""
    hot = _make_hotness(tmp_path)
    hot.record(0, torch.tensor([[0, 1], [1, 3]], dtype=torch.int32))
    hot._host_counts += hot.counts.numpy()  # 模拟退出前最后一次成功的周期排空
    hot.counts = _DeadCudaCounts()  # 此刻 CUDA 已不可用
    hot.save()  # 不得外抛

    payload = json.loads((tmp_path / "hotstats.json").read_text(encoding="utf-8"))
    assert np.array_equal(
        np.asarray(payload["counts"], dtype=np.int64),
        np.array([[1, 2, 0, 1], [0, 0, 0, 0]]),
    )


def test_maybe_flush_writes_json_periodically(monkeypatch, tmp_path):
    """周期排空顺带原子写 JSON：崩溃/SIGKILL 最多丢一个间隔的统计。"""
    hot = _make_hotness(tmp_path, flush_interval_s=60.0)
    clock = {"t": hot._last_flush}
    monkeypatch.setattr("freetoken.moe.hotness.time.monotonic", lambda: clock["t"])

    hot.record(0, torch.tensor([[0, 1], [1, 3]], dtype=torch.int32))
    clock["t"] += 61.0
    assert hot.maybe_flush() is True
    payload = json.loads((tmp_path / "hotstats.json").read_text(encoding="utf-8"))
    assert payload["counts"] == [[1, 2, 0, 1], [0, 0, 0, 0]]

    hot.record(1, torch.tensor([[2, 2]], dtype=torch.int32))
    clock["t"] += 61.0
    assert hot.maybe_flush() is True
    payload = json.loads((tmp_path / "hotstats.json").read_text(encoding="utf-8"))
    assert payload["counts"] == [[1, 2, 0, 1], [0, 0, 2, 0]]
    assert not (tmp_path / "hotstats.json.tmp").exists()  # 原子替换无残留
