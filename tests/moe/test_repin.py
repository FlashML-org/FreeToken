"""动态重钉（--hot-expert-repin-*，设计文档 §10）：决策、迁移不变式与 GPU 域切换。

覆盖三层：

* ``plan_repin_swaps`` 纯宿主决策：迟滞边界、max_swaps 上限、平局取小 id、
  冷集中无候选不换、零热度候选不换、参数校验；
* ``OffloadMoeCache.swap_pinned_experts`` 行级交换的映射不变式（slot_for_id ↔
  cold_row 互逆、钉住槽无重复、bank 行内容 == 换入/换出专家参考权重、三源 gather
  索引与 cold_row 宿主镜像同步刷新）与交换后的合并查询路由（oracle 驱动）；
* ``HotExpertRepinManager``：墙钟门控 + 窗口就绪门控 + 执行与日志；
* GPU（skipif no CUDA）：A→B 域切换按窗口迁移热集、迁移后 3000 步真实 lru_ensure
  钉住恒驻、迁移后 decode 输出与等效静态钉住配置逐位一致。
"""

import logging

import numpy as np
import pytest
import torch

import freetoken.moe.hot_pin  # noqa: F401 -- 须在 collection 期导入：pytest 9 的 caplog 只把
# 捕获 handler 附加到"进入测试阶段时已在 loggerDict 里"的非传播 logger；本文件的测试函数
# 都在函数内 import hot_pin，单独运行本文件时若此处不导入，caplog 断言会因 handler 缺席为空
from freetoken.distributed import set_tp_info, try_get_tp_info

from .test_hot_pin import (  # 复用钉住测试的装配助手与 flashlib oracle
    _apply_copy_plan,
    _bf16_offload_layer,
    _init_pins,
    _lru_ensure_oracle,
    _load_pinned_slot_contents,
    _make_pinned_cache,
)

LRU_FLOOR = 512  # init_hot_pins 的 LRU 区绝对下限（max(2E, 512) 的 512 项）


def _init_tp():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _make_hotness(num_layers=1, num_experts=16, window_interval_s=10.0, device="cpu"):
    from freetoken.moe.hotness import ExpertHotness

    return ExpertHotness(
        num_layers=num_layers,
        num_experts=num_experts,
        device=torch.device(device),
        out_path=None,  # 动态重钉 alone：只喂窗口，不落盘
        flush_interval_s=3600.0,
        window_interval_s=window_interval_s,
    )


# ---------------------------------------------------------------------------
# plan_repin_swaps：纯宿主决策
# ---------------------------------------------------------------------------


def test_plan_repin_swaps_hysteresis_boundary():
    """迟滞边界：候选 EMA == 受害 × gain 恰好交换（>=），低于一个可感知的量则不换。"""
    from freetoken.moe.hot_pin import plan_repin_swaps

    # 候选 2 的 EMA=30，钉住 0 的 EMA=20：30 == 20*1.5 → 交换
    ema = np.array([[20.0, 100.0, 30.0, 0.0]])
    assert plan_repin_swaps(ema, [[0, 1]], gain=1.5, max_swaps=8) == {0: [(2, 0)]}
    # 29.9 < 30 → 不换（top-2 仍是 {1, 2}，2 也在内但迟滞不过线）
    ema[0, 2] = 29.9
    assert plan_repin_swaps(ema, [[0, 1]], gain=1.5, max_swaps=8) == {}


def test_plan_repin_swaps_cap_and_ordering():
    """max_swaps 上限与排序：候选按 EMA 降序、受害按 EMA 升序配对，平局取小 id。"""
    from freetoken.moe.hot_pin import plan_repin_swaps

    # 钉住 [0, 1, 2, 3]（EMA 均 10），冷 4..7（EMA 均 100）：4 个候选、平局小 id 优先
    ema = np.array([[10.0] * 4 + [100.0] * 4])
    plan = plan_repin_swaps(ema, [[0, 1, 2, 3]], gain=1.5, max_swaps=8)
    # top-4 = 冷 4,5,6,7（EMA 100），受害升序平局取小 id -> (4,0) (5,1) (6,2) (7,3)
    assert plan == {0: [(4, 0), (5, 1), (6, 2), (7, 3)]}
    # max_swaps=2 截断
    plan = plan_repin_swaps(ema, [[0, 1, 2, 3]], gain=1.5, max_swaps=2)
    assert plan == {0: [(4, 0), (5, 1)]}
    # 候选平局取小 id：冷 6 与 7 同为 100，top-2 只有一个空位时先取 6
    ema2 = np.array([[10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0]])
    plan = plan_repin_swaps(ema2, [[0, 1]], gain=1.5, max_swaps=8)
    assert plan == {0: [(6, 0), (7, 1)]}


def test_plan_repin_swaps_no_candidate_and_zero_ema():
    """钉住集已是 top-K 时不换；候选零热度不换（零热度不构成"更热"证据）。"""
    from freetoken.moe.hot_pin import plan_repin_swaps

    # 钉住 [0, 1] 恰为 top-2：无候选
    ema = np.array([[50.0, 40.0, 30.0, 1.0]])
    assert plan_repin_swaps(ema, [[0, 1]], gain=1.5, max_swaps=8) == {}
    # top-2 = {1, 2}，候选 2 的 EMA=0：零热度不换（受害 EMA 也是 0 也不换）
    ema = np.array([[0.0, 5.0, 0.0, 0.0]])
    assert plan_repin_swaps(ema, [[0, 1]], gain=1.5, max_swaps=8) == {}
    # 全零 EMA：不换
    ema = np.zeros((1, 4))
    assert plan_repin_swaps(ema, [[0, 1]], gain=1.5, max_swaps=8) == {}
    # 多层：空钉住层（CPU 解码层）跳过，仅热层出现在计划里
    ema = np.array([[10.0, 100.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    plan = plan_repin_swaps(ema, [[0], []], gain=1.5, max_swaps=8)
    assert plan == {0: [(1, 0)]}


def test_plan_repin_swaps_validates_params():
    """gain < 1、max_swaps < 1、形状不符抛 ValueError。"""
    from freetoken.moe.hot_pin import plan_repin_swaps

    ema = np.zeros((1, 4))
    with pytest.raises(ValueError, match="gain"):
        plan_repin_swaps(ema, [[0]], gain=0.9, max_swaps=8)
    with pytest.raises(ValueError, match="max_swaps"):
        plan_repin_swaps(ema, [[0]], gain=1.5, max_swaps=0)
    with pytest.raises(ValueError, match="不一致"):
        plan_repin_swaps(ema, [[0], [1]], gain=1.5, max_swaps=8)


# ---------------------------------------------------------------------------
# swap_pinned_experts：行级交换不变式
# ---------------------------------------------------------------------------


def test_swap_pinned_experts_invariants():
    """交换后：slot_for_id ↔ cold_row 互逆、pin 槽无重复、bank 行/槽内容与换入换出
    专家参考权重一致、三源 gather 索引与 cold_row 宿主镜像同步刷新。"""
    _init_tp()
    L, E = 1, 16
    pins = (0, 1)
    cache = _make_pinned_cache(num_layers=L, num_experts=E, pins=pins)
    _init_pins(cache, pins=pins, k_per_layer=[2])
    _load_pinned_slot_contents(cache)
    # 预填钉住槽 usage（模拟已运行过若干步）
    cache.usage[cache.pin_slots[0, :2].long()] = 7
    cache.step.fill_(99)

    cache.swap_pinned_experts(0, [(4, 0), (5, 1)])

    # 钉住集翻转：pin_ids/pin_slots 值改写、shape 不变
    assert cache.pin_ids.shape == (L, 2)
    assert cache.pin_ids[0].tolist() == [4, 5]
    assert cache.pin_slots[0].tolist() == [cache.pin_base, cache.pin_base + 1]
    # slot_for_id ↔ cold_row 互逆：钉住 ⇔ cold_row == -1 ⇔ slot_for_id ∈ 顶部区
    sf = cache.slot_for_id[0].cpu().tolist()
    cr = cache.cold_row[0].cpu().tolist()
    for e in range(E):
        if e in (4, 5):
            assert cr[e] == -1 and sf[e] >= cache.pin_base, e
        else:
            assert cr[e] >= 0 and sf[e] == -1, e
    # 冷行号是冷集上的双射（恰为 0..E-K-1 的一个排列）
    assert sorted(r for r in cr if r >= 0) == list(range(E - 2))
    assert cr[0] == 2 and cr[1] == 3  # 被替换者回填到换入者的旧冷行
    # id_of_slot 反向一致、钉住槽无重复
    assert cache.id_of_slot[cache.pin_base].item() == 4
    assert cache.id_of_slot[cache.pin_base + 1].item() == 5
    assert len({cache.id_of_slot[s].item() for s in range(cache.pin_base, cache.cache_size)}) == 2
    # usage 刷成当前 step（flashlib 不可驱逐语义的自我续期衔接）
    assert cache.usage[cache.pin_slots[0, :2].long()].tolist() == [99, 99]
    # bank 行内容 == 换出专家参考权重（host-to-host 写回，检查宿主 bank 的行）；
    # 钉住槽 == 换入专家指纹（slot cache 的顶部槽）
    for _per_layer, bank_cache in cache.banks:
        assert bank_cache[cache.pin_base].mean().item() == 4.0  # 换入专家 4 -> 顶部槽
        assert bank_cache[cache.pin_base + 1].mean().item() == 5.0
    for role_i, (host_rows, _bank_cache) in enumerate(cache.banks):
        assert host_rows[0][2].mean().item() == 0.0  # 换出专家 0 -> 行 2
        assert host_rows[0][3].mean().item() == 1.0  # 换出专家 1 -> 行 3
        # 其余冷行未被扰动（对照宿主 bank 原指纹）
        assert host_rows[0][0].mean().item() == 2.0 and host_rows[0][1].mean().item() == 3.0
    # 三源 gather 索引与 cold_row 宿主镜像刷新
    assert cache._pin_gather_dst[0].tolist() == [4, 5]
    assert cache._pin_gather_src[0].tolist() == [cache.pin_base, cache.pin_base + 1]
    assert cache._pin_cold_dst[0].tolist() == [e for e in range(E) if e not in (4, 5)]
    assert cache._pin_cold_src[0].tolist() == list(range(E - 2))
    np.testing.assert_array_equal(cache._cold_row_np[0], np.array(cr, dtype=np.int32))
    # 复用同一 scratch（第二次交换不再分配更大缓冲）
    bufs = cache._repin_scratch_bufs
    cache.swap_pinned_experts(0, [(6, 4)])
    assert cache._repin_scratch_bufs is bufs
    assert cache.pin_ids[0].tolist() == [6, 5]
    for host_rows, bank_cache in cache.banks:
        # 4 的字节写回 6 的旧冷行（pins=(4,5) 下 6 的冷行为 4），6 上位到原 4 的钉住槽
        assert host_rows[0][4].mean().item() == 4.0
        assert bank_cache[cache.pin_base].mean().item() == 6.0


def test_swap_pinned_experts_rejects_bad_pairs():
    """候选不在冷集 / 被替换者不在钉住区 / 重复对：拒绝且不产生部分交换。"""
    _init_tp()
    cache = _make_pinned_cache(num_layers=1, num_experts=16, pins=(0, 1))
    _init_pins(cache, pins=(0, 1), k_per_layer=[2])
    _load_pinned_slot_contents(cache)
    with pytest.raises(ValueError, match="不在冷集"):
        cache.swap_pinned_experts(0, [(0, 1)])  # 0 是钉住专家，不是候选
    with pytest.raises(ValueError, match="不在钉住区"):
        cache.swap_pinned_experts(0, [(2, 3)])  # 3 是冷专家，没有钉住槽
    with pytest.raises(ValueError, match="重复"):
        cache.swap_pinned_experts(0, [(2, 0), (3, 0)])
    # 拒绝后映射原封不动
    assert cache.pin_ids[0].tolist() == [0, 1]
    assert cache.cold_row[0, 2].item() == 0


def test_post_swap_merged_query_routing(monkeypatch):
    """交换后的合并查询路由：新钉住集恒命中恒驻，被换出专家按新冷行正确换入。"""
    from freetoken.moe import offload_kernels

    monkeypatch.setattr(offload_kernels, "lru_ensure", _lru_ensure_oracle)
    L, E = 2, 16
    cache = _make_pinned_cache(num_layers=L, num_experts=E, pins=(0, 1))
    _init_pins(cache, pins=(0, 1), k_per_layer=[2, 2])
    _load_pinned_slot_contents(cache)
    cache.swap_pinned_experts(0, [(4, 0), (5, 1)])
    new_pins = [cache.pin_ids[l, : cache.pin_counts[l]].tolist() for l in range(L)]
    assert new_pins[0] == [4, 5] and new_pins[1] == [0, 1]
    pinned_slots = {
        (l, e): int(cache.slot_for_id[l, e].item()) for l in range(L) for e in new_pins[l]
    }

    rng = np.random.default_rng(31)
    for step in range(1500):
        layer_id = step % L
        raw = torch.from_numpy(rng.integers(0, E, size=(2, 2)).astype(np.int32))
        raw.view(-1)[0] = new_pins[layer_id][0]
        ids = raw.clone()
        cache.ensure_experts(layer_id, ids)
        for (l, e), slot in pinned_slots.items():
            assert int(cache.id_of_slot[slot].item()) == l * E + e, (step, l, e)
            assert float(cache.banks[0][1][slot].mean().item()) == float(l * 100 + e)
        _apply_copy_plan(cache, layer_id)
        for i in range(raw.numel()):
            e = int(raw.view(-1)[i].item())
            slot = int(ids.view(-1)[i].item())
            if e in new_pins[layer_id]:
                assert slot == pinned_slots[(layer_id, e)], (step, e)
            else:
                assert float(cache.banks[0][1][slot].mean().item()) == float(layer_id * 100 + e), (step, e)


# ---------------------------------------------------------------------------
# fp8 bank（真实 NVFP4/FP8 的 bank dtype）
# ---------------------------------------------------------------------------


def test_swap_pinned_experts_fp8_banks():
    """fp8 bank 上的行级交换：index 系内核对 Float8_e4m3fn 未实现，交换必须走
    逐行 slice copy_（真实 NVFP4 演示发现的首个生产路径崩溃，本用例防止回退）。"""
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    L, E, H, I = 1, 16, 8, 4
    pins = [0, 1]
    dev = torch.device("cpu")
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=E + max(2 * E, LRU_FLOOR) + 8, device=dev)
    gate_up, down = [], []
    for l in range(L):
        gu = torch.zeros(E - len(pins), 2 * I, H, dtype=torch.float8_e4m3fn)
        dn = torch.zeros(E - len(pins), H, I, dtype=torch.float8_e4m3fn)
        for e in range(E):
            if e in pins:
                continue
            r = e - len(pins)
            gu[r] = float(e)  # 指纹 ≤ 15，fp8 e4m3 精确表示
            dn[r] = float(e)
        gate_up.append(gu)
        down.append(dn)
    cache.set_bank_sources({"gate_up": gate_up, "down": down}, per_layer_rows=[E - len(pins)])
    _init_pins(cache, pins=tuple(pins), k_per_layer=[len(pins)])
    _load_pinned_slot_contents(cache)
    cache.swap_pinned_experts(0, [(4, 0), (5, 1)])
    assert cache.pin_ids[0].tolist() == [4, 5]
    for host_rows, bank_cache in cache.banks:
        assert host_rows[0][2].float().mean().item() == 0.0  # 换出专家 0 -> 行 2
        assert host_rows[0][3].float().mean().item() == 1.0  # 换出专家 1 -> 行 3
        assert bank_cache[cache.pin_base].float().mean().item() == 4.0
        assert bank_cache[cache.pin_base + 1].float().mean().item() == 5.0


# ---------------------------------------------------------------------------
# HotExpertRepinManager：门控 + 执行
# ---------------------------------------------------------------------------


def test_repin_manager_gates_and_execution(monkeypatch, caplog):
    """墙钟未到 / 窗口未就绪 / 无迁移都不触发；A→B 窗口到达后一次换齐并打日志。"""
    import freetoken.moe.hot_pin as hot_pin_mod
    import freetoken.moe.hotness as hotness_mod
    from freetoken.moe.hot_pin import HotExpertRepinManager

    _init_tp()
    cache = _make_pinned_cache(num_layers=1, num_experts=16, pins=(0, 1))
    _init_pins(cache, pins=(0, 1), k_per_layer=[2])
    _load_pinned_slot_contents(cache)
    hot = _make_hotness(window_interval_s=10.0)

    clock = {"t": 1000.0}
    monkeypatch.setattr(hotness_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(hot_pin_mod.time, "monotonic", lambda: clock["t"])
    hot._last_flush = clock["t"]  # __init__ 取真实时钟，构造后对齐到受控时钟
    manager = HotExpertRepinManager(cache, hot, interval_s=10.0, gain=1.5, max_swaps=8)

    # A 域流量进首个窗口：0/1 各 100 次
    for _ in range(50):
        hot.record(0, torch.tensor([[0, 1, 0, 1]], dtype=torch.int32))
    # 窗口未就绪（首次排空只是开窗）：到点也直接 False
    clock["t"] = 1012.0
    assert hot.maybe_flush() is True
    assert hot.has_window is False
    clock["t"] = 1023.0
    assert manager.maybe_repin() is False
    assert cache.pin_ids[0].tolist() == [0, 1]

    # 第二次排空封口 A 域窗口：钉住集仍是最热 → 无迁移
    clock["t"] = 1024.0
    assert hot.maybe_flush() is True  # 窗口 #1 封口 = A 域计数
    assert hot.has_window
    clock["t"] = 1035.0  # 墙钟到点
    assert manager.maybe_repin() is False  # EMA top-2 仍是钉住的 {0, 1}
    assert cache.pin_ids[0].tolist() == [0, 1]

    # B 域窗口：4/5 的 EMA 压倒钉住者（迟滞线 = 50*1.5 = 75，B 窗计数 200 -> EMA 100）
    for _ in range(100):
        hot.record(0, torch.tensor([[4, 5, 4, 5]], dtype=torch.int32))
    clock["t"] = 1047.0
    assert hot.maybe_flush() is True  # 窗口 #2 封口 = B 域计数，EMA = 0.5*A + 0.5*B
    with caplog.at_level(logging.INFO, logger="freetoken.moe.hot_pin"):
        assert manager.maybe_repin() is True
    assert cache.pin_ids[0].tolist() == [4, 5]
    assert cache.slot_for_id[0, 4].item() == cache.pin_base
    assert cache.cold_row[0, 0].item() >= 0
    assert "dynamic repin" in caplog.text and "0 -> 4" in caplog.text and "1 -> 5" in caplog.text

    # 墙钟门：刚触发过，同一时刻再调不重复触发
    clock["t"] = 1048.0
    assert manager.maybe_repin() is False
    clock["t"] = 1058.0
    assert manager.maybe_repin() is False  # 到点但窗口无迁移（B 已是钉住集）

    # 参数校验
    with pytest.raises(ValueError, match="interval"):
        HotExpertRepinManager(cache, hot, interval_s=0.0, gain=1.5, max_swaps=8)
    with pytest.raises(ValueError, match="gain"):
        HotExpertRepinManager(cache, hot, interval_s=10.0, gain=0.9, max_swaps=8)


# ---------------------------------------------------------------------------
# 运行中调 K：set_target_k / apply_target_k（固定容量布局 + 活跃计数动态）
# ---------------------------------------------------------------------------


def test_manager_target_k_grow_shrink(monkeypatch, caplog):
    """扩容等窗口、缩容免窗口；每层目标 clamp 到 [floors, K_cap]；扩缩后映射不变式
    （slot_for_id ↔ cold_row 互逆、空闲行池记账、哨兵、填充行）与合并查询端到端。"""
    import freetoken.moe.hot_pin as hot_pin_mod
    import freetoken.moe.hotness as hotness_mod
    from freetoken.moe import offload_kernels
    from freetoken.moe.hot_pin import HotExpertRepinManager
    from freetoken.moe.offload_cache import _PIN_USAGE_SENTINEL

    monkeypatch.setattr(offload_kernels, "lru_ensure", _lru_ensure_oracle)
    _init_tp()
    L, E = 2, 16
    cache = _make_pinned_cache(
        num_layers=L, num_experts=E, pins=(0, 1), k_per_layer=[2, 2],
        cache_size=2 * L * 4 + max(2 * E, LRU_FLOOR) + 8,
    )
    _init_pins(cache, pins=(0, 1), k_per_layer=[2, 2], pin_capacity=4)
    _load_pinned_slot_contents(cache)
    hot = _make_hotness(num_layers=L, num_experts=E, window_interval_s=10.0)

    clock = {"t": 1000.0}
    monkeypatch.setattr(hotness_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(hot_pin_mod.time, "monotonic", lambda: clock["t"])
    hot._last_flush = clock["t"]  # __init__ 取真实时钟，构造后对齐到受控时钟
    manager = HotExpertRepinManager(cache, hot, interval_s=10.0, gain=1.5, max_swaps=8)

    # 越界目标：就地拒绝（poller 侧转告警）
    with pytest.raises(ValueError, match="超出"):
        manager.set_target_k(5)
    with pytest.raises(ValueError, match="超出"):
        manager.set_target_k(-1)

    # 扩容无窗口：整体等待，目标保留
    manager.set_target_k(4)
    assert manager.apply_target_k() is False
    assert manager.target_k == 4
    assert cache.pin_counts == [2, 2]

    # A 域窗口：2/3 热（两层同流量），两个完整窗口后扩到 K=4
    for _ in range(50):
        hot.record(0, torch.tensor([[2, 3, 2, 3]], dtype=torch.int32))
        hot.record(1, torch.tensor([[2, 3, 2, 3]], dtype=torch.int32))
    clock["t"] = 1011.0
    assert hot.maybe_flush() is True  # 开窗
    clock["t"] = 1022.0
    assert hot.maybe_flush() is True  # 封口
    with caplog.at_level(logging.INFO, logger="freetoken.moe.hot_pin"):
        assert manager.apply_target_k() is True
    assert manager.target_k is None  # 目标消费后清空
    assert cache.pin_counts == [4, 4]
    assert "dynamic pin-k" in caplog.text
    for l in range(L):
        assert cache.pin_ids[l].tolist() == [0, 1, 2, 3]  # 扩到满容量：无填充尾
        # 映射不变式：钉住 ⇔ cold_row == -1 ⇔ slot_for_id ∈ 容量区
        cr = cache.cold_row[l].cpu().tolist()
        sf = cache.slot_for_id[l].cpu().tolist()
        for e in range(E):
            if e < 4:
                assert cr[e] == -1 and cache.pin_base + l * 4 <= sf[e] < cache.pin_base + (l + 1) * 4, (l, e)
            else:
                assert cr[e] >= 0 and sf[e] == -1, (l, e)
        # 冷行号是冷集 × 非空闲行的双射：行 0/1 已让给空闲池
        assert sorted(r for r in cr if r >= 0) == list(range(2, E - 2))
        assert cache._pin_free_rows[l] == {0, 1}
        # 钉住槽指纹 == 换入专家（bank 冷行 H2D）
        for j, e in enumerate([0, 1, 2, 3]):
            s = int(cache.pin_slots[l, j].item())
            assert int(cache.id_of_slot[s].item()) == l * E + e
            for _per_layer, bank_cache in cache.banks:
                assert bank_cache[s].mean().item() == float(l * 100 + e)
        # 三源 gather 索引与宿主冷行镜像刷新
        assert cache._pin_gather_dst[l].tolist() == [0, 1, 2, 3]
        assert cache._pin_gather_num[l].tolist() == [4]
        np.testing.assert_array_equal(cache._cold_row_np[l], np.array(cr, dtype=np.int32))

    # 合并查询端到端（oracle）：新钉恒命中，填充 dup 无害（无额外 fetch、无越界）
    rng = np.random.default_rng(5)
    for step in range(300):
        layer_id = step % L
        raw = torch.from_numpy(rng.integers(0, E, size=(1, 4)).astype(np.int32))
        raw.view(-1)[0] = step % 4  # 轮询全部四个钉住专家
        ids = raw.clone()
        cache.ensure_experts(layer_id, ids)
        for e in range(4):
            s = int(cache.slot_for_id[layer_id, e].item())
            assert s >= cache.pin_base and int(cache.id_of_slot[s].item()) == layer_id * E + e, (step, e)
        _apply_copy_plan(cache, layer_id)

    # 缩容免窗口：全新无窗口 hotness 的管理器也能缩（尾部 3 换出到空闲行）
    hot2 = _make_hotness(num_layers=L, num_experts=E, window_interval_s=10.0)
    manager2 = HotExpertRepinManager(cache, hot2, interval_s=10.0, gain=1.5, max_swaps=8)
    manager2.set_target_k(3)
    # 未到墙钟也应用（目标 K 不受墙钟门控，首个 idle 安全点即生效）
    assert manager2.maybe_repin(now=clock["t"]) is True
    assert cache.pin_counts == [3, 3]
    for l in range(L):
        cr = cache.cold_row[l].cpu().tolist()
        assert cr[3] == 0  # 换出到最小空闲行（空闲池 {0,1} 取 0）
        assert cache._pin_free_rows[l] == {1}
        assert cache.slot_for_id[l, 3].item() == -1
        s3 = int(cache.pin_slots[l, 3].item())
        assert int(cache.id_of_slot[s3].item()) == -1
        assert int(cache.usage[s3].item()) == _PIN_USAGE_SENTINEL  # 空容量槽对 argmin 隐形
        assert cache.pin_ids[l].tolist() == [0, 1, 2, 0]
        # 换出字节回宿主 bank（D2H）：行 0 现在装专家 3
        for per_layer, _bank_cache in cache.banks:
            assert per_layer[l][0].mean().item() == float(l * 100 + 3)

    # 已在目标上：无变更也消费目标
    manager2.set_target_k(3)
    assert manager2.apply_target_k() is False
    assert manager2.target_k is None
    # 下界 clamp：目标 1 < floor 2 -> 每层缩到 2（floor），不会低于加载期钉住数
    manager2.set_target_k(1)
    assert manager2.apply_target_k() is True
    assert cache.pin_counts == [2, 2]
    for l in range(L):
        assert cache._pin_free_rows[l] == set()
        assert int(cache.cold_row[l, 2].item()) == 1  # 依次取空闲行池剩余行
    # 低于下界的换出被 cache 原语拒绝
    with pytest.raises(ValueError, match="区间内"):
        cache.unpin_tail_experts(0, 1)


def test_target_k_grow_defers_until_window_ready(monkeypatch):
    """扩容目标在窗口就绪前持续等待（目标保留），窗口齐备后的首个 idle 落地。"""
    import freetoken.moe.hotness as hotness_mod
    from freetoken.moe.hot_pin import HotExpertRepinManager

    _init_tp()
    L, E = 1, 16
    cache = _make_pinned_cache(num_layers=L, num_experts=E, pins=(0,), k_per_layer=[1],
                               cache_size=L * 2 + max(2 * E, LRU_FLOOR) + 8)
    _init_pins(cache, pins=(0,), k_per_layer=[1], pin_capacity=2)
    _load_pinned_slot_contents(cache)
    hot = _make_hotness(num_layers=L, num_experts=E, window_interval_s=10.0)
    clock = {"t": 1000.0}
    monkeypatch.setattr(hotness_mod.time, "monotonic", lambda: clock["t"])
    hot._last_flush = clock["t"]  # __init__ 取真实时钟，构造后对齐到受控时钟
    manager = HotExpertRepinManager(cache, hot, interval_s=10.0, gain=1.5, max_swaps=8)
    manager.set_target_k(2)
    assert manager.apply_target_k() is False
    assert manager.target_k == 2
    # 窗口只有零热度候选：仍不扩（零热度不构成钉住证据），目标保留
    for _ in range(50):
        hot.record(0, torch.tensor([[0, 0]], dtype=torch.int32))
    clock["t"] = 1011.0
    hot.maybe_flush()
    clock["t"] = 1022.0
    hot.maybe_flush()
    assert hot.has_window
    assert manager.apply_target_k() is False
    assert manager.target_k == 2
    # 换成专家 1 的真实热度后落地
    for _ in range(50):
        hot.record(0, torch.tensor([[1, 1]], dtype=torch.int32))
    clock["t"] = 1033.0
    hot.maybe_flush()
    assert manager.apply_target_k() is True
    assert cache.pin_counts == [2] and cache.pin_ids[0, :2].tolist() == [0, 1]
    assert manager.target_k is None


def test_target_k_real_service_shape_converges(monkeypatch, caplog):
    """真实服务形态回归（2026-09-28 FP8 服务扩容死循环）：K=32/K_cap=61。
    修复前 apply_target_k 把绝对目标当 want 传给 _growth_candidates（pin_k=40 时
    选 40 个候选），install 校验在 32+29>=61 处每次拒绝第 30 个候选且零副作用，
    目标因异常保留、每个 idle 原样重试，永不收敛。修复后：按剩余容量截断 want，
    窗口热度不足的余量保留目标跨 idle 续传，全部层落地才清空目标；缩容 61->32
    同样收敛。"""
    import freetoken.moe.hot_pin as hot_pin_mod
    import freetoken.moe.hotness as hotness_mod
    from freetoken.moe.hot_pin import HotExpertRepinManager

    _init_tp()
    L, E, K0, CAP = 2, 128, 32, 61
    cache = _make_pinned_cache(
        num_layers=L, num_experts=E, pins=tuple(range(K0)), k_per_layer=[K0, K0],
        cache_size=L * CAP + max(2 * E, LRU_FLOOR) + 8,
    )
    _init_pins(cache, pins=tuple(range(K0)), k_per_layer=[K0, K0], pin_capacity=CAP)
    _load_pinned_slot_contents(cache)
    hot = _make_hotness(num_layers=L, num_experts=E, window_interval_s=10.0)

    clock = {"t": 1000.0}
    monkeypatch.setattr(hotness_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(hot_pin_mod.time, "monotonic", lambda: clock["t"])
    hot._last_flush = clock["t"]  # __init__ 取真实时钟，构造后对齐到受控时钟
    manager = HotExpertRepinManager(cache, hot, interval_s=10.0, gain=1.5, max_swaps=8)
    manager.set_target_k(CAP)
    # 窗口未就绪：目标保留（真实服务首个热度窗口封口前的形态）
    assert manager.apply_target_k() is False
    assert manager.target_k == CAP

    def seal_window(layer_experts: list[int], t: float) -> None:
        """喂一个完整窗口的每层流量并封口（与既有 target_k 用例同一窗口节奏）。"""
        row = torch.tensor([layer_experts], dtype=torch.int32)
        for _ in range(50):
            for l in range(L):
                hot.record(l, row)
        clock["t"] = t
        assert hot.maybe_flush() is True  # 开窗
        clock["t"] = t + 11.0
        assert hot.maybe_flush() is True  # 封口

    with caplog.at_level(logging.INFO, logger="freetoken.moe.hot_pin"):
        # 第一窗只有 10 个非零热度候选：装 10 个、shortfall 19，目标保留待续传
        seal_window(list(range(32, 42)), 1011.0)
        assert manager.apply_target_k() is True
        assert cache.pin_counts == [42, 42]
        assert manager.target_k == CAP  # 幂等可续：半应用状态保留目标
        # 第二窗补齐余下 19 个候选：一次 apply 收敛到 K_cap，目标清空
        seal_window(list(range(42, 61)), 1033.0)
        assert manager.apply_target_k() is True
        assert cache.pin_counts == [CAP, CAP]
        assert manager.target_k is None

    pin_k_logs = [r.getMessage() for r in caplog.records if "dynamic pin-k" in r.getMessage()]
    assert any("target K=61 applied" in m for m in pin_k_logs)
    final_log = pin_k_logs[-1]
    assert "target K=61 applied" in final_log and "热度不足" not in final_log

    for l in range(L):
        # 61 个钉住 id 唯一、映射互逆（钉住 <=> cold_row == -1 <=> 槽在容量区）
        ids = cache.pin_ids[l].tolist()
        assert len(set(ids)) == CAP and ids[:K0] == list(range(K0))
        assert sorted(ids[K0:]) == list(range(K0, CAP))
        cr = cache.cold_row[l].cpu().tolist()
        sf = cache.slot_for_id[l].cpu().tolist()
        for e in range(E):
            if e < CAP:
                assert cr[e] == -1 and cache.pin_base + l * CAP <= sf[e] < cache.pin_base + (l + 1) * CAP, (l, e)
            else:
                assert cr[e] >= 0 and sf[e] == -1, (l, e)
        # 余下冷专家（CAP..E-1）保持原始冷行（e - K0）：装入释放的行进了空闲池
        assert sorted(r for r in cr if r >= 0) == list(range(CAP - K0, E - K0))
        # 缩 K 释放的 bank 行恰为装入专家让出的行（e - K0），空闲池记账一致
        assert cache._pin_free_rows[l] == set(range(CAP - K0))
        np.testing.assert_array_equal(cache._cold_row_np[l], np.array(cr, dtype=np.int32))

    # 缩容回 K0（真实验证的 61 -> 32 方向）：免窗口、尾部换出、目标消费
    manager.set_target_k(K0)
    assert manager.apply_target_k() is True
    assert cache.pin_counts == [K0, K0]
    assert manager.target_k is None
    for l in range(L):
        assert cache.pin_ids[l, :K0].tolist() == list(range(K0))
        cr = cache.cold_row[l].cpu().tolist()
        assert all(cr[e] >= 0 for e in range(K0, E)) and all(cr[e] == -1 for e in range(K0))
        assert cache._pin_free_rows[l] == set()


def test_install_pinned_experts_truncates_at_capacity():
    """install 契约（真实服务事故的第二道防线）：超容量截断到剩余容量并返回实际
    装入（不 raise），越界/不在冷集/层内重复才 raise 且零副作用。"""
    _init_tp()
    L, E = 1, 16
    cache = _make_pinned_cache(num_layers=L, num_experts=E, pins=(0,), k_per_layer=[1],
                               cache_size=4 + max(2 * E, LRU_FLOOR) + 8)
    _init_pins(cache, pins=(0,), k_per_layer=[1], pin_capacity=4)
    _load_pinned_slot_contents(cache)
    # 剩余容量 3、候选 5 个：截断装入前 3 个，恰好到满容量（含 >= 边界）
    assert cache.install_pinned_experts(0, [2, 3, 4, 5, 6]) == [2, 3, 4]
    assert cache.pin_counts == [4]
    assert cache.pin_ids[0].tolist() == [0, 2, 3, 4]
    assert cache._pin_free_rows[0] == {1, 2, 3}  # 装入专家让出的冷行（e - 1）
    for j, e in enumerate([0, 2, 3, 4]):
        s = int(cache.pin_slots[0, j].item())
        assert int(cache.id_of_slot[s].item()) == e
        for _per_layer, bank_cache in cache.banks:
            assert bank_cache[s].mean().item() == float(e)
    # 已满容量再装：返回空列表而非报错
    assert cache.install_pinned_experts(0, [5]) == []
    assert cache.pin_counts == [4]

    # 非法输入在写入前整体拒绝（校验循环先于任何字节/映射改动）
    fresh = _make_pinned_cache(num_layers=L, num_experts=E, pins=(0,), k_per_layer=[1],
                               cache_size=4 + max(2 * E, LRU_FLOOR) + 8)
    _init_pins(fresh, pins=(0,), k_per_layer=[1], pin_capacity=4)
    with pytest.raises(ValueError, match="重复"):
        fresh.install_pinned_experts(0, [2, 2, 3])  # 层内重复（冷行快照查不出，须显式拒绝）
    with pytest.raises(ValueError, match="越界"):
        fresh.install_pinned_experts(0, [16])
    with pytest.raises(ValueError, match="不在冷集"):
        fresh.install_pinned_experts(0, [0])  # 已是钉住专家
    assert fresh.pin_counts == [1]
    assert fresh.pin_ids[0, :1].tolist() == [0]
    assert fresh.cold_row[0, 2].item() >= 0 and fresh._pin_free_rows[0] == set()


# ---------------------------------------------------------------------------
# GPU：动态 K 值更新对已捕获 decode 图立即生效（hybrid 内核）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pin_k_dynamic_k_on_captured_hybrid_graph_gpu(monkeypatch):
    """捕获带钉住的 ensure_experts_hybrid 进 CUDA graph，idle 扩容（装新钉）后
    replay：新钉专家立即恒命中容量槽（不被改写 -1）、旧钉不受影响、扩容前该专家
    的 LRU 残留副本槽已去映射；缩容后 replay 按恢复的冷行换入。"""
    import freetoken.moe.hot_pin as hot_pin_mod
    import freetoken.moe.hotness as hotness_mod
    from freetoken.moe.hot_pin import HotExpertRepinManager
    from freetoken.moe.offload_cache import _PIN_USAGE_SENTINEL

    L, E = 1, 16
    dev = torch.device("cuda")
    cache = _make_pinned_cache(num_layers=L, num_experts=E, pins=(0, 1), device="cuda",
                               decode_target="hybrid", dtype=torch.bfloat16,
                               cache_size=L * 4 + max(2 * E, LRU_FLOOR) + 8)
    _init_pins(cache, pins=(0, 1), k_per_layer=[2], pin_capacity=4)
    _load_pinned_slot_contents(cache)
    torch.cuda.synchronize()

    clock = {"t": 1000.0}
    monkeypatch.setattr(hotness_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(hot_pin_mod.time, "monotonic", lambda: clock["t"])
    hot = _make_hotness(num_layers=L, num_experts=E, window_interval_s=10.0, device="cuda")
    hot._last_flush = clock["t"]
    cache.hotness = hot
    manager = HotExpertRepinManager(cache, hot, interval_s=10.0, gain=1.5, max_swaps=8)
    cache.repin_manager = manager

    # 预热 + 捕获：路由含未来新钉专家 2 与冷专家 5（捕获时 2 是 miss → 有 LRU 残留槽）
    warm = torch.full((1, 4), 9, dtype=torch.int32, device=dev)
    cache.ensure_experts_hybrid(0, warm)
    torch.cuda.synchronize()
    raw = torch.tensor([[0, 1, 2, 5]], dtype=torch.int32, device=dev)
    graph = torch.cuda.CUDAGraph()
    ids = torch.empty_like(raw)
    ids.copy_(raw)
    cache.ensure_experts_hybrid(0, ids)
    cache.copy_missing()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        cache.ensure_experts_hybrid(0, ids)
        cache.copy_missing()
    torch.cuda.synchronize()
    stale_slot = int(cache.slot_for_id[0, 2].item())
    assert 0 <= stale_slot < cache.pin_base

    # A 域窗口：2/3 热 → idle 扩容到 K=4（装 2/3 进各层空闲容量槽）
    hot.record(0, torch.tensor([[2, 3]] * 100, dtype=torch.int32, device=dev))
    clock["t"] = 1011.0
    assert hot.maybe_flush() is True
    clock["t"] = 1022.0
    assert hot.maybe_flush() is True
    manager.set_target_k(4)
    assert manager.apply_target_k() is True
    torch.cuda.synchronize()
    assert cache.pin_counts == [4]
    assert cache.pin_ids[0, :4].tolist() == [0, 1, 2, 3]
    pin_slots = [int(cache.pin_slots[0, j].item()) for j in range(4)]
    assert int(cache.slot_for_id[0, 2].item()) == pin_slots[2]
    assert int(cache.id_of_slot[stale_slot].item()) == -1  # 残留副本槽已去映射
    for _per_layer, bank_cache in cache.banks:
        assert bank_cache[pin_slots[2]].mean().item() == 2.0
        assert bank_cache[pin_slots[3]].mean().item() == 3.0

    # 扩容后 replay：新钉 2 恒命中容量槽、旧钉不变、5 首次 replay 换入后恒命中
    for step in range(20):
        ids.copy_(raw)
        graph.replay()
        torch.cuda.synchronize()
        got = ids.cpu().tolist()[0]
        assert got[:3] == pin_slots[:3], (step, got)
        assert got[3] >= 0, (step, got)
        assert float(cache.banks[0][1][got[3]].mean().item()) == 5.0, (step, got)

    # 缩容到 K=3（尾部 3 换出，2 保持钉住）→ replay：3 空槽盖哨兵，2/5 照常服务
    manager.set_target_k(3)
    assert manager.apply_target_k() is True
    torch.cuda.synchronize()
    assert cache.pin_counts == [3]
    assert int(cache.slot_for_id[0, 3].item()) == -1
    assert int(cache.id_of_slot[pin_slots[3]].item()) == -1
    assert int(cache.usage[pin_slots[3]].item()) == _PIN_USAGE_SENTINEL
    row3 = int(cache.cold_row[0, 3].item())
    assert row3 >= 0
    for per_layer, _bank_cache in cache.banks:
        assert per_layer[0][row3].mean().item() == 3.0  # 换出字节回宿主

    # 缩容后 replay：2/5 仍恒命中（映射与字节未动），路由输出合法
    for step in range(20):
        ids.copy_(raw)
        graph.replay()
        torch.cuda.synchronize()
        got = ids.cpu().tolist()[0]
        assert got[0] == pin_slots[0] and got[1] == pin_slots[1], (step, got)
        assert got[2] == pin_slots[2], (step, got)  # 2 未被缩掉，仍恒命中
        assert got[3] >= 0 and float(cache.banks[0][1][got[3]].mean().item()) == 5.0, (step, got)


# ---------------------------------------------------------------------------
# GPU（真实 flashlib lru_ensure + 真实 bf16 GEMM）
# ---------------------------------------------------------------------------


def _decode_output(layer, cache, ids, weights, hidden):
    """固定路由下的一次 decode 输出。

    fused_topk 换成固定路由；fused decode GEMM 换成确定性参考实现（槽位字节指纹的
    加权组合）——测试仓惯例（test_offload 同款）：合成 bank 的行宽达不到真实 fused
    kernel 的启动下限，而等价性检验的实质是"有效槽位字节 + 路由"逐位一致，两个
    cache 走同一参考函数，torch.equal 即可证明。ensure/copy_missing 仍是真实 kernel。
    """
    import freetoken.layers.moe as moe_mod
    import freetoken.moe.fused as fused_mod

    layer.offload_cache = cache

    def fake_topk(*, hidden_states, gating_output, topk, renormalize):
        return weights.to(hidden_states.device), ids.to(hidden_states.device)

    def ref_decode(hidden_states, w1, w2, topk_weights, topk_ids, activation,
                   apply_router_weight_on_input, act_alpha=1.0, act_limit=float("inf")):
        k = topk_ids.shape[1]
        feats = w1.float().mean(dim=(1, 2)) + w2.float().mean(dim=(1, 2))  # [S] 槽位指纹
        sel = feats[topk_ids.long().view(-1)].view(-1, k) * topk_weights.float()
        return (sel.sum(dim=1, keepdim=True) * hidden_states.float()).to(hidden_states.dtype)

    orig_topk, orig_decode = moe_mod.fused_topk, fused_mod.fused_experts_decode_impl
    moe_mod.fused_topk = fake_topk
    fused_mod.fused_experts_decode_impl = ref_decode
    try:
        out = layer.decode_forward(hidden)
    finally:
        moe_mod.fused_topk = orig_topk
        fused_mod.fused_experts_decode_impl = orig_decode
    torch.cuda.synchronize()
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dynamic_repin_domain_switch_gpu(monkeypatch):
    """GPU 端到端：A→B 域切换后热集按窗口迁移；迁移后 3000 步真实 lru_ensure 新钉住
    恒驻、被换出专家按新冷行正确换入；迁移后 decode 输出与等效静态钉住配置逐位一致。"""
    import freetoken.moe.hot_pin as hot_pin_mod
    import freetoken.moe.hotness as hotness_mod
    from freetoken.moe.hot_pin import HotExpertRepinManager

    L, E = 1, 16
    dev = torch.device("cuda")
    cache = _make_pinned_cache(num_layers=L, num_experts=E, pins=(0, 1), device="cuda", dtype=torch.bfloat16)
    _init_pins(cache, pins=(0, 1), k_per_layer=[2])
    _load_pinned_slot_contents(cache)
    torch.cuda.synchronize()

    clock = {"t": 1000.0}
    monkeypatch.setattr(hotness_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(hot_pin_mod.time, "monotonic", lambda: clock["t"])
    hot = _make_hotness(num_layers=L, num_experts=E, window_interval_s=10.0, device="cuda")
    hot._last_flush = clock["t"]
    cache.hotness = hot
    manager = HotExpertRepinManager(cache, hot, interval_s=10.0, gain=1.5, max_swaps=8)
    cache.repin_manager = manager

    # A 域窗口（钉住者仍最热）：两次排空封口后无迁移
    hot.record(0, torch.tensor([[0, 1]] * 50, dtype=torch.int32, device=dev))
    clock["t"] = 1011.0
    assert hot.maybe_flush() is True  # 开窗
    clock["t"] = 1022.0
    assert hot.maybe_flush() is True  # 封口窗口 #1 = A 域计数
    assert hot.has_window
    clock["t"] = 1033.0
    assert manager.maybe_repin() is False

    # B 域窗口：4/5 上位（迟滞线 = 25*1.5 = 37.5，B 窗计数 100 -> EMA 50 过线）
    hot.record(0, torch.tensor([[4, 5]] * 100, dtype=torch.int32, device=dev))
    clock["t"] = 1044.0
    assert hot.maybe_flush() is True
    assert manager.maybe_repin() is True
    torch.cuda.synchronize()
    assert cache.pin_ids[0].tolist() == [4, 5]
    new_pins = cache.pin_ids[0].tolist()
    assert cache.slot_for_id[0, 4].item() == cache.pin_base
    assert cache.slot_for_id[0, 5].item() == cache.pin_base + 1
    for _per_layer, bank_cache in cache.banks:
        assert bank_cache[cache.pin_base].mean().item() == 4.0
        assert bank_cache[cache.pin_base + 1].mean().item() == 5.0

    # 迁移期间（交换刚完成、未跑任何 ensure）的一次 decode：与等效静态配置一致
    layer_dyn = _bf16_offload_layer(0, E, 2, 8, 4)
    ref = _make_pinned_cache(num_layers=L, num_experts=E, pins=(4, 5), device="cuda", dtype=torch.bfloat16)
    _init_pins(ref, pins=(4, 5), k_per_layer=[2])
    _load_pinned_slot_contents(ref)
    torch.cuda.synchronize()
    layer_ref = _bf16_offload_layer(0, E, 2, 8, 4)
    hidden = torch.randn(3, 8, device=dev, dtype=torch.bfloat16)
    weights = torch.tensor([[0.6, 0.4]] * 3, dtype=torch.float32)
    for ids_choices in ([(4, 0), (4, 0)],):  # 迁移期间：换入者 + 被换出者混合路由
        ids = torch.tensor([list(ids_choices[0])] * 3, dtype=torch.int32)
        got = _decode_output(layer_dyn, cache, ids, weights, hidden)
        want = _decode_output(layer_ref, ref, ids, weights, hidden)
        assert torch.equal(got.cpu(), want.cpu()), ids_choices

    # 迁移后 3000 步随机 query（真实 lru_ensure + fused copy）：新钉住恒驻、
    # 被换出专家 0/1 从新冷行换入（指纹校验）
    rng = np.random.default_rng(47)
    for step in range(3000):
        raw = torch.from_numpy(rng.integers(0, E, size=(1, 4)).astype(np.int32)).cuda()
        raw.view(-1)[0] = new_pins[step % 2]
        ids = raw.clone()
        cache.ensure_experts(0, ids)
        cache.copy_missing()
        torch.cuda.synchronize()
        for j, e in enumerate(new_pins):
            slot = cache.pin_base + j
            assert int(cache.id_of_slot[slot].item()) == e, (step, e)
            assert float(cache.banks[0][1][slot].mean().item()) == float(e), (step, e)
        for i in range(raw.numel()):
            e = int(raw.view(-1)[i].item())
            slot = int(ids.view(-1)[i].item())
            if e in new_pins:
                assert slot == cache.pin_base + new_pins.index(e), (step, e)
            else:
                assert float(cache.banks[0][1][slot].mean().item()) == float(e), (step, e)

    # 迁移后再一次 decode（含被换出者 + 换入者）：仍与静态基线一致
    for ids_rows in ([[4, 0], [5, 1], [4, 5]], [[0, 1], [0, 5], [2, 4]]):
        ids = torch.tensor(ids_rows, dtype=torch.int32)
        got = _decode_output(layer_dyn, cache, ids, weights, hidden)
        want = _decode_output(layer_ref, ref, ids, weights, hidden)
        assert torch.equal(got.cpu(), want.cpu()), ids_rows


# ---------------------------------------------------------------------------
# CUDA graph 捕获 + 重钉（真实 flashlib 合并查询的 graph 重放）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_swap_then_captured_graph_replay_gpu():
    """真实系统的 decode 走 CUDA graph 捕获的合并查询；重钉只改张量值（shape 不变），
    graph 重放必须读新值且不产生非法访问。交换后：映射新钉住集重放恒命中、被换出
    专家经新冷行换入、钉住槽映射不漂移。"""
    L, E = 2, 16
    dev = torch.device("cuda")
    cache = _make_pinned_cache(num_layers=L, num_experts=E, pins=(0, 1), device="cuda", dtype=torch.bfloat16)
    pins_matrix = _init_pins(cache, pins=(0, 1), k_per_layer=[2, 2])
    _load_pinned_slot_contents(cache)
    torch.cuda.synchronize()

    # 预热：为每个路由形状分配合并查询缓冲（真实系统在 eager warmup 完成）
    warm = torch.full((1, 4), 9, dtype=torch.int32, device=dev)
    cache.ensure_experts(0, warm)
    cache.copy_missing()
    torch.cuda.synchronize()

    # 路由混合：交换后的两个钉住专家（4/5）+ 两个被换出专家（0/1）
    raw = torch.tensor([[4, 5, 0, 1]], dtype=torch.int32, device=dev)
    graph = torch.cuda.CUDAGraph()
    ids = torch.empty_like(raw)
    ids.copy_(raw)
    cache.ensure_experts(0, ids)
    cache.copy_missing()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        cache.ensure_experts(0, ids)
        cache.copy_missing()
    torch.cuda.synchronize()

    # 重钉：层 0 的 (4 -> 0)、(5 -> 1)（换入 4/5，换出 0/1）
    cache.swap_pinned_experts(0, [(4, 0), (5, 1)])
    torch.cuda.synchronize()
    new_pins = cache.pin_ids[0].tolist()
    assert new_pins == [4, 5]

    # 交换后重放：每次重放前用原始路由覆盖（等价于 fused_topk 在 graph 内重写）
    pin_slots0 = {e: int(cache.slot_for_id[0, e].item()) for e in new_pins}
    for step in range(50):
        ids.copy_(raw)
        graph.replay()
        torch.cuda.synchronize()
        # 钉住专家恒命中钉住槽；被换出专家 0/1 正确换入（冷行 2/3，指纹校验）
        got = ids.cpu().tolist()[0]
        for e, slot in pin_slots0.items():
            assert slot in got, (step, e, got)
        for i, e in enumerate(raw.cpu().tolist()[0]):
            slot = got[i]
            fingerprint = float(cache.banks[0][1][slot].mean().item())
            if e in new_pins:
                assert slot == pin_slots0[e] and fingerprint == float(e), (step, e)
            else:
                assert fingerprint == float(e), (step, e, slot)


def test_target_k_grows_from_cumulative_hotness_without_window(monkeypatch):
    """扩容选点在首个热度窗口封口前退回全量累计热度（首批流量排空即可扩），
    不再等待 window_interval_s；选点与 EMA 同序（降序、零热度不选）。"""
    import freetoken.moe.hot_pin as hot_pin_mod
    import freetoken.moe.hotness as hotness_mod
    from freetoken.moe.hot_pin import HotExpertRepinManager

    _init_tp()
    L, E, K0, CAP = 2, 128, 32, 61
    cache = _make_pinned_cache(
        num_layers=L, num_experts=E, pins=tuple(range(K0)), k_per_layer=[K0, K0],
        cache_size=L * CAP + max(2 * E, LRU_FLOOR) + 8,
    )
    _init_pins(cache, pins=tuple(range(K0)), k_per_layer=[K0, K0], pin_capacity=CAP)
    _load_pinned_slot_contents(cache)
    hot = _make_hotness(num_layers=L, num_experts=E, window_interval_s=10.0)

    clock = {"t": 1000.0}
    monkeypatch.setattr(hotness_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(hot_pin_mod.time, "monotonic", lambda: clock["t"])
    hot._last_flush = clock["t"]
    # 未封口窗口：record 一批流量（不封口）+ 一次排空 → 全量累计有值
    row = torch.tensor([[K0 + 5, K0 + 9, K0 + 2]], dtype=torch.int32)
    for l in range(L):
        for _ in range(10):
            hot.record(l, row)
    clock["t"] += 61.0
    assert hot.maybe_flush() is True
    assert not hot.has_window

    manager = HotExpertRepinManager(cache, hot, interval_s=3600.0, gain=1.5, max_swaps=8)
    manager.set_target_k(K0 + 3)
    assert manager.apply_target_k() is True  # 无窗口也直接扩容
    assert all(c == K0 + 3 for c in cache.pin_counts)
    # 装入的正是累计热度最高的未钉专家（K0+5 / K0+9 / K0+2 中按 EMA 排序取 3 个）
    for l in range(L):
        assert set(cache.pinned_id_lists()[l]) - set(range(K0)) <= {K0 + 2, K0 + 5, K0 + 9}
    # 全量累计路径消费目标后不再重触发
    assert manager.target_k is None
