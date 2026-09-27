"""显存钉住热点专家（--hot-expert-list）：契约、冷压缩 bank、slot 顶部区与钉住 LRU。

覆盖四层：

* pin list / stats 双入口的读取校验与 cold_row 构建（``moe.hot_pin``，纯 CPU）；
* ``build_expert_banks`` 的冷压缩（bank 行数/内容、钉住暂存回调、无钉住回归）；
* ``OffloadMoeCache`` 的钉住装配（顶部区映射、reset/rebuild 重钉、记账）与
  合并查询 ensure（CPU 上用 flashlib ``_seq`` 语义的纯 python oracle 驱动数千步）；
* GPU（skipif no CUDA）：真实 flashlib lru_ensure、hybrid kernel 与 CPU 镜像逐位
  对拍、materialize 钉住跳过 + prefill 槽位重映射、CPU executor 冷行 id 重映射。
"""

import json
import logging

import numpy as np
import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info

LRU_FLOOR = 512  # init_hot_pins 的 LRU 区绝对下限（max(2E, 512) 的 512 项）


def _init_tp():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _bf16_offload_layer(layer_id: int, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int):
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig

    return OffloadMoELayer(
        layer_id, num_experts, top_k, hidden_size, intermediate_size,
        quant_config=NoQuantConfig(), prefix=f"model.layers.{layer_id}.mlp.experts",
    )


def _write_pin_list(path, pins, num_layers, num_experts, slots=None, version=1):
    """按选点工具 write_pin_list 的 schema 手工组装一个 pin list 文件。"""
    slots = slots if slots is not None else max(len(p) for p in pins)
    payload = {
        "schema_version": version,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "per_layer_slots": slots,
        "pins": [{"layer": i, "experts": list(p)} for i, p in enumerate(pins)],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def _write_stats(path, counts, num_layers, num_experts):
    payload = {
        "schema_version": 1,
        "meta": {"num_layers": num_layers, "num_experts": num_experts, "model_path": "/tmp/m"},
        "counts": counts,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def _lru_ensure_oracle(query, slot_of_id, id_of_slot, lru_usage, lru_step, out_indices,
                       src_indices, dst_indices, num_copy, stats=None, id_base=0, **_):
    """flashlib ``_seq`` 策略的纯 python 镜像（受害槽 argmin(usage, slot)、
    usage == step 不可驱逐、重复查询折叠去重）。仅在 CPU 测试里替换 triton kernel。"""
    k = query.numel()
    num_cached = id_of_slot.numel()
    q = query.long() + id_base
    s = slot_of_id[q].tolist()
    step = int(lru_step.item()) + 1
    lru_step.fill_(step)
    usage = lru_usage.tolist()
    for i in range(k):
        if s[i] >= 0:
            usage[s[i]] = step
    first_seen: dict[int, int] = {}
    for i in range(k):
        first_seen.setdefault(int(q[i]), i)
    missing = sorted(e for e, i in first_seen.items() if s[i] == -1)
    num_copy.fill_(len(missing))
    for rank, e in enumerate(missing):
        victim = min(range(num_cached), key=lambda c: (usage[c], c))
        old = int(id_of_slot[victim].item())
        if old >= 0:
            slot_of_id.view(-1)[old] = -1
        id_of_slot[victim] = e
        slot_of_id.view(-1)[e] = victim
        usage[victim] = step
        dst_indices[rank] = victim
        src_indices[rank] = e - id_base
    out = [
        s[i] if s[i] >= 0 else int(dst_indices[missing.index(int(q[i]))].item()) for i in range(k)
    ]
    out_indices.copy_(torch.tensor(out, dtype=torch.int32))
    lru_usage.copy_(torch.tensor(usage, dtype=lru_usage.dtype))


def _make_pinned_cache(num_layers=2, num_experts=16, pins=(3, 7), k_per_layer=None,
                       cache_size=None, device="cpu", dims=(8, 4), dtype=torch.float32):
    """带冷压缩 bank（尚未装配钉住）的 bf16 cache。

    bank 内容按指纹填充：冷行 (layer, expert) 的两个 bank 行均为标量 l*100+e，
    行均值即来源指纹；未写行（不该存在）预填 1000+l 便于暴露取错行。
    """
    from freetoken.moe.offload_cache import OffloadMoeCache

    _init_tp()
    H, I = dims
    k_per_layer = k_per_layer if k_per_layer is not None else [len(pins)] * num_layers
    total_pins = sum(k_per_layer)
    cache_size = cache_size if cache_size is not None else total_pins + max(2 * num_experts, LRU_FLOOR) + 8
    dev = torch.device(device)
    cache = OffloadMoeCache(
        num_layers=num_layers, num_experts=num_experts, cache_size=cache_size, device=dev,
    )
    gate_up, down = [], []
    for l in range(num_layers):
        k = k_per_layer[l]
        layer_pins = list(pins[:k]) if k else []
        gu = torch.full((num_experts - k, 2 * I, H), float(1000 + l), dtype=dtype)
        dn = torch.full((num_experts - k, H, I), float(1000 + l), dtype=dtype)
        for e in range(num_experts):
            if e in layer_pins:
                continue
            r = _cold_rank(e, layer_pins)
            gu[r] = float(l * 100 + e)
            dn[r] = float(l * 100 + e)
        gate_up.append(gu.pin_memory() if dev.type == "cuda" else gu)
        down.append(dn.pin_memory() if dev.type == "cuda" else dn)
    cache.set_bank_sources(
        {"gate_up": gate_up, "down": down},
        per_layer_rows=[num_experts - k for k in k_per_layer],
    )
    return cache


def _cold_rank(expert, layer_pin_list):
    """专家在层内的冷行号（pinned -> -1）。"""
    if expert in layer_pin_list:
        return -1
    return expert - sum(1 for e in layer_pin_list if e < expert)


def _init_pins(cache, pins=(3, 7), k_per_layer=None):
    """用与 hot_pin 相同的契约装配钉住；返回每层钉住列表。"""
    k_per_layer = k_per_layer if k_per_layer is not None else [len(pins)] * cache.num_layers
    from freetoken.moe.hot_pin import cold_row_from_pins

    pins_matrix = [list(pins[:k]) if k else [] for k in k_per_layer]
    cold_row = cold_row_from_pins(pins_matrix, cache.num_experts)
    k_max = max(k_per_layer)
    pin_ids = torch.zeros((cache.num_layers, k_max), dtype=torch.int32)
    for l, row in enumerate(pins_matrix):
        if row:
            pin_ids[l, : len(row)] = torch.tensor(row, dtype=torch.int32)
    cache.init_hot_pins(pin_ids, k_per_layer, cold_row)
    return pins_matrix


def _apply_copy_plan(cache, layer_id):
    """CPU 上模拟 fused copy_missing：按计划把冷 bank 行拷进对应槽。"""
    n = int(cache.num_indices.item())
    for i in range(n):
        slot = int(cache.evict_slots[i].item())
        row = int(cache.src_indices[i].item())
        for per_layer, bank_cache in cache.banks:
            bank_cache[slot] = per_layer[layer_id][row]


def _load_pinned_slot_contents(cache):
    """真实系统里由 pin_sink arena 装载；测试里直接把钉住槽写成来源指纹。"""
    for l in range(cache.num_layers):
        for j in range(cache.pin_counts[l]):
            slot = int(cache.pin_slots[l, j].item())
            e = int(cache.pin_ids[l, j].item())
            for _, bank_cache in cache.banks:
                bank_cache[slot] = float(l * 100 + e)


# ---------------------------------------------------------------------------
# pin list / stats 契约
# ---------------------------------------------------------------------------


def test_load_pin_list_roundtrip_and_errors(tmp_path):
    """合法 pin list 读回原序；几何/版本/重复等坏文件报 ValueError。"""
    from freetoken.moe.hot_pin import _load_pin_list

    path = _write_pin_list(tmp_path / "pins.json", [[7, 2], [0, 5]], num_layers=2, num_experts=8)
    assert _load_pin_list(str(path), 2, 8) == [[7, 2], [0, 5]]

    _write_pin_list(tmp_path / "bad_dims.json", [[7, 2], [0, 5]], num_layers=2, num_experts=4)
    with pytest.raises(ValueError, match="不一致"):
        _load_pin_list(str(tmp_path / "bad_dims.json"), 2, 8)
    _write_pin_list(tmp_path / "bad_ver.json", [[7, 2], [0, 5]], 2, 8, version=2)
    with pytest.raises(ValueError, match="schema_version"):
        _load_pin_list(str(tmp_path / "bad_ver.json"), 2, 8)
    _write_pin_list(tmp_path / "dup.json", [[2, 2], [0, 5]], 2, 8)
    with pytest.raises(ValueError, match="重复"):
        _load_pin_list(str(tmp_path / "dup.json"), 2, 8)
    _write_pin_list(tmp_path / "oob.json", [[8, 2], [0, 5]], 2, 8)
    with pytest.raises(ValueError, match="越界"):
        _load_pin_list(str(tmp_path / "oob.json"), 2, 8)
    # 误把 stats 文件当 pin list：给出需要 --hot-expert-slots 的可操作提示
    _write_stats(tmp_path / "stats.json", [[0] * 8, [0] * 8], 2, 8)
    with pytest.raises(ValueError, match="--hot-expert-slots"):
        _load_pin_list(str(tmp_path / "stats.json"), 2, 8)


def test_cold_row_is_inverse_of_pins():
    """cold_row：pinned -> -1，冷专家 -> 按 id 升序的行号；无钉层为恒等。"""
    from freetoken.moe.hot_pin import cold_row_from_pins

    cr = cold_row_from_pins([[7, 2], []], 8)
    assert cr[0].tolist() == [0, 1, -1, 2, 3, 4, 5, -1]
    assert cr[1].tolist() == list(range(8))
    # 互逆性：每个冷行号恰对应一个专家
    for e in range(8):
        r = int(cr[0, e])
        if r >= 0:
            assert [i for i in range(8) if int(cr[0, i]) == r] == [e]


def test_resolve_hot_pin_plan_entries_and_cpu_skip(tmp_path):
    """pin list / stats+slots 双入口、CPU 解码层剥离、空计划与错误用法。"""
    from freetoken.moe.hot_pin import resolve_hot_pin_plan

    # 两者都未设置 -> None
    assert resolve_hot_pin_plan(None, None, 2, 8) is None
    # slots 缺 list
    with pytest.raises(ValueError, match="--hot-expert-list"):
        resolve_hot_pin_plan(None, 2, 2, 8)
    # stats + slots：内部选点（每层 top-2，计数降序平局取小 id）
    stats = _write_stats(tmp_path / "s.json", [[9, 9, 1, 1, 0, 0, 0, 0], [5] * 8], 2, 8)
    plan = resolve_hot_pin_plan(str(stats), 2, 2, 8)
    assert plan.pins == [[0, 1], [0, 1]]
    # pin list 单独入口
    path = _write_pin_list(tmp_path / "p.json", [[6], [1]], 2, 8, slots=1)
    plan = resolve_hot_pin_plan(str(path), None, 2, 8)
    assert plan.pins == [[6], [1]]
    # stats 几何不匹配
    _write_stats(tmp_path / "m.json", [[0] * 4, [0] * 4], 2, 4)
    with pytest.raises(ValueError, match="不一致"):
        resolve_hot_pin_plan(str(tmp_path / "m.json"), 1, 2, 8)
    # CPU 解码层剥离 + 记录在 skipped
    plan = resolve_hot_pin_plan(str(path), None, 2, 8, cpu_layer_ids=frozenset({0}))
    assert plan.pins == [[], [1]]
    assert plan.skipped == {0: [6]}
    # 全部层都被剥离 -> None
    assert resolve_hot_pin_plan(str(path), None, 2, 8, cpu_layer_ids=frozenset({0, 1})) is None
    # slots < 1
    with pytest.raises(ValueError, match="--hot-expert-slots"):
        resolve_hot_pin_plan(str(path), 0, 2, 8)


# ---------------------------------------------------------------------------
# 冷压缩 bank
# ---------------------------------------------------------------------------


def test_build_expert_banks_cold_compression_and_pin_staging():
    """钉住层 bank 压缩为 [E-K] 行且内容按冷行号对号入座；钉住权重按 pin list 序
    进入 per-layer 暂存并经 sink 移交（每层一次）；无钉住时行为与原先一致。"""
    from freetoken.moe.expert_banks import build_expert_banks

    _init_tp()
    L, E, H, I = 2, 8, 16, 8
    layer = _bf16_offload_layer(0, E, 2, H, I)
    ref_gu = torch.arange(E * 2 * I * H, dtype=torch.float32).reshape(E, 2 * I, H).bfloat16()
    ref_dn = torch.arange(E * H * I, dtype=torch.float32).reshape(E, H, I).bfloat16()
    pieces = []
    for l in range(L):
        gu = ref_gu + l * 1000
        dn = ref_dn + l * 1000
        pieces.append((l, 0, E // 2, {"gate_up": gu[: E // 2], "down": dn[: E // 2]}))
        pieces.append((l, E // 2, E, {"gate_up": gu[E // 2:], "down": dn[E // 2:]}))

    pin_sets = {0: [7, 2], 1: [0, 5]}
    sinks = []
    banks = build_expert_banks(
        layer.quant_method, L, pieces, device=torch.device("cpu"),
        pin_sets=pin_sets, pin_sink=lambda l, s: sinks.append((l, s)),
    )
    assert banks.cold_row.shape == (L, E)
    assert banks.cold_row[0].tolist() == [0, 1, -1, 2, 3, 4, 5, -1]
    assert banks.sources["gate_up"][0].shape[0] == E - 2
    assert banks.sources["gate_up"][1].shape[0] == E - 2
    # 冷行内容 == 对应专家 piece 行；钉住权重进入暂存的 pin list 序行
    for l in range(L):
        for e in range(E):
            r = int(banks.cold_row[l, e])
            gu_ref = (ref_gu[e].float() + l * 1000).bfloat16()
            if r >= 0:
                assert torch.equal(banks.sources["gate_up"][l][r], gu_ref), (l, e)
            else:
                j = pin_sets[l].index(e)
                stage = dict(sinks)[l]
                assert torch.equal(stage["gate_up"][j], gu_ref), (l, e)
    # 每层恰好移交一次，暂存形状 [K, *row]
    assert [l for l, _ in sinks] == [0, 1]
    assert dict(sinks)[0]["gate_up"].shape == (2, 2 * I, H)

    # 无钉住回归：全量行数 + cold_row None
    banks2 = build_expert_banks(layer.quant_method, L, pieces, device=torch.device("cpu"))
    assert banks2.cold_row is None
    assert all(t.shape[0] == E for t in banks2.sources["gate_up"])

    # K >= E 拒绝（冷 bank 至少 1 行）
    with pytest.raises(ValueError, match="钉住数"):
        build_expert_banks(layer.quant_method, L, pieces, device=torch.device("cpu"),
                           pin_sets={0: list(range(E))}, pin_sink=lambda *a: None)


# ---------------------------------------------------------------------------
# OffloadMoeCache 钉住装配
# ---------------------------------------------------------------------------


def test_init_hot_pins_maps_accounting_and_floors(caplog):
    """预填映射（slot_for_id/id_of_slot 互逆、钉住槽在顶部区）、记账字节与
    overlap / LRU 地板 / K > E 校验。"""
    cache = _make_pinned_cache(num_layers=2, num_experts=16, pins=(3, 7))
    with caplog.at_level(logging.INFO, logger="freetoken.moe.offload_cache"):
        pins_matrix = _init_pins(cache, pins=(3, 7))
    total_pins = 4  # 2 层 x K=2
    assert cache.pin_base == cache.cache_size - total_pins
    for l in range(2):
        for j, e in enumerate(pins_matrix[l]):
            slot = cache.pin_base + l * 2 + j
            assert int(cache.slot_for_id[l, e].item()) == slot
            assert int(cache.id_of_slot[slot].item()) == l * 16 + e
    # 记账：钉住字节 = 每层 2 专家 x 两 bank 行字节（dims=(8,4)：8*8*4 + 8*4*4）
    row_bytes = 2 * 4 * 8 * 4 + 8 * 4 * 4
    assert cache.pinned_bytes() == total_pins * row_bytes
    assert "pinned" in caplog.text and "LRU region" in caplog.text
    assert f"{cache.pinned_bytes() / 2**20:.1f} MiB" in caplog.text

    # overlap 拒绝（错误信息说明将由三源组装解除）
    cache2 = _make_pinned_cache(num_layers=1, num_experts=16, pins=(3,))
    cache2.prefill_overlap = True
    with pytest.raises(ValueError, match="overlap|三源"):
        _init_pins(cache2, pins=(3,), k_per_layer=[1])
    # LRU 地板拒绝：P + max(2E, 512) 放不下
    from freetoken.moe.hot_pin import cold_row_from_pins

    small = _make_pinned_cache(num_layers=1, num_experts=16, pins=(3, 1), cache_size=64, k_per_layer=[2])
    with pytest.raises(ValueError, match="LRU"):
        small.init_hot_pins(
            torch.tensor([[3, 1]], dtype=torch.int32), [2],
            cold_row_from_pins([[3, 1]], 16),
        )
    # K > E 拒绝
    with pytest.raises(ValueError, match="专家数"):
        small.init_hot_pins(torch.ones((1, 20), dtype=torch.int32), [20], cold_row_from_pins([[]], 16))


def test_reset_and_rebuild_restore_pin_maps():
    """reset/rebuild 清空全部映射后必须立即重钉（钉住专家永远命中不变式）。"""
    from freetoken.moe import offload_kernels

    cache = _make_pinned_cache(num_layers=2, num_experts=16, pins=(3, 7))
    pins_matrix = _init_pins(cache, pins=(3, 7))

    def fake_reset(c):
        # 模拟 _reset_cache_kernel：清空映射与时钟
        c.slot_for_id.fill_(-1)
        c.id_of_slot.fill_(-1)
        c.usage.zero_()
        c.step.zero_()

    orig = offload_kernels.reset_cache
    offload_kernels.reset_cache = fake_reset
    try:
        cache.reset()
    finally:
        offload_kernels.reset_cache = orig
    for l in range(2):
        for j, e in enumerate(pins_matrix[l]):
            assert int(cache.slot_for_id[l, e].item()) == cache.pin_base + l * 2 + j

    # rebuild：cache_size 变化 -> 几何重解算 + 重钉
    cache.rebuild(cache.cache_size + 16)
    assert cache.pin_base == cache.cache_size - 4
    for l in range(2):
        for j, e in enumerate(pins_matrix[l]):
            assert int(cache.slot_for_id[l, e].item()) == cache.pin_base + l * 2 + j
    # rebuild 地板：钉住区扣掉后 LRU 不足 -> 拒绝
    with pytest.raises(ValueError, match="LRU"):
        cache.validate_rebuild(4 + 100)


# ---------------------------------------------------------------------------
# 合并查询 ensure（flashlib 路径，CPU oracle 驱动）
# ---------------------------------------------------------------------------


def test_merged_query_ensure_pins_survive_lru(monkeypatch):
    """数千步随机路由后钉住槽 id_of_slot 不变、钉住专家恒命中、冷专家 LRU 正常且
    copy 计划经 cold_row remap 取到正确的冷行（指纹校验全链路）。"""
    from freetoken.moe import offload_kernels

    monkeypatch.setattr(offload_kernels, "lru_ensure", _lru_ensure_oracle)
    L, E = 2, 16
    pins = (3, 7)
    cache = _make_pinned_cache(num_layers=L, num_experts=E, pins=pins)
    pins_matrix = _init_pins(cache, pins=pins)
    _load_pinned_slot_contents(cache)
    pinned_slots = {
        (l, e): int(cache.slot_for_id[l, e].item()) for l in range(L) for e in pins_matrix[l]
    }

    rng = np.random.default_rng(11)
    for step in range(3000):
        layer_id = step % L
        raw = torch.from_numpy(rng.integers(0, E, size=(2, 2)).astype(np.int32))
        raw.view(-1)[0] = pins_matrix[layer_id][0]  # 保证钉住专家每步都被查询
        ids = raw.clone()
        cache.ensure_experts(layer_id, ids)
        # 钉住槽映射与内容永不变
        for (l, e), slot in pinned_slots.items():
            assert int(cache.id_of_slot[slot].item()) == l * E + e, (step, l, e)
            assert float(cache.banks[0][1][slot].mean().item()) == float(l * 100 + e)
        _apply_copy_plan(cache, layer_id)
        # 路由改写：钉住专家命中钉住槽；冷专家槽经 remap 计划拷入了正确的行
        for i in range(raw.numel()):
            e = int(raw.view(-1)[i].item())
            slot = int(ids.view(-1)[i].item())
            if e in pins_matrix[layer_id]:
                assert slot == pinned_slots[(layer_id, e)], (step, e)
            else:
                assert float(cache.banks[0][1][slot].mean().item()) == float(layer_id * 100 + e), (step, e)
        # 计划条目只含冷行号（钉住专家不产生 copy 项）
        n = int(cache.num_indices.item())
        for i in range(n):
            assert 0 <= int(cache.src_indices[i].item()) < E - len(pins_matrix[layer_id])


# ---------------------------------------------------------------------------
# hybrid 路径（CPU 镜像原生跑；GPU 与镜像对拍见 CUDA 用例）
# ---------------------------------------------------------------------------


def test_hybrid_cpu_mirror_with_pins():
    """CPU 镜像 + 钉住：受害槽不取顶部区、src_indices 写冷行号、溢出 miss 重写 -1，
    数千步后钉住槽 id_of_slot 不变。"""
    L, E = 2, 16
    pins = (5, 9, 1)
    cache = _make_pinned_cache(
        num_layers=L, num_experts=E, pins=pins,
        cache_size=3 + max(2 * E, LRU_FLOOR) + 8,
    )
    pins_matrix = _init_pins(cache, pins=pins)
    pinned_slots = {
        (l, e): int(cache.slot_for_id[l, e].item()) for l in range(L) for e in pins_matrix[l]
    }
    rng = np.random.default_rng(23)
    for step in range(3000):
        layer_id = step % L
        ids = torch.from_numpy(rng.integers(0, E, size=(1, 4)).astype(np.int32))
        cache.ensure_experts_hybrid(layer_id, ids)
        for (l, e), slot in pinned_slots.items():
            assert int(cache.id_of_slot[slot].item()) == l * E + e, (step, l, e)
            assert int(cache.slot_for_id[l, e].item()) == slot
        n = int(cache.num_indices.item())
        for i in range(n):
            slot = int(cache.evict_slots[i].item())
            assert slot < cache.pin_base, (step, slot)
            row = int(cache.src_indices[i].item())
            assert 0 <= row < E - len(pins_matrix[layer_id]), (step, row)
        for s in ids.view(-1).tolist():
            assert s == -1 or 0 <= s < cache.cache_size


# ---------------------------------------------------------------------------
# GPU（真实 flashlib / triton kernel）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_flashlib_merged_query_pin_never_evicted_gpu():
    """真实 flashlib lru_ensure：合并查询数千步后钉住槽内容不变，冷专家按 cold_row
    正确换入（fused copy_missing 全链路）。"""
    L, E = 2, 16
    pins = (3, 7)
    cache = _make_pinned_cache(num_layers=L, num_experts=E, pins=pins, device="cuda")
    pins_matrix = _init_pins(cache, pins=pins)
    _load_pinned_slot_contents(cache)
    torch.cuda.synchronize()
    pinned_slots = {
        (l, e): int(cache.slot_for_id[l, e].item()) for l in range(L) for e in pins_matrix[l]
    }

    rng = np.random.default_rng(5)
    for step in range(2000):
        layer_id = step % L
        raw = torch.from_numpy(rng.integers(0, E, size=(1, 4)).astype(np.int32)).cuda()
        raw.view(-1)[0] = pins_matrix[layer_id][0]
        ids = raw.clone()
        cache.ensure_experts(layer_id, ids)
        cache.copy_missing()
        torch.cuda.synchronize()
        for (l, e), slot in pinned_slots.items():
            assert int(cache.id_of_slot[slot].item()) == l * E + e, (step, l, e)
            assert float(cache.banks[0][1][slot].mean().item()) == float(l * 100 + e), (step, l, e)
        for i in range(raw.numel()):
            e = int(raw.view(-1)[i].item())
            slot = int(ids.view(-1)[i].item())
            if e in pins_matrix[layer_id]:
                assert slot == pinned_slots[(layer_id, e)], (step, e)
            else:
                assert float(cache.banks[0][1][slot].mean().item()) == float(layer_id * 100 + e), (step, e)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_kernel_with_pins_matches_cpu_mirror_gpu():
    """真实 hybrid kernel（pin_base 排除 + cold_row src）与 CPU 镜像逐位对拍。"""
    L, E = 2, 16
    pins = (5, 9, 1)
    gpu = _make_pinned_cache(num_layers=L, num_experts=E, pins=pins, device="cuda")
    ref = _make_pinned_cache(num_layers=L, num_experts=E, pins=pins, device="cpu")
    pins_matrix = _init_pins(gpu, pins=pins)
    _init_pins(ref, pins=pins)
    rng = np.random.default_rng(9)
    for step in range(200):
        layer_id = step % L
        ids = torch.from_numpy(rng.integers(0, E, size=(1, 4)).astype(np.int32))
        g, c = ids.clone().cuda(), ids.clone()
        gpu.ensure_experts_hybrid(layer_id, g)
        ref.ensure_experts_hybrid(layer_id, c)
        torch.cuda.synchronize()
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert torch.equal(gpu.usage.cpu(), ref.usage.cpu())
        n = int(gpu.num_indices.item())
        assert n == int(ref.num_indices.item())
        # 计划条目（前 n 项）逐位一致；计划之外的尾部是无效垃圾，不比较
        assert torch.equal(gpu.evict_slots[:n].cpu(), ref.evict_slots[:n].cpu())
        assert torch.equal(gpu.src_indices[:n].cpu(), ref.src_indices[:n].cpu())
        # 钉住槽永不被驱逐（P = 2 层 x 3）
        for l in range(L):
            for j, e in enumerate(pins_matrix[l]):
                assert int(gpu.slot_for_id[l, e].item()) == gpu.pin_base + l * 3 + j


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_materialize_with_pins_skips_pinned_and_installs_bytes_gpu(monkeypatch):
    """materialize：冷专家进 [0,E)（计划按冷行压缩）、钉住专家的暂存位是空槽且其
    字节由 D2D 安装；prefill 保持原始形态（原始 id 直通、views 取前 E 行、n=E）。"""
    _init_tp()
    L, E, H, I = 2, 16, 32, 16
    pins = (3, 7)
    cache = _make_pinned_cache(num_layers=L, num_experts=E, pins=pins, device="cuda", dims=(H, I))
    pins_matrix = _init_pins(cache, pins=pins)
    _load_pinned_slot_contents(cache)
    layer_id = 1
    layer = _bf16_offload_layer(layer_id, E, 3, H, I)
    layer.offload_cache = cache

    cache.materialize_layer(layer_id)
    cache.copy_missing()
    torch.cuda.synchronize()
    n = int(cache.num_indices.item())
    assert n == E - len(pins_matrix[layer_id])
    # 计划按冷行压缩：第 i 项 -> 槽位 == 第 i 个冷专家 id（slot == expert id），src == i
    cold_ids = [e for e in range(E) if e not in pins_matrix[layer_id]]
    for i, e in enumerate(cold_ids):
        assert int(cache.evict_slots[i].item()) == e
        assert int(cache.src_indices[i].item()) == i
        assert float(cache.banks[0][1][e].mean().item()) == float(layer_id * 100 + e)
    # 钉住映射原封不动；暂存位安装了钉住字节且在映射上是空槽
    for j, e in enumerate(pins_matrix[layer_id]):
        assert int(cache.slot_for_id[layer_id, e].item()) == cache.pin_base + layer_id * 2 + j
        assert int(cache.id_of_slot[e].item()) == -1
        assert int(cache.usage[e].item()) == 0
        for _per_layer, bank_cache in cache.banks:
            assert torch.equal(bank_cache[e], bank_cache[int(cache.pin_slots[layer_id, j])]), e

    # prefill 保持原始形态：原始 id 直通、views 取前 E 行、n = E
    topk_weights = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32)
    topk_ids = torch.tensor([[3, 7, 0]], dtype=torch.int32)  # 两个钉住 + 一个冷专家
    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda *, hidden_states, gating_output, topk, renormalize: (topk_weights, topk_ids),
    )
    captured = {}

    def fake_gemm(cache_, hidden, w, ids, *, views, n, alphas, is_prefill):
        captured.update(ids=ids.clone(), views=views, n=n, alphas=alphas, prefill=is_prefill)
        return hidden

    monkeypatch.setattr(layer, "_expert_gemm", fake_gemm)
    layer.prefill_forward(torch.randn(1, H), torch.randn(1, E))
    assert captured["prefill"] is True
    assert captured["n"] == E
    assert captured["alphas"] is None
    assert captured["views"][0].data_ptr() == cache.bank_caches["gate_up"].data_ptr()
    assert captured["views"][0].shape[0] == E
    assert captured["ids"].tolist() == [[3, 7, 0]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_executor_decode_submit_remaps_ids_to_cold_rows():
    """CPU executor：发往 CPU 的专家 id 经 cold_row 重映射为冷行号（-1 原样保留）。"""
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    L, E = 1, 16
    pins = (2, 10)
    cache = _make_pinned_cache(num_layers=L, num_experts=E, pins=pins, device="cuda", dtype=torch.bfloat16)
    pins_matrix = _init_pins(cache, pins=pins)
    executor = CpuMoeExecutor(
        cache, top_k=3, activation="silu", apply_router_weight_on_input=False,
        num_threads=2, max_tokens=4, device=torch.device("cuda"),
    )
    H = int(cache.bank_sources["gate_up"][0].shape[2])
    hidden = torch.randn(2, H, device="cuda")
    weights = torch.rand(2, 3, device="cuda")
    ids = torch.tensor([[2, 10, 5], [0, -1, 2]], dtype=torch.int32, device="cuda")
    executor.decode_submit(0, hidden, weights, ids)
    torch.cuda.synchronize()
    got = executor._io[2]["ids"].cpu().tolist()
    cr = cache.cold_row[0].cpu().tolist()
    assert got == [[cr[2], cr[10], cr[5]], [cr[0], -1, cr[2]]]
