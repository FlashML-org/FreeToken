"""prefill overlap 三源组装（显存钉住 × --moe-prefill-overlap/hit-d2d 共存）。

核心验收：同一批 prefill chunk（含"冷专家部分命中"与"钉住专家全命中"）下，
四种装配的缓冲字节与逐 token 输出完全一致：

* 钉住 + overlap + hit-D2D split（miss 经 cold_row remap 进 batch，小 bank 组合填充；
  本机缺 CUDA 13 nvcc 时以 ctypes cudaMemcpyAsync 垫片替代 batch 语义）；
* 钉住 + overlap + hit-D2D 不可用回退（组合填充：冷行自 bank + 钉住行自顶部槽）；
* 钉住 + 无 overlap（materialize + 钉住 D2D 安装，stage1 路径回归）；
* 无钉住 + overlap（上游基线，验证重构未改变原语义）。
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# gate_up [2I, H] bf16 = 262144 B >= _SMALL_BANK_FEAT_BYTES（走 gather + batch remap 的
# "大 bank"路径）；down [H, I] bf16 = 131072 B < 阈值（走组合填充的"小 bank"路径）。
# 一个 cache 同时覆盖两条装配路径。
NUM_LAYERS, E, H, I = 2, 8, 512, 128
PINS = [[3, 5], [2, 6]]  # 每层独立钉住集
P = sum(len(p) for p in PINS)
CACHE_SIZE = 2 * E + 512 + P  # LRU 地板 max(2E,512)=512 + 钉住区 -> pin_base = 2E + 512
TOKENS, TOPK = 2, 2  # 每个 chunk 每层的路由展开为 [TOKENS, TOPK]（共 4 个路由项）

# 每个 chunk 的每层路由：pinned（恒命中）+ 冷专家 hit + 冷专家 miss + 双缓冲借用区
# 里的陈旧命中（slot < 2E，必须重取）。chunk B 用连续冷 id 的长 run 验证 remap 的
# coalescing（L0 [4,6,7]、L1 [4,5]）。
CHUNK_IDS = [
    [[3, 1, 2, 4], [2, 3, 7, 4]],
    [[3, 4, 6, 7], [6, 3, 4, 5]],
]
# 预热种子：hit 区槽位（>= 2E）与借用区槽位（< 2E，字节将被投毒）
WARM_SEEDS = [{1: 2 * E, 4: 5}, {3: 2 * E + 1, 4: 6}]
# split 路径的命中统计（按整行 slot 映射计，不按路由）：每次预取的 hit = 该层全部
# slot >= 2E 的专家 = 2 个钉住 + 存活的 warm 种子。chunk A/B 每层恰 3 hit -> 4 次预取
# 共 12；借用区种子（slot 5/6）在 chunk A 预取时即被失效，不计入。
SPLIT_HIT_ROWS = 12


def _init_tp() -> None:
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _batch_memcpy_or_shim():
    """可用的 batch-memcpy 入口；真 binding 不可用时退化为 ctypes cudaMemcpyAsync。

    真路径（CUDA 13 工具链可 JIT）即生产语义；垫片按相同签名逐条目 enqueue 异步
    拷贝（batch 条目两两独立、顺序无关），保证 split 路径在任何环境可测。
    """
    from freetoken.kernel.batch_memcpy import load_batch_memcpy

    try:
        return load_batch_memcpy()
    except Exception:  # noqa: BLE001 -- 本机缺 CUDA 13 nvcc 等环境缺口，走垫片
        pass
    candidates = sorted(
        glob.glob(
            os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cu*", "lib", "libcudart.so*")
        )
        + glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*"))
    )
    assert candidates, "no libcudart found for the batch-memcpy shim"
    rt = ctypes.CDLL(candidates[-1])

    def shim(dst_ptrs: torch.Tensor, src_ptrs: torch.Tensor, sizes: torch.Tensor, stream: int) -> None:
        """按 batch-memcpy 语义逐条目 enqueue cudaMemcpyAsync（kind=Default）。"""
        for dp, sp, nb in zip(dst_ptrs.tolist(), src_ptrs.tolist(), sizes.tolist()):
            rc = rt.cudaMemcpyAsync(
                ctypes.c_void_p(dp),
                ctypes.c_void_p(sp),
                ctypes.c_size_t(nb),
                ctypes.c_int(4),  # cudaMemcpyDefault（UVA 自动判向）
                ctypes.c_void_p(stream),
            )
            assert rc == 0, f"cudaMemcpyAsync failed: cuda error {rc}"

    return shim


def _cold_rank(expert: int, layer_pins: list[int]) -> int:
    """专家在层内的冷行号（pinned -> -1）。"""
    if expert in layer_pins:
        return -1
    return expert - sum(1 for e in layer_pins if e < expert)


def _build_cache(
    pins: list[list[int]] | None,
    overlap: bool,
    hit_d2d: bool = True,
    device: str = "cuda",
) -> OffloadMoeCache:
    """带指纹冷压缩 bank 的 cache（四个装配臂共用，bank 内容逐位一致）。

    指纹：冷行 (l, r) 与钉住顶部槽均为标量 l*100+e；未写行预填 1000+l（bf16 舍入
    后仍远大于指纹值），取错行/漏行必然暴露。
    """
    _init_tp()
    dev = torch.device(device)
    k_per_layer = [len(p) for p in pins] if pins is not None else [0] * NUM_LAYERS
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS,
        num_experts=E,
        cache_size=CACHE_SIZE,
        device=dev,
        prefill_overlap=overlap,
        prefill_hit_d2d=hit_d2d,
    )
    gate_up, down = [], []
    for l in range(NUM_LAYERS):
        k = k_per_layer[l]
        layer_pins = list(pins[l]) if pins is not None else []
        gu = torch.full((E - k, 2 * I, H), float(1000 + l), dtype=torch.bfloat16)
        dn = torch.full((E - k, H, I), float(1000 + l), dtype=torch.bfloat16)
        for e in range(E):
            r = _cold_rank(e, layer_pins)
            if r >= 0:
                gu[r] = float(l * 100 + e)
                dn[r] = float(l * 100 + e)
        gate_up.append(gu.pin_memory() if dev.type == "cuda" else gu)
        down.append(dn.pin_memory() if dev.type == "cuda" else dn)
    cache.set_bank_sources(
        {"gate_up": gate_up, "down": down},
        per_layer_rows=[E - k for k in k_per_layer],
    )
    if pins is not None:
        from freetoken.moe.hot_pin import cold_row_from_pins

        counts = [len(p) for p in pins]
        pin_ids = torch.zeros((NUM_LAYERS, max(counts)), dtype=torch.int32)
        for l, row in enumerate(pins):
            if row:
                pin_ids[l, : len(row)] = torch.tensor(row, dtype=torch.int32)
        cache.init_hot_pins(pin_ids, counts, cold_row_from_pins(pins, E))
        # 钉住权重入顶部区（真实系统由 pin_sink arena 承载）
        for l in range(NUM_LAYERS):
            for j, e in enumerate(pins[l]):
                slot = int(cache.pin_slots[l, j].item())
                for _, bank_cache in cache.banks:
                    bank_cache[slot] = float(l * 100 + e)
    return cache


def _seed_warm_hits(cache: OffloadMoeCache) -> None:
    """预热部分冷专家（hit 区 + 借用区各一）并把借用区槽位字节投毒。

    借用区命中（slot < 2E）必须被分类为 miss 并从 bank 重取：投毒字节出现在
    buffer 即说明分类或取数错误。hit 区种子写入指纹字节（D2D gather 的取数源）。
    """
    for l, seeds in enumerate(WARM_SEEDS):
        for e, slot in seeds.items():
            cache.slot_for_id[l, e] = slot
            cache.id_of_slot[slot] = l * E + e
            if slot >= 2 * E:
                for _, bank_cache in cache.banks:
                    bank_cache[slot] = float(l * 100 + e)
    for _, bank_cache in cache.banks:
        bank_cache[: 2 * E].fill_(float("nan"))


def _expected_layer(cache: OffloadMoeCache, layer_id: int) -> tuple[torch.Tensor, ...]:
    """组合后的整层指纹（buffer/slots 的 position == 专家 id）：每行 = l*100+e。"""
    outs = []
    for per_layer, _cache in cache.banks:
        row = per_layer[layer_id]
        expected = torch.full((E, *row.shape[1:]), float(1000 + layer_id), dtype=row.dtype)
        for e in range(E):
            expected[e] = float(layer_id * 100 + e)
        outs.append(expected)
    return tuple(outs)


def _ref_forward(
    views: tuple[torch.Tensor, ...], x: torch.Tensor, w: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    """bf16 视图上的逐 token 参考输出（fp32 计算：silu(gu@x) 再 dn 投影）。

    四臂的字节一致性隐含输出一致性；此函数把验收落成"逐 token 输出一致"的
    原文形式（任何装配错误都会同时打破两者）。
    """
    import torch.nn.functional as F

    gu, dn = views[0].float().cpu(), views[1].float().cpu()
    xf, wf = x.float().cpu(), w.float().cpu()
    out = torch.zeros(TOKENS, dn.shape[1])  # down [E, H, I] -> 输出维度 H
    for t in range(TOKENS):
        for k in range(TOPK):
            e = int(ids[t, k])
            gate, up = gu[e, : gu.shape[1] // 2], gu[e, gu.shape[1] // 2 :]
            inter = F.silu(gate @ xf[t]) * (up @ xf[t])  # [I]
            out[t] += wf[t, k] * (dn[e] @ inter)  # [H]
    return out


def _check_views(
    cache: OffloadMoeCache, snapshots: list[list[tuple[torch.Tensor, ...]]]
) -> list[list[torch.Tensor]]:
    """视图字节 == 组合指纹；同时跑参考前向返回逐 token 输出。"""
    outs = []
    gen = torch.Generator().manual_seed(7)
    for chunk, per_chunk in zip(CHUNK_IDS, snapshots):
        x = torch.randn(TOKENS, H, generator=gen)
        w = torch.softmax(torch.rand(TOKENS, TOPK, generator=gen), dim=-1)
        outs.append([])
        for layer_id, views in enumerate(per_chunk):
            for got, want in zip(views, _expected_layer(cache, layer_id)):
                assert torch.equal(got.cpu(), want), (layer_id, got.shape)
            outs[-1].append(
                _ref_forward(
                    views, x, w, torch.tensor(chunk[layer_id], dtype=torch.int32).view(TOKENS, TOPK)
                )
            )
    return outs


def _run_overlap_chunks(cache: OffloadMoeCache) -> list[list[tuple[torch.Tensor, ...]]]:
    """跑两个 prefill chunk，返回每 chunk 每层的 bank 视图快照（clone 自缓冲）。"""
    snapshots: list[list[tuple[torch.Tensor, ...]]] = []
    for _chunk in CHUNK_IDS:
        cache.begin_prefill()
        snapshots.append([])
        for layer_id in range(NUM_LAYERS):
            views = cache.wait_prefill_layer(layer_id)
            torch.cuda.synchronize()
            snapshots[-1].append(tuple(v.clone() for v in views))
            cache.release_prefill_layer(layer_id)
    return snapshots


def _run_materialize_chunks(cache: OffloadMoeCache) -> list[list[tuple[torch.Tensor, ...]]]:
    """非 overlap materialize 臂：与 overlap 臂同形的视图快照。"""
    snapshots: list[list[tuple[torch.Tensor, ...]]] = []
    for _chunk in CHUNK_IDS:
        snapshots.append([])
        for layer_id in range(NUM_LAYERS):
            cache.materialize_layer(layer_id)
            cache.copy_missing()
            torch.cuda.synchronize()
            views = tuple(v.clone() for v in cache.bank_views(E))
            for got, want in zip(views, _expected_layer(cache, layer_id)):
                assert torch.equal(got.cpu(), want), (layer_id, got.shape)
            snapshots[-1].append(views)
    return snapshots


# ---------------------------------------------------------------------------
# 四臂装配 + 三方一致性（核心验收）
# ---------------------------------------------------------------------------


@CUDA
def test_pins_overlap_split_matches_sources():
    """钉住 + overlap + hit-D2D split：miss 冷行 remap、钉住/命中 D2D gather、
    小 bank 组合填充全部就位，缓冲字节与指纹一致。"""
    cache = _build_cache(PINS, overlap=True, hit_d2d=True)
    cache._batch_memcpy = _batch_memcpy_or_shim()  # 预置入口，绕过本机 JIT 缺口
    _seed_warm_hits(cache)
    snapshots = _run_overlap_chunks(cache)
    assert cache._prefill_hit_d2d_active
    assert cache.prefill_hit_rows == SPLIT_HIT_ROWS
    assert cache.prefill_total_rows == len(CHUNK_IDS) * NUM_LAYERS * E
    _check_views(cache, snapshots)


@CUDA
def test_pins_overlap_compose_fallback_matches_sources(monkeypatch):
    """hit-D2D 不可用回退：整层组合填充（冷行自 bank + 钉住行自顶部槽）字节正确。"""
    cache = _build_cache(PINS, overlap=True, hit_d2d=True)
    monkeypatch.setattr(cache, "_hit_d2d_usable", lambda: False)
    _seed_warm_hits(cache)
    snapshots = _run_overlap_chunks(cache)
    assert not cache._prefill_hit_d2d_active
    _check_views(cache, snapshots)


@CUDA
def test_hit_d2d_fallback_logs_reason(monkeypatch, caplog):
    """钉住 + hit-d2d 请求但 batch 不可用：一次性回退告警（含原因），随后走组合填充。"""
    cache = _build_cache(PINS, overlap=True, hit_d2d=True)
    monkeypatch.setattr(cache, "_batch_memcpy", False)  # 强制 resolve 失败
    with caplog.at_level(logging.WARNING, logger="freetoken.moe.offload_cache"):
        cache.begin_prefill()
    assert not cache._prefill_hit_d2d_active
    assert "cudaMemcpyBatchAsync is unavailable" in caplog.text
    assert cache._hit_d2d_fallback_logged
    _seed_warm_hits(cache)
    _check_views(cache, _run_overlap_chunks(cache))


@CUDA
def test_pins_materialize_matches_sources():
    """钉住 + 无 overlap（stage1 materialize + 钉住 D2D 安装路径）字节正确。"""
    cache = _build_cache(PINS, overlap=False)
    _seed_warm_hits(cache)
    _check_views(cache, _run_materialize_chunks(cache))


@CUDA
def test_no_pins_overlap_split_regression():
    """无钉住 + overlap 基线：split 重构（miss remap 分支仅钉住时启用）不改变原语义。"""
    cache = _build_cache(None, overlap=True, hit_d2d=True)
    assert cache.pin_ids is None and cache._cold_row_np is None
    cache._batch_memcpy = _batch_memcpy_or_shim()
    _seed_warm_hits(cache)
    snapshots = _run_overlap_chunks(cache)
    assert cache._prefill_hit_d2d_active
    assert cache.prefill_hit_rows == 4  # 只有 warm 种子是 hit
    _check_views(cache, snapshots)


@CUDA
def test_prefill_overlap_pin_three_way_consistency():
    """核心验收：钉住+overlap（split）、钉住+overlap（回退组合填充）、钉住+无
    overlap（materialize）、无钉住基线——同一批 chunk 的视图字节与逐 token 输出
    完全一致（确定性内核，等价温度 0）。"""
    arms = {
        "pins+overlap+split": (_build_cache(PINS, overlap=True), "overlap"),
        "pins+overlap+fallback": (_build_cache(PINS, overlap=True), "fallback"),
        "pins+no-overlap": (_build_cache(PINS, overlap=False), "materialize"),
        "no-pins+overlap": (_build_cache(None, overlap=True), "overlap"),
    }
    caches: dict[str, OffloadMoeCache] = {}
    results = {}
    for name, (cache, mode) in arms.items():
        caches[name] = cache
        _seed_warm_hits(cache)
        if mode == "overlap":
            cache._batch_memcpy = _batch_memcpy_or_shim()
            snapshots = _run_overlap_chunks(cache)
        elif mode == "fallback":
            cache._batch_memcpy = _batch_memcpy_or_shim()
            cache._hit_d2d_usable = lambda: False  # noqa: B023 -- 测试臂内固定
            snapshots = _run_overlap_chunks(cache)
        else:
            snapshots = _run_materialize_chunks(cache)
        results[name] = (snapshots, _check_views(cache, snapshots))

    ref_name = "pins+no-overlap"
    ref_snapshots, ref_outs = results[ref_name]
    for name, (snapshots, outs) in results.items():
        if name == ref_name:
            continue
        for chunk_i in range(len(CHUNK_IDS)):
            for layer_id in range(NUM_LAYERS):
                for got, want in zip(snapshots[chunk_i][layer_id], ref_snapshots[chunk_i][layer_id]):
                    assert torch.equal(got.cpu(), want.cpu()), (name, chunk_i, layer_id)
                assert torch.equal(outs[chunk_i][layer_id], ref_outs[chunk_i][layer_id]), (
                    name,
                    chunk_i,
                    layer_id,
                )
    # 三方一致性的"三方"主体（钉住+overlap 的两种实现 + 钉住+无overlap）逐一对照
    # 无钉住基线之外，split 与 fallback 两种 overlap 实现也必须互为逐位副本
    for chunk_i in range(len(CHUNK_IDS)):
        for layer_id in range(NUM_LAYERS):
            assert torch.equal(
                results["pins+overlap+split"][1][chunk_i][layer_id],
                results["pins+overlap+fallback"][1][chunk_i][layer_id],
            )
