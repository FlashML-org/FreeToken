"""Hybrid decode's bandwidth-matched fetch split.

Covers the two halves of --moe-hybrid-max-fetch auto: the profile reader that turns
`ft bench bw` kernel bandwidths into a fetch fraction, and the ensure kernel's
per-step integer split (GPU kernel vs CPU reference mirror, and the balance rule).
"""

import json
import os

import pytest
import torch

from freetoken.moe.bench_profile import default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction
from freetoken.moe.offload_cache import OffloadMoeCache

Q = 1 << 16


def _balanced_fetch(num_missing: int, frac_q16: int) -> int:
    """Reference split: F ~ frac * misses, rounded to whichever integer neighbor
    minimizes the slower overlapped side (fetch ~ F*(1-frac), CPU ~ (M-F)*frac)."""
    lo = (num_missing * frac_q16) >> 16
    cost = lambda f: max(f * (Q - frac_q16), (num_missing - f) * frac_q16)  # noqa: E731
    return min(num_missing, lo if cost(lo) <= cost(lo + 1) else lo + 1)


def test_balanced_fetch_tracks_fraction():
    # The split follows fetched : cpu = pcie : (cpu - pcie) up to integer rounding, and
    # never over/under-shoots by more than one expert.
    for frac in (0.1, 0.415, 0.454, 0.7, 1.0):
        q = round(frac * Q)
        for m in range(0, 65):
            f = _balanced_fetch(m, q)
            assert 0 <= f <= m
            assert abs(f - frac * m) <= 1.0
    # ceil would over-fetch here (the regression this rule fixed): 41.5% of 3 misses is
    # 1.24 -> fetching 2 makes the PCIe side ~1.6x slower than balance; keep it at 1.
    assert _balanced_fetch(3, round(0.415 * Q)) == 1
    assert _balanced_fetch(4, round(0.415 * Q)) == 2


def test_load_hybrid_fetch_fraction(tmp_path):
    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtype_kernels": {
            "bf16": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            "nvfp4_x": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                        "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
        "workloads": {
            "m": {"kernels": {"ds_fp4": {"cpu_moe_gbs": 80.0, "pcie_gather_gbs": 50.0}}}
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert load_hybrid_fetch_fraction("bf16", path=str(path)) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert load_hybrid_fetch_fraction("nvfp4_x", path=str(path)) == pytest.approx(0.25)
    # per-model fallback when there is no per-dtype entry for the format
    assert load_hybrid_fetch_fraction("ds_fp4", path=str(path)) == pytest.approx(0.625)
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path)) is None
    # a profile from different hardware is ignored
    assert load_hybrid_fetch_fraction("bf16", gpu_name="OTHER", path=str(path)) is None


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, verdict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"gpu": {"name": name}, "dtypes": {"bf16": verdict}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "hybrid")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "hybrid"
    assert load_backend_recommendation("bf16", gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "offload")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "offload"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fraction_gpu_matches_cpu_reference():
    torch.manual_seed(0)
    num_experts, cache_size, top_k, frac = 32, 40, 8, 0.415

    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts, hybrid_fetch_fraction=frac,
        )

    gpu, ref = make(), make()
    frac_q16 = round(frac * Q)
    for step in range(64):
        ids = torch.randperm(num_experts)[:top_k].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()  # a CPU ids tensor drives the reference path
        gpu.ensure_experts_hybrid(0, g)
        ref.ensure_experts_hybrid(0, c)
        missing = int(gpu.num_missing_full.item())
        fetched = int(gpu.num_indices.item())
        assert missing == int(ref.num_missing_full.item())
        assert fetched == int(ref.num_indices.item()) == _balanced_fetch(missing, frac_q16)
        # slot rewrites (hit/fetched -> slot, overflow -> -1) and LRU state stay identical
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert (g >= 0).sum().item() == len(set(ids.tolist())) - (missing - fetched)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fixed_cap_unchanged():
    # fraction 0 (no profile / explicit --moe-hybrid-max-fetch) keeps the fixed cap.
    cache = OffloadMoeCache(
        num_layers=1, num_experts=32, cache_size=40, device=torch.device("cuda"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=1,
    )
    ids = torch.arange(8, dtype=torch.int32).cuda()
    cache.ensure_experts_hybrid(0, ids)
    assert int(cache.num_missing_full.item()) == 8
    assert int(cache.num_indices.item()) == 1


def test_set_fetch_params_updates_attrs_and_device_values():
    """set_fetch_params 的 Q16 换算与属性/设备值一致性（CPU 张量即可验证）。

    Business Logic（为什么需要这个测试）:
        运行中调参（--tune-file）只经 set_fetch_params 写入 hybrid 拉取参数；设备
        张量（内核按指针读取的权威值）与 Python 属性（CPU 镜像与测试读取的镜像值）
        必须始终一致，且 q16 公式与 ensure_experts_hybrid 入口完全相同，否则
        GPU/CPU 两路分流会分叉。

    Code Logic（这个测试做什么）:
        在 CPU 设备构造 hybrid cache，依次验证：构造时设备张量已镜像初始属性；
        set_fetch_params(7, 0.415) 后属性与 fetch_params[0]/[1] 同步为新值；旧语义
        set_fetch_params(1, 0.0) = 固定 cap 1、无比例分流；比例越界被夹界（>1 饱和
        到 1<<16、<0 夹到 0）。另验证非 hybrid 模式 fetch_params 保持 None 且
        set_fetch_params 只更新属性不崩溃。
    """
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=6, device=torch.device("cpu"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=1,
    )
    # 构造时设备张量镜像初始属性（无 profile 旧语义：cap 1、无比例）
    assert cache.fetch_params is not None
    assert cache.fetch_params.tolist() == [1, 0]
    cache.set_fetch_params(7, 0.415)
    assert cache.hybrid_max_fetch == 7
    assert cache.hybrid_fetch_fraction == pytest.approx(0.415)
    assert cache.fetch_params.tolist() == [7, round(0.415 * Q)]
    # 无 profile 旧语义保留：set_fetch_params(1, 0.0) = 固定 cap 1、无比例分流
    cache.set_fetch_params(1, 0.0)
    assert cache.hybrid_max_fetch == 1
    assert cache.fetch_params.tolist() == [1, 0]
    # 夹界与 ensure_experts_hybrid 入口公式一致：>1 饱和、<0 归零
    cache.set_fetch_params(3, 1.5)
    assert cache.fetch_params[1].item() == 1 << 16
    cache.set_fetch_params(3, -0.5)
    assert cache.fetch_params[1].item() == 0

    # 非 hybrid 模式：无 fetch_params 张量，set_fetch_params 只更新属性
    gpu_cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=6, device=torch.device("cpu")
    )
    assert gpu_cache.fetch_params is None
    gpu_cache.set_fetch_params(2, 0.5)
    assert gpu_cache.hybrid_max_fetch == 2
    assert gpu_cache.hybrid_fetch_fraction == pytest.approx(0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fetch_params_update_applies_to_captured_graph():
    """运行中 set_fetch_params 必须对已捕获 decode 图立即生效（本功能验收核心）。

    Business Logic（为什么需要这个测试）:
        hybrid 的 cap/比例原先作为内核标量实参在 CUDA graph 捕获时被冻结，运行中
        改 Python 属性无效；改为 fetch_params 设备张量按指针读取后，必须做到
        "不重捕获、只改值"即可让同一条已捕获图的每步拉取量按新比例变化。

    Code Logic（这个测试做什么）:
        构造 hybrid cache（fraction=A=0.25），以固定 expert_ids 把
        ensure_experts_hybrid 捕获进 CUDA graph（eager 预跑完成 Triton JIT 后捕获）；
        每轮 replay 前用图外 eager 的 cache.reset() 恢复冷态并回写 expert_ids（内核
        原地改写），replay 后统计被改写为 -1 的溢出 miss 数。先断言 A 比例的溢出数
        符合带宽数学期望，然后不重捕获、set_fetch_params(fraction=B=0.75) 后再
        replay 同一 graph，断言溢出数按 B 比例变化且两轮不同。
    """
    torch.manual_seed(0)
    num_experts, cache_size, top_k = 32, 40, 8
    frac_a, frac_b = 0.25, 0.75
    missing = top_k  # 冷态下互异 ids 全为 miss
    cache = OffloadMoeCache(
        num_layers=1, num_experts=num_experts, cache_size=cache_size,
        device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
        hybrid_max_fetch=num_experts, hybrid_fetch_fraction=frac_a,
    )
    ids_orig = torch.arange(top_k, dtype=torch.int32).cuda()
    ids_buf = ids_orig.clone()
    # eager 预跑：Triton JIT 编译必须发生在捕获之前
    cache.ensure_experts_hybrid(0, ids_buf)
    cache.reset()
    ids_buf.copy_(ids_orig)
    # 官方推荐的侧流预热 -> 捕获模式
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        cache.ensure_experts_hybrid(0, ids_buf)
    torch.cuda.current_stream().wait_stream(side)
    cache.reset()
    ids_buf.copy_(ids_orig)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache.ensure_experts_hybrid(0, ids_buf)

    def replay_overflow() -> tuple[int, int]:
        """从冷态 replay 一次已捕获图，返回 (溢出 -1 数, 实际拉取数)。"""
        ids_buf.copy_(ids_orig)
        cache.reset()
        graph.replay()
        torch.cuda.synchronize()
        overflow = int((ids_buf == -1).sum())
        return overflow, int(cache.num_indices.item())

    q16_a = round(frac_a * Q)
    overflow_a, fetched_a = replay_overflow()
    assert int(cache.num_missing_full.item()) == missing
    assert overflow_a == missing - _balanced_fetch(missing, q16_a)
    assert fetched_a == _balanced_fetch(missing, q16_a)

    # 关键断言：不重捕获，只改 fetch_params 的值，同一条已捕获图按新比例分流
    cache.set_fetch_params(cache.hybrid_max_fetch, frac_b)
    assert cache.fetch_params[1].item() == round(frac_b * Q)
    q16_b = round(frac_b * Q)
    overflow_b, fetched_b = replay_overflow()
    assert overflow_b == missing - _balanced_fetch(missing, q16_b)
    assert fetched_b == _balanced_fetch(missing, q16_b)
    assert overflow_b != overflow_a


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_set_fetch_params_on_inference_mode_constructed_cache():
    """launch.py 在 torch.inference_mode() 下构造 Scheduler：cache 缓冲因此是
    inference tensor，运行中（模式外）的 set_fetch_params 就地写必须仍可用
    （写路径显式进入该模式），否则 --tune-file 轮询每轮静默失败。"""
    with torch.inference_mode():
        cache = OffloadMoeCache(
            num_layers=1, num_experts=32, cache_size=40, device=torch.device("cuda"),
            quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=32, hybrid_fetch_fraction=0.375,
        )
    assert int(cache.fetch_params[1]) == round(0.375 * Q)
    # 模式外（模拟轮询线程）更新：修复前在此抛 Inplace-update 运行时错误
    cache.set_fetch_params(32, 0.5)
    assert cache.hybrid_fetch_fraction == 0.5
    assert int(cache.fetch_params[1]) == round(0.5 * Q)
    # 新值对已捕获的图生效：fraction 0.5 下 8 个全 miss 应取平衡拉取数
    ids = torch.arange(8, dtype=torch.int32).cuda()
    cache.ensure_experts_hybrid(0, ids)
    assert int(cache.num_indices.item()) == _balanced_fetch(8, round(0.5 * Q))
