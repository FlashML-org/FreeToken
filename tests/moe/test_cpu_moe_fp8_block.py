"""CPU MoE executor -- 128x128 block-FP8 专家（Qwen3.6-35B-A3B-FP8 等 checkpoint 的
``--moe-strategy cpu``/``hybrid`` 解码；#534 解除 cpu_format=None 的限制）。

CPU GEMV（csrc/cpu_moe/cpu_moe_ext.cpp 的 ``dot_fp8block_*``）读取与 GPU offload
路径逐字节一致的 e4m3 + padded bf16 scale bank，在 K 循环内解量化（每 128-K 块
部分和 x 该行块 scale，fp32 累加，bf16 中间/输出）。本文件用 float64 解量化参考
对拍：e4m3 解码本身是精确的，差异只剩 fp32 归约顺序与 bf16 舍入（各 ~2^-8 相对
量级）；参数化覆盖 padded / 非 padded scale 步长、多 expert 多 token、scalar 与
AVX2 两档 ISA。GPU 部分与产线 triton fp8_block 解码 kernel 在同一份 bank 上对拍
（无 CUDA 跳过）。

padded scale 列填入巨大哨兵值：kernel 一旦误读该列，输出会立刻爆掉使断言失败。
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as Fn

from freetoken.kernel.aot_models import fp8_block_scale_pad

FP8 = torch.float8_e4m3fn
BLOCK = 128
# padded scale 列的哨兵值：绝不能被 kernel 读到（读到即一个 128 宽部分和 x ~1e4）
_PAD_SENTINEL = 999.0


def _make_fp8_block_cache(L, E, H, I, seed=0, top_k=4):
    """
    Business Logic（为什么需要这个函数）:
        CPU executor 需要"随机但合法"的 fp8_block bank 来做数值对拍，形状契约必须
        与产线 GPU bank 完全一致（含 padded scale 行宽），否则测的不是产线路径。

    Code Logic（这个函数做什么）:
        权重由真实值量化到 e4m3（绝不产生 NaN 码），scale 为 bf16 正数且行宽经
        ``fp8_block_scale_pad`` 填充、padded 列填 ``_PAD_SENTINEL``；返回
        SimpleNamespace cache（bank 按层切分为 [E, ...]）。
    """
    torch.manual_seed(seed)
    S = L * E

    def rows(out_rows, in_rows):
        w = (torch.randn(S, out_rows, in_rows) * 0.05).to(FP8)
        scale_cols = in_rows // BLOCK
        pad_cols = fp8_block_scale_pad(out_rows // BLOCK, scale_cols)
        scale = 0.001 + 0.01 * torch.rand(S, out_rows // BLOCK, pad_cols, dtype=torch.bfloat16)
        scale[:, :, scale_cols:] = _PAD_SENTINEL  # padded 列绝不能被读
        return w.contiguous(), scale.contiguous()

    gate_up, gate_up_scale = rows(2 * I, H)
    down, down_scale = rows(H, I)
    return SimpleNamespace(
        quant_format="fp8_block",
        bank_sources={
            "gate_up": list(gate_up.split(E)), "gate_up_scale": list(gate_up_scale.split(E)),
            "down": list(down.split(E)), "down_scale": list(down_scale.split(E)),
        },
        num_layers=L,
        num_experts=E,
        top_k=top_k,
        decode_target="cpu",
        cpu_executor=None,
    )


def _dequant(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[E, N, K] e4m3 + [E, N//128, pad(K//128)] bf16 scale -> [E, N, K] float64
    精确解量化（scale 列按 r//128、k//128 展开；超出 K//128 的列是 padding，丢弃）。"""
    K = w.shape[2]
    s = scale[:, :, : K // BLOCK].to(torch.float64)
    s_full = s.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
    return w.to(torch.float64) * s_full


def _reference(cache, layer, hidden, ids, w) -> torch.Tensor:
    """
    Business Logic（为什么需要这个函数）:
        CPU GEMV 的对拍锚点必须是"语义上与 GPU 一致"的高精度参考，而不是 GPU kernel
        自身（否则 GPU 的 bug 会两边一致地漏掉）。

    Code Logic（这个函数做什么）:
        在 float64 解量化权重上重放解码：swiglu 乘积处一次 bf16 舍入（对齐 C++ 的
        舍入点）、router 权重乘在 down 输出上、路由求和后仍为 f64（与 bf16 输出
        对比时容差由 bf16 舍入主导）。-1 路由跳过。
    """
    gu = _dequant(cache.bank_sources["gate_up"][layer], cache.bank_sources["gate_up_scale"][layer])
    dn = _dequant(cache.bank_sources["down"][layer], cache.bank_sources["down_scale"][layer])
    E, two_i, H = gu.shape
    I = two_i // 2
    bs, top_k = ids.shape
    xf = hidden.to(torch.float64)
    out = torch.zeros(bs, H, dtype=torch.float64)
    for t in range(bs):
        for k in range(top_k):
            e = int(ids[t, k])
            if e < 0:
                continue
            gate = gu[e, :I] @ xf[t]
            up = gu[e, I:] @ xf[t]
            g = (Fn.silu(gate.float()).double() * up).bfloat16().double()
            out[t] += float(w[t, k]) * (dn[e] @ g)
    return out


def _make_executor(cache, bs, *, activation="silu", isa=None):
    """在 CPU 设备上构造 CpuMoeExecutor（无 CUDA 上下文依赖）；``isa`` 通过
    FREETOKEN_CPU_MOE_ISA 降档（scalar / avx2），构造后恢复环境。"""
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    if isa is not None:
        os.environ["FREETOKEN_CPU_MOE_ISA"] = isa
    try:
        return CpuMoeExecutor(
            cache,
            top_k=cache.top_k,
            activation=activation,
            apply_router_weight_on_input=False,
            num_threads=4,
            max_tokens=bs,
            device=torch.device("cpu"),
        )
    finally:
        if isa is not None:
            os.environ.pop("FREETOKEN_CPU_MOE_ISA", None)


def _run_cpu_decode(ex, layer, hidden, ids, w) -> torch.Tensor:
    """直接驱动 C++ worker 池（eager run_task）：CPU 设备没有可挂 host-func 节点的
    CUDA stream，故绕开 decode() 的图捕获外壳，喂入 pinned 等价的连续张量。"""
    bs = hidden.shape[0]
    io = {
        "x": hidden.to(torch.bfloat16).contiguous(),
        "ids": ids.to(torch.int32).contiguous(),
        "w": w.to(torch.float32).contiguous(),
        "y": torch.empty(bs, ex.H, dtype=torch.bfloat16),
    }
    task = ex._ext.create_task(
        layer, bs, io["x"].data_ptr(), io["ids"].data_ptr(), io["w"].data_ptr(), io["y"].data_ptr()
    )
    ex._ext.run_task(task)
    return io["y"]


def _rel(err: torch.Tensor, ref: torch.Tensor) -> float:
    return (err - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)


@pytest.mark.parametrize("H,I", [(256, 128), (128, 128), (1024, 512), (2048, 512)])
@pytest.mark.parametrize("isa", ["avx2", "scalar"])
def test_cpu_decode_fp8_block_matches_f64_reference(H, I, isa):
    """fp8_block CPU GEMV vs float64 解量化参考：覆盖 padded（(256,128)/(128,128)）
    与非 padded（(1024,512)、真实 Qwen3.6-35B 几何 (2048,512)）scale 步长，以及
    scalar / AVX2 两档 ISA。(H, I) 解析、padded 哨兵列不可读，一并由对拍保证。"""
    L, E, top_k, bs = 2, 8, 4, 3
    layer = 1
    cache = _make_fp8_block_cache(L, E, H, I, seed=H + I, top_k=top_k)

    ex = _make_executor(cache, bs, isa=isa)
    assert ex.isa.startswith(isa), ex.isa
    assert (ex.H, ex.I) == (H, I)

    torch.manual_seed(41)
    hidden = torch.randn(bs, H)
    ids = torch.randint(0, E, (bs, top_k), dtype=torch.int32)
    w = torch.rand(bs, top_k)

    out = _run_cpu_decode(ex, layer, hidden, ids, w)
    ref = _reference(cache, layer, hidden, ids, w)
    rel = _rel(out.float(), ref.float())
    assert rel < 1e-2, f"fp8_block H={H} I={I} isa={isa} rel err {rel}"


def test_cpu_decode_fp8_block_skips_negative_ids():
    """-1 路由（hybrid 后端分给 GPU 的专家）不得触达 bank：输出必须等于只算剩余
    单路由的解码结果。"""
    L, E, H, I, top_k, bs = 1, 8, 256, 128, 4, 1
    cache = _make_fp8_block_cache(L, E, H, I, seed=7, top_k=top_k)
    ex = _make_executor(cache, bs)

    hidden = torch.randn(bs, H)
    w = torch.rand(bs, top_k)
    ids_full = torch.randint(0, E, (bs, top_k), dtype=torch.int32)
    ids_skip = ids_full.clone()
    ids_skip[0, 1:] = -1

    out = _run_cpu_decode(ex, 0, hidden, ids_skip, w)
    ref = _reference(cache, 0, hidden, ids_skip, w)
    assert _rel(out.float(), ref.float()) < 1e-2
    out_one = _run_cpu_decode(ex, 0, hidden, ids_skip[:, :1].contiguous(), w[:, :1].contiguous())
    assert _rel(out.float(), out_one.float()) < 1e-2


def test_cpu_decode_fp8_block_rejects_unaligned_geometry():
    """128x128 块布局要求 H、I 均为 128 的倍数：几何不齐时 executor 必须大声失败，
    而不是错读 scale 行。"""
    L, E, H = 1, 4, 256
    I = 64  # 128-K 块除不开（I // 128 == 0）
    cache = _make_fp8_block_cache(L, E, H, 128, seed=3, top_k=2)
    cache.bank_sources = {
        "gate_up": list(torch.zeros(L * E, 2 * I, H, dtype=FP8).split(E)),
        "gate_up_scale": list(torch.zeros(
            L * E, 2 * I // BLOCK, fp8_block_scale_pad(2 * I // BLOCK, H // BLOCK),
            dtype=torch.bfloat16).split(E)),
        "down": list(torch.zeros(L * E, H, I, dtype=FP8).split(E)),
        "down_scale": list(torch.zeros(
            L * E, H // BLOCK, fp8_block_scale_pad(H // BLOCK, I // BLOCK),
            dtype=torch.bfloat16).split(E)),
    }
    with pytest.raises((RuntimeError, AssertionError)):
        _make_executor(cache, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("bs", [1, 4])
def test_cpu_decode_fp8_block_matches_gpu_triton(bs):
    """CPU fp8_block GEMV vs 产线 triton 内联解量化解码 kernel：同一份 bank 上
    （两侧都 fp32 累加，容差收紧）。"""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_fp8_block import fused_experts_decode_fp8_block

    L, E, H, I, top_k = 2, 8, 256, 128, 4
    layer = 1
    cache = _make_fp8_block_cache(L, E, H, I, seed=11, top_k=top_k)
    dev = torch.device("cuda")

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=4,
        max_tokens=bs,
        device=dev,
    )

    torch.manual_seed(500 + bs)
    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.randint(0, E, (bs, top_k), device=dev, dtype=torch.int32)
    w = torch.rand(bs, top_k, device=dev)

    out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    gpu_out = fused_experts_decode_fp8_block(
        hidden, b["gate_up"][layer].to(dev), b["gate_up_scale"][layer].to(dev),
        b["down"][layer].to(dev), b["down_scale"][layer].to(dev),
        w, ids, "silu", False,
    ).float()

    rel = _rel(out.cpu(), gpu_out.cpu())
    assert rel < 2e-2, f"fp8_block bs={bs} cpu-vs-gpu rel err {rel}"


def test_fp8_block_kernel_selects_for_hybrid():
    """#534 回归：fp8_block 专家 kernel 声明 CPU 执行格式后，kernel 选择器在
    cpu/hybrid 解码目标下必须可用，而不是抛 KernelSelectionError。"""
    from freetoken.layers.quantization.moe.fp8_block import TritonFp8BlockMoEKernel
    from freetoken.moe.cpu_executor import _WFMT_IDS

    kernel = TritonFp8BlockMoEKernel()
    assert kernel.cpu_format == "fp8_block"
    assert _WFMT_IDS[kernel.cpu_format] == 5  # cpu_moe_ext.cpp 的 WF_FP8_BLOCK
    for decode_target in ("cpu", "hybrid", "gpu"):
        cfg = SimpleNamespace(
            num_experts=256, hidden=2048, intermediate=512, top_k=8, tp_rank=0, tp_size=1,
            scheme=None, activation="silu", alpha=1.0, beta=0.0, limit=None, interleaved=False,
            has_bias=False, apply_router_weight_on_input=False, strategy="offload",
            decode_target=decode_target,
        )
        assert kernel.unusable_reason(cfg) is None, decode_target
