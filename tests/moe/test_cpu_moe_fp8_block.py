"""Block-FP8 CPU MoE GEMV vs FreeToken's Triton block-fp8 decode kernel.

The CPU path widens e4m3 to bf16 in-register and reduces with `dpbf16`, where the
reference dequantizes to fp32 and reduces in fp32. e4m3 -> bf16 is exact (3 mantissa
bits into 7, and bf16's exponent range covers e4m3's whole span), so the products are
identical and only the summation order differs -- same latitude the bf16 path takes.

Two things this is really guarding:
  * the scale is indexed [row // 128][k // 128] and *multiplies*, despite upstream
    naming it ``weight_scale_inv``. Inverting it, or transposing the two axes, still
    produces plausible-looking output.
  * exp == 0 is subnormal (m * 2^-9) and does not follow the exponent-rebias bit trick
    the normal range uses, so it is blended in from a table.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

BLK = 128


def _nb(n: int) -> int:
    return (n + BLK - 1) // BLK


def _make_fp8_block_cache(L, E, H, I, seed=0):
    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.kernel.aot_models import fp8_block_scale_pad

    torch.manual_seed(seed)
    S = L * E

    def rows(OUT, IN):
        w = alloc_pinned_tensor(S, OUT, IN, dtype=torch.float8_e4m3fn)
        w.copy_((torch.randn(S, OUT, IN) * 6.0).to(torch.float8_e4m3fn))
        s = alloc_pinned_tensor(
            S, _nb(OUT), fp8_block_scale_pad(_nb(OUT), _nb(IN)), dtype=torch.bfloat16
        )
        s.copy_((0.01 + 0.02 * torch.rand_like(s)).to(torch.bfloat16))
        return w, s

    gu, gus = rows(2 * I, H)
    dn, dns = rows(H, I)
    return SimpleNamespace(
        quant_format="fp8_block",
        bank_sources={
            "gate_up": list(gu.split(E)), "gate_up_scale": list(gus.split(E)),
            "down": list(dn.split(E)), "down_scale": list(dns.split(E)),
        },
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


def test_cpu_fp8_block_accepts_qwen38_padded_scale_bank():
    from freetoken.kernel.aot_models import fp8_block_scale_pad
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    H, I = 2560, 640
    cache = _make_fp8_block_cache(1, 1, H, I)
    assert cache.bank_sources["down_scale"][0].shape[1:] == (
        _nb(H), fp8_block_scale_pad(_nb(H), _nb(I))
    ) == (20, 6)
    CpuMoeExecutor(
        cache,
        top_k=1,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=1,
        max_tokens=1,
        device=torch.device("cpu"),
    )


def _make_full_range_fp8_block_cache(H, I, seed=0):
    from freetoken.kernel.aot_models import fp8_block_scale_pad
    from freetoken.kernel.pinned import alloc_pinned_tensor

    torch.manual_seed(seed)
    finite_codes = torch.tensor([c for c in range(256) if c not in (0x7F, 0xFF)], dtype=torch.uint8)

    def rows(OUT, IN):
        count = OUT * IN
        codes = finite_codes.repeat((count + finite_codes.numel() - 1) // finite_codes.numel())[:count]
        codes = codes[torch.randperm(count)].reshape(1, OUT, IN)
        weights = alloc_pinned_tensor(1, OUT, IN, dtype=torch.float8_e4m3fn)
        weights.copy_(codes.view(torch.float8_e4m3fn))
        scales = alloc_pinned_tensor(
            1, _nb(OUT), fp8_block_scale_pad(_nb(OUT), _nb(IN)), dtype=torch.bfloat16
        )
        scales.copy_((0.01 + 0.09 * torch.rand_like(scales)).to(torch.bfloat16))
        return weights, scales

    gu, gus = rows(2 * I, H)
    dn, dns = rows(H, I)
    return SimpleNamespace(
        quant_format="fp8_block",
        bank_sources={
            "gate_up": [gu], "gate_up_scale": [gus],
            "down": [dn], "down_scale": [dns],
        },
        num_layers=1,
        num_experts=1,
        decode_target="cpu",
        cpu_executor=None,
    )


def _reference_fp8_gemv(weights, scales, x):
    rows, cols = weights.shape[-2:]
    scale = scales[0, :_nb(rows), :_nb(cols)].to(torch.float64)
    scale = scale.repeat_interleave(BLK, 0).repeat_interleave(BLK, 1)[:rows, :cols]
    return (weights[0].to(torch.float64) * scale) @ x.to(torch.float64)


def test_cpu_fp8_block_matches_float64_reference():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    H, I = 259, 257  # ragged K for both GEMVs; 257 down rows exercise many reductions
    cache = _make_full_range_fp8_block_cache(H, I, seed=20260904)
    ex = CpuMoeExecutor(
        cache,
        top_k=1,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=1,
        max_tokens=1,
        device=torch.device("cpu"),
    )

    hidden = torch.randn(1, H, dtype=torch.bfloat16)
    io = ex._io_for(1)
    io["x"].copy_(hidden)
    io["ids"].zero_()
    io["w"].fill_(1.0)
    ex._ext.run_task(ex._task_for(0, 1))
    cpu_out = io["y"][0].float().to(torch.float64)

    banks = cache.bank_sources
    gate_up = _reference_fp8_gemv(banks["gate_up"][0], banks["gate_up_scale"][0], hidden[0])
    intermediate = (torch.nn.functional.silu(gate_up[:I]) * gate_up[I:])
    expected = _reference_fp8_gemv(banks["down"][0], banks["down_scale"][0], intermediate)
    rel = (cpu_out - expected).abs().max() / (expected.abs().max() + 1e-6)
    # The executor stores its result as bf16. Across 257 output rows, the measured
    # max relative error was 3.45e-3; this margin leaves room for reduction order
    # without accepting the materially larger error from a narrow accumulator.
    assert rel < 5e-3, f"max relative error {rel.item()}"


@pytest.mark.parametrize("isa", ["scalar", "avx2", "avx512", "avx512bf16"])
def test_cpu_fp8_block_isa_override(monkeypatch, isa):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    cache = _make_fp8_block_cache(1, 1, 256, 128, seed=17)
    monkeypatch.delenv("FREETOKEN_CPU_MOE_ISA", raising=False)
    auto = CpuMoeExecutor(
        cache, top_k=1, activation="silu", apply_router_weight_on_input=False,
        num_threads=1, max_tokens=1, device=torch.device("cpu"),
    )
    names = ("scalar", "avx2", "avx512f", "avx512bf16")
    auto_rank = names.index(auto.isa)
    monkeypatch.setenv("FREETOKEN_CPU_MOE_ISA", isa)
    forced = CpuMoeExecutor(
        cache, top_k=1, activation="silu", apply_router_weight_on_input=False,
        num_threads=1, max_tokens=1, device=torch.device("cpu"),
    )
    assert names.index(forced.isa) == min(names.index({
        "scalar": "scalar", "avx2": "avx2", "avx512": "avx512f",
        "avx512bf16": "avx512bf16",
    }[isa]), auto_rank)


@pytest.mark.parametrize("bs", [1, 2, 5])
def test_cpu_fp8_block_matches_gpu_decode_kernel(bs):
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_fp8_block import fused_experts_decode_fp8_block

    L, E, H, I, top_k = 2, 8, 256, 128, 4
    layer = 1
    dev = torch.device("cuda")
    cache = _make_fp8_block_cache(L, E, H, I, seed=bs)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    gpu_out = fused_experts_decode_fp8_block(
        hidden,
        cache.bank_sources["gate_up"][layer].to(dev),
        cache.bank_sources["gate_up_scale"][layer].to(dev),
        cache.bank_sources["down"][layer].to(dev),
        cache.bank_sources["down_scale"][layer].to(dev),
        w, ids.clone(), "silu", False,
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 2e-2, f"bs={bs} rel err {rel.item()}"


def test_subnormal_and_sign_decode_exactly():
    """Check PyTorch's e4m3-to-bf16 conversion used by the CPU decoder.

    The C++ decoder is exercised by ``test_cpu_fp8_block_matches_float64_reference``.
    """

    codes = torch.arange(256, dtype=torch.uint8)
    ref = codes.view(torch.float8_e4m3fn).float()
    finite = torch.isfinite(ref)
    # bf16 is exact for e4m3, so a round-trip through it must be lossless.
    got = ref.to(torch.bfloat16).float()
    assert torch.equal(got[finite], ref[finite]), "e4m3 does not round-trip through bf16"
    # subnormals (exp == 0, m != 0) are the values the bit trick cannot produce
    sub = (codes & 0x78) == 0
    assert finite[sub].all() and (ref[sub].abs().max() < 2.0 ** -6)
