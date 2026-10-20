"""Block-FP8 CPU MoE GEMV vs a float64 reference and FreeToken's Triton block-fp8 decode kernel.

Two things this is really guarding:
  * the scale is indexed [row // 128][k // 128] and *multiplies*, despite upstream naming it ``weight_scale_inv``. Inverting it, or transposing the two axes, still produces plausible-looking output.
  * the SIMD kernels decode e4m3 by reading each byte as fp16 bits. Subnormals carry too little weight in a full-range GEMV to show a decode error, so they get their own case.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

BLK = 128


def _nb(n: int) -> int:
    return (n + BLK - 1) // BLK


def _make_fp8_block_cache(L, E, H, I, seed=0):
    from freetoken.kernel.aot_models import fp8_block_scale_pad

    if torch.cuda.is_available():
        from freetoken.kernel.pinned import alloc_pinned_tensor
    else:
        def alloc_pinned_tensor(*shape, dtype):
            return torch.empty(*shape, dtype=dtype)

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
    io = ex._io
    io["x"].copy_(hidden)
    io["ids"].zero_()
    io["w"].fill_(1.0)
    ex._ext.run(0, 1)
    cpu_out = io["y"][0].float().to(torch.float64)
    banks = cache.bank_sources
    gate_up = _reference_fp8_block_gemv(banks["gate_up"][0], banks["gate_up_scale"][0], hidden[0])
    intermediate = torch.nn.functional.silu(gate_up[:I]) * gate_up[I:]
    expected = _reference_fp8_block_gemv(banks["down"][0], banks["down_scale"][0], intermediate)
    rel = (cpu_out - expected).abs().max() / (expected.abs().max() + 1e-6)
    assert rel < 5e-3, f"max relative error {rel.item()}"


# every finite e4m3 code, and the subnormal ones (exp == 0) with both signs
_FINITE_CODES = [c for c in range(256) if c not in (0x7F, 0xFF)]
_SUBNORMAL_CODES = [c for c in range(256) if c & 0x78 == 0]


def _make_coded_fp8_block_cache(H, I, codes, seed=0):
    from freetoken.kernel.aot_models import fp8_block_scale_pad

    if torch.cuda.is_available():
        from freetoken.kernel.pinned import alloc_pinned_tensor
    else:
        def alloc_pinned_tensor(*shape, dtype):
            return torch.empty(*shape, dtype=dtype)

    torch.manual_seed(seed)
    pool = torch.tensor(codes, dtype=torch.uint8)

    def rows(OUT, IN):
        count = OUT * IN
        picked = pool.repeat((count + pool.numel() - 1) // pool.numel())[:count]
        picked = picked[torch.randperm(count)].reshape(1, OUT, IN)
        weights = alloc_pinned_tensor(1, OUT, IN, dtype=torch.float8_e4m3fn)
        weights.copy_(picked.view(torch.float8_e4m3fn))
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


def _reference_fp8_block_gemv(weights, scales, x):
    rows, cols = weights.shape[-2:]
    scale = scales[0, :_nb(rows), :_nb(cols)].to(torch.float64)
    scale = scale.repeat_interleave(BLK, 0).repeat_interleave(BLK, 1)[:rows, :cols]
    return (weights[0].to(torch.float64) * scale) @ x.to(torch.float64)


@pytest.mark.parametrize("codes", ["finite", "subnormal"])
@pytest.mark.parametrize("isa", ["scalar", "avx2", "avx512"])
def test_cpu_fp8_block_matches_float64_reference(monkeypatch, isa, codes):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    H, I = 259, 257  # ragged K for both GEMVs; 257 down rows exercise many reductions
    cache = _make_coded_fp8_block_cache(H, I, _FINITE_CODES if codes == "finite" else _SUBNORMAL_CODES, seed=20260904)
    monkeypatch.setenv("FREETOKEN_CPU_MOE_ISA", isa)
    ex = CpuMoeExecutor(
        cache,
        top_k=1,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=1,
        max_tokens=1,
        device=torch.device("cpu"),
    )
    tiers = {"scalar": 0, "avx2": 1, "avx512": 2}
    if tiers[isa] > ("scalar", "avx2", "avx512f", "avx512bf16").index(ex.isa):
        pytest.skip(f"requested {isa}, host/build selected {ex.isa}")

    hidden = torch.randn(1, H, dtype=torch.bfloat16)
    io = ex._io
    io["x"].copy_(hidden)
    io["ids"].zero_()
    io["w"].fill_(1.0)
    ex._ext.run(0, 1)
    cpu_out = io["y"][0].float().to(torch.float64)

    banks = cache.bank_sources
    gate_up = _reference_fp8_block_gemv(banks["gate_up"][0], banks["gate_up_scale"][0], hidden[0])
    intermediate = (torch.nn.functional.silu(gate_up[:I]) * gate_up[I:])
    expected = _reference_fp8_block_gemv(banks["down"][0], banks["down_scale"][0], intermediate)
    rel = (cpu_out - expected).abs().max() / (expected.abs().max() + 1e-6)
    # The executor stores bf16; measured max relative error is 3.3e-3 (finite codes) and 1.2e-3 (subnormal codes).
    # The bound leaves room for reduction order, not for a narrow accumulator.
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


# FREETOKEN_CPU_MOE_ISA tier -> the fp8_block kernel it selects; the avx512bf16 tier selects the avx512f kernel, so it is not repeated here.
# The env var caps *down* only, so a tier the CPU or the build lacks selects a lower kernel, which turns into a skip instead of testing the same kernel twice.
_ISA_TIERS = {"scalar": "scalar", "avx2": "avx2", "avx512": "avx512f"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("isa", list(_ISA_TIERS))
@pytest.mark.parametrize("bs", [1, 2, 5])
def test_cpu_fp8_block_matches_gpu_decode_kernel(bs, isa, monkeypatch):
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_fp8_block import fused_experts_decode_fp8_block

    L, E, H, I, top_k = 2, 8, 256, 128, 4
    layer = 1
    dev = torch.device("cuda")
    cache = _make_fp8_block_cache(L, E, H, I, seed=bs)

    # Set before construction: the executor latches its dot kernel there.
    monkeypatch.setenv("FREETOKEN_CPU_MOE_ISA", isa)
    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
    )
    if ex.isa != _ISA_TIERS[isa]:
        pytest.skip(f"asked for {isa}, host/build selected {ex.isa}: not supported here")

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
    assert rel < 2e-2, f"bs={bs} isa={ex.isa} rel err {rel.item()}"


def test_fp16_bits_carry_every_e4m3_byte():
    """A byte b read as fp16 bits (b & 0x80) << 8 | (b & 0x7F) << 7 is its e4m3 value times 2^-8, which the SIMD kernels rely on."""
    codes = torch.arange(256, dtype=torch.int32)
    ref = codes.to(torch.uint8).view(torch.float8_e4m3fn).float()
    bits = ((codes & 0x80) << 8) | ((codes & 0x7F) << 7)
    got = torch.where(bits >= 0x8000, bits - 0x10000, bits).to(torch.int16).view(torch.float16).float() * 256
    finite = torch.isfinite(ref)
    assert torch.equal(got[finite], ref[finite])
    assert torch.equal(got[~finite].abs(), torch.full((2,), 480.0))  # 0x7F / 0xFF, as e4m3_decode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("bs", [1, 3])
@pytest.mark.parametrize("accumulate", [False, True])
def test_cpu_fp8_block_graph_distinct_layer_inputs(bs, accumulate):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    L, E, H, I, top_k = 3, 8, 256, 128, 4
    cache = _make_fp8_block_cache(L, E, H, I, seed=7)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ex = CpuMoeExecutor(
            cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
            num_threads=4, max_tokens=bs, device=torch.device("cuda"),
        )
        hidden = torch.randn(L, bs, H, device="cuda", dtype=torch.bfloat16)
        ids = torch.zeros(L, bs, top_k, device="cuda", dtype=torch.int32)
        weights = torch.rand(L, bs, top_k, device="cuda")
        gpu_out = torch.randn_like(hidden)

        def forward():
            outputs = []
            for layer in range(L):
                pending = ex.decode_submit(layer, hidden[layer], weights[layer], ids[layer])
                outputs.append(ex.decode_sync(pending, add_to=gpu_out[layer].clone() if accumulate else None))
            return outputs

        forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = forward()

        for step in range(3):
            torch.manual_seed(101 + step)
            hidden.copy_(torch.randn(L, bs, H, dtype=torch.bfloat16))
            new_ids = torch.randint(0, E, (L, bs, top_k), dtype=torch.int32)
            new_ids[step].fill_(-1)
            new_ids[(step + 1) % L, :, 0] = -1
            ids.copy_(new_ids)
            weights.copy_(torch.rand(L, bs, top_k))
            gpu_out.copy_(torch.randn(L, bs, H, dtype=torch.bfloat16))
            # Do not run eager first: it would prefill the shared host IO with this step's data.
            graph.replay()
            stream.synchronize()
            eager = forward()
            stream.synchronize()
            for got, expected in zip(captured, eager):
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
            skipped = gpu_out[step] if accumulate else torch.zeros_like(gpu_out[step])
            torch.testing.assert_close(captured[step], skipped, rtol=0, atol=0)
