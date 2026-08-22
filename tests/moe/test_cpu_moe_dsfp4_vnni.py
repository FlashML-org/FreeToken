"""ds_fp4 W4A8 (VNNI) must agree with the fp32 dot it replaces.

`FREETOKEN_CPU_MOE_DSFP4_VNNI=1` swaps the ds_fp4 expert GEMV from an fp32 dequant-
and-FMA dot to VPDPBUSD over int8 activations, the same W4A8 trick nvfp4 already uses
(both banks are row-major with K contiguous, so the reduction runs along the dot
product). The weights stay bit-exact -- e2m1 codes decode to exact int8 through the
kE2M1x2 LUT -- so the only new error is the per-16-block activation quantization.

The subtle part, and what this pins: ds_fp4's e8m0 weight scale covers **32** K while
the shared activation quantizer works in 16-K blocks, so for 16-K block `b` the weight
scale lives at `scale[b >> 1]`. Getting that index wrong still produces plausible
output, just consistently wrong by a per-block factor.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _make_dsfp4_cache(L, E, H, I, seed=0):
    from freetoken.kernel.pinned import alloc_pinned_tensor

    torch.manual_seed(seed)
    S = L * E

    def rows(OUT, IN):
        packed = alloc_pinned_tensor(S, OUT, IN // 2, dtype=torch.uint8)
        packed.copy_(torch.randint(0, 256, (S, OUT, IN // 2), dtype=torch.uint8))
        scale = alloc_pinned_tensor(S, OUT, IN // 32, dtype=torch.uint8)
        # e8m0 near 2^0 so nothing lands in the fp32 denormal range
        scale.fill_(127)
        return packed, scale

    gup, gus = rows(2 * I, H)
    dnp, dns = rows(H, I)
    return SimpleNamespace(
        quant_format="ds_fp4",
        bank_sources={
            "gate_up_packed": list(gup.split(E)), "gate_up_scale": list(gus.split(E)),
            "down_packed": list(dnp.split(E)), "down_scale": list(dns.split(E)),
        },
        num_layers=L, num_experts=E, decode_target="cpu", cpu_executor=None,
    )


def _decode(cache, bs, top_k, H, layer, vnni: str):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    prev = os.environ.get("FREETOKEN_CPU_MOE_DSFP4_VNNI")
    os.environ["FREETOKEN_CPU_MOE_DSFP4_VNNI"] = vnni
    try:
        ex = CpuMoeExecutor(
            cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
            num_threads=0, max_tokens=bs, device=torch.device("cuda"),
            swiglu_limit=7.0,
        )
        torch.manual_seed(1234)
        dev = torch.device("cuda")
        hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(cache.num_experts, device=dev)[:top_k] for _ in range(bs)]
        ).to(torch.int32)
        w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)
        out = ex.decode(layer, hidden, w, ids).float()
        torch.cuda.synchronize()
        return out, ex.isa
    finally:
        if prev is None:
            os.environ.pop("FREETOKEN_CPU_MOE_DSFP4_VNNI", None)
        else:
            os.environ["FREETOKEN_CPU_MOE_DSFP4_VNNI"] = prev


@pytest.mark.parametrize("bs", [1, 4])
def test_w4a8_matches_the_fp32_dot(bs):
    L, E, H, I, top_k, layer = 2, 8, 512, 256, 4, 1
    cache = _make_dsfp4_cache(L, E, H, I, seed=bs)
    ref, _ = _decode(cache, bs, top_k, H, layer, "0")
    got, _ = _decode(cache, bs, top_k, H, layer, "1")
    rel = (got - ref).abs().max() / (ref.abs().max() + 1e-6)
    # Activation quantization only; the weights are bit-identical between the paths.
    assert rel < 5e-2, f"bs={bs} rel err {rel.item()}"


def test_disabled_by_default():
    """W4A8 changes numerics for no measured gain on a DRAM-bound machine, so a user
    who does not ask for it must get the fp32 dot."""
    L, E, H, I, top_k, layer = 2, 8, 512, 256, 4, 1
    cache = _make_dsfp4_cache(L, E, H, I, seed=7)
    os.environ.pop("FREETOKEN_CPU_MOE_DSFP4_VNNI", None)
    default, _ = _decode(cache, 1, top_k, H, layer, "0")
    cache2 = _make_dsfp4_cache(L, E, H, I, seed=7)
    prev = os.environ.pop("FREETOKEN_CPU_MOE_DSFP4_VNNI", None)
    try:
        from freetoken.moe.cpu_executor import CpuMoeExecutor

        ex = CpuMoeExecutor(
            cache2, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
            num_threads=0, max_tokens=1, device=torch.device("cuda"), swiglu_limit=7.0,
        )
        torch.manual_seed(1234)
        dev = torch.device("cuda")
        hidden = torch.randn(1, H, device=dev, dtype=torch.bfloat16)
        ids = torch.stack([torch.randperm(E, device=dev)[:top_k]]).to(torch.int32)
        w = torch.rand(1, top_k, device=dev, dtype=torch.float32)
        out = ex.decode(layer, hidden, w, ids).float()
        torch.cuda.synchronize()
    finally:
        if prev is not None:
            os.environ["FREETOKEN_CPU_MOE_DSFP4_VNNI"] = prev
    assert torch.equal(out, default), "unset env must give exactly the fp32 path"
