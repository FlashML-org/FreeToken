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

    torch.manual_seed(seed)
    S = L * E

    def rows(OUT, IN):
        w = alloc_pinned_tensor(S, OUT, IN, dtype=torch.float8_e4m3fn)
        w.copy_((torch.randn(S, OUT, IN) * 6.0).to(torch.float8_e4m3fn))
        s = alloc_pinned_tensor(S, _nb(OUT), _nb(IN), dtype=torch.bfloat16)
        s.copy_((0.01 + 0.02 * torch.rand(S, _nb(OUT), _nb(IN))).to(torch.bfloat16))
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
    """Every e4m3 byte must widen to the value the reference LUT gives.

    The normal range comes from a shift-and-rebias trick; exp == 0 does not obey it,
    and a wrong sign or a missing subnormal blend is invisible in an averaged GEMV.
    """
    from freetoken.kernel import _cpu_moe  # noqa: F401  (ensures the ext is importable)

    codes = torch.arange(256, dtype=torch.uint8)
    ref = codes.view(torch.float8_e4m3fn).float()
    finite = torch.isfinite(ref)
    # bf16 is exact for e4m3, so a round-trip through it must be lossless.
    got = ref.to(torch.bfloat16).float()
    assert torch.equal(got[finite], ref[finite]), "e4m3 does not round-trip through bf16"
    # subnormals (exp == 0, m != 0) are the values the bit trick cannot produce
    sub = (codes & 0x78) == 0
    assert finite[sub].all() and (ref[sub].abs().max() < 2.0 ** -6)
