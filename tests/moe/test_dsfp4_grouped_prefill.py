"""ds_fp4 grouped prefill GEMM vs the per-route decode GEMV.

Both consume the same FP8-round-tripped activations and dequantize the same
banks; they differ only in fp32 accumulation order (tl.dot tile reduction vs
the GEMV's sequential K-walk), so each individual GEMM must agree to <=1 bf16
ulp -- the fp64 truth of a mismatching element sits on a bf16 rounding midpoint
and the two orders land on adjacent values. Through the full chain the down
activation is FP8-re-quantized, so a 1-ulp gate_up seed can flip a per-128-block
amax across a power of two and shift that block's quant step: the chained
comparison is statistical (H100 measured: >=99.3% bit-same), not elementwise.

E=256/top6 matches the production routing geometry. Production runs one static
config (BLOCK_M=64, no autotune); the BLOCK_M=16 parametrization keeps the
smaller mma tile compiling and correct for future per-card config changes.
"""

import pytest
import torch

E, H, I, TOP_K = 256, 1024, 512, 6
LIMIT = 7.0


def _banks(device):
    g = torch.Generator(device="cpu").manual_seed(0)

    def u8(*shape, low=0, high=256):
        return torch.randint(low, high, shape, dtype=torch.uint8, generator=g).to(device)

    return (
        u8(E, 2 * I, H // 2),
        u8(E, 2 * I, H // 32, low=118, high=134),
        u8(E, H, I // 2),
        u8(E, H, I // 32, low=118, high=134),
    )


def _routing(T, device):
    g = torch.Generator(device="cpu").manual_seed(1)
    x = (torch.randn(T, H, dtype=torch.bfloat16, generator=g) * 0.1).to(device)
    slots = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(T)])
    slots = slots.to(device=device, dtype=torch.int32).contiguous()
    w = torch.rand(T, TOP_K, generator=g)
    w = (w / w.sum(-1, keepdim=True)).float().to(device).contiguous()
    return x, slots, w


def _ulp_bf16(ref: torch.Tensor) -> torch.Tensor:
    mag = ref.abs().clamp_min(2**-126)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7)


def _assert_one_ulp(out, ref):
    diff = (out.float() - ref.float()).abs()
    # 1 ulp at the element's magnitude, plus fp32-ordering noise for elements
    # whose value nearly cancels relative to the K-walk partials.
    tol = torch.maximum(_ulp_bf16(ref), torch.full_like(diff, 0.25))
    assert (diff <= tol).all(), (diff.max().item(), diff.argmax().item())
    assert (out == ref).float().mean().item() > 0.995


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("block_m", [16, 64])
def test_grouped_gemms_within_one_ulp_of_gemv(block_m):
    from freetoken.kernel.triton.dsv4.fp8_linear import (
        act_quant_fp8_inplace,
        act_quant_fp8_roundtrip,
    )
    from freetoken.moe.fused import moe_align_block_size
    from freetoken.moe.fused_ds_fp4 import _grouped_decode, _grouped_prefill

    device = "cuda"
    gup, gus, dp, ds = _banks(device)
    T = 1024 if block_m == 64 else 256
    x, slots, w = _routing(T, device)
    cfg = dict(
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8,
        num_warps=8, num_stages=1,
    )
    si, ei, ntpp = moe_align_block_size(slots, block_m, E)
    tw = w.reshape(-1)

    # gate_up: token-indexed A, no routed weight
    xq = act_quant_fp8_roundtrip(x, 128)
    ref = _grouped_decode(xq, gup, gus, slots, None, a_row_is_route=False, mul_routed_weight=False)
    out = torch.empty_like(ref)
    _grouped_prefill(xq, gup, gus, out, tw, si, ei, ntpp, T * TOP_K, TOP_K, False, cfg)
    torch.cuda.synchronize()
    _assert_one_ulp(out, ref)

    # down: route-indexed A, routed weight applied
    act = (torch.randn(T * TOP_K, I, dtype=torch.bfloat16, device=device) * 0.3).contiguous()
    act_quant_fp8_inplace(act, 128)
    ref = _grouped_decode(act, dp, ds, slots, w, a_row_is_route=True, mul_routed_weight=True)
    out = torch.empty_like(ref)
    _grouped_prefill(act, dp, ds, out, tw, si, ei, ntpp, T * TOP_K, 1, True, cfg)
    torch.cuda.synchronize()
    _assert_one_ulp(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("T", [256, 1024])  # BLOCK_M=16 and BLOCK_M=64 wrapper tiers
def test_full_chain_statistics(T):
    import freetoken.moe.fused_ds_fp4 as fmod

    device = "cuda"
    gup, gus, dp, ds = _banks(device)
    x, slots, w = _routing(T, device)
    ref = fmod.routed_experts_fp4(x, slots.clone(), w, gup, gus, dp, ds, LIMIT)
    out = fmod.routed_experts_fp4_prefill(x, slots.clone(), w, gup, gus, dp, ds, LIMIT, E)
    torch.cuda.synchronize()
    bit_same = (ref == out).float().mean().item()
    assert bit_same > 0.985, bit_same
    diff = (out.float() - ref.float()).abs()
    denom = ref.float().pow(2).mean().sqrt()
    assert (diff.pow(2).mean().sqrt() / denom).item() < 5e-3
    assert diff.max().item() < 0.25 * ref.float().abs().max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sparse_chunk_falls_back_to_gemv():
    import freetoken.moe.fused_ds_fp4 as fmod

    device = "cuda"
    gup, gus, dp, ds = _banks(device)
    # 43 * 6 = 258 routes < _GROUPED_MIN_ROUTES: the wrapper IS the GEMV path
    x, slots, w = _routing(43, device)
    ref = fmod.routed_experts_fp4(x, slots.clone(), w, gup, gus, dp, ds, LIMIT)
    out = fmod.routed_experts_fp4_prefill(x, slots.clone(), w, gup, gus, dp, ds, LIMIT, E)
    assert torch.equal(ref, out)
