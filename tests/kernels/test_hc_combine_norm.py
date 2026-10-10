"""Pins for the fused hc combine + grouped RMSNorm path (hc_combine_norm).

hc_combine_norm replaces the standalone pair (hc_combine -> grouped_gemma_rmsnorm)
in the Qwen4-exp decoder loop. The swap is only sound if it is BITWISE identical:
the hyper-connection streams feed every later layer, and a bf16 flip here compounds
into different greedy tokens by the end of the stack. We assert torch.equal, not
assert_close: an approx-tolerance test would not have caught the regression this
path shipped with at first attempt -- the fused kernel reduced squares over the
[8, block] tile view (a two-axis tree) while the standalone kernel reduces the
flat row tree; fp32 ulps that differ between trees flip bf16 rounding. Both paths
on the same data must return the same bytes: fails before the reduction-tree fix
in _hc_combine_norm_kernel, passes after.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.hc import hc_combine, grouped_gemma_rmsnorm, hc_combine_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HC, H = 4, 2560  # model geometry: the reduction tree specializes on the block layout
DIM = HC * H
EPS = 1e-6


def _inputs(seed: int, T: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    bf = lambda shape: torch.randn(shape, generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    return (
        bf((T, DIM)),  # residual streams R
        bf((T, H)),  # block output y
        bf((T, HC)),  # inject scores s
        (torch.randn(DIM, generator=g, device="cuda", dtype=torch.float32) * 0.1).to(torch.bfloat16),  # norm weight
    )


@pytest.mark.parametrize("T", [1, 8])
@pytest.mark.parametrize("seed", range(6))
def test_combine_norm_matches_split_bitwise(seed: int, T: int) -> None:
    R, y, s, w = _inputs(seed, T)
    r2_split = hc_combine(R, y, s, HC)
    rn_split = grouped_gemma_rmsnorm(r2_split, w, EPS, HC)
    r2_fused, rn_fused = hc_combine_norm(R, y, s, w, EPS, HC)
    assert torch.equal(r2_fused, r2_split)
    assert torch.equal(rn_fused, rn_split)


@pytest.mark.parametrize("T", [1, 8])
@pytest.mark.parametrize("seed", range(6))
def test_gated_combine_norm_matches_split_bitwise(seed: int, T: int) -> None:
    # The MoE epilogue (routed + gate * shared, fp32 math, bf16 store) folded into
    # the fused combine+norm: elementwise only, so it must agree with the split
    # chain (shared_gate_mul_add -> hc_combine_norm) to the byte.
    from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add

    R, routed, s, w = _inputs(seed, T)
    g = torch.Generator(device="cuda").manual_seed(seed + 1000)
    shared = torch.randn((T, H), generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    gate = torch.rand(T, generator=g, device="cuda", dtype=torch.float32)
    y_split = shared_gate_mul_add(routed, shared, gate)
    r2_split, rn_split = hc_combine_norm(R, y_split, s, w, EPS, HC)
    r2_fused, rn_fused = hc_combine_norm(R, routed, s, w, EPS, HC, shared=shared, gate=gate)
    assert torch.equal(r2_fused, r2_split)
    assert torch.equal(rn_fused, rn_split)
