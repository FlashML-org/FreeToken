"""MXFP8 (block-32 e8m0) W8A16 linear and routed experts: kernels vs the dequant reference, plus the
Gemma (1+w) norm and swigluoai activation the MiniMax-M3 modules ride on."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

DEV = "cuda"


def _make_mxfp8(N: int, K: int, seed: int = 0):
    torch.manual_seed(seed)
    wf = torch.randn(N, K, device=DEV) * 0.05
    codes = torch.randint(110, 132, (N, K // 32), device=DEV, dtype=torch.uint8)
    descale = torch.exp2(codes.float() - 127.0)
    w8 = (
        (wf.view(N, -1, 32) / descale.unsqueeze(-1)).clamp(-448, 448).view(N, K)
    ).to(torch.float8_e4m3fn)
    return w8, codes


# 1 = m1 kernel; 2..256 = dot GEMV across its M_TILE buckets {16,32,64,128,256}
# (17/33/129 exercise row-padding masks); 257/300 = dequant+cuBLAS past
# _GEMV_MAX_M (257 pins the boundary).
@pytest.mark.parametrize("M", [1, 2, 8, 16, 17, 33, 64, 129, 256, 257, 300])
# (511, 6144): N not a BLOCK_N multiple (n-mask tail); (640, 6112): K a multiple
# of the 32-wide scale block but not of BLOCK_K=128 (k-mask + OOB scale codes).
@pytest.mark.parametrize("N,K", [(512, 6144), (9216, 6144), (640, 6144), (511, 6144), (640, 6112)])
def test_mxfp8_linear_matches_dequant_reference(M: int, N: int, K: int):
    from freetoken.kernel.triton.mxfp8_linear import mxfp8_dequant, mxfp8_linear

    w8, codes = _make_mxfp8(N, K, seed=M)
    ref_w = mxfp8_dequant(w8, codes, torch.float32)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    y = mxfp8_linear(x, w8, codes)
    y_ref = (x.float() @ ref_w.t()).to(torch.bfloat16)
    rel = (y.float() - y_ref.float()).abs().max() / y_ref.float().abs().max().clamp(min=1e-6)
    assert rel.item() < 2e-2, rel.item()


# (160, 96): K is a multiple of the 32-wide scale group but not of the K tiles (k-mask + masked scale codes), N not a multiple of the N tiles.
# 5, 60, 300 and 800 tokens put 2.5, 30, 150 and 400 routes on each expert, one per token block size of the prefill kernel.
@pytest.mark.parametrize("H,I", [(256, 128), (160, 96)])
@pytest.mark.parametrize("M", [5, 60, 300, 800])
@pytest.mark.parametrize("activation,alpha,limit", [("silu", 1.0, float("inf")), ("swiglu_clamp", 1.0, 0.5)])
def test_mxfp8_moe_matches_dequant_reference(H: int, I: int, M: int, activation: str, alpha: float, limit: float):
    from freetoken.kernel.triton.mxfp8_linear import mxfp8_dequant
    from freetoken.kernel.triton.mxfp8_moe import fused_experts_decode_mxfp8, fused_experts_mxfp8
    from freetoken.layers import gated_act_and_mul

    E, top_k = 4, 2
    gu, gus = _make_mxfp8(E * 2 * I, H, seed=M)
    dn, dns = _make_mxfp8(E * H, I, seed=M + 1)
    banks = (gu.view(E, 2 * I, H), gus.view(E, 2 * I, -1), dn.view(E, H, I), dns.view(E, H, -1))
    gu_ref = mxfp8_dequant(gu, gus, torch.float32).view(E, 2 * I, H)
    dn_ref = mxfp8_dequant(dn, dns, torch.float32).view(E, H, I)
    x = torch.randn(M, H, device=DEV, dtype=torch.bfloat16) * 8
    ids = torch.stack([torch.randperm(E, device=DEV)[:top_k] for _ in range(M)]).to(torch.int32)
    w = torch.rand(M, top_k, device=DEV)

    def reference(ids):
        ref = torch.zeros(M, H, device=DEV)
        for t, k in zip(*torch.nonzero(ids >= 0, as_tuple=True)):
            e = ids[t, k]
            h = (x[t].float() @ gu_ref[e].T).to(x.dtype).view(1, -1)
            a = torch.empty(1, I, device=DEV, dtype=x.dtype)
            gated_act_and_mul(activation, h, a, alpha=alpha, limit=limit)
            ref[t] += w[t, k] * (a[0].float() @ dn_ref[e].T)
        return ref

    def rel(y, ref):
        return ((y.float() - ref).abs().max() / ref.abs().max().clamp(min=1e-6)).item()

    ref = reference(ids)
    assert rel(fused_experts_mxfp8(x, *banks, w, ids, E, activation, alpha, limit), ref) < 2e-2
    assert rel(fused_experts_decode_mxfp8(x, *banks, w, ids, activation, alpha, limit), ref) < 2e-2
    # hybrid decode hands the GPU a route split with holes
    holes = ids.clone()
    holes[::2, 0] = -1
    assert rel(fused_experts_decode_mxfp8(x, *banks, w, holes, activation, alpha, limit), reference(holes)) < 2e-2


def test_gemma_plus_one_norm_matches_flashinfer_semantics():
    """Triton fallback vs the (1+w) definition; per-head 3D strided in-place."""
    from freetoken.kernel.triton.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

    torch.manual_seed(0)
    x = torch.randn(64, 6144, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(6144, device=DEV, dtype=torch.bfloat16) * 0.1

    def ref(v):
        vf = v.float()
        inv = torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (vf * inv * (1.0 + w.float())).to(torch.bfloat16)

    out = gemma_rmsnorm(x, w, 1e-6)
    assert (out.float() - ref(x).float()).abs().max().item() < 2e-2

    a, r = x.clone(), x.clone()
    gemma_fused_add_rmsnorm(a, r, w, 1e-6)
    assert torch.equal(r, (x.float() + x.float()).to(torch.bfloat16))
    assert (a.float() - ref(r).float()).abs().max().item() < 2e-2

    # per-head strided in-place (the fused-qkv slice pattern)
    q = torch.randn(16, 9216, device=DEV, dtype=torch.bfloat16)
    wq = torch.randn(128, device=DEV, dtype=torch.bfloat16) * 0.1
    qh = q[:, :8192].view(16, 64, 128)
    ref3 = torch.empty_like(qh)
    for h in range(64):
        vf = qh[:, h].float()
        inv = torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + 1e-6)
        ref3[:, h] = (vf * inv * (1.0 + wq.float())).to(torch.bfloat16)
    gemma_rmsnorm(qh, wq, 1e-6, out=qh)
    assert (qh.float() - ref3.float()).abs().max().item() < 2e-2


def test_swigluoai_and_mul_uninterleaved():
    from freetoken.layers import swigluoai_and_mul

    torch.manual_seed(1)
    d = 3072
    x = torch.randn(500, 2 * d, device=DEV, dtype=torch.bfloat16) * 3
    gate, up = x[:, :d].float(), x[:, d:].float()
    alpha, limit = 1.702, 7.0
    g = gate.clamp(max=limit)
    ref = g * torch.sigmoid(g * alpha) * (up.clamp(-limit, limit) + 1.0)
    out = swigluoai_and_mul(x, alpha=alpha, limit=limit)
    assert (out.float() - ref).abs().max().item() < 0.15
