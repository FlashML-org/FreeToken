"""``kernel/triton/dsv4/fp8_linear.block_fp8_linear`` against a torch transcription of the reference
``act_quant`` + ``fp8_gemm`` (ue8m0 activation scales, ``block x block`` e8m0 weight scales), for the
DeepSeek-V4 block (128) and the DeepSeek-V4.1 block (32), on the GEMV (M=1) and GEMM (M>1) paths."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu


def _pow2_ceil_scale(amax: torch.Tensor, qmax: float) -> torch.Tensor:
    """``2 ** ceil(log2(amax / qmax))`` -- the reference ``fast_round_scale``."""
    return torch.exp2(torch.ceil(torch.log2(amax / qmax)))


def quantize_weight(w: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per ``block x block`` ue8m0-scaled e4m3 weight, scale as e8m0 codes."""
    n, k = w.shape
    blocks = w.float().view(n // block, block, k // block, block).permute(0, 2, 1, 3)
    amax = blocks.abs().amax(dim=(-1, -2)).clamp_min(1e-4)
    scale = _pow2_ceil_scale(amax, 448.0)  # [n//block, k//block]
    q = (blocks / scale[..., None, None]).clamp(-448.0, 448.0).to(FP8)
    w_fp8 = q.permute(0, 2, 1, 3).reshape(n, k)
    return w_fp8, scale.to(E8M0)


def act_quant_reference(x: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    groups = x.float().view(m, k // block, block)
    amax = groups.abs().amax(dim=-1).clamp_min(1e-4)
    scale = _pow2_ceil_scale(amax, 448.0)
    q = (groups / scale[..., None]).clamp(-448.0, 448.0).to(FP8)
    return q.view(m, k), scale


def linear_reference(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor, block: int) -> torch.Tensor:
    m, k = x.shape
    n = w_fp8.shape[0]
    a_fp8, a_scale = act_quant_reference(x, block)
    a = a_fp8.float().view(m, k // block, block)
    w = w_fp8.float().view(n // block, block, k // block, block)
    ws = w_scale.float()  # [n//block, k//block]
    out = torch.zeros(m, n, dtype=torch.float32, device=x.device)
    for kb in range(k // block):
        wk = w[:, :, kb, :].reshape(n, block)  # [n, block]
        partial = a[:, kb, :] @ wk.t()  # [m, n]
        out += partial * a_scale[:, kb, None] * ws[:, kb].repeat_interleave(block)[None, :]
    return out


@pytest.mark.parametrize("block", [32, 128])
@pytest.mark.parametrize("m", [1, 2, 5, 8, 9, 64])
def test_block_fp8_linear_matches_reference(block: int, m: int):
    from freetoken.kernel.triton.dsv4.fp8_linear import block_fp8_linear

    torch.manual_seed(0)
    # K // block = 5 slabs exercises the GEMV split that must divide the slab count; N spans two GEMM N-tiles
    k, n = 5 * block, 256 if block == 128 else 224
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 3
    w_fp8, w_scale = quantize_weight(torch.randn(n, k, device="cuda", dtype=torch.bfloat16), block)

    got = block_fp8_linear(x, w_fp8, w_scale, block=block).float()
    want = linear_reference(x, w_fp8, w_scale, block)
    # bf16 output; the fp32 reduction order inside the MMA may differ from the slab loop
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2 * want.abs().max().item())


def test_block_argument_must_match_the_scale_shape():
    from freetoken.kernel.triton.dsv4.fp8_linear import block_fp8_linear

    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    w_fp8, w_scale = quantize_weight(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16), 32)
    with pytest.raises(AssertionError):
        block_fp8_linear(x, w_fp8, w_scale, block=128)


@pytest.mark.parametrize("block", [32, 128])
def test_grouped_w8a16_gemv_matches_the_dequantized_einsum(block):
    """``wo_a`` at decode: the fp8 payload dequantized in-kernel against a bf16 activation must equal
    the reference's bf16 einsum over the dequantized weight (fp32 accumulation, order aside)."""
    from freetoken.kernel.triton.dsv4.fp8_linear import dequant_block_fp8, grouped_w8a16_gemv

    G, r, d = 4, 256, 1024
    torch.manual_seed(block)
    w = torch.randn(G * r, d, device="cuda") * 0.05
    w_fp8, scale = quantize_weight(w, block)
    x = torch.randn(G, d, device="cuda", dtype=torch.bfloat16)
    deq = dequant_block_fp8(w_fp8, scale, block=block)
    ref_deq = (w_fp8.float() * torch.exp2(scale.view(torch.uint8).float() - 127.0).repeat_interleave(block, 0).repeat_interleave(block, 1)).to(torch.bfloat16)
    assert torch.equal(deq, ref_deq)
    for t in (1, 3):
        xt = torch.randn(t, G, d, device="cuda", dtype=torch.bfloat16) if t > 1 else x.unsqueeze(0)
        want = torch.einsum("tgd,grd->tgr", xt, ref_deq.view(G, r, d)).flatten(1)
        got = grouped_w8a16_gemv(xt, w_fp8, scale, block=block)
        assert got.shape == (t, G * r) and got.dtype == torch.bfloat16
        torch.testing.assert_close(got.float(), want.float(), atol=2e-2, rtol=2e-2)
