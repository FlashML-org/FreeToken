"""The block-fp8 MoE kernels against a torch reference, for each gate-up epilogue they accept."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, H, I, TOPK, M = 4, 256, 256, 2, 8
BLOCK = 128


def _quant_block(w):
    n, k = w.shape
    blocks = w.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / 448.0
    q = (blocks / scale).to(torch.float8_e4m3fn).view(n, k)
    return q, scale.squeeze(1).squeeze(-1).to(torch.bfloat16)


def _dequant(q, s):
    n, k = q.shape
    return (q.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK) * s.float()[:, None, :, None]).view(n, k)


def _experts():
    torch.manual_seed(0)
    gu = [_quant_block(torch.randn(2 * I, H, device="cuda") / H**0.5) for _ in range(E)]
    dn = [_quant_block(torch.randn(H, I, device="cuda") / I**0.5) for _ in range(E)]
    stack = lambda xs, i: torch.stack([x[i] for x in xs]).contiguous()
    return stack(gu, 0), stack(gu, 1), stack(dn, 0), stack(dn, 1)


def _routing():
    ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(M)]).to(torch.int32)
    w = torch.softmax(torch.randn(M, TOPK, device="cuda"), dim=-1)
    return w, ids


def _reference(x, gate_up, gate_up_scale, down, down_scale, w, ids, activation, alpha, limit):
    from freetoken.layers import gated_act_and_mul

    out = torch.zeros(M, H, device="cuda", dtype=torch.float32)
    for e in range(E):
        gu = _dequant(gate_up[e], gate_up_scale[e])
        dn = _dequant(down[e], down_scale[e])
        for t, k in zip(*torch.nonzero(ids == e, as_tuple=True)):
            h = (x[t].float() @ gu.T).to(x.dtype).view(1, -1)
            a = torch.empty(1, I, device="cuda", dtype=x.dtype)
            gated_act_and_mul(activation, h, a, alpha=alpha, limit=limit)
            out[t] += w[t, k] * (a[0].float() @ dn.T)
    return out.to(x.dtype)


@pytest.mark.parametrize("activation, alpha, limit", [("silu", 1.0, float("inf")), ("swiglu_clamp", 1.0, 0.5), ("gelu_tanh", 1.0, float("inf"))])
def test_fp8_block_moe_epilogues_match_the_reference(activation, alpha, limit):
    from freetoken.kernel.triton.fp8_blockscale_moe import fused_experts_decode_fp8_blockscale, fused_experts_fp8_blockscale

    gate_up, gate_up_scale, down, down_scale = _experts()
    w, ids = _routing()
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    ref = _reference(x, gate_up, gate_up_scale, down, down_scale, w, ids, activation, alpha, limit).float()
    plain = _reference(x, gate_up, gate_up_scale, down, down_scale, w, ids, "silu", 1.0, float("inf")).float()
    if activation != "silu":
        assert not torch.allclose(ref, plain, rtol=1e-2, atol=1e-3), "the epilogue under test must change the result"

    decode = fused_experts_decode_fp8_blockscale(x, gate_up, gate_up_scale, down, down_scale, w, ids, activation, alpha, limit).float()
    # W8A16 decode: only bf16 accumulation-order differences remain
    assert torch.nn.functional.cosine_similarity(decode.flatten(), ref.flatten(), dim=0) > 0.999
    assert (decode - ref).abs().max() <= 2e-2 * ref.abs().max() + 1e-3

    prefill = fused_experts_fp8_blockscale(x, gate_up, gate_up_scale, down, down_scale, w, ids, E, activation, alpha, limit).float()
    # W8A8 prefill quantizes the activations per 128-group, so the tolerance is the fp8 activation error
    assert torch.nn.functional.cosine_similarity(prefill.flatten(), ref.flatten(), dim=0) > 0.99
    assert (prefill - ref).abs().max() <= 8e-2 * ref.abs().max() + 1e-3
