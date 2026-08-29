from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def _reference(hidden, fn, base, scale, norm_eps=1e-5, hc_eps=1e-6, iters=20):
    hc = hidden.shape[-2]
    flat = hidden.flatten(-2).float()
    normed = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    pre_w, post_w, comb_w = F.linear(normed, fn).split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = base.split([hc, hc, hc * hc])
    pre = torch.sigmoid(pre_w * scale[0] + pre_b) + hc_eps
    post = 2 * torch.sigmoid(post_w * scale[1] + post_b)
    comb = torch.softmax(
        comb_w.reshape(*comb_w.shape[:-1], hc, hc) * scale[2] + comb_b.view(hc, hc),
        -1,
    )
    comb = comb + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    collapsed = (pre.unsqueeze(-1) * hidden.float()).sum(-2).to(hidden.dtype)
    return collapsed, post, comb


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mhc_fused_path_matches_release_equations_on_ampere():
    from freetoken.models.glm5_next.hc import HyperConnection, collapse_head

    torch.manual_seed(53)
    device = torch.device("cuda")
    module = HyperConnection(hidden_size=128).to(device)
    with torch.no_grad():
        module.fn.normal_(std=0.02)
        module.base.normal_(std=0.02)
        module.scale.copy_(torch.tensor([0.7, 0.9, 1.1], device=device))
    hidden = torch.randn(2, 3, 4, 128, device=device, dtype=torch.bfloat16)

    collapsed, post, comb = module.mix(hidden)
    ref_collapsed, ref_post, ref_comb = _reference(
        hidden, module.fn, module.base, module.scale
    )
    torch.testing.assert_close(collapsed, ref_collapsed, rtol=0, atol=0.02)
    torch.testing.assert_close(post, ref_post, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(comb, ref_comb, rtol=2e-5, atol=2e-5)

    block_output = torch.randn_like(collapsed)
    got = module.combine(hidden, block_output, post, comb)
    expected = post.to(hidden.dtype).unsqueeze(-1) * block_output.unsqueeze(-2)
    expected += torch.matmul(comb.to(hidden.dtype).transpose(-1, -2), hidden)
    # The fused kernel accumulates in fp32 before one bf16 store; the eager release
    # expression rounds the multiply and matmul separately (at most two bf16 ulps here).
    torch.testing.assert_close(got, expected, rtol=0, atol=0.04)
    torch.testing.assert_close(collapse_head(hidden), hidden.mean(dim=-2), rtol=0, atol=0)
