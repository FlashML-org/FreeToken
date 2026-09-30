from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _ref_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


@pytest.mark.parametrize("with_scale", [False, True])
@pytest.mark.parametrize("shape", [(3, 256), (2, 3, 256), (3, 5376)])
def test_gemma_rmsnorm_uses_sgl_kernel_semantics(
    with_scale: bool,
    shape: tuple[int, ...],
):
    from freetoken.layers import GemmaRMSNorm

    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    norm = GemmaRMSNorm(shape[-1], eps=1e-6, with_scale=with_scale)
    weight = None
    if with_scale:
        weight = torch.randn((shape[-1],), device="cuda", dtype=torch.bfloat16)
        norm.weight = weight

    out = norm.forward(x)
    ref = _ref_rmsnorm(x, weight, eps=1e-6)

    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("hidden_size", [8, 5376])
def test_gemma_add_rmsnorm_uses_sgl_kernel_semantics(hidden_size: int):
    from freetoken.layers import GemmaRMSNorm

    torch.manual_seed(4)
    x = torch.randn((2, hidden_size), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((2, hidden_size), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden_size,), device="cuda", dtype=torch.bfloat16)
    norm = GemmaRMSNorm(hidden_size, eps=1e-6)
    norm.weight = weight

    out, residual_out = norm.forward_add_residual(x.clone(), residual.clone())
    ref_residual = x.float() + residual.float()
    ref = _ref_rmsnorm(ref_residual.to(x.dtype), weight, eps=1e-6)

    torch.testing.assert_close(residual_out.float(), ref_residual, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("tokens, hidden_size", [(2, 256), (2, 5376), (80, 2816)])
def test_gemma_dual_rmsnorm_residual_scalar_matches_reference(
    tokens: int,
    hidden_size: int,
):
    from freetoken.kernel.triton.gemma4_fused import gemma_dual_rmsnorm_residual_scalar

    torch.manual_seed(7)
    x1 = torch.randn((tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    x2 = torch.randn((tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn((hidden_size,), device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn((hidden_size,), device="cuda", dtype=torch.bfloat16)
    w3 = torch.randn((hidden_size,), device="cuda", dtype=torch.bfloat16)
    scalar = torch.randn((1,), device="cuda", dtype=torch.bfloat16)

    out = gemma_dual_rmsnorm_residual_scalar(
        x1,
        w1,
        x2,
        w2,
        w3,
        residual,
        scalar,
        1e-6,
        1e-6,
        1e-6,
    )

    combined = _ref_rmsnorm(x1, w1, 1e-6).float() + _ref_rmsnorm(x2, w2, 1e-6).float()
    ref = (residual.float() + _ref_rmsnorm(combined.to(x1.dtype), w3, 1e-6).float()) * scalar.float()
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_gemma4_router_uses_sgl_kernel_topk_softmax_semantics():
    from freetoken.models.gemma4 import Gemma4Router

    cfg = SimpleNamespace(
        hidden_size=16,
        rms_norm_eps=1e-6,
        num_experts=128,
        num_experts_per_tok=8,
    )
    torch.manual_seed(3)
    router = Gemma4Router(cfg)
    logits = torch.randn((5, cfg.num_experts), device="cuda", dtype=torch.bfloat16)
    per_expert_scale = (
        torch.rand((cfg.num_experts,), device="cuda", dtype=torch.bfloat16) + 0.5
    )
    router.per_expert_scale = per_expert_scale
    router.scale = torch.ones((cfg.hidden_size,), device="cuda", dtype=torch.bfloat16)
    router.norm.forward = lambda x: x
    router.proj.forward = lambda h: logits

    weights, ids = router.forward(torch.zeros((5, cfg.hidden_size), device="cuda"))

    ref_ids = torch.argsort(logits.float(), dim=-1, descending=True, stable=True)[:, :cfg.num_experts_per_tok]  # Match the router's lower-ID-first tie policy for equal BF16 logits.
    topk_logits = logits.float().gather(-1, ref_ids)  # Compute reference probabilities for those exact expert IDs.
    ref_weights = torch.softmax(topk_logits, dim=-1) * per_expert_scale[ref_ids].float()

    torch.testing.assert_close(ids, ref_ids.to(torch.int32))
    torch.testing.assert_close(weights, ref_weights, rtol=2e-4, atol=2e-4)
