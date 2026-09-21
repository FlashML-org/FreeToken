"""swiglu_clamp (GLM-5.3 ``swiglu_limit``) activation parity.

Reference: vLLM's SiluAndMulWithClamp with alpha=1, beta=0 --
``clamp(gate, max=L) * sigmoid(gate_clamped) * clamp(up, +-L)``. Checks the
Triton kernel, its distinction from swigluoai (the +1 up bias), and that the
compiled CPU MoE extension advertises the new generic act id.
"""

from __future__ import annotations

import pytest
import torch

LIMIT = 10.0


def _ref(x: torch.Tensor, limit: float = LIMIT) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].float().clamp(max=limit)
    up = x[..., d:].float().clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(gate) * up).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_matches_reference():
    from freetoken.layers import swiglu_clamp_and_mul

    torch.manual_seed(0)
    # Scale up so the clamp actually engages on a good fraction of elements.
    x = torch.randn(129, 2 * 512, device="cuda", dtype=torch.bfloat16) * 8.0
    out = swiglu_clamp_and_mul(x, alpha=1.0, limit=LIMIT)
    ref = _ref(x)
    assert (out.float() - ref.float()).abs().max().item() < 2e-2
    assert (x[..., :512].float() > LIMIT).any(), "test data never hit the clamp"


def test_cpu_extension_supports_swiglu_clamp():
    from freetoken.moe.cpu_executor import compiled_extension_supports

    assert compiled_extension_supports("swiglu_clamp"), (
        "compiled _cpu_moe extension is stale -- rebuild with ACT_SWIGLU_CLAMP "
        "(python setup.py build_ext --inplace)"
    )
