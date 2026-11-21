"""Router and expert activation against the references checked against transformers #51050
(``Inkling2LoopTopKRouter``, ``Inkling2LoopSoftcappedSwiGLU``)."""

from __future__ import annotations

import pytest
import torch

from .reference import route as ref_route
from .reference import softcap_swiglu as ref_swiglu


def test_route_matches_reference():
    from freetoken.models.inkling2_loop.moe import route

    g = torch.Generator().manual_seed(0)
    x = torch.randn(97, 2304, generator=g).to(torch.bfloat16)
    weight = torch.randn(128, 2304, generator=g) * 0.02
    # a bias large enough to change the pick: selection uses score + bias, the weights the bare score
    bias = torch.randn(128, generator=g) * 0.2
    w, ids = route(x, weight, bias, 8, 8.0)
    order = ids.long().argsort(-1)
    w_ref, ids_ref = ref_route(x, weight, bias, 8, 8.0)
    assert torch.equal(ids.long().gather(1, order), ids_ref)
    assert torch.allclose(w.gather(1, order), w_ref, atol=1e-6)
    assert not torch.equal(ids_ref, ref_route(x, weight, torch.zeros_like(bias), 8, 8.0)[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")
def test_softcap_swiglu_kernel():
    from freetoken.layers import gated_act_and_mul

    g = torch.Generator().manual_seed(1)
    # gate past its cap of 4 and up past 25 on many elements, where the caps bend the result
    x = (torch.randn(257, 2 * 1152, generator=g) * torch.tensor([8.0] * 1152 + [40.0] * 1152)).to("cuda", torch.bfloat16)
    out = torch.empty(257, 1152, device="cuda", dtype=torch.bfloat16)
    gated_act_and_mul("swiglu_softcap", x, out, alpha=4.0, limit=25.0)
    ref = ref_swiglu(x[:, :1152], x[:, 1152:], 4.0, 25.0)
    assert ((out.float() - ref).abs() / ref.abs().clamp(min=1.0)).max().item() < 1e-2
