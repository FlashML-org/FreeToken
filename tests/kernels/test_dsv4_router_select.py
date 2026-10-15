"""Fused gate selection (``kernel/triton/dsv4/router``) against the torch expression it replaces, for
every score function, with and without top-k weight normalization, at decode and prefill widths."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def reference(scores, bias, topk, score_func, normalize, route_scale):
    if score_func == "softmax":
        s = scores.softmax(dim=-1)
    elif score_func == "sigmoid":
        s = scores.sigmoid()
    else:
        s = F.softplus(scores).sqrt()
    ids = (s + bias).topk(topk, dim=-1)[1]
    w = s.gather(1, ids)
    if normalize and topk > 1:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w * route_scale, ids


@pytest.mark.parametrize("score_func", ["sqrtsoftplus", "sigmoid", "softmax"])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("T", [1, 3, 300])
def test_router_select_matches_torch(score_func, normalize, T):
    from freetoken.kernel.triton.dsv4.router import router_select

    g = torch.Generator(device="cpu").manual_seed(T)
    E, K = 384, 6
    scores = (torch.randn(T, E, generator=g) * 3).cuda()
    bias = (torch.randn(E, generator=g) * 0.1).cuda()
    w, ids = router_select(scores, bias, K, score_func=score_func, normalize=normalize, route_scale=2.5)
    w_ref, ids_ref = reference(scores, bias, K, score_func, normalize, 2.5)
    assert torch.equal(ids.sort(dim=-1).values, ids_ref.to(torch.int32).sort(dim=-1).values)
    # same expert set; compare the weights per expert (selection order may differ only on exact ties)
    order = ids.argsort(dim=-1)
    order_ref = ids_ref.argsort(dim=-1)
    torch.testing.assert_close(w.gather(1, order), w_ref.gather(1, order_ref), atol=1e-6, rtol=1e-5)


def test_router_select_breaks_ties_toward_the_lowest_expert():
    from freetoken.kernel.triton.dsv4.router import router_select

    scores = torch.zeros(1, 384, device="cuda")  # every expert ties
    bias = torch.zeros(384, device="cuda")
    _, ids = router_select(scores, bias, 6, score_func="sigmoid", normalize=True, route_scale=1.0)
    assert ids.tolist() == [[0, 1, 2, 3, 4, 5]]
