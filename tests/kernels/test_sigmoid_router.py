"""Compare fused routing with the original eager torch expressions."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def reference(logits, bias, k, *, renormalize, n_group=1, topk_group=1, routed_scale=1.0, eps=1e-20):
    scores = logits.float().sigmoid()
    choice = scores + bias.float()
    if n_group > 1:
        grouped = choice.reshape(len(logits), n_group, logits.shape[1] // n_group)
        group_scores = grouped.topk(2, dim=-1).values.sum(-1)
        group_ids = group_scores.topk(topk_group, dim=-1).indices
        mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, group_ids, True)
        choice = grouped.masked_fill(~mask[..., None], -float("inf")).reshape_as(choice)
    ids = choice.topk(k, dim=-1).indices
    weights = scores.gather(1, ids)
    if renormalize:
        weights = weights / (weights.sum(-1, keepdim=True) + eps)
    return (weights * routed_scale).contiguous(), ids.to(torch.int32).contiguous()


def assert_routes(actual, expected):
    w, ids = actual
    wr, ir = expected
    assert ids.dtype == torch.int32 and w.dtype == torch.float32
    assert ids.is_contiguous() and w.is_contiguous()
    torch.testing.assert_close(ids, ir, rtol=0, atol=0)
    torch.testing.assert_close(w, wr, rtol=1e-5, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("tokens", [0, 1, 7, 257])
@pytest.mark.parametrize("experts,k,groups,keep", [(128, 4, 1, 1), (288, 8, 1, 1), (192, 6, 8, 3), (90, 5, 3, 2)])
@pytest.mark.parametrize("normalize,eps,scale", [(False, 0.0, 1.0), (True, 0.0, 1.0), (True, 1e-20, 2.5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sigmoid_router(tokens, experts, k, groups, keep, normalize, eps, scale, dtype):
    from freetoken.kernel.triton.sigmoid_router import fused_topk_sigmoid

    torch.manual_seed(42)
    logits = torch.randn(tokens, experts * 2, device="cuda", dtype=dtype)[:, ::2]
    bias = (torch.randn(experts * 2, device="cuda") * 0.1)[::2]
    kwargs = {"renormalize": normalize, "n_group": groups, "topk_group": keep, "eps": eps, "routed_scale": scale}
    assert_routes(fused_topk_sigmoid(logits, bias, k, **kwargs), reference(logits, bias, k, **kwargs))


def test_bias_is_only_for_selection_and_group_scores_use_two_experts():
    from freetoken.kernel.triton.sigmoid_router import fused_topk_sigmoid

    logits = torch.zeros(1, 6, device="cuda")
    # Group 0 has the largest individual score; group 1 has the largest top-2 sum.
    bias = torch.tensor([8., 0., 5., 4., 1., 0.], device="cuda")
    w, ids = fused_topk_sigmoid(logits, bias, 2, renormalize=False, n_group=3, topk_group=1)
    assert ids.tolist() == [[2, 3]]
    torch.testing.assert_close(w, torch.full_like(w, 0.5), rtol=0, atol=0)


@pytest.mark.parametrize("groups,keep", [(1, 1), (3, 2)])
def test_exact_ties_choose_lowest_ids(groups, keep):
    from freetoken.kernel.triton.sigmoid_router import fused_topk_sigmoid

    x = torch.zeros(2, 18, device="cuda")
    w, ids = fused_topk_sigmoid(x, x[0], 5, renormalize=True, n_group=groups, topk_group=keep)
    assert ids.tolist() == [list(range(5))] * 2
    torch.testing.assert_close(w, torch.full_like(w, 0.2))


def test_near_ties_and_selecting_every_expert():
    from freetoken.kernel.triton.sigmoid_router import fused_topk_sigmoid

    x = torch.zeros(1, 18, device="cuda")
    bias = torch.arange(18, device="cuda", dtype=torch.float32) * 1e-5
    for k in (1, 5, 18):
        kwargs = {"renormalize": True, "n_group": 3, "topk_group": 3}
        assert_routes(fused_topk_sigmoid(x, bias, k, **kwargs), reference(x, bias, k, **kwargs))


@pytest.mark.parametrize("eps", [0.0, 1e-20])
@pytest.mark.parametrize("k", [1, 4])
def test_zero_sum_and_top_one_normalization(eps, k):
    from freetoken.kernel.triton.sigmoid_router import fused_topk_sigmoid

    x = torch.tensor([[-1000.] * 8, [0.] * 8], device="cuda")
    b = torch.arange(8, device="cuda", dtype=torch.float32)
    kwargs = {"renormalize": True, "eps": eps}
    assert_routes(fused_topk_sigmoid(x, b, k, **kwargs), reference(x, b, k, **kwargs))


def test_graph_replay_updates_inputs_and_token_limit():
    from freetoken.kernel.triton.sigmoid_router import fused_topk_sigmoid

    torch.manual_seed(77)
    x = torch.randn(7, 90, device="cuda")
    b = torch.randn(90, device="cuda")
    limit = torch.tensor(7, dtype=torch.int32, device="cuda")
    kwargs = {"renormalize": True, "n_group": 3, "topk_group": 2}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused_topk_sigmoid(x, b, 5, num_token_non_padded=limit, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w, ids = fused_topk_sigmoid(x, b, 5, num_token_non_padded=limit, **kwargs)
    for count in (3, 0, 7, 1):
        x.normal_()
        b.normal_()
        limit.fill_(count)
        w.fill_(float("nan"))
        ids.fill_(999)
        graph.replay()
        wr, ir = reference(x, b, 5, **kwargs)
        assert_routes((w[:count], ids[:count]), (wr[:count], ir[:count]))
        assert torch.all(ids[count:] == -1)
        assert torch.all(w[count:] == 0)
