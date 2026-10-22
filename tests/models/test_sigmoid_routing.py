"""Exercise each model's _route and the post-load FP32 router storage."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_fp32_router_load_preserves_values():
    from freetoken.layers.linear import Fp32RouterLinear

    with torch.device("cuda"):
        gate = Fp32RouterLinear(32, 16, has_bias=False)
    gate.weight = gate.weight.bfloat16()
    source = torch.randn_like(gate.weight)
    x = torch.randn(3, 32, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x.float(), source.float())
    state = {"gate.weight": source}
    gate.load_state_dict(state, prefix="gate")
    assert not state
    assert gate.weight.dtype == torch.float32
    torch.testing.assert_close(gate.weight, source.float(), rtol=0, atol=0)
    ptr = gate.weight.data_ptr()
    for _ in range(2):
        torch.testing.assert_close(gate.forward(x), expected, rtol=0, atol=0)
        assert gate.weight.data_ptr() == ptr


@pytest.mark.parametrize("family", ["glm4_moe", "glm_moe_dsa", "glm5_next", "minimax_m2", "minimax_m3"])
@pytest.mark.parametrize("normalize", [False, True])
def test_model_route_matches_original(family, normalize):
    import importlib

    names = {"glm4_moe": "Glm4MoeSparseBlock", "glm_moe_dsa": "GlmMoeDsaSparseBlock",
                 "glm5_next": "Glm5NextSparseBlock", "minimax_m2": "MiniMaxM2SparseMoeBlock",
                 "minimax_m3": "MiniMaxM3SparseMoeBlock"}
    cls = getattr(importlib.import_module(f"freetoken.models.{family}.moe"), names[family])
    torch.manual_seed(21)
    x = torch.randn(3, 32, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(18, 32, device="cuda", dtype=torch.bfloat16).float()
    logits = torch.nn.functional.linear(x.float(), weight)
    bias = torch.randn(18, device="cuda")
    if family == "minimax_m2":
        # M2 preserves its original BF16 gate output and selection-bias storage.
        logits = logits.bfloat16()
        bias = bias.bfloat16()
    gate = SimpleNamespace(weight=weight, forward=lambda h: torch.nn.functional.linear(h.float(), weight))
    obj = SimpleNamespace(gate=gate, e_score_correction_bias=bias, top_k=3,
                          norm_topk_prob=normalize, routed_scaling_factor=2.5, n_group=3, topk_group=2)
    actual = cls._route(obj, logits if family == "minimax_m2" else x)
    scores = logits.float().sigmoid()
    choice = scores + bias
    if family.startswith("glm"):
        grouped = choice.reshape(3, 3, 6)
        group_scores = grouped.topk(2, dim=-1).values.sum(-1)
        group_ids = group_scores.topk(2, dim=-1).indices
        mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, group_ids, True)
        choice = grouped.masked_fill(~mask[..., None], -float("inf")).reshape_as(choice)
    ids = choice.topk(3, dim=-1).indices
    w = scores.gather(1, ids)
    eps = 0.0 if family == "minimax_m2" else 1e-20
    if normalize:
        w = w / (w.sum(-1, keepdim=True) + eps)
    if family != "minimax_m2":
        w = w * 2.5
    torch.testing.assert_close(actual[1], ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(actual[0], w, rtol=1e-5, atol=1e-6)


def test_parent_load_and_graph_replay_fp32_gate():
    from freetoken.layers.base import BaseOP
    from freetoken.layers.linear import Fp32RouterLinear

    parent = BaseOP()
    with torch.device("cuda"):
        parent.gate = Fp32RouterLinear(32, 16, has_bias=False)
    parent.gate.weight = parent.gate.weight.bfloat16()
    source = torch.randn_like(parent.gate.weight)
    parent.load_state_dict({"gate.weight": source})
    assert parent.gate.weight.dtype == torch.float32
    x = torch.randn(3, 32, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            parent.gate.forward(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = parent.gate.forward(x)
    pointer = parent.gate.weight.data_ptr()
    for _ in range(3):
        x.normal_()
        graph.replay()
        expected = torch.nn.functional.linear(x.float(), source.float())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert parent.gate.weight.data_ptr() == pointer
