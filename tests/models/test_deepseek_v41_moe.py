from types import SimpleNamespace

import pytest
import torch

from freetoken.models.deepseek_v41.moe import DSV41OffloadMoELayer, Gate, clamped_swiglu


@pytest.mark.parametrize("score_func", ["softmax", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("topk,norm", [(1, True), (2, True), (2, False)])
def test_router_uses_temperature_and_bias_only_for_selection(score_func, topk, norm):
    args = SimpleNamespace(n_activated_experts=topk, score_func=score_func, gate_temp=2.0,
                           norm_topk_prob=norm, route_scale=1.5, n_routed_experts=3,
                           dim=2, vision_enabled=True)
    gate = Gate(0, args)
    gate.weight.data.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 1.]]))
    gate.bias.data.copy_(torch.tensor([0., 10., 0.]))
    gate.bias_vl.data.copy_(torch.tensor([0., 0., 10.]))
    x = torch.tensor([[2., -1.], [2., -1.]], dtype=torch.bfloat16)
    mask = torch.tensor([False, True])
    scores = torch.tensor([[1., -.5, -1.5], [1., -.5, -1.5]])
    if score_func == "softmax":
        scores = scores.exp() / scores.exp().sum(-1, keepdim=True)
    elif score_func == "sigmoid":
        scores = 1 / (1 + (-scores).exp())
    else:
        scores = torch.logaddexp(torch.zeros_like(scores), scores).sqrt()
    expected_ids = torch.tensor([[1, 0], [2, 0]])[:, :topk]
    expected = scores.gather(1, expected_ids)
    if norm and topk > 1:
        expected = expected / (expected.sum(-1, keepdim=True) + 1e-20)
    actual, ids = gate(x, mask)
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(actual, expected * 1.5)


def test_swiglu_clamps_gate_only_above_and_up_both_sides():
    gate = torch.tensor([-12., 12., 12., 2.])
    up = torch.tensor([12., -12., 2., 3.])
    clipped_gate = torch.tensor([-12., 10., 10., 2.])
    clipped_up = torch.tensor([10., -10., 2., 3.])
    expected = clipped_gate / (1 + (-clipped_gate).exp()) * clipped_up
    torch.testing.assert_close(clamped_swiglu(gate, up, 10.), expected)


@pytest.mark.parametrize("strategy,decode_target", [("offload", "gpu"), ("cpu", "cpu"), ("hybrid", "hybrid")])
def test_native_expert_method_receives_clamp_and_execution_policy(strategy, decode_target):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.quantization import QuantBackend, QuantKind, set_quant_backend

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    set_quant_backend(QuantBackend.parse("moe.nvfp4=triton"))
    args = SimpleNamespace(n_layers=1, index_source_layers=(), engram_layer_ids=(),
                           n_routed_experts=2, n_activated_experts=1, dim=32, moe_inter_dim=32,
                           norm_topk_prob=True, swiglu_limit=10.0)
    layer = DSV41OffloadMoELayer(0, args, strategy=strategy, decode_target=decode_target)
    method = layer.quant_method
    assert method.kind is QuantKind.NVFP4 and method.kernel.name == "triton"
    assert (method.cfg.activation, method.cfg.alpha, method.cfg.limit) == ("swiglu_clamp", 1.0, 10.0)
    assert (layer.alpha, layer.limit) == (1.0, 10.0)
    assert (method.cfg.strategy, method.cfg.decode_target) == (strategy, decode_target)
    assert not method.scheme.has("input_scale")
    assert set(method.layout()) == {"gate_up", "gate_up_scale", "gate_up_global",
                                    "down", "down_scale", "down_global"}
    assert not layer.state_dict()
