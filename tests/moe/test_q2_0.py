"""External expert overrides must preserve the trunk's independent schemes."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from freetoken.layers.quantization import QuantConfig, QuantKind
from freetoken.layers.quantization.configs.base import ExpertSourceConfig, NoQuantConfig
from freetoken.layers.quantization.moe.base import MoEConfig
from freetoken.layers.quantization.moe.q2_0 import Q20MoEMethod
from freetoken.layers.quantization.names import NameMap


def test_q2_override_preserves_trunk_and_shared_experts():
    quant = ExpertSourceConfig(NoQuantConfig(NameMap(roots=(("model.layers", "model.language_model.layers"),))))
    for name in ("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.shared_expert.gate_up_proj",
                 "model.layers.0.mlp.shared_expert_gate", "model.layers.0.mlp.gate", "lm_head"):
        assert quant.scheme_for(name) is None
    scheme = quant.scheme_for("model.layers.0.mlp.experts")
    assert scheme.kind is QuantKind.Q2_0
    assert quant.scheme_for_name("model.language_model.layers.0.mlp.experts.511.down_proj") == scheme


def test_q2_layout_packing_preserves_expert_rows_without_expansion():
    cfg = MoEConfig(num_experts=3, hidden=64, intermediate=64, top_k=2, strategy="offload")
    method = Q20MoEMethod(cfg)
    layout = method.layout()
    assert sum(torch.empty(spec.shape, dtype=spec.dtype).numel() for spec in layout.values()) == 3 * 64 * 64 // 64 * 18
    pieces = {role: torch.randint(0, 256, (3, 64, 18), dtype=torch.uint8) for role in ("gate", "up", "down")}
    out = {role: torch.empty((3, *spec.shape), dtype=spec.dtype) for role, spec in layout.items()}
    method.pack(pieces, out)
    torch.testing.assert_close(out["gate_up"][:, :64], pieces["gate"], rtol=0, atol=0)
    torch.testing.assert_close(out["gate_up"][:, 64:], pieces["up"], rtol=0, atol=0)
    torch.testing.assert_close(out["down"], pieces["down"], rtol=0, atol=0)
    assert method.cpu_format is None
    for changes in ({"decode_target": "cpu"}, {"strategy": "resident"}, {"hidden": 65},
                    {"dtype": torch.float16}, {"apply_router_weight_on_input": True}):
        with pytest.raises(RuntimeError):
            Q20MoEMethod(replace(cfg, **changes))


def test_q2_source_is_explicit_and_architecture_scoped():
    config = SimpleNamespace(model_type="qwen4_exp", freetoken_expert_source={"format": "q2_0"})
    assert QuantConfig.from_hf(config).scheme_for("model.layers.0.mlp.experts").kind is QuantKind.Q2_0
    config.model_type = "qwen3_5_moe"
    with pytest.raises(ValueError):
        QuantConfig.from_hf(config)


def _quantized_weights(experts, rows, columns):
    scales = (torch.rand(experts, rows, columns // 64, device='cuda') * .06 - .03).half()
    scales[0, 0, 0] = 0
    codes = torch.randint(0, 4, (experts, rows, columns), device='cuda', dtype=torch.uint8)
    blocks = torch.empty((experts, rows, columns // 64, 18), dtype=torch.uint8, device='cuda')
    blocks[..., :2] = scales.contiguous().view(torch.uint8).reshape(experts, rows, columns // 64, 2)
    fields = codes.reshape(experts, rows, columns // 64, 16, 4)
    blocks[..., 2:] = fields[..., 0] | fields[..., 1] << 2 | fields[..., 2] << 4 | fields[..., 3] << 6
    # Reference values come from the integer grid, not the kernel's decoder.
    reference = ((codes.float() - 1) * scales.float().repeat_interleave(64, -1)).bfloat16()
    return blocks.reshape(experts, rows, columns // 64 * 18), reference


def _reference(x, gate_up, down, ids, weights):
    output = torch.zeros(x.shape, dtype=torch.float32, device='cuda')
    for token in range(x.shape[0]):
        for route in range(ids.shape[1]):
            expert = ids[token, route]
            gate, up = (gate_up[expert].float() @ x[token].float()).bfloat16().float().chunk(2)
            active = (torch.nn.functional.silu(gate) * up).bfloat16().float()
            result = (down[expert].float() @ active).bfloat16().float()
            output[token] += result * weights[token, route]
    return output.bfloat16()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('tokens,topk,remap', [(1, 1, False), (7, 3, False), (19, 3, False), (5, 3, True)])
def test_q2_vector_matches_reference_with_signed_scales_and_cache_slots(tokens, topk, remap):
    from freetoken.moe.fused_q2_0 import fused_experts_q2_0

    torch.manual_seed(719)
    gate_up, gu_ref = _quantized_weights(8, 128, 128)
    down, down_ref = _quantized_weights(8, 128, 64)
    x = torch.randn(tokens, 128, device='cuda', dtype=torch.bfloat16)
    ids = torch.randint(0, 8, (tokens, topk), device='cuda', dtype=torch.int32)
    weights = torch.softmax(torch.randn(tokens, topk, device='cuda'), -1)
    expected = _reference(x, gu_ref, down_ref, ids, weights)
    if remap:
        permutation = torch.randperm(8, device='cuda')
        gate_up, down = gate_up[permutation], down[permutation]
        ids = permutation.argsort()[ids.long()].int()
    actual = fused_experts_q2_0(x, gate_up, down, weights, ids)
    torch.testing.assert_close(actual, expected, atol=.002, rtol=.02)
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_q2_vector_cuda_graph_replay_reads_updated_activations():
    from freetoken.moe.fused_q2_0 import fused_experts_q2_0

    torch.manual_seed(723)
    gate_up, gu_ref = _quantized_weights(8, 128, 128)
    down, down_ref = _quantized_weights(8, 128, 64)
    x = torch.randn(1, 128, device='cuda', dtype=torch.bfloat16)
    ids = torch.tensor([[5, 1, 5]], device='cuda', dtype=torch.int32)
    weights = torch.tensor([[.25, .5, .25]], device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused_experts_q2_0(x, gate_up, down, weights, ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused_experts_q2_0(x, gate_up, down, weights, ids)
    x.add_(.25)
    graph.replay()
    torch.testing.assert_close(actual, _reference(x, gu_ref, down_ref, ids, weights), atol=.002, rtol=.02)
