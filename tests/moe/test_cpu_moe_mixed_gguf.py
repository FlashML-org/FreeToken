from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_mixed_gguf_cpu_executor_builds_zero_copy_views_and_dispatches(monkeypatch):
    import freetoken.moe.cpu_executor as mod
    from freetoken.models.gguf.dequant import GGML_BF16, GGML_Q4_0, row_bytes

    created = []

    class FakeExecutor:
        def __init__(self, cache, **kwargs):
            self.cache = cache
            self.kwargs = kwargs
            created.append(self)

        def decode(self, layer_id, *args):
            return self.cache.quant_format, layer_id

        def decode_submit(self, layer_id, *args):
            return self.cache.quant_format, layer_id

        def decode_sync(self, pending):
            return pending

        def raise_if_unhealthy(self):
            return None

    monkeypatch.setattr(mod, "CpuMoeExecutor", FakeExecutor)
    experts, hidden, intermediate = 2, 64, 32
    q4_gu = torch.empty(
        experts * 2 * intermediate * row_bytes(hidden, GGML_Q4_0),
        dtype=torch.uint8,
    ).view(experts, -1)
    q4_dn = torch.empty(
        experts * hidden * row_bytes(intermediate, GGML_Q4_0),
        dtype=torch.uint8,
    ).view(experts, -1)
    bf_gu = torch.empty(
        experts * 2 * intermediate * hidden * 2, dtype=torch.uint8
    ).view(experts, -1)
    bf_dn = torch.empty(
        experts * hidden * intermediate * 2, dtype=torch.uint8
    ).view(experts, -1)
    cache = SimpleNamespace(
        num_layers=2,
        num_experts=experts,
        gguf_expert_types=(
            (GGML_Q4_0, GGML_Q4_0),
            (GGML_BF16, GGML_BF16),
        ),
        expert_hidden_size=hidden,
        expert_intermediate_size=intermediate,
        bank_sources={"gate_up": [q4_gu, bf_gu], "down": [q4_dn, bf_dn]},
    )

    executor = mod.MixedGgufCpuMoeExecutor(
        cache,
        top_k=2,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=1,
        max_tokens=1,
        device=torch.device("cpu"),
    )
    assert {item.cache.quant_format for item in created} == {"q4_0", "bf16"}
    q4 = next(item.cache for item in created if item.cache.quant_format == "q4_0")
    bf16 = next(item.cache for item in created if item.cache.quant_format == "bf16")
    assert q4.bank_sources["gate_up"][0].shape == (
        experts,
        2 * intermediate,
        row_bytes(hidden, GGML_Q4_0),
    )
    assert bf16.bank_sources["down"][1].shape == (experts, hidden, intermediate)
    assert q4.bank_sources["gate_up"][0].data_ptr() == q4_gu.data_ptr()
    assert bf16.bank_sources["down"][1].data_ptr() == bf_dn.data_ptr()
    assert executor.decode(0) == ("q4_0", 0)
    assert executor.decode(1) == ("bf16", 1)
    assert executor.decode_sync(executor.decode_submit(1)) == ("bf16", 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mixed_gguf_bf16_layer_matches_native_fused_experts():
    from freetoken.models.gguf.dequant import GGML_BF16
    from freetoken.moe.fused import fused_experts_impl
    from freetoken.moe.fused_gguf import fused_experts_gguf

    experts, hidden, intermediate, tokens, top_k = 4, 64, 32, 3, 2
    torch.manual_seed(8)
    gate_up = torch.randn(
        experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16
    )
    down = torch.randn(
        experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16
    )
    gate_up_bytes = gate_up.contiguous().view(torch.uint8).reshape(experts, -1)
    down_bytes = down.contiguous().view(torch.uint8).reshape(experts, -1)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [2, 3], [1, 2]], device="cuda", dtype=torch.int32)
    weights = torch.rand(tokens, top_k, device="cuda", dtype=torch.float32)

    original = x.clone()
    got = fused_experts_gguf(
        x,
        gate_up_bytes,
        down_bytes,
        weights,
        ids,
        "silu",
        gate_up_type=GGML_BF16,
        down_type=GGML_BF16,
        gate_up_rows=2 * intermediate,
        down_rows=hidden,
    )
    expected = fused_experts_impl(
        x.clone(), gate_up, down, weights, ids, "silu", False
    )
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
