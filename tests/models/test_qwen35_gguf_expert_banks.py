"""Shape and registration checks for Qwen's mixed GGUF routed-expert banks."""

from freetoken.models.gguf.dequant import GGML_Q4_K, GGML_Q5_K, row_bytes
from freetoken.models.qwen3_5_moe.gguf import _expert_specs
from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT, _BANK_SCHEMAS


class _Config:
    """Small geometry carrier matching Qwen3.6-35B-A3B's routed MoE."""

    num_experts = 256
    hidden_size = 2048
    moe_intermediate_size = 512


def test_qwen_mixed_gguf_bank_shapes_preserve_each_tensor_encoding():
    """Gate/up and down rows keep their distinct Q4_K and Q5_K byte strides."""
    specs = _expert_specs(_Config())
    gate_shape, gate_dtype = specs["gate_up"]
    down_shape, down_dtype = specs["down"]

    assert gate_shape == (256, 1024, row_bytes(2048, GGML_Q4_K))
    assert down_shape == (256, 2048, row_bytes(512, GGML_Q5_K))
    assert str(gate_dtype) == "torch.uint8"
    assert str(down_dtype) == "torch.uint8"


def test_qwen_mixed_gguf_bank_budget_matches_the_two_exact_row_layouts():
    """Cache planning counts Q4_K gate/up bytes and Q5_K down bytes separately."""
    hidden, intermediate = 2048, 512
    expected = 2 * intermediate * row_bytes(hidden, GGML_Q4_K) + hidden * row_bytes(
        intermediate, GGML_Q5_K
    )
    assert _BANK_SCHEMAS["q4_k_q5_k"] == ("gate_up", "down")
    assert _BANK_BYTES_PER_EXPERT["q4_k_q5_k"](hidden, intermediate) == expected
