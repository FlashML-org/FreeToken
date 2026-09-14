"""Block-FP8 routed-expert MoE dispatch (offload `_expert_gemm` "fp8_block" branch).

The experts live in the offload cache as block-fp8 (fp8-e4m3 weight + bf16 per-128x128
``weight_scale_inv``) -- half the resident/host bytes of bf16, so the cache holds ~2x more
experts. The grouped GEMMs read the routed experts' fp8 rows directly and dequantize inside
the K-loop (``kernel/triton/fp8_blockscale_moe``), so no gather/copy or separate bf16 dequant
of the experts is ever materialized. Decode shapes are static per captured batch size, so the
path stays CUDA-graph capturable. Same entry points are reused by the resident (non-offload)
``Fp8ResidentMoE`` -- ``topk_ids`` index expert rows in the resident case and cache slots in
the offload case; both index the bank tensors identically.
"""

from __future__ import annotations


def fused_experts_fp8_block(
    hidden_states, gate_up, gate_up_scale, down, down_scale,
    topk_weights, topk_ids, num_experts, activation="silu",
    apply_router_weight_on_input=False, act_alpha=1.0, act_limit=float("inf"),
):
    """Prefill: W8A8 fused grouped GEMM over the materialized-layer banks
    (``[num_experts, ...]``, position == expert id)."""
    assert not apply_router_weight_on_input
    from freetoken.kernel.triton.fp8_blockscale_moe import fused_experts_fp8_blockscale

    return fused_experts_fp8_blockscale(
        hidden_states, gate_up, gate_up_scale, down, down_scale,
        topk_weights, topk_ids, num_experts, activation, act_alpha, act_limit,
    )


def fused_experts_decode_fp8_block(
    hidden_states, gate_up, gate_up_scale, down, down_scale,
    topk_weights, topk_ids, activation="silu", apply_router_weight_on_input=False,
    act_alpha=1.0, act_limit=float("inf"),
):
    """Decode: W8A16 fused inline-dequant grouped GEMV -- reads the routed experts' fp8 rows
    directly (``topk_ids`` index the banks) and dequantizes in the K-loop. CUDA-graph safe."""
    assert not apply_router_weight_on_input
    from freetoken.kernel.triton.fp8_blockscale_moe import fused_experts_decode_fp8_blockscale

    return fused_experts_decode_fp8_blockscale(
        hidden_states, gate_up, gate_up_scale, down, down_scale,
        topk_weights, topk_ids, activation, act_alpha, act_limit,
    )


__all__ = ["fused_experts_fp8_block", "fused_experts_decode_fp8_block"]
