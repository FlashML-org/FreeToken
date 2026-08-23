"""MXFP4 fused-MoE kernels: transposed split-K GEMV decode + grouped
``_t`` prefill over ``[E, K//2, N]``-layout blocks/scales (N innermost, shared
between the resident and offload paths).
"""

from __future__ import annotations

import torch

from freetoken.moe.fused import moe_align_block_size, try_get_optimal_moe_config

# Token-count threshold used by GptOssMxfp4TritonMoELayer.forward to dispatch: batches at
# or below this bound take the gather decode path (no sort), larger batches take the
# grouped/sorted prefill kernel. GPT-OSS decode targets small batches (max running
# requests), matching the gather path.
MXFP4_DECODE_MAX_TOKENS = 16


def gpt_oss_swiglu(
    gate_up: torch.Tensor, *, alpha: float, limit: float | None
) -> torch.Tensor:
    gate = gate_up[..., ::2]
    up = gate_up[..., 1::2]
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1.0)


def kimi_situ(
    gate_up: torch.Tensor, *, beta: float = 4.0, linear_beta: float = 25.0
) -> torch.Tensor:
    """Kimi-K3 SiTU over the interleaved MXFP4 expert output ``gate, up`` pairs."""
    if beta <= 0 or linear_beta <= 0:
        raise ValueError("SiTU beta values must be positive")
    gate = gate_up[..., ::2]
    up = gate_up[..., 1::2]
    return (
        beta
        * torch.tanh(gate / beta)
        * torch.sigmoid(gate)
        * linear_beta
        * torch.tanh(up / linear_beta)
    )


def _activate_interleaved(
    gate_up: torch.Tensor,
    *,
    activation: str,
    act_alpha: float,
    act_limit: float | None,
) -> torch.Tensor:
    if activation in {"gpt_oss_swiglu", "swigluoai"}:
        return gpt_oss_swiglu(gate_up, alpha=act_alpha, limit=act_limit)
    if activation == "situ":
        if act_limit is None:
            raise ValueError("SiTU requires linear_beta")
        return kimi_situ(gate_up, beta=act_alpha, linear_beta=act_limit)
    raise NotImplementedError(f"MXFP4 expert activation {activation!r} is unsupported")


def dequant_mxfp4_blocks(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if blocks.dtype != torch.uint8:
        raise TypeError("MXFP4 blocks must be uint8")
    if scales.dtype != torch.uint8:
        raise TypeError("MXFP4 scales must be uint8")
    if blocks.shape[-1] != 16:
        raise ValueError("MXFP4 block pack dimension must be 16 bytes")
    if blocks.shape[:-1] != scales.shape:
        raise ValueError("MXFP4 blocks/scales shape mismatch")

    nibbles = torch.stack((blocks & 0x0F, blocks >> 4), dim=-1).reshape(
        *blocks.shape[:-1],
        32,
    )
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=blocks.device,
        dtype=torch.float32,
    )
    unpacked = values[nibbles.long()]
    # E8M0 reserves 0xff.  Match the split-K and CPU production kernels by
    # clamping that byte to the largest finite exponent instead of producing a
    # path-dependent infinity in the Python/grouped reference.
    scale = torch.exp2(scales.float().clamp_(max=254) - 127).unsqueeze(-1)
    dequantized = unpacked * scale
    return dequantized.reshape(*blocks.shape[:-2], blocks.shape[-2] * 32).to(out_dtype)


def run_mxfp4_prefill_experts_t(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_blocks_t: torch.Tensor,  # [E, H//2, 2*I]
    gate_up_scales_t: torch.Tensor,  # [E, H//32, 2*I]
    gate_up_bias: torch.Tensor,  # [E, 2*I]
    down_blocks_t: torch.Tensor,  # [E, I//2, H]
    down_scales_t: torch.Tensor,  # [E, I//32, H]
    down_bias: torch.Tensor,  # [E, H]
    *,
    top_k: int,
    hidden_act_alpha: float,
    swiglu_limit: float | None,
    activation: str = "gpt_oss_swiglu",
) -> torch.Tensor:
    """Prefill experts using the transposed weight layout shared with split-K decode
    ([E, K//2, N] blocks, [E, K//32, N] scales, N innermost). Uses
    mxfp4_fused_moe_kernel_t_triton for the grouped GEMM."""
    from freetoken.kernel import (
        moe_sum_reduce_triton,
        mxfp4_fused_moe_kernel_t_triton,
    )

    if not hidden_states.is_cuda:
        raise RuntimeError("GPT-OSS MXFP4 MoE requires the Triton CUDA kernel")
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()
    if not topk_ids.is_contiguous():
        topk_ids = topk_ids.contiguous()

    num_tokens = hidden_states.shape[0]
    num_weight_experts = gate_up_blocks_t.shape[0]
    local_intermediate_size = gate_up_blocks_t.shape[2] // 2  # N = 2*I on the last axis
    hidden_size = hidden_states.shape[-1]
    config = try_get_optimal_moe_config(
        (num_weight_experts, 2 * local_intermediate_size, hidden_size),
        (num_weight_experts, hidden_size, local_intermediate_size),
        top_k,
        num_tokens,
    )
    if config["BLOCK_SIZE_K"] % 32 != 0:
        config = {**config, "BLOCK_SIZE_K": 64}

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        config["BLOCK_SIZE_M"],
        num_weight_experts,
    )

    gate_up = torch.empty(
        (num_tokens, top_k, 2 * local_intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    mxfp4_fused_moe_kernel_t_triton(
        hidden_states,
        gate_up_blocks_t,
        gate_up_scales_t,
        gate_up_bias,
        gate_up,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        top_k,
        config,
        compute_type=hidden_states.dtype,
    )

    activated = _activate_interleaved(
        gate_up.view(num_tokens * top_k, 2 * local_intermediate_size),
        activation=activation,
        act_alpha=hidden_act_alpha,
        act_limit=swiglu_limit,
    ).contiguous()

    down = torch.empty(
        (num_tokens, top_k, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    mxfp4_fused_moe_kernel_t_triton(
        activated,
        down_blocks_t,
        down_scales_t,
        down_bias,
        down,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        1,
        config,
        compute_type=hidden_states.dtype,
    )

    output = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(down, output)
    return output


def _transpose_mxfp4_for_decode(
    blocks: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF MXFP4 [E, N, K//32, 16] blocks / [E, N, K//32] scales -> split-K GEMV
    layout: blocks_t [E, K//2, N], scales_t [E, K//32, N] (N innermost/contiguous)."""
    num_experts, n, k_blocks, _ = blocks.shape
    blocks_t = (
        blocks.reshape(num_experts, n, k_blocks * 16).permute(0, 2, 1).contiguous()
    )
    scales_t = scales.permute(0, 2, 1).contiguous()
    return blocks_t, scales_t


def _decode_split_count(routes: int, k_groups: int, target_programs: int) -> int:
    """Pick the split-K count so the GEMV launches ~target_programs program-rows.
    More routes => fewer K-splits needed for occupancy. Matches the tuned
    decode values at M=1 (gate_up ~45, down ~18)."""
    return max(1, min(k_groups, -(-target_programs // max(routes, 1))))


def run_mxfp4_splitk_decode_experts(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_blocks_t: torch.Tensor,
    gate_up_scales_t: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_blocks_t: torch.Tensor,
    down_scales_t: torch.Tensor,
    down_bias: torch.Tensor,
    *,
    top_k: int,
    hidden_act_alpha: float,
    swiglu_limit: float | None,
    activation: str = "gpt_oss_swiglu",
) -> torch.Tensor:
    """Split-K GEMV MoE decode over TRANSPOSED MXFP4 weights:
    gate_up_blocks_t [E, H//2, 2I], gate_up_scales_t [E, H//32, 2I], bias [E, 2I];
    down_blocks_t [E, I//2, H], down_scales_t [E, I//32, H], bias [E, H].
    Targets small token counts (M <= MXFP4_DECODE_MAX_TOKENS)."""
    from freetoken.kernel import mxfp4_splitk_gemv_triton

    if not hidden_states.is_cuda:
        raise RuntimeError("GPT-OSS MXFP4 MoE requires the Triton CUDA kernel")
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()

    M = hidden_states.shape[0]
    H = hidden_states.shape[-1]
    two_I = gate_up_blocks_t.shape[2]
    local_intermediate_size = two_I // 2
    routes = M * top_k
    device = hidden_states.device
    compute_type = hidden_states.dtype

    route_experts = topk_ids.reshape(-1).to(torch.int64)
    route_weights = topk_weights.reshape(-1).contiguous()
    if M == 1:
        # all routes share the single token: broadcast it (stride_xe=0), no gather.
        routed_x = hidden_states
        gu_stride_xe = 0
    else:
        route_tokens = torch.arange(M, device=device).repeat_interleave(top_k)
        routed_x = hidden_states.index_select(0, route_tokens).contiguous()
        gu_stride_xe = routed_x.stride(0)

    gu_splits = _decode_split_count(routes, H // 32, target_programs=180)
    gate_up_out = mxfp4_splitk_gemv_triton(
        routed_x,
        gate_up_blocks_t,
        gate_up_scales_t,
        gate_up_bias,
        route_experts,
        N=two_I,
        K=H,
        stride_xe=gu_stride_xe,
        num_splits=gu_splits,
    )

    hidden_out = (
        _activate_interleaved(
            gate_up_out,
            activation=activation,
            act_alpha=hidden_act_alpha,
            act_limit=swiglu_limit,
        )
        .to(compute_type)
        .contiguous()
    )

    dp_splits = _decode_split_count(
        routes, local_intermediate_size // 32, target_programs=72
    )
    down_out = mxfp4_splitk_gemv_triton(
        hidden_out,
        down_blocks_t,
        down_scales_t,
        down_bias,
        route_experts,
        N=H,
        K=local_intermediate_size,
        stride_xe=hidden_out.stride(0),
        num_splits=dp_splits,
        expert_wts=route_weights,
    )

    return down_out.view(M, top_k, H).sum(dim=1).to(compute_type)


__all__ = [
    "MXFP4_DECODE_MAX_TOKENS",
    "dequant_mxfp4_blocks",
    "gpt_oss_swiglu",
    "kimi_situ",
    "run_mxfp4_prefill_experts_t",
    "run_mxfp4_splitk_decode_experts",
]
