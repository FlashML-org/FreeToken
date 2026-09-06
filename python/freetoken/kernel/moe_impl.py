from typing import Any, Dict, Tuple

import torch


def fused_moe_kernel_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: torch.dtype,
) -> None:
    import triton
    import triton.language as tl

    from .triton.fused_moe import fused_moe_kernel

    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    padded_size = 0
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    K = B.shape[2] - padded_size
    if K % config["BLOCK_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False
    dtype = tl.bfloat16 if compute_type == torch.bfloat16 else tl.float16
    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2] - padded_size,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # type: ignore
        top_k=top_k,  # type: ignore
        compute_type=dtype,  # type: ignore
        even_Ks=even_Ks,  # type: ignore
        **config,
    )


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import triton

    from .triton.fused_moe import (
        moe_align_block_size_init_kernel,
        moe_align_block_size_stage1_kernel,
        moe_align_block_size_stage2_kernel,
        moe_align_block_size_stage3_kernel,
        moe_align_block_size_stage4_kernel,
    )

    assert topk_ids.is_contiguous()
    assert topk_ids.dtype == torch.int32
    assert topk_ids.dim() == 2

    # Keep the same extra sentinel expert slot expected by the existing
    # sgl_kernel wrapper call site.
    effective_num_experts = num_experts + 1
    numel = topk_ids.numel()
    if numel < effective_num_experts:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + effective_num_experts * (block_size - 1)

    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (triton.cdiv(max_num_tokens_padded, block_size),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    tokens_cnts = torch.empty(
        (effective_num_experts + 1, effective_num_experts),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    cumsum = torch.empty(
        (effective_num_experts + 1,),
        dtype=torch.int32,
        device=topk_ids.device,
    )

    init_block = 1024
    init_numel = max(
        sorted_token_ids.numel(),
        expert_ids.numel(),
        tokens_cnts.numel(),
        cumsum.numel(),
    )
    moe_align_block_size_init_kernel[(triton.cdiv(init_numel, init_block),)](
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        sorted_token_ids.numel(),
        expert_ids.numel(),
        tokens_cnts.numel(),
        cumsum.numel(),
        numel,
        BLOCK_SIZE=init_block,
    )

    grid = (effective_num_experts,)
    tokens_per_thread = triton.cdiv(numel, effective_num_experts)
    moe_align_block_size_stage1_kernel[grid](
        topk_ids,
        tokens_cnts,
        effective_num_experts,
        numel,
        tokens_per_thread,
    )
    moe_align_block_size_stage2_kernel[grid](
        tokens_cnts,
        effective_num_experts,
    )
    moe_align_block_size_stage3_kernel[(1,)](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        effective_num_experts,
        block_size,
    )
    moe_align_block_size_stage4_kernel[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        effective_num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_pad


def moe_sum_reduce_triton(input: torch.Tensor, output: torch.Tensor) -> None:
    import triton

    from .triton.fused_moe import moe_sum_reduce_kernel

    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    block_m = 1 if token_num <= 16 else 2
    block_dim = min(triton.next_power_of_2(hidden_dim), 1024)
    grid = (triton.cdiv(token_num, block_m), triton.cdiv(hidden_dim, block_dim))
    moe_sum_reduce_kernel[grid](
        input,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        output,
        output.stride(0),
        output.stride(1),
        token_num,
        topk_num,
        hidden_dim,
        BLOCK_M=block_m,
        BLOCK_DIM=block_dim,
        NUM_STAGE=4,
    )


def moe_weighted_sum_reduce_triton(
    input: torch.Tensor, weights: torch.Tensor, output: torch.Tensor
) -> None:
    """Apply route weights and reduce ``[tokens, topk, hidden]`` in one Triton launch."""
    import triton

    from .triton.fused_moe import moe_weighted_sum_reduce_kernel

    assert input.is_contiguous()
    assert weights.is_contiguous()
    assert output.is_contiguous()
    token_num, topk_num, hidden_dim = input.shape
    assert weights.shape == (token_num, topk_num)
    assert output.shape == (token_num, hidden_dim)

    block_m = 1 if token_num <= 16 else 2
    block_dim = min(triton.next_power_of_2(hidden_dim), 1024)
    grid = (triton.cdiv(token_num, block_m), triton.cdiv(hidden_dim, block_dim))
    moe_weighted_sum_reduce_kernel[grid](
        input,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        weights,
        weights.stride(0),
        weights.stride(1),
        output,
        output.stride(0),
        output.stride(1),
        token_num,
        topk_num,
        hidden_dim,
        BLOCK_M=block_m,
        BLOCK_DIM=block_dim,
        NUM_STAGE=4,
    )


def mxfp4_fused_moe_kernel_t_triton(
    A: torch.Tensor,
    B_blocks_t: torch.Tensor,   # transposed layout [E, K//2, N] (uint8, N innermost)
    B_scales_t: torch.Tensor,   # transposed layout [E, K//32, N] (uint8)
    bias: torch.Tensor,         # [E, N]
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: torch.dtype,
) -> None:
    """Grouped/sorted MXFP4 fused-MoE GEMM over the TRANSPOSED weight layout
    [E, K//2, N] / [E, K//32, N]. Reuses the stride-parameterized
    mxfp4_fused_moe_kernel; only the shape/stride mapping differs from the HF
    driver. N is innermost (stride_bn=1), which keeps the N-axis tile loads
    coalesced in the prefill GEMM."""
    import triton
    import triton.language as tl

    from .triton.mxfp4_moe import mxfp4_fused_moe_kernel

    assert A.is_contiguous()
    assert B_blocks_t.is_contiguous()
    assert B_scales_t.is_contiguous()
    assert bias.is_contiguous()
    assert C.is_contiguous()
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert B_blocks_t.dtype == torch.uint8
    assert B_scales_t.dtype == torch.uint8

    E, K_half, N = B_blocks_t.shape
    K = K_half * 2
    assert B_scales_t.shape == (E, K // 32, N)
    assert A.shape[1] == K
    assert bias.shape == (E, N)
    assert C.shape[-1] == N
    assert config["BLOCK_SIZE_K"] % 32 == 0

    if compute_type == torch.float32:
        dtype = tl.float32
    elif compute_type == torch.bfloat16:
        dtype = tl.bfloat16
    else:
        dtype = tl.float16

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    mxfp4_fused_moe_kernel[grid](
        A,
        B_blocks_t,
        B_scales_t,
        bias,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B_blocks_t.stride(0),   # stride_be
        B_blocks_t.stride(2),   # stride_bn  (N axis, contiguous)
        B_blocks_t.stride(1),   # stride_bk2 (K-byte axis)
        B_scales_t.stride(0),   # stride_se
        B_scales_t.stride(2),   # stride_sn  (N axis, contiguous)
        B_scales_t.stride(1),   # stride_sk32 (scale-K axis)
        bias.stride(0),
        bias.stride(1),
        C.stride(-2),
        C.stride(-1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # type: ignore
        top_k=top_k,  # type: ignore
        compute_type=dtype,  # type: ignore
        **config,
    )


def gpt_oss_swiglu_triton(
    gate_up: torch.Tensor,
    out: torch.Tensor,
    *,
    alpha: float,
    limit: float,
    compute_type: torch.dtype,
) -> None:
    import triton
    import triton.language as tl

    from .triton.mxfp4_moe import gpt_oss_swiglu_kernel

    assert gate_up.is_contiguous()
    assert out.is_contiguous()
    assert gate_up.shape[0] == out.shape[0]
    assert gate_up.shape[1] == 2 * out.shape[1]

    dtype = tl.bfloat16 if compute_type == torch.bfloat16 else tl.float16
    block_size = 128
    grid = (out.shape[0], triton.cdiv(out.shape[1], block_size))
    gpt_oss_swiglu_kernel[grid](
        gate_up,
        out,
        out.shape[0],
        out.shape[1],
        gate_up.stride(0),
        gate_up.stride(1),
        out.stride(0),
        out.stride(1),
        alpha,
        limit,
        BLOCK_SIZE=block_size,  # type: ignore
        compute_type=dtype,  # type: ignore
        num_warps=4,  # type: ignore
    )


def fused_moe_decode_kernel_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
    config: Dict[str, Any],
    compute_type: torch.dtype,
) -> None:
    import triton
    import triton.language as tl

    from .triton.decode_moe import fused_moe_decode_kernel

    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert topk_weights.stride(1) == 1
    assert topk_ids.stride(1) == 1
    assert topk_weights.shape == topk_ids.shape

    M, top_k = topk_ids.shape
    total_routes = M * top_k
    assert C.shape[0] == M and C.shape[1] == top_k
    if a_row_is_route:
        assert A.shape[0] == total_routes
    else:
        assert A.shape[0] == M

    K = B.shape[2]
    if compute_type == torch.float32:
        dtype = tl.float32
    elif compute_type == torch.bfloat16:
        dtype = tl.bfloat16
    else:
        dtype = tl.float16
    even_Ks = K % config["BLOCK_SIZE_K"] == 0
    grid = (total_routes, triton.cdiv(B.shape[1], config["BLOCK_SIZE_N"]))

    fused_moe_decode_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        topk_ids,
        total_routes,
        B.shape[1],
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],  # type: ignore
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],  # type: ignore
        TOP_K=top_k,  # type: ignore
        A_ROW_IS_ROUTE=a_row_is_route,  # type: ignore
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # type: ignore
        compute_type=dtype,  # type: ignore
        even_Ks=even_Ks,  # type: ignore
        num_warps=config.get("num_warps", 4),  # type: ignore
    )


def gpt_oss_fused_routing(
    logits: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused softmax-over-top-k router for GPT-OSS (norm_topk_prob=True).

    Returns (topk_weights[float32], topk_ids[int32]). Selecting top-k by logit and
    softmaxing over the selected k is equivalent to softmax-over-all + renormalize.
    """
    import triton

    from .triton.mxfp4_moe import gpt_oss_routing_kernel

    assert logits.is_cuda and logits.dim() == 2
    tokens, num_experts = logits.shape
    assert top_k <= num_experts
    assert num_experts <= 1024, "routing kernel BLOCK_E (next pow2 of E) must be <= 1024"
    logits = logits.float().contiguous()
    topk_weights = torch.empty((tokens, top_k), dtype=torch.float32, device=logits.device)
    topk_ids = torch.empty((tokens, top_k), dtype=torch.int32, device=logits.device)
    if tokens == 0:
        return topk_weights, topk_ids
    # tl.sort requires at least 2 elements; pad to 2 when there is only 1 expert
    block_e = max(2, triton.next_power_of_2(num_experts))
    gpt_oss_routing_kernel[(tokens,)](
        logits,
        topk_weights,
        topk_ids,
        logits.stride(0),
        E=num_experts,  # type: ignore
        K=top_k,  # type: ignore
        BLOCK_E=block_e,  # type: ignore
        num_warps=1,  # type: ignore
    )
    return topk_weights, topk_ids


_FP4_LUT_FLOATS = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]
_fp4_lut_cache: Dict[Any, "torch.Tensor"] = {}


def get_fp4_lut(device: "torch.device") -> torch.Tensor:
    cached = _fp4_lut_cache.get(device)
    if cached is None:
        cached = torch.tensor(_FP4_LUT_FLOATS, dtype=torch.float32, device=device)
        _fp4_lut_cache[device] = cached
    return cached


def mxfp4_splitk_gemv_triton(
    x: torch.Tensor,
    w_blocks_t: torch.Tensor,
    w_scales_t: torch.Tensor,
    bias: torch.Tensor | None,
    expert_ids: torch.Tensor,
    N: int,
    K: int,
    stride_xe: int,
    num_splits: int,
    *,
    out: torch.Tensor | None = None,
    partial: torch.Tensor | None = None,
    expert_wts: torch.Tensor | None = None,
    block_n: int = 64,
    num_warps: int = 1,
) -> torch.Tensor:
    """Split-K expert GEMV over MXFP4 weights stored transposed as [E, K//2, N]
    (blocks) and [E, K//32, N] (scales). ``x`` is [routes, K] (or broadcast via
    stride_xe=0); ``expert_ids`` is [routes]. Returns [routes, N] bf16.
    """
    import triton

    from .triton.mxfp4_moe import mxfp4_splitk_gemv_kernel, mxfp4_splitk_reduce_kernel

    routes = expert_ids.shape[0]
    if partial is None:
        partial = torch.empty(routes * num_splits, N, dtype=torch.float32, device=x.device)
    if out is None:
        out = torch.empty(routes, N, dtype=torch.bfloat16, device=x.device)
    has_bias = bias is not None and bias.numel() > 0
    bias_arg = bias if has_bias else x
    bias_stride = bias.stride(0) if has_bias else 0
    k_groups = K // 32
    kgps = triton.cdiv(k_groups, num_splits)
    lut = get_fp4_lut(x.device)

    grid = (triton.cdiv(N, block_n), routes * num_splits)
    mxfp4_splitk_gemv_kernel[grid](
        x, w_blocks_t, w_scales_t, bias_arg, expert_ids, partial, lut, N, K,
        stride_xe, w_blocks_t.stride(0), w_blocks_t.stride(1),
        w_scales_t.stride(0), w_scales_t.stride(1), bias_stride, N,
        HAS_BIAS=has_bias, BLOCK_N=block_n,  # type: ignore
        NUM_K_SPLITS=num_splits, K_GROUPS_PER_SPLIT=kgps,  # type: ignore
        num_warps=num_warps,
    )
    has_wts = expert_wts is not None
    rgrid = (triton.cdiv(N, block_n), routes)
    mxfp4_splitk_reduce_kernel[rgrid](
        partial, out, N, expert_wts if has_wts else x,
        HAS_EXPERT_WTS=has_wts, NUM_K_SPLITS=num_splits, BLOCK_N=block_n,  # type: ignore
        num_warps=num_warps,
    )
    return out
