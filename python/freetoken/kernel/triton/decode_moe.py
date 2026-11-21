import triton
import triton.language as tl


@triton.jit
def fused_moe_decode_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    total_routes: int,
    N: int,
    K: int,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_ck,
    stride_cn,
    stride_tw_m,
    stride_tw_k,
    stride_tid_m,
    stride_tid_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TOP_K: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
):
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_row = route_id if A_ROW_IS_ROUTE else token_id

    expert_id = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k)
    a_ptrs = a_ptr + a_row * stride_am + offs_k * stride_ak
    b_ptrs = (
        b_ptr
        + expert_id.to(tl.int64) * stride_be
        + offs_n[None, :] * stride_bn
        + offs_k[:, None] * stride_bk
    )

    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if even_Ks:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        else:
            k_mask = offs_k < K - k_start * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=k_mask, other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.sum(a[:, None] * b.to(a.dtype), axis=0)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
        accumulator = accumulator * weight

    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(
        c_ptrs,
        accumulator.to(compute_type),
        mask=(route_id < total_routes) & (offs_n < N),
    )
