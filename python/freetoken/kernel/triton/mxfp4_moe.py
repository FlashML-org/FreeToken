import triton
import triton.language as tl

# ── Split-K expert GEMV (transposed weight layout [E, K_bytes, N]) ──────────
# The MXFP4 MoE decode/prefill at small token counts is a memory-bound GEMV per
# (route, expert); splitting the K
# reduction across NUM_K_SPLITS programs gives the occupancy needed at M=1, and
# storing the weights with N innermost (stride 1) makes the per-program load
# coalesced. fp4 values come from a 16-entry LUT; the e8m0 scale is built by
# placing the exponent into the float32 exponent field (no SFU exp2).


@triton.jit
def _fp4_table_lut(nibble, lut_ptr):
    return tl.load(lut_ptr + nibble)


@triton.jit
def _e8m0_scale(s_val):
    exponent = tl.minimum(tl.maximum(s_val.to(tl.int32), 0), 254)
    return (exponent << 23).to(tl.float32, bitcast=True)


@triton.jit
def mxfp4_splitk_gemv_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    bias_ptr,
    expert_ids_ptr,
    out_ptr,
    lut_ptr,
    N,
    K: tl.constexpr,
    stride_xe,
    stride_we,
    stride_wk,
    stride_se,
    stride_sk,
    stride_be,
    stride_oe,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_K_SPLITS: tl.constexpr,
    K_GROUPS_PER_SPLIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_ek = tl.program_id(1)
    pid_e = pid_ek // NUM_K_SPLITS
    pid_k = pid_ek % NUM_K_SPLITS

    expert_id = tl.load(expert_ids_ptr + pid_e)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    k_groups = K // 32
    kg0 = pid_k * K_GROUPS_PER_SPLIT
    kg1 = tl.minimum(kg0 + K_GROUPS_PER_SPLIT, k_groups)

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    w_base = expert_id * stride_we
    s_base = expert_id * stride_se
    x_base = pid_e * stride_xe

    for kg in range(kg0, kg1):
        s_val = tl.load(s_ptr + s_base + kg * stride_sk + offs_n, mask=mask_n, other=0)
        scale_f = _e8m0_scale(s_val)
        for kk in tl.static_range(16):
            k_packed = kg * 16 + kk
            w_byte = tl.load(
                w_ptr + w_base + k_packed * stride_wk + offs_n, mask=mask_n, other=0
            ).to(tl.int32)
            lo = w_byte & 0x0F
            hi = (w_byte >> 4) & 0x0F
            x_lo = tl.load(x_ptr + x_base + kg * 32 + kk * 2).to(tl.float32)
            x_hi = tl.load(x_ptr + x_base + kg * 32 + kk * 2 + 1).to(tl.float32)
            acc += _fp4_table_lut(lo, lut_ptr) * scale_f * x_lo
            acc += _fp4_table_lut(hi, lut_ptr) * scale_f * x_hi

    if HAS_BIAS and pid_k == 0:
        acc += tl.load(
            bias_ptr + expert_id * stride_be + offs_n, mask=mask_n, other=0.0
        ).to(tl.float32)

    out_row = pid_e * NUM_K_SPLITS + pid_k
    tl.store(out_ptr + out_row * stride_oe + offs_n, acc, mask=mask_n)


@triton.jit
def mxfp4_splitk_reduce_kernel(
    partial_ptr,
    out_ptr,
    N,
    expert_wts_ptr,
    HAS_EXPERT_WTS: tl.constexpr,
    NUM_K_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_e = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    ewt = tl.load(expert_wts_ptr + pid_e) if HAS_EXPERT_WTS else 1.0
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k in range(NUM_K_SPLITS):
        acc += tl.load(
            partial_ptr + (pid_e * NUM_K_SPLITS + k) * N + offs_n,
            mask=mask_n,
            other=0.0,
        )
    if HAS_EXPERT_WTS:
        acc *= ewt
    tl.store(out_ptr + pid_e * N + offs_n, acc.to(tl.bfloat16), mask=mask_n)


@triton.jit
def _dequant_fp4_lut(nibble):
    sign_bit = (nibble >> 3) & 1
    exp_bits = (nibble >> 1) & 3
    man_bit = nibble & 1

    is_subnormal = exp_bits == 0
    mantissa = 1.0 + man_bit.to(tl.float32) * 0.5
    exponent = tl.exp2((exp_bits - 1).to(tl.float32))
    value = tl.where(is_subnormal, man_bit.to(tl.float32) * 0.5, mantissa * exponent)
    return tl.where(sign_bit != 0, -value, value)


@triton.jit
def mxfp4_fused_moe_kernel(
    a_ptr,
    b_blocks_ptr,
    b_scales_ptr,
    bias_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk2,
    stride_se,
    stride_sn,
    stride_sk32,
    stride_bias_e,
    stride_bias_n,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_route_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_route = tl.load(sorted_token_ids_ptr + offs_route_id).to(tl.int64)
    route_mask = offs_route < num_valid_tokens

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    route_rows = offs_route // top_k

    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    a_ptrs = a_ptr + route_rows[:, None] * stride_am + offs_k[None, :] * stride_ak

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_start + offs_k
        k_byte_offsets = k_offsets // 2
        scale_offsets = k_offsets // 32

        a = tl.load(
            a_ptrs,
            mask=route_mask[:, None] & (k_offsets[None, :] < K),
            other=0.0,
        )

        packed = tl.load(
            b_blocks_ptr
            + expert_id * stride_be
            + offs_n[None, :] * stride_bn
            + k_byte_offsets[:, None] * stride_bk2,
            mask=(offs_n[None, :] < N) & (k_offsets[:, None] < K),
            other=0,
        ).to(tl.int32)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        nibble = tl.where((k_offsets[:, None] & 1) == 0, low, high)

        scale_u8 = tl.load(
            b_scales_ptr
            + expert_id * stride_se
            + offs_n[None, :] * stride_sn
            + scale_offsets[:, None] * stride_sk32,
            mask=(offs_n[None, :] < N) & (k_offsets[:, None] < K),
            other=127,
        )
        # 0xff is reserved by E8M0. Keep this grouped path bit-contract aligned
        # with split-K and the CPU executor, both of which clamp to 254.
        scale_u8 = tl.minimum(scale_u8.to(tl.int32), 254)
        scale = tl.exp2(scale_u8.to(tl.float32) - 127.0)
        b = (_dequant_fp4_lut(nibble) * scale).to(compute_type)

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak

    bias = tl.load(
        bias_ptr + expert_id * stride_bias_e + offs_n * stride_bias_n,
        mask=offs_n < N,
        other=0.0,
    ).to(tl.float32)
    accumulator += bias[None, :]

    if MUL_ROUTED_WEIGHT:
        route_weight = tl.load(
            topk_weights_ptr + offs_route, mask=route_mask, other=0.0
        )
        accumulator *= route_weight[:, None]

    c_ptrs = c_ptr + stride_cm * offs_route[:, None] + stride_cn * offs_n[None, :]
    c_mask = route_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(compute_type), mask=c_mask)


@triton.jit
def gpt_oss_swiglu_kernel(
    gate_up_ptr,
    out_ptr,
    total_routes,
    intermediate_size: tl.constexpr,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    alpha: tl.constexpr,
    limit: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    compute_type: tl.constexpr,
):
    route_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offs_i = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (route_id < total_routes) & (offs_i < intermediate_size)

    gate = tl.load(
        gate_up_ptr + route_id * stride_in_m + (2 * offs_i) * stride_in_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        gate_up_ptr + route_id * stride_in_m + (2 * offs_i + 1) * stride_in_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    gate = tl.minimum(gate, limit)
    up = tl.minimum(tl.maximum(up, -limit), limit)
    activated = gate * (1.0 / (1.0 + tl.exp(-(gate * alpha)))) * (up + 1.0)

    tl.store(
        out_ptr + route_id * stride_out_m + offs_i * stride_out_n,
        activated.to(compute_type),
        mask=mask,
    )


# REUSE-NOTE: trimmed local copy of gemma4's _gemma4_routing_kernel
# (kernel/triton/gemma4_fused.py) without per_expert_scale. See the REUSE-NOTE
# there; consolidate into one shared fused-routing kernel later.
@triton.jit
def gpt_oss_routing_kernel(
    logits_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    stride_logits_t,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    valid = offs_e < E

    logits = tl.load(
        logits_ptr + token_id * stride_logits_t + offs_e,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)

    min_i32 = -2147483648
    logit_bits = logits.to(tl.int32, bitcast=True)
    sign = logit_bits >> 31
    key = tl.where(sign == 0, logit_bits ^ -1, logit_bits ^ min_i32)
    key = tl.where(valid, key, 0x7FFFFFFF)
    packed = ((key.to(tl.int64) & 0x00000000FFFFFFFF) << 32) | offs_e.to(tl.int64)

    sorted_packed = tl.sort(packed, descending=False)
    all_keys = ((sorted_packed >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    all_ids = (sorted_packed & 0x00000000FFFFFFFF).to(tl.int32)

    sign_k = all_keys >> 31
    all_bits = tl.where(sign_k < 0, all_keys ^ -1, all_keys ^ min_i32)
    all_logits = all_bits.to(tl.float32, bitcast=True)

    top_mask = offs_e < K
    max_l = tl.max(tl.where(top_mask, all_logits, -float("inf")), axis=0)
    raw_exp = tl.where(top_mask, tl.exp(all_logits - max_l), 0.0)
    denom = tl.sum(raw_exp, axis=0)
    weights = raw_exp / tl.where(denom > 0.0, denom, 1.0)

    out_off = token_id * K + offs_e
    tl.store(topk_weights_ptr + out_off, weights, mask=top_mask)
    tl.store(topk_ids_ptr + out_off, all_ids, mask=top_mask)
