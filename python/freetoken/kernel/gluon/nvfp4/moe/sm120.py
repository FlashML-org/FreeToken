"""NVFP4 W4A16 MoE prefill GEMM for sm_120: a loader warp (TMA weights, cp.async scales and tokens), decode warps and mma.sync warps.

Same contract as the Triton prefill GEMM; :mod:`freetoken.kernel.nvfp4_moe` picks it for BLOCK_SIZE_M 64 and 128 and sets its constexprs.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from ..utils import e2m1x8_bf16x2, scale_bf16x2


@gluon.jit
def _tile(t, num_pid_m, num_pid_n, GROUP_SIZE_M: gl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    first_pid_m = (t // num_pid_in_group) * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((t % num_pid_in_group) % group_size_m)
    pid_n = (t % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@gluon.jit
def _load_partition(a_ptr, ids_ptr, eid_ptr, w_desc, scale_ptr, bufs, bars, N, nk, num_tiles, num_pid_m, num_valid,
                    stride_am, ks4, n_rows, top_k: gl.constexpr, K_TAIL: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
                    PERSISTENT: gl.constexpr):
    BN: gl.constexpr = bufs[1].type.shape[1]
    SPC: gl.constexpr = 256 // (bufs[1].type.shape[2] * 2)
    if PERSISTENT:
        # it / jt count chunks and scale blocks across tiles, so the rings keep their phase from one tile to the next
        it = 0
        jt = 0
        for t in range(gl.program_id(0), num_tiles, gl.num_programs(0)):
            pid_m, pid_n = _tile(t, num_pid_m, gl.cdiv(N, BN), GROUP_SIZE_M)
            _load_tile(a_ptr, ids_ptr, eid_ptr, w_desc, scale_ptr, bufs, bars, pid_m, pid_n, N, nk, it, jt, num_valid,
                       stride_am, ks4, n_rows, top_k, K_TAIL)
            it += nk
            jt += gl.cdiv(nk, SPC)
    else:
        pid_m, pid_n = _tile(gl.program_id(0), num_pid_m, gl.cdiv(N, BN), GROUP_SIZE_M)
        _load_tile(a_ptr, ids_ptr, eid_ptr, w_desc, scale_ptr, bufs, bars, pid_m, pid_n, N, nk, 0, 0, num_valid,
                   stride_am, ks4, n_rows, top_k, K_TAIL)


@gluon.jit
def _load_tile(a_ptr, ids_ptr, eid_ptr, w_desc, scale_ptr, bufs, bars, pid_m, pid_n, N, nk, it, jt, num_valid,
               stride_am, ks4, n_rows, top_k: gl.constexpr, K_TAIL: gl.constexpr):
    x_bufs, w_bufs, s_bufs, d_bufs = bufs
    x_ready, x_empty, w_ready, w_empty, s_ready, s_empty, d_ready = bars
    KS: gl.constexpr = w_bufs.type.shape[2] // 8
    XS: gl.constexpr = x_bufs.type.shape[0]
    BM: gl.constexpr = x_bufs.type.shape[1] // KS
    BK: gl.constexpr = KS * 16
    WS: gl.constexpr = w_bufs.type.shape[0]
    KB: gl.constexpr = w_bufs.type.shape[2]
    SS: gl.constexpr = s_bufs.type.shape[0]
    BN: gl.constexpr = s_bufs.type.shape[1]
    SPC: gl.constexpr = 256 // BK

    # 16 bytes per cp.async; adjacent lanes read one token's contiguous K
    NWL: gl.constexpr = gl.num_warps()
    xl: gl.constexpr = gl.BlockedLayout([1, 1, 8], [KS, 16 // KS, 2], [1, NWL, 1], [2, 0, 1])
    offs_s = gl.arange(0, KS, layout=gl.SliceLayout(1, gl.SliceLayout(2, xl)))
    offs_c = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, xl)))
    cols = gl.expand_dims(gl.expand_dims(offs_s * 16, 1), 2) + gl.expand_dims(gl.expand_dims(offs_c, 0), 1)

    # scale rows are K / 16 bytes: 16-byte copies when K % 256 == 0, else 4-byte copies zeroed past the row end
    SV: gl.constexpr = 4 if K_TAIL else 16
    sll: gl.constexpr = gl.BlockedLayout([1, SV], [32 * SV // 16, 16 // SV], [NWL, 1], [1, 0])
    s_cols = gl.arange(0, 16, layout=gl.SliceLayout(0, sll))
    # both forms equal K // 16; the second tells the compiler the row is a multiple of 16 bytes
    row_len = ks4 * 4 if K_TAIL else (ks4 // 4) * 16

    w_row = gl.load(eid_ptr + pid_m) * N + pid_n * BN
    offs_m = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(0, gl.SliceLayout(2, xl)))
    tok = gl.load(ids_ptr + offs_m)
    tok_mask = gl.expand_dims(gl.expand_dims(tok < num_valid, 0), 2)
    rows = gl.expand_dims(gl.expand_dims((tok // top_k).to(gl.int64) * stride_am, 0), 2)
    x_ptrs = a_ptr + rows + cols
    s_rows = w_row + gl.arange(0, BN, layout=gl.SliceLayout(1, sll))
    s_ptrs = scale_ptr + s_rows.to(gl.int64)[:, None] * row_len + s_cols[None, :]
    s_row_ok = (s_rows < n_rows)[:, None]
    for kk in range(nk):
        if kk % SPC == 0:
            jl = kk // SPC
            j = jt + jl
            ss = j % SS
            mbarrier.wait(s_empty.index(ss), (j // SS & 1) ^ 1)
            async_copy.async_load(s_bufs.index(ss), s_ptrs + jl * 16, mask=s_row_ok & (jl * 16 + s_cols < row_len)[None, :])
            async_copy.mbarrier_arrive(s_ready.index(ss))
            mbarrier.arrive(s_ready.index(ss))
        i = it + kk
        ws = i % WS
        mbarrier.wait(w_empty.index(ws), (i // WS & 1) ^ 1)
        mbarrier.expect(w_ready.index(ws), w_desc.block_type.nbytes)
        tma.async_load(w_desc, [w_row, kk * KB], w_ready.index(ws), w_bufs.index(ws))
        xs = i % XS
        mbarrier.wait(x_empty.index(xs), (i // XS & 1) ^ 1)
        async_copy.async_load(x_bufs.index(xs).reshape([KS, BM, 16]), x_ptrs + kk * BK, mask=tok_mask)
        async_copy.mbarrier_arrive(x_ready.index(xs))
        mbarrier.arrive(x_ready.index(xs))


@gluon.jit
def _decode_partition(a_ptr, ids_ptr, eid_ptr, w_desc, scale_ptr, bufs, bars, N, nk, num_tiles, num_pid_m, num_valid,
                      stride_am, ks4, n_rows, top_k: gl.constexpr, K_TAIL: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
                      PERSISTENT: gl.constexpr):
    SPC: gl.constexpr = 256 // (bufs[1].type.shape[2] * 2)
    if PERSISTENT:
        it = 0
        jt = 0
        for t in range(gl.program_id(0), num_tiles, gl.num_programs(0)):
            _decode_tile(bufs, bars, nk, it, jt, K_TAIL)
            it += nk
            jt += gl.cdiv(nk, SPC)
    else:
        _decode_tile(bufs, bars, nk, 0, 0, K_TAIL)


@gluon.jit
def _decode_tile(bufs, bars, nk, it, jt, K_TAIL: gl.constexpr):
    x_bufs, w_bufs, s_bufs, d_bufs = bufs
    x_ready, x_empty, w_ready, w_empty, s_ready, s_empty, d_ready = bars
    WS: gl.constexpr = w_bufs.type.shape[0]
    BN: gl.constexpr = w_bufs.type.shape[1]
    KB: gl.constexpr = w_bufs.type.shape[2]
    BK: gl.constexpr = KB * 2
    SS: gl.constexpr = s_bufs.type.shape[0]
    SB: gl.constexpr = BK // 16
    DS: gl.constexpr = d_bufs.type.shape[0]
    SPC: gl.constexpr = 256 // BK
    NW: gl.constexpr = gl.num_warps()

    # one thread per row, so the k16 slice stores of a warp hit 32 consecutive rows
    ql: gl.constexpr = gl.BlockedLayout([1, 2 * SB], [32, 1], [NW, 1], [1, 0])
    q3l: gl.constexpr = gl.BlockedLayout([1, SB, 2], [32, 1, 1], [NW, 1, 1], [2, 1, 0])
    qsl: gl.constexpr = gl.SliceLayout(2, q3l)
    w32_layout: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=w_bufs.type.layout.swizzle_byte_width,
                                                     element_bitwidth=32, rank=2)
    d32_layout: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=32, rank=2)
    w32 = w_bufs._reinterpret(gl.int32, [WS, BN, KB // 4], w32_layout)
    d32 = d_bufs._reinterpret(gl.int32, [DS, SB * BN, 8], d32_layout)

    # with K % 256 != 0 the last scale block covers fewer than SPC chunks
    for jb in range(gl.cdiv(nk, SPC)):
        j = jt + jb
        ss = j % SS
        mbarrier.wait(s_ready.index(ss), j // SS & 1)
        for c in gl.static_range(SPC):
            kk = jb * SPC + c
            if not K_TAIL or kk < nk:
                i = it + kk
                ws = i % WS
                mbarrier.wait(w_ready.index(ws), i // WS & 1)
                q = w32.index(ws).load(ql)
                s = s_bufs.index(ss).slice(c * SB, SB, dim=1).load(qsl)
                fence_async_shared()
                mbarrier.arrive(w_empty.index(ws))
                q = gl.convert_layout(gl.reshape(q, [BN, SB, 2]), q3l, assert_trivial=True).to(gl.uint32, bitcast=True)
                w0, w1, w2, w3 = e2m1x8_bf16x2(q, gl.expand_dims(scale_bf16x2(s), 2))
                # join appends the new axis last, so this flattens to w0, w1, w2, w3; then k16 slice major for the MMA warps
                w = gl.reshape(gl.join(gl.join(w0, w2), gl.join(w1, w3)), [BN, SB, 8])
                w = gl.permute(w, (1, 0, 2)).to(gl.int32, bitcast=True)

                ds = i % DS
                # the MMA warps release x and d slots together on x_empty
                mbarrier.wait(x_empty.index(ds), (i // DS & 1) ^ 1)
                d32.index(ds).reshape([SB, BN, 8]).store(w)
                mbarrier.arrive(d_ready.index(ds))
        fence_async_shared()
        mbarrier.arrive(s_empty.index(ss))


@gluon.jit
def _mma_chunks(bufs, bars, acc, it, nk, al: gl.constexpr, bl: gl.constexpr, BN: gl.constexpr, BM: gl.constexpr):
    x_bufs, w_bufs, s_bufs, d_bufs = bufs
    x_ready, x_empty, w_ready, w_empty, s_ready, s_empty, d_ready = bars
    KS: gl.constexpr = w_bufs.type.shape[2] // 8
    XS: gl.constexpr = x_bufs.type.shape[0]
    for kk in range(nk):
        i = it + kk
        xs = i % XS
        mbarrier.wait(d_ready.index(xs), i // XS & 1)
        mbarrier.wait(x_ready.index(xs), i // XS & 1)
        for k16 in gl.static_range(KS):
            a = d_bufs.index(xs).slice(k16 * BN, BN).load(al)
            b = x_bufs.index(xs).slice(k16 * BM, BM).permute((1, 0)).load(bl)
            acc = mma_v2(a, b, acc)
        mbarrier.arrive(x_empty.index(xs))
    return acc


@gluon.jit
def _mma_partition(bufs, bars, ids_ptr, eid_ptr, global_ptr, c_ptr, topk_weights_ptr, N, nk, num_tiles, num_pid_m,
                   num_valid, stride_ge, stride_cm, stride_tw, mma: gl.constexpr, BN: gl.constexpr, BM: gl.constexpr,
                   MUL_ROUTED_WEIGHT: gl.constexpr, GROUP_SIZE_M: gl.constexpr, PERSISTENT: gl.constexpr):
    al: gl.constexpr = gl.DotOperandLayout(0, mma, 2)
    bl: gl.constexpr = gl.DotOperandLayout(1, mma, 2)
    if PERSISTENT:
        it = 0
        for t in range(gl.program_id(0), num_tiles, gl.num_programs(0)):
            acc = _mma_chunks(bufs, bars, gl.zeros([BN, BM], gl.float32, mma), it, nk, al, bl, BN, BM)
            it += nk
            # the epilogue runs here while the loader and decode warps already fill the rings for the next tile
            # stored straight from the MMA layout, as a convert_layout needs scratch shared memory on top of the rings
            pid_m, pid_n = _tile(t, num_pid_m, gl.cdiv(N, BN), GROUP_SIZE_M)
            slot = gl.load(eid_ptr + pid_m)
            offs_n = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
            offs_t = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
            _epilogue(acc, ids_ptr, global_ptr, c_ptr, topk_weights_ptr, slot, offs_n, offs_t, N, num_valid,
                      stride_ge, stride_cm, stride_tw, MUL_ROUTED_WEIGHT)
    else:
        return _mma_chunks(bufs, bars, gl.zeros([BN, BM], gl.float32, mma), 0, nk, al, bl, BN, BM)


@gluon.jit
def _epilogue(acc, ids_ptr, global_ptr, c_ptr, topk_weights_ptr, slot, offs_n, offs_t, N, num_valid,
              stride_ge, stride_cm, stride_tw, MUL_ROUTED_WEIGHT: gl.constexpr):
    route = gl.load(ids_ptr + offs_t)
    route_mask = route < num_valid
    n_mask = offs_n < N
    g = gl.load(global_ptr + slot.to(gl.int64) * stride_ge + offs_n, mask=n_mask, other=0.0).to(gl.float32)
    # 256 undoes the 2^-8 that the decode leaves in every weight
    acc = acc * (g * 256.0)[:, None]
    if MUL_ROUTED_WEIGHT:
        wt = gl.load(topk_weights_ptr + route * stride_tw, mask=route_mask, other=0.0)
        acc = acc * wt[None, :]
    c_ptrs = c_ptr + route.to(gl.int64)[None, :] * stride_cm + offs_n[:, None]
    gl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=route_mask[None, :] & n_mask[:, None])


SCALE_SMEM_LAYOUT = gl.constexpr(gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=2))


@gluon.jit
def _prefill_nvfp4_moe_kernel(
    a_ptr, w_desc, scale_ptr, global_ptr, c_ptr, topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens, ks4, n_rows,
    stride_am, stride_ge, stride_cm, stride_tw,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    STAGES: gl.constexpr,
    W_STAGES: gl.constexpr,
    PERSISTENT: gl.constexpr,
    K_TAIL: gl.constexpr,
    DECODE_WARPS: gl.constexpr,
    MMA_WARPS_N: gl.constexpr,
    LOAD_REGS: gl.constexpr,
    DECODE_REGS: gl.constexpr,
    MUL_ROUTED_WEIGHT: gl.constexpr,
    top_k: gl.constexpr,
):
    BM: gl.constexpr = BLOCK_SIZE_M
    BN: gl.constexpr = BLOCK_SIZE_N
    BK: gl.constexpr = BLOCK_SIZE_K
    live_m = gl.cdiv(gl.load(num_tokens_post_padded_ptr), BM)
    if PERSISTENT:
        num_pid_m = live_m
    else:
        # the grid covers the most m-blocks moe_align can produce; each CTA takes the tile of its index
        num_pid_m = gl.cdiv(EM, BM)
        pid_m, pid_n = _tile(gl.program_id(0), num_pid_m, gl.cdiv(N, BN), GROUP_SIZE_M)
        if pid_m >= live_m:
            return
    num_tiles = num_pid_m * gl.cdiv(N, BN)
    nk = K // BK

    # x and the decoded weights are stored as k16 slices, [stage, (k // 16) * rows, 16], and share STAGES slots
    x_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, 16], gl.bfloat16)
    d_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BN, 16], gl.bfloat16)
    x_bufs = gl.allocate_shared_memory(gl.bfloat16, [STAGES, (BK // 16) * BM, 16], x_layout)
    w_bufs = gl.allocate_shared_memory(gl.uint8, [W_STAGES, BN, BK // 2], w_desc.layout)
    s_bufs = gl.allocate_shared_memory(gl.uint8, [2, BN, 16], SCALE_SMEM_LAYOUT)
    d_bufs = gl.allocate_shared_memory(gl.bfloat16, [STAGES, (BK // 16) * BN, 16], d_layout)
    bufs = (x_bufs, w_bufs, s_bufs, d_bufs)

    x_ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    x_empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    w_ready = gl.allocate_shared_memory(gl.int64, [W_STAGES, 1], mbarrier.MBarrierLayout())
    w_empty = gl.allocate_shared_memory(gl.int64, [W_STAGES, 1], mbarrier.MBarrierLayout())
    s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    s_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    d_ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES):
        mbarrier.init(x_ready.index(i), count=1)
        mbarrier.init(x_empty.index(i), count=1)
        mbarrier.init(d_ready.index(i), count=1)
    for i in gl.static_range(W_STAGES):
        mbarrier.init(w_ready.index(i), count=1)
        mbarrier.init(w_empty.index(i), count=1)
    for i in gl.static_range(2):
        mbarrier.init(s_ready.index(i), count=1)
        mbarrier.init(s_empty.index(i), count=1)
    bars = (x_ready, x_empty, w_ready, w_empty, s_ready, s_empty, d_ready)

    NW: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[MMA_WARPS_N, NW // MMA_WARPS_N],
                                                  instr_shape=[16, 8])
    # tuples go inline: a tuple stored in a variable turns its constexprs into runtime values
    acc = gl.warp_specialize([
        (_mma_partition, (bufs, bars, sorted_token_ids_ptr, expert_ids_ptr, global_ptr, c_ptr, topk_weights_ptr, N, nk,
                          num_tiles, num_pid_m, num_valid_tokens, stride_ge, stride_cm, stride_tw, mma, BN, BM,
                          MUL_ROUTED_WEIGHT, GROUP_SIZE_M, PERSISTENT)),
        (_load_partition, (a_ptr, sorted_token_ids_ptr, expert_ids_ptr, w_desc, scale_ptr, bufs, bars, N, nk, num_tiles,
                           num_pid_m, num_valid_tokens, stride_am, ks4, n_rows, top_k, K_TAIL, GROUP_SIZE_M, PERSISTENT)),
        (_decode_partition, (a_ptr, sorted_token_ids_ptr, expert_ids_ptr, w_desc, scale_ptr, bufs, bars, N, nk, num_tiles,
                             num_pid_m, num_valid_tokens, stride_am, ks4, n_rows, top_k, K_TAIL, GROUP_SIZE_M, PERSISTENT)),
    ], [1, DECODE_WARPS], [LOAD_REGS, DECODE_REGS])

    if not PERSISTENT:
        # a coalesced store layout; the rings are idle by now, so the conversion's scratch fits
        out_l: gl.constexpr = gl.BlockedLayout([8, 1], [BN // 8, 32 * 8 // BN], [1, NW], [0, 1])
        pid_m, pid_n = _tile(gl.program_id(0), num_pid_m, gl.cdiv(N, BN), GROUP_SIZE_M)
        slot = gl.load(expert_ids_ptr + pid_m)
        offs_n = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(1, out_l))
        offs_t = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(0, out_l))
        _epilogue(gl.convert_layout(acc, out_l), sorted_token_ids_ptr, global_ptr, c_ptr, topk_weights_ptr, slot, offs_n,
                  offs_t, N, num_valid_tokens, stride_ge, stride_cm, stride_tw, MUL_ROUTED_WEIGHT)


def supports(a: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, c: torch.Tensor) -> bool:
    """Whether :func:`prefill_gemm` can run this GEMM on an sm_120 GPU."""
    K = packed.shape[2] * 2
    # the TMA weight loads and 16-byte cp.async need contiguous, 16-byte aligned rows
    return (a.dtype == torch.bfloat16 and K % 64 == 0
            and packed.is_contiguous() and scale.is_contiguous() and c.is_contiguous()
            and packed.data_ptr() % 16 == 0 and scale.data_ptr() % 16 == 0
            and a.stride(1) == 1 and a.stride(0) % 8 == 0 and a.data_ptr() % 16 == 0)


def prefill_gemm(a, packed, scale, glob, c, topk_weights_flat, sorted_ids, expert_ids, num_tokens_post_padded,
                 num_valid_tokens, top_k, mul_routed_weight, cfg):
    """Launch the kernel with the tile / pipeline constexprs in ``cfg``; K must be a multiple of cfg["BLOCK_SIZE_K"]."""
    S, N, KB = packed.shape
    K = KB * 2
    BM, BN, BK = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
    w_layout = gl.NVMMASharedLayout.get_default_for([BN, BK // 2], gl.uint8)
    w_desc = TensorDescriptor.from_tensor(packed.view(S * N, KB), [BN, BK // 2], w_layout)
    max_tiles = triton.cdiv(sorted_ids.shape[0], BM) * triton.cdiv(N, BN)
    if cfg["PERSISTENT"]:
        max_tiles = min(max_tiles, torch.cuda.get_device_properties(a.device).multi_processor_count)
    _prefill_nvfp4_moe_kernel[(max_tiles,)](
        a, w_desc, scale.view(torch.uint8), glob, c, topk_weights_flat, sorted_ids, expert_ids, num_tokens_post_padded,
        N, K, sorted_ids.shape[0], num_valid_tokens, K // 64, S * N,
        a.stride(0), glob.stride(0), c.stride(1), topk_weights_flat.stride(0),
        K_TAIL=K % 256 != 0, MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k, **cfg,
    )
