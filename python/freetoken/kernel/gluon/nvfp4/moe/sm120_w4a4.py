"""NVFP4 W4A4 MoE prefill GEMMs for sm_120: NVFP4 weights times NVFP4 activations on mma.sync kind::mxf4nvf4.

The activations are quantized under static global scales (the checkpoint's input_scale), so each GEMM input is e2m1 codes, per-16 e4m3 scales and one fp32 scalar.
:func:`gate_up_gemm` applies the gated activation in its epilogue and writes the down GEMM's NVFP4 input; :func:`down_gemm` scales by the routing weight and adds each route into its token's output row with bf16 atomics (the sum order, and so the result, varies from run to run).
Routing follows the W4A16 kernels (moe_align sorted ids, expert per M block, swap-AB accumulator [BN, BM]); :mod:`freetoken.kernel.nvfp4_moe` picks the constexprs.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from ..utils import mma_nvf4, scale_layout

# gated activations of the gate_up epilogue, as gated_act_and_mul computes them
ACTIVATIONS = {"silu": 0, "gelu_tanh": 1, "swigluoai": 2, "swiglu_clamp": 3}

SCALE_SMEM_LAYOUT = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))


@gluon.jit
def _tile(t, num_pid_m, num_pid_n, GROUP_SIZE_M: gl.constexpr, N_GROUP: gl.constexpr):
    if N_GROUP > 0:
        # N-outer strips of N_GROUP tiles: concurrent CTAs add into a narrow output strip that stays in L2
        per_strip = num_pid_m * N_GROUP
        strip = t // per_strip
        rem = t % per_strip
        width = min(N_GROUP, num_pid_n - strip * N_GROUP)
        return rem // width, strip * N_GROUP + rem % width
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    first_pid_m = (t // num_pid_in_group) * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    return first_pid_m + ((t % num_pid_in_group) % group_size_m), (t % num_pid_in_group) // group_size_m


@gluon.jit
def _scale_ptrs(ws_ptr, xs_ptr, ids_ptr, w_row, w_row_up, x_div, pid_m, num_valid, ks4, n_rows,
                BM: gl.constexpr, BN: gl.constexpr, K_TAIL: gl.constexpr, GATED: gl.constexpr, NWL: gl.constexpr):
    # scale rows are K / 16 bytes: 16-byte copies when K % 256 == 0, else 4-byte copies zeroed past the row end
    SV: gl.constexpr = 4 if K_TAIL else 16
    sll: gl.constexpr = gl.BlockedLayout([1, SV], [32 * SV // 16, 16 // SV], [NWL, 1], [1, 0])
    s_cols = gl.arange(0, 16, layout=gl.SliceLayout(0, sll))
    # both forms equal K // 16; the second tells the compiler the row is a multiple of 16 bytes
    row_len = ks4 * 4 if K_TAIL else (ks4 // 4) * 16
    r = gl.arange(0, BN, layout=gl.SliceLayout(1, sll))
    if GATED:
        sw_rows = gl.where(r < BN // 2, w_row + r, w_row_up + r - BN // 2)
    else:
        sw_rows = w_row + r
    sw_ptrs = ws_ptr + sw_rows.to(gl.int64)[:, None] * row_len + s_cols[None, :]
    sw_ok = (sw_rows < n_rows)[:, None]
    sx_tok = gl.load(ids_ptr + pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, sll)))
    # the x scale rows are contiguous K / 16 bytes too; row_len keeps the divisibility a stride argument loses
    sx_ptrs = xs_ptr + ((sx_tok // x_div).to(gl.int64) * row_len)[:, None] + s_cols[None, :]
    sx_ok = (sx_tok < num_valid)[:, None]
    return sw_ptrs, sw_ok, sx_ptrs, sx_ok, s_cols, row_len


@gluon.jit
def _load_tile(xq_ptr, xs_ptr, ids_ptr, eid_ptr, w_desc, ws_ptr, bufs, bars, pid_m, pid_n, N, nk, it, jt, num_valid,
               stride_xq, ks4, n_rows, x_div, K_TAIL: gl.constexpr, GATED: gl.constexpr):
    x_bufs, w_bufs, sw_bufs, sx_bufs = bufs
    ready, empty, s_ready, s_empty = bars
    S: gl.constexpr = x_bufs.type.shape[0]
    BM: gl.constexpr = x_bufs.type.shape[1]
    KB: gl.constexpr = x_bufs.type.shape[2]
    BN: gl.constexpr = w_bufs.type.shape[1]
    SPC: gl.constexpr = 128 // KB
    NWL: gl.constexpr = gl.num_warps()

    if GATED:
        # rows [0, BN/2) of the tile are gate rows, [BN/2, BN) the matching up rows N/2 further down
        w_row = gl.load(eid_ptr + pid_m) * N + pid_n * (BN // 2)
        w_row_up = w_row + N // 2
    else:
        w_row = gl.load(eid_ptr + pid_m) * N + pid_n * BN
        w_row_up = w_row

    # x rows: 16 bytes per cp.async, KB // 16 lanes per row
    xl: gl.constexpr = gl.BlockedLayout([1, 16], [32 * 16 // KB, KB // 16], [NWL, 1], [1, 0])
    tok = gl.load(ids_ptr + pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, xl)))
    x_ptrs = xq_ptr + ((tok // x_div).to(gl.int64) * stride_xq)[:, None] + gl.arange(0, KB, layout=gl.SliceLayout(0, xl))[None, :]
    x_mask = (tok < num_valid)[:, None]
    sw_ptrs, sw_ok, sx_ptrs, sx_ok, s_cols, row_len = _scale_ptrs(
        ws_ptr, xs_ptr, ids_ptr, w_row, w_row_up, x_div, pid_m, num_valid, ks4, n_rows, BM, BN, K_TAIL, GATED, NWL)

    for kk in range(nk):
        if kk % SPC == 0:
            jl = kk // SPC
            j = jt + jl
            ss = j % 2
            mbarrier.wait(s_empty.index(ss), (j // 2 & 1) ^ 1)
            col_ok = (jl * 16 + s_cols < row_len)[None, :]
            async_copy.async_load(sw_bufs.index(ss), sw_ptrs + jl * 16, mask=sw_ok & col_ok)
            async_copy.async_load(sx_bufs.index(ss), sx_ptrs + jl * 16, mask=sx_ok & col_ok)
            async_copy.mbarrier_arrive(s_ready.index(ss))
            mbarrier.arrive(s_ready.index(ss))
        c = it + kk
        st = c % S
        mbarrier.wait(empty.index(st), (c // S & 1) ^ 1)
        # the cp.async arrive goes first: it raises the pending count before the TMA expect can complete the phase
        async_copy.async_load(x_bufs.index(st), x_ptrs + kk * KB, mask=x_mask)
        async_copy.mbarrier_arrive(ready.index(st))
        if GATED:
            mbarrier.expect(ready.index(st), 2 * w_desc.block_type.nbytes)
            tma.async_load(w_desc, [w_row, kk * KB], ready.index(st), w_bufs.index(st).slice(0, BN // 2))
            tma.async_load(w_desc, [w_row_up, kk * KB], ready.index(st), w_bufs.index(st).slice(BN // 2, BN // 2))
        else:
            mbarrier.expect(ready.index(st), w_desc.block_type.nbytes)
            tma.async_load(w_desc, [w_row, kk * KB], ready.index(st), w_bufs.index(st))


@gluon.jit
def _load_partition(xq_ptr, xs_ptr, ids_ptr, eid_ptr, w_desc, ws_ptr, bufs, bars, N, nk, num_pid_m, num_valid,
                    stride_xq, ks4, n_rows, x_div, K_TAIL: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
                    GATED: gl.constexpr, N_GROUP: gl.constexpr):
    BN: gl.constexpr = bufs[1].type.shape[1]
    pid_m, pid_n = _tile(gl.program_id(0), num_pid_m, gl.cdiv(N, BN), GROUP_SIZE_M, N_GROUP)
    _load_tile(xq_ptr, xs_ptr, ids_ptr, eid_ptr, w_desc, ws_ptr, bufs, bars, pid_m, pid_n, N, nk, 0, 0, num_valid,
               stride_xq, ks4, n_rows, x_div, K_TAIL, GATED)


@gluon.jit
def _mma_partition(bufs, bars, nk, mma: gl.constexpr, wsl: gl.constexpr, xsl: gl.constexpr, K_TAIL: gl.constexpr):
    x_bufs, w_bufs, sw_bufs, sx_bufs = bufs
    ready, empty, s_ready, s_empty = bars
    S: gl.constexpr = x_bufs.type.shape[0]
    BM: gl.constexpr = x_bufs.type.shape[1]
    KB: gl.constexpr = x_bufs.type.shape[2]
    BN: gl.constexpr = w_bufs.type.shape[1]
    SB: gl.constexpr = KB // 8
    SPC: gl.constexpr = 128 // KB
    al: gl.constexpr = gl.DotOperandLayout(0, mma, 4)
    bl: gl.constexpr = gl.DotOperandLayout(1, mma, 4)
    acc = gl.zeros([BN, BM], gl.float32, mma)
    # with K % 256 != 0 the last scale block covers fewer than SPC chunks
    for jb in range(gl.cdiv(nk, SPC)):
        ss = jb % 2
        mbarrier.wait(s_ready.index(ss), jb // 2 & 1)
        for c in gl.static_range(SPC):
            kk = jb * SPC + c
            if not K_TAIL or kk < nk:
                st = kk % S
                mbarrier.wait(ready.index(st), kk // S & 1)
                a = w_bufs.index(st).load(al)
                b = x_bufs.index(st).permute((1, 0)).load(bl)
                sw = sw_bufs.index(ss).slice(c * SB, SB, dim=1).load(wsl).to(gl.float8e4nv, bitcast=True)
                sx = sx_bufs.index(ss).slice(c * SB, SB, dim=1).load(xsl).to(gl.float8e4nv, bitcast=True)
                mbarrier.arrive(empty.index(st))
                acc = mma_nvf4(a, sw, b, sx, acc)
        mbarrier.arrive(s_empty.index(ss))
    return acc


@gluon.jit
def _e2m1x2(hi, lo):
    # cvt packs its first source into the high nibble
    return gl.inline_asm_elementwise("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $1, $2; cvt.u16.u8 $0, t; }",
                                     "=h,r,r", [hi, lo], gl.uint16, True, 1)


@gluon.jit
def _gated_epilogue(acc, slot, pid_m, pid_n, wg_ptr, g_in, ids_ptr, q_ptr, qs_ptr, g_out, N, num_valid, stride_ge,
                    ACT: gl.constexpr, act_alpha, act_limit):
    """act(gate, up) of the tile's BN / 2 columns as NVFP4 under the global g_out: q [routes, N / 4], qs [routes, N / 32]."""
    BN: gl.constexpr = acc.type.shape[0]
    BM: gl.constexpr = acc.type.shape[1]
    H2: gl.constexpr = BN // 2
    G: gl.constexpr = H2 // 16
    NW: gl.constexpr = gl.num_warps()
    out_l: gl.constexpr = acc.type.layout
    r = gl.arange(0, BN, layout=gl.SliceLayout(1, out_l))
    col = gl.where(r < H2, pid_n * H2 + r, N // 2 + pid_n * H2 + r - H2)
    wg = gl.load(wg_ptr + slot.to(gl.int64) * stride_ge + col).to(gl.float32) * g_in
    # bf16 like the W4A16 gate_up output; n-contiguous 16-byte runs, swizzled so the lanes of a warp spread over banks
    h_l: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [0, 1])
    h_s = gl.allocate_shared_memory(gl.bfloat16, [BN, BM], h_l, (acc * wg[:, None]).to(gl.bfloat16))
    # one thread holds 16 consecutive columns of one route
    TG: gl.constexpr = min(32, G)
    ql2: gl.constexpr = gl.BlockedLayout([16, 1], [TG, 32 // TG], [1, NW], [0, 1])
    l4: gl.constexpr = gl.BlockedLayout([1, 8, 2, 1], [TG, 1, 1, 32 // TG], [1, 1, 1, NW], [2, 1, 0, 3])
    gate = gl.convert_layout(gl.reshape(h_s.slice(0, H2).load(ql2), [G, 8, 2, BM]), l4, assert_trivial=True).to(gl.float32)
    up = gl.convert_layout(gl.reshape(h_s.slice(H2, H2).load(ql2), [G, 8, 2, BM]), l4, assert_trivial=True).to(gl.float32)
    if ACT == 1:
        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        # tanh(y) = 1 - 2 / (exp(2y) + 1)
        h = 0.5 * gate * (2.0 - 2.0 / (gl.exp(2.0 * inner) + 1.0)) * up
    elif ACT == 2 or ACT == 3:
        gate = gl.minimum(gate, act_limit)
        up = gl.minimum(gl.maximum(up, -act_limit), act_limit)
        if ACT == 2:
            up = up + 1.0
        h = gate / (1.0 + gl.exp(-gate * act_alpha)) * up
    else:
        h = gate / (1.0 + gl.exp(-gate)) * up
    h = h.to(gl.bfloat16).to(gl.float32)
    amax = gl.max(gl.max(gl.abs(h), axis=2), axis=1)
    sc = (amax / (6.0 * g_out)).to(gl.float8e4nv)
    sf = sc.to(gl.float32) * g_out
    inv = gl.where(sf > 0, 1.0 / sf, 0.0)
    v = h * gl.expand_dims(gl.expand_dims(inv, 1), 2)
    lo, hi = gl.split(gl.permute(v, (0, 1, 3, 2)))
    q = _e2m1x2(hi, lo).to(gl.uint8)
    ql: gl.constexpr = q.type.layout
    g_i = gl.arange(0, G, layout=gl.SliceLayout(1, gl.SliceLayout(2, ql)))
    b_i = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(2, ql)))
    rq = gl.load(ids_ptr + pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(0, gl.SliceLayout(1, ql))))
    q_cols = pid_n * (H2 // 2) + g_i[:, None] * 8 + b_i[None, :]
    q_ptrs = q_ptr + (rq.to(gl.int64) * (N // 4))[None, None, :] + q_cols[:, :, None]
    gl.store(q_ptrs, q, mask=(rq < num_valid)[None, None, :])
    sl: gl.constexpr = sc.type.layout
    sg = gl.arange(0, G, layout=gl.SliceLayout(1, sl))
    rs = gl.load(ids_ptr + pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(0, sl)))
    s_ptrs = qs_ptr + (rs.to(gl.int64) * (N // 32))[None, :] + (pid_n * G + sg)[:, None]
    gl.store(s_ptrs, sc.to(gl.uint8, bitcast=True), mask=(rs < num_valid)[None, :])


@gluon.jit
def _w4a4_moe_kernel(
    xq_ptr, xs_ptr, g_in_ptr, w_desc, ws_ptr, wg_ptr, out_ptr, topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    q_ptr, qs_ptr, g_out_ptr, act_alpha, act_limit,
    N, K, EM, num_valid_tokens, ks4, n_rows,
    stride_xq, stride_ge, stride_out, stride_tw,
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr, N_GROUP: gl.constexpr, STAGES: gl.constexpr, MMA_WARPS_N: gl.constexpr,
    LOAD_WARPS: gl.constexpr, LOAD_REGS: gl.constexpr, K_TAIL: gl.constexpr, GATED: gl.constexpr,
    ACT: gl.constexpr, top_k: gl.constexpr, wsl: gl.constexpr, xsl: gl.constexpr,
):
    BM: gl.constexpr = BLOCK_SIZE_M
    BN: gl.constexpr = BLOCK_SIZE_N
    KB: gl.constexpr = BLOCK_SIZE_K // 2
    num_pid_m = gl.cdiv(EM, BM)
    pid_m, pid_n = _tile(gl.program_id(0), num_pid_m, gl.cdiv(N, BN), GROUP_SIZE_M, N_GROUP)
    if pid_m >= gl.cdiv(gl.load(num_tokens_post_padded_ptr), BM):
        return
    nk = K // BLOCK_SIZE_K
    # gate_up reads one x row per token, down one row per route
    x_div = top_k if GATED else 1

    x_bufs = gl.allocate_shared_memory(gl.uint8, [STAGES, BM, KB], gl.NVMMASharedLayout.get_default_for([BM, KB], gl.uint8))
    w_bufs = gl.allocate_shared_memory(gl.uint8, [STAGES, BN, KB], w_desc.layout)
    sw_bufs = gl.allocate_shared_memory(gl.uint8, [2, BN, 16], SCALE_SMEM_LAYOUT)
    sx_bufs = gl.allocate_shared_memory(gl.uint8, [2, BM, 16], SCALE_SMEM_LAYOUT)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    s_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(empty.index(i), count=1)
    for i in gl.static_range(2):
        mbarrier.init(s_ready.index(i), count=1)
        mbarrier.init(s_empty.index(i), count=1)

    NW: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[MMA_WARPS_N, NW // MMA_WARPS_N],
                                                  instr_shape=[16, 8])
    # tuples go inline: a tuple stored in a variable turns its constexprs into runtime values
    acc = gl.warp_specialize([
        (_mma_partition, ((x_bufs, w_bufs, sw_bufs, sx_bufs), (ready, empty, s_ready, s_empty), nk, mma, wsl, xsl, K_TAIL)),
        (_load_partition, (xq_ptr, xs_ptr, sorted_token_ids_ptr, expert_ids_ptr, w_desc, ws_ptr,
                           (x_bufs, w_bufs, sw_bufs, sx_bufs), (ready, empty, s_ready, s_empty), N, nk, num_pid_m,
                           num_valid_tokens, stride_xq, ks4, n_rows, x_div, K_TAIL, GROUP_SIZE_M, GATED, N_GROUP)),
    ], [LOAD_WARPS], [LOAD_REGS])

    out_l: gl.constexpr = gl.BlockedLayout([8, 1], [BN // 8, 32 * 8 // BN], [1, NW], [0, 1])
    acc = gl.convert_layout(acc, out_l)
    slot = gl.load(expert_ids_ptr + pid_m)
    g_in = gl.load(g_in_ptr)
    if GATED:
        _gated_epilogue(acc, slot, pid_m, pid_n, wg_ptr, g_in, sorted_token_ids_ptr, q_ptr, qs_ptr, gl.load(g_out_ptr),
                        N, num_valid_tokens, stride_ge, ACT, act_alpha, act_limit)
    else:
        offs_n = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(1, out_l))
        n_mask = offs_n < N
        route = gl.load(sorted_token_ids_ptr + pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(0, out_l)))
        route_mask = route < num_valid_tokens
        wg = gl.load(wg_ptr + slot.to(gl.int64) * stride_ge + offs_n, mask=n_mask, other=0.0).to(gl.float32) * g_in
        tw = gl.load(topk_weights_ptr + route * stride_tw, mask=route_mask, other=0.0)
        acc = acc * wg[:, None] * tw[None, :]
        out_ptrs = out_ptr + (route // top_k).to(gl.int64)[None, :] * stride_out + offs_n[:, None]
        gl.atomic_add(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=route_mask[None, :] & n_mask[:, None], sem="relaxed")


# Persistent down GEMM: the 256 x 128 tile and 8 MMA warps of the kernel above, with the routing-weight scale and the atomics on
# separate epilogue warps. The accumulator is two 128 x 128 halves that go to the epilogue one at a time through one 32 KB buffer,
# so the next tile's mainloop overlaps the atomics of the last half.


@gluon.jit
def _num_tiles(ntpp_ptr, N, BM: gl.constexpr, BN: gl.constexpr):
    num_pid_m = gl.cdiv(gl.load(ntpp_ptr), BM)
    num_pid_n = gl.cdiv(N, BN)
    return num_pid_m, num_pid_n, num_pid_m * num_pid_n


@gluon.jit
def _down_load_partition(xq_ptr, xs_ptr, ids_ptr, eid_ptr, ntpp_ptr, w_desc, ws_ptr, bufs, bars, N, nk, num_valid,
                         stride_xq, ks4, n_rows, K_TAIL: gl.constexpr, N_GROUP: gl.constexpr, GROUP_SIZE_M: gl.constexpr):
    BM: gl.constexpr = bufs[0].type.shape[1]
    KB: gl.constexpr = bufs[0].type.shape[2]
    BN: gl.constexpr = bufs[1].type.shape[1]
    SPC: gl.constexpr = 128 // KB
    num_pid_m, num_pid_n, num_tiles = _num_tiles(ntpp_ptr, N, BM, BN)
    # it / jt count chunks and scale blocks across tiles, so the rings keep their phase from one tile to the next
    it = 0
    jt = 0
    for t in range(gl.program_id(0), num_tiles, gl.num_programs(0)):
        pid_m, pid_n = _tile(t, num_pid_m, num_pid_n, GROUP_SIZE_M, N_GROUP)
        _load_tile(xq_ptr, xs_ptr, ids_ptr, eid_ptr, w_desc, ws_ptr, (bufs[0], bufs[1], bufs[2], bufs[3]),
                   (bars[0], bars[1], bars[2], bars[3]), pid_m, pid_n, N, nk, it, jt, num_valid, stride_xq, ks4,
                   n_rows, 1, K_TAIL, False)
        it += nk
        jt += gl.cdiv(nk, SPC)


@gluon.jit
def _down_epi_partition(ids_ptr, ntpp_ptr, out_ptr, bufs, bars, N, num_valid, stride_out, top_k: gl.constexpr,
                        N_GROUP: gl.constexpr, GROUP_SIZE_M: gl.constexpr):
    h_buf = bufs[4]
    acc_ready, acc_empty = bars[4], bars[5]
    HB: gl.constexpr = h_buf.type.shape[0]
    BM: gl.constexpr = h_buf.type.shape[1]
    BN: gl.constexpr = 2 * HB
    CH: gl.constexpr = 64
    NWE: gl.constexpr = gl.num_warps()
    # 8 consecutive n of one route per thread: one 16-byte bf16 atomic; 64 rows per pass keep the registers low
    el: gl.constexpr = gl.BlockedLayout([8, 1], [CH // 8, 32 * 8 // CH], [1, NWE], [0, 1])
    offs_cl = gl.arange(0, CH, layout=gl.SliceLayout(1, el))
    offs_tl = gl.arange(0, BM, layout=gl.SliceLayout(0, el))
    num_pid_m, num_pid_n, num_tiles = _num_tiles(ntpp_ptr, N, BM, BN)
    hc = 0
    for t in range(gl.program_id(0), num_tiles, gl.num_programs(0)):
        pid_m, pid_n = _tile(t, num_pid_m, num_pid_n, GROUP_SIZE_M, N_GROUP)
        route = gl.load(ids_ptr + pid_m * BM + offs_tl)
        rmask = (route < num_valid)[None, :]
        rows = (route // top_k).to(gl.int64)[None, :] * stride_out
        for half in gl.static_range(2):
            mbarrier.wait(acc_ready.index(0), hc & 1)
            for ch in gl.static_range(HB // CH):
                offs_n = pid_n * BN + half * HB + ch * CH + offs_cl
                h = h_buf.slice(ch * CH, CH).load(el)
                gl.atomic_add(out_ptr + rows + offs_n[:, None], h, mask=rmask & (offs_n < N)[:, None], sem="relaxed")
            mbarrier.arrive(acc_empty.index(0))
            hc += 1


@gluon.jit
def _down_mma_partition(bufs, bars, ids_ptr, eid_ptr, ntpp_ptr, wg_ptr, g_in_ptr, tw_ptr, N, nk, num_valid, stride_ge,
                        stride_tw, mma: gl.constexpr, wsl: gl.constexpr, xsl: gl.constexpr, K_TAIL: gl.constexpr,
                        N_GROUP: gl.constexpr, GROUP_SIZE_M: gl.constexpr):
    x_bufs, w_bufs, sw_bufs, sx_bufs, h_buf = bufs
    ready, empty, s_ready, s_empty, acc_ready, acc_empty = bars
    S: gl.constexpr = x_bufs.type.shape[0]
    BM: gl.constexpr = x_bufs.type.shape[1]
    KB: gl.constexpr = x_bufs.type.shape[2]
    BN: gl.constexpr = w_bufs.type.shape[1]
    HB: gl.constexpr = BN // 2
    SB: gl.constexpr = KB // 8
    SPC: gl.constexpr = 128 // KB
    al: gl.constexpr = gl.DotOperandLayout(0, mma, 4)
    bl: gl.constexpr = gl.DotOperandLayout(1, mma, 4)
    offs_nl = gl.arange(0, HB, layout=gl.SliceLayout(1, mma))
    offs_tl = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
    g_in = gl.load(g_in_ptr)
    num_pid_m, num_pid_n, num_tiles = _num_tiles(ntpp_ptr, N, BM, BN)
    it = 0
    jt = 0
    hc = 0
    for t in range(gl.program_id(0), num_tiles, gl.num_programs(0)):
        pid_m, pid_n = _tile(t, num_pid_m, num_pid_n, GROUP_SIZE_M, N_GROUP)
        acc0 = gl.zeros([HB, BM], gl.float32, mma)
        acc1 = gl.zeros([HB, BM], gl.float32, mma)
        for jb in range(gl.cdiv(nk, SPC)):
            j = jt + jb
            ss = j % 2
            mbarrier.wait(s_ready.index(ss), j // 2 & 1)
            for c in gl.static_range(SPC):
                kk = jb * SPC + c
                if not K_TAIL or kk < nk:
                    i = it + kk
                    st = i % S
                    mbarrier.wait(ready.index(st), i // S & 1)
                    b = x_bufs.index(st).permute((1, 0)).load(bl)
                    sx = sx_bufs.index(ss).slice(c * SB, SB, dim=1).load(xsl).to(gl.float8e4nv, bitcast=True)
                    a0 = w_bufs.index(st).slice(0, HB).load(al)
                    a1 = w_bufs.index(st).slice(HB, HB).load(al)
                    sws = sw_bufs.index(ss).slice(c * SB, SB, dim=1)
                    sw0 = sws.slice(0, HB).load(wsl).to(gl.float8e4nv, bitcast=True)
                    sw1 = sws.slice(HB, HB).load(wsl).to(gl.float8e4nv, bitcast=True)
                    mbarrier.arrive(empty.index(st))
                    acc0 = mma_nvf4(a0, sw0, b, sx, acc0)
                    acc1 = mma_nvf4(a1, sw1, b, sx, acc1)
            mbarrier.arrive(s_empty.index(ss))
        it += nk
        jt += gl.cdiv(nk, SPC)
        slot = gl.load(eid_ptr + pid_m)
        route = gl.load(ids_ptr + pid_m * BM + offs_tl)
        rmask = route < num_valid
        xs_ = gl.load(tw_ptr + route * stride_tw, mask=rmask, other=0.0) * g_in
        wrow = wg_ptr + slot.to(gl.int64) * stride_ge
        n0 = pid_n * BN + offs_nl
        wg0 = gl.load(wrow + n0, mask=n0 < N, other=0.0).to(gl.float32)
        h0 = (acc0 * wg0[:, None] * xs_[None, :]).to(gl.bfloat16)
        mbarrier.wait(acc_empty.index(0), (hc & 1) ^ 1)
        h_buf.store(h0)
        mbarrier.arrive(acc_ready.index(0))
        hc += 1
        n1 = n0 + HB
        wg1 = gl.load(wrow + n1, mask=n1 < N, other=0.0).to(gl.float32)
        h1 = (acc1 * wg1[:, None] * xs_[None, :]).to(gl.bfloat16)
        mbarrier.wait(acc_empty.index(0), (hc & 1) ^ 1)
        h_buf.store(h1)
        mbarrier.arrive(acc_ready.index(0))
        hc += 1


@gluon.jit
def _w4a4_down_persistent_kernel(
    xq_ptr, xs_ptr, g_in_ptr, w_desc, ws_ptr, wg_ptr, out_ptr, topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, num_valid_tokens, ks4, n_rows, stride_xq, stride_ge, stride_out, stride_tw,
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr, STAGES: gl.constexpr,
    K_TAIL: gl.constexpr, N_GROUP: gl.constexpr, GROUP_SIZE_M: gl.constexpr, MMA_WARPS_N: gl.constexpr,
    LOAD_WARPS: gl.constexpr, LOAD_REGS: gl.constexpr, EPI_WARPS: gl.constexpr, EPI_REGS: gl.constexpr,
    top_k: gl.constexpr, wsl: gl.constexpr, xsl: gl.constexpr,
):
    BM: gl.constexpr = BLOCK_SIZE_M
    BN: gl.constexpr = BLOCK_SIZE_N
    KB: gl.constexpr = BLOCK_SIZE_K // 2
    nk = K // BLOCK_SIZE_K
    x_bufs = gl.allocate_shared_memory(gl.uint8, [STAGES, BM, KB], gl.NVMMASharedLayout.get_default_for([BM, KB], gl.uint8))
    w_bufs = gl.allocate_shared_memory(gl.uint8, [STAGES, BN, KB], w_desc.layout)
    sw_bufs = gl.allocate_shared_memory(gl.uint8, [2, BN, 16], SCALE_SMEM_LAYOUT)
    sx_bufs = gl.allocate_shared_memory(gl.uint8, [2, BM, 16], SCALE_SMEM_LAYOUT)
    # n-contiguous and swizzled, so the epilogue's 8-row reads and the MMA-layout stores spread over banks
    h_buf = gl.allocate_shared_memory(gl.bfloat16, [BN // 2, BM], gl.SwizzledSharedLayout(8, 1, 8, [0, 1]))
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    s_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_ready = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    acc_empty = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(empty.index(i), count=1)
    for i in gl.static_range(2):
        mbarrier.init(s_ready.index(i), count=1)
        mbarrier.init(s_empty.index(i), count=1)
    mbarrier.init(acc_ready.index(0), count=1)
    mbarrier.init(acc_empty.index(0), count=1)

    NW: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[MMA_WARPS_N, NW // MMA_WARPS_N],
                                                  instr_shape=[16, 8])
    gl.warp_specialize([
        (_down_mma_partition, ((x_bufs, w_bufs, sw_bufs, sx_bufs, h_buf), (ready, empty, s_ready, s_empty, acc_ready, acc_empty),
                               sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr, wg_ptr, g_in_ptr,
                               topk_weights_ptr, N, nk, num_valid_tokens, stride_ge, stride_tw, mma, wsl, xsl, K_TAIL,
                               N_GROUP, GROUP_SIZE_M)),
        (_down_load_partition, (xq_ptr, xs_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr, w_desc,
                                ws_ptr, (x_bufs, w_bufs, sw_bufs, sx_bufs, h_buf),
                                (ready, empty, s_ready, s_empty, acc_ready, acc_empty), N, nk, num_valid_tokens,
                                stride_xq, ks4, n_rows, K_TAIL, N_GROUP, GROUP_SIZE_M)),
        (_down_epi_partition, (sorted_token_ids_ptr, num_tokens_post_padded_ptr, out_ptr,
                               (x_bufs, w_bufs, sw_bufs, sx_bufs, h_buf), (ready, empty, s_ready, s_empty, acc_ready, acc_empty),
                               N, num_valid_tokens, stride_out, top_k, N_GROUP, GROUP_SIZE_M)),
    ], [LOAD_WARPS, EPI_WARPS], [LOAD_REGS, EPI_REGS])


def supports(H: int, I: int) -> bool:
    """Whether the GEMMs take a layer of this hidden / intermediate size."""
    # K is a whole number of 64-wide k blocks for both GEMMs; the gated tile pairs 64 gate rows with 64 up rows at least
    return H % 64 == 0 and I % 64 == 0


def _weight_desc(packed, box_n, BK):
    S, N, KB = packed.shape
    layout = gl.NVMMASharedLayout.get_default_for([box_n, BK // 2], gl.uint8)
    return TensorDescriptor.from_tensor(packed.view(S * N, KB), [box_n, BK // 2], layout)


def _scale_layouts(cfg, w_rows):
    nw, wn = cfg["num_warps"], cfg["MMA_WARPS_N"]
    kg = cfg["BLOCK_SIZE_K"] // 16
    return scale_layout(0, w_rows, kg, [wn, nw // wn]), scale_layout(1, cfg["BLOCK_SIZE_M"], kg, [wn, nw // wn])


def gate_up_gemm(xq, xs, g_in, packed, scale, glob, q, qs, g_out, sorted_ids, expert_ids, num_tokens_post_padded,
                 num_valid_tokens, top_k, activation, act_alpha, act_limit, cfg):
    """q, qs = NVFP4 under g_out of act(gate, up), where [gate; up] = (xq * xs * g_in)[route // top_k] W4A4 packed[expert]^T * glob."""
    S, N, KB = packed.shape
    K = KB * 2
    BM, BN, BK = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
    assert K % BK == 0 and (N // 2) % (BN // 2) == 0 and xs.stride(0) == K // 16
    assert q.stride(0) == N // 4 and qs.stride(0) == N // 32
    wsl, xsl = _scale_layouts(cfg, BN)
    grid = triton.cdiv(sorted_ids.shape[0], BM) * triton.cdiv(N, BN)
    _w4a4_moe_kernel[(grid,)](
        xq, xs.view(torch.uint8), g_in, _weight_desc(packed, BN // 2, BK), scale.view(torch.uint8), glob, None, None,
        sorted_ids, expert_ids, num_tokens_post_padded, q, qs.view(torch.uint8), g_out, act_alpha, act_limit,
        N, K, sorted_ids.shape[0], num_valid_tokens, K // 64, S * N, xq.stride(0), glob.stride(0), 0, 0,
        K_TAIL=K % 256 != 0, GATED=True, ACT=ACTIVATIONS[activation], top_k=top_k, wsl=wsl, xsl=xsl, **cfg,
    )


def down_gemm(xq, xs, g_in, packed, scale, glob, out, topk_weights_flat, sorted_ids, expert_ids, num_tokens_post_padded,
              num_valid_tokens, top_k, cfg):
    """out[route // top_k] += (xq * xs * g_in)[route] W4A4 packed[expert]^T * glob * topk_weights_flat[route]; out starts zeroed."""
    S, N, KB = packed.shape
    K = KB * 2
    cfg = dict(cfg)
    persistent = cfg.pop("PERSISTENT")
    BM, BN, BK = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
    assert K % BK == 0 and xs.stride(0) == K // 16
    args = (xq, xs.view(torch.uint8), g_in, _weight_desc(packed, BN, BK), scale.view(torch.uint8), glob, out,
            topk_weights_flat, sorted_ids, expert_ids, num_tokens_post_padded)
    if persistent:
        wsl, xsl = _scale_layouts(cfg, BN // 2)
        grid = torch.cuda.get_device_properties(out.device).multi_processor_count
        _w4a4_down_persistent_kernel[(grid,)](
            *args, N, K, num_valid_tokens, K // 64, S * N, xq.stride(0), glob.stride(0), out.stride(0),
            topk_weights_flat.stride(0), K_TAIL=K % 256 != 0, top_k=top_k, wsl=wsl, xsl=xsl, **cfg,
        )
        return
    wsl, xsl = _scale_layouts(cfg, BN)
    grid = triton.cdiv(sorted_ids.shape[0], BM) * triton.cdiv(N, BN)
    _w4a4_moe_kernel[(grid,)](
        *args, None, None, None, 1.0, 1.0,
        N, K, sorted_ids.shape[0], num_valid_tokens, K // 64, S * N, xq.stride(0), glob.stride(0), out.stride(0),
        topk_weights_flat.stride(0), K_TAIL=K % 256 != 0, GATED=False, ACT=0, top_k=top_k, wsl=wsl, xsl=xsl, **cfg,
    )
