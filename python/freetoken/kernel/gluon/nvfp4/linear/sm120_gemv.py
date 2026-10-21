"""NVFP4 W4A16 GEMV and skinny GEMM for sm_120 on the row-major weight: [N, K/2] e2m1 codes, flat 128x4 e4m3 block scales, one fp32 global.

:func:`gemv_fma` decodes each code word once on the CUDA cores and reuses it for every token; :func:`gemv_mma` runs bf16 mma.sync with the
weight rows on M and the tokens on N. Both keep K a constexpr with the K loop unrolled and load the activation with evict_last, so it
stays in L2 while the weight stream passes; the weight keeps the default policy, as evict_first on it slows the kernels after it in a
model step (lines it would have evicted stay in L2 for them to evict). :func:`gemv_config` picks the kernel and tile for a shape.
"""

import functools

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from ..utils import e2m1x8_bf16x2, scale_bf16x2

# one code word (k = 8w + 0..7) times its f16x2 scale pair -> 8 fp32; e2m1 times e4m3 is exact in f16
_DECODE = gl.constexpr("""
{
.reg .b8 b0, b1, b2, b3;
.reg .b32 h0, h1, h2, h3;
.reg .b16 l0, l1, l2, l3, l4, l5, l6, l7;
mov.b32 {b0, b1, b2, b3}, $8;
cvt.rn.f16x2.e2m1x2 h0, b0;
cvt.rn.f16x2.e2m1x2 h1, b1;
cvt.rn.f16x2.e2m1x2 h2, b2;
cvt.rn.f16x2.e2m1x2 h3, b3;
mul.rn.f16x2 h0, h0, $9;
mul.rn.f16x2 h1, h1, $9;
mul.rn.f16x2 h2, h2, $9;
mul.rn.f16x2 h3, h3, $9;
mov.b32 {l0, l1}, h0;
mov.b32 {l2, l3}, h1;
mov.b32 {l4, l5}, h2;
mov.b32 {l6, l7}, h3;
cvt.f32.f16 $0, l0;
cvt.f32.f16 $1, l1;
cvt.f32.f16 $2, l2;
cvt.f32.f16 $3, l3;
cvt.f32.f16 $4, l4;
cvt.f32.f16 $5, l5;
cvt.f32.f16 $6, l6;
cvt.f32.f16 $7, l7;
}
""")


@gluon.jit
def _scale_row(row, K_TILES: gl.constexpr):
    # 128x4 atoms: [row tile, k tile, row % 32, (row % 128) // 32, group % 4]
    return (row // 128).to(gl.int64) * (K_TILES * 512) + (row % 32) * 16 + (row % 128) // 32 * 4


@gluon.jit
def _gemv_fma_kernel(x_ptr, w_ptr, s_ptr, g_ptr, out_ptr, T, stride_x, stride_o,
                     K: gl.constexpr, T_C: gl.constexpr, R: gl.constexpr, V: gl.constexpr):
    W: gl.constexpr = gl.num_warps()
    BN: gl.constexpr = R * W
    BKW: gl.constexpr = 32 * V
    KW: gl.constexpr = K // 8
    K_TILES: gl.constexpr = (K // 16 + 3) // 4
    # a warp owns R whole rows; each lane holds V code words and their 8 k per row and per token
    L4: gl.constexpr = gl.BlockedLayout([T_C, R, V, 8], [1, 1, 32, 1], [1, W, 1, 1], [3, 2, 1, 0])
    La: gl.constexpr = gl.SliceLayout(3, L4)
    Lw: gl.constexpr = gl.SliceLayout(0, La)
    Lx: gl.constexpr = gl.SliceLayout(1, L4)
    Lxj: gl.constexpr = gl.SliceLayout(1, La)

    # 4 consecutive CTA rows are rows r, r + 32, r + 64, r + 96: one 16-byte chunk of every 128x4 scale atom
    flat = gl.program_id(0) * BN + gl.arange(0, BN, layout=gl.SliceLayout(1, Lw))
    row = (flat // 128) * 128 + (flat % 128) // 4 + (flat % 4) * 32
    wcol = gl.arange(0, BKW, layout=gl.SliceLayout(0, Lw))
    w_base = w_ptr + row.to(gl.int64)[:, None] * KW + wcol[None, :]
    s_row = _scale_row(row, K_TILES)
    xt = gl.arange(0, T_C, layout=gl.SliceLayout(1, gl.SliceLayout(2, Lx)))
    xw = gl.arange(0, BKW, layout=gl.SliceLayout(0, gl.SliceLayout(2, Lx)))
    xj = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, Lx)))
    x_off = xt[:, None, None] * stride_x + (xw * 8)[None, :, None] + xj[None, None, :]
    x_ok = (xt < T)[:, None, None]

    acc = gl.zeros([T_C, BN, BKW], gl.float32, La)
    for ph in gl.static_range(KW // BKW):
        words = gl.load(w_base + ph * BKW)
        grp = (ph * BKW + wcol) // 2
        sc = gl.load(s_ptr + s_row[:, None] + (grp // 4 * 512 + grp % 4)[None, :])
        s16 = sc.to(gl.float8e4nv, bitcast=True).to(gl.float16).to(gl.uint16, bitcast=True).to(gl.uint32)
        s2 = (s16 | (s16 << 16)).to(gl.int32, bitcast=True)
        v0, v1, v2, v3, v4, v5, v6, v7 = gl.inline_asm_elementwise(_DECODE, "=r,=r,=r,=r,=r,=r,=r,=r,r,r", [words, s2],
                                                                   dtype=(gl.float32,) * 8, is_pure=True, pack=1)
        xa = gl.load(x_ptr + ph * BKW * 8 + x_off, mask=x_ok, other=0.0, eviction_policy="evict_last").to(gl.float32)
        # in-thread split of the 8 k of each word: x0..x7 line up with v0..v7
        ev, od = gl.split(gl.reshape(xa, [T_C, BKW, 4, 2]))
        e0, e1 = gl.split(gl.reshape(ev, [T_C, BKW, 2, 2]))
        o0, o1 = gl.split(gl.reshape(od, [T_C, BKW, 2, 2]))
        x0, x4 = gl.split(e0)
        x2, x6 = gl.split(e1)
        x1, x5 = gl.split(o0)
        x3, x7 = gl.split(o1)
        acc += v0[None] * gl.convert_layout(x0, Lxj)[:, None, :]
        acc += v1[None] * gl.convert_layout(x1, Lxj)[:, None, :]
        acc += v2[None] * gl.convert_layout(x2, Lxj)[:, None, :]
        acc += v3[None] * gl.convert_layout(x3, Lxj)[:, None, :]
        acc += v4[None] * gl.convert_layout(x4, Lxj)[:, None, :]
        acc += v5[None] * gl.convert_layout(x5, Lxj)[:, None, :]
        acc += v6[None] * gl.convert_layout(x6, Lxj)[:, None, :]
        acc += v7[None] * gl.convert_layout(x7, Lxj)[:, None, :]
    res = gl.sum(acc, axis=2) * gl.load(g_ptr)
    Lo: gl.constexpr = gl.SliceLayout(2, La)
    ot = gl.arange(0, T_C, layout=gl.SliceLayout(1, Lo))
    orow = gl.convert_layout(row, gl.SliceLayout(0, Lo))
    gl.store(out_ptr + ot[:, None] * stride_o + orow[None, :], res.to(out_ptr.dtype.element_ty), mask=(ot < T)[:, None])


@gluon.jit
def _to_bf16(q):
    lo = (q & 0xFFFF).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    hi = (q >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    return gl.join(lo, hi)


@gluon.jit
def _kperm(t, A: gl.constexpr, B: gl.constexpr):
    # k_phys = 32 lid + 4 s + 2 h + e -> k_mma = 16 s + 8 h + 2 lid + e: a lane's contiguous 32 k become its fragment slots
    t = gl.reshape(t, [A, B, 4, 8, 2, 2])
    t = gl.permute(t, (0, 1, 3, 4, 2, 5))
    return gl.reshape(t, [A, B, 128])


@gluon.jit
def _gemv_mma_kernel(x_ptr, w_ptr, s_ptr, g_ptr, out_ptr, T, stride_x, stride_o,
                     K: gl.constexpr, T_C: gl.constexpr, BM: gl.constexpr, Lc: gl.constexpr, Lx: gl.constexpr, mma: gl.constexpr):
    KW: gl.constexpr = gl.num_warps()
    NB: gl.constexpr = K // 128
    K_TILES: gl.constexpr = (K // 16 + 3) // 4
    al: gl.constexpr = gl.DotOperandLayout(0, mma, 2)
    bl: gl.constexpr = gl.DotOperandLayout(1, mma, 2)

    # batched [KW, BM, T_C] mma: warp w takes the 128-k blocks w, w + KW, ...; the warps' partials are summed at the end
    row0 = gl.program_id(0) * BM
    cw = gl.arange(0, KW, layout=gl.SliceLayout(1, gl.SliceLayout(2, Lc)))
    cm = gl.arange(0, BM, layout=gl.SliceLayout(0, gl.SliceLayout(2, Lc)))
    cj = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, Lc)))
    row = row0 + cm
    c_off = (row.to(gl.int64) * (K // 8))[None, :, None] + (cw * 16)[:, None, None] + cj[None, None, :]
    grp = cw[:, None, None] * 8 + (cj // 2)[None, None, :]
    s_off = _scale_row(row, K_TILES)[None, :, None] + (grp // 4 * 512 + grp % 4)
    xw = gl.arange(0, KW, layout=gl.SliceLayout(1, gl.SliceLayout(2, Lx)))
    xt = gl.arange(0, T_C, layout=gl.SliceLayout(0, gl.SliceLayout(2, Lx)))
    xk = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(1, Lx)))
    x_off = (xt * stride_x)[None, :, None] + (xw * 128)[:, None, None] + xk[None, None, :]
    x_ok = (xt < T)[None, :, None]

    acc = gl.zeros([KW, BM, T_C], gl.float32, mma)
    for it in gl.static_range((NB + KW - 1) // KW):
        b0 = it * KW
        if (it + 1) * KW <= NB:
            q = gl.load(w_ptr + c_off + b0 * 16)
            sc = gl.load(s_ptr + s_off + b0 * 1024)
            xa = gl.load(x_ptr + x_off + b0 * 128, mask=x_ok, other=0.0, eviction_policy="evict_last")
        else:
            ok = (cw + b0 < NB)[:, None, None]
            q = gl.load(w_ptr + c_off + b0 * 16, mask=ok, other=0)
            sc = gl.load(s_ptr + s_off + b0 * 1024, mask=ok, other=0)
            xa = gl.load(x_ptr + x_off + b0 * 128, mask=x_ok & (xw + b0 < NB)[:, None, None], other=0.0, eviction_policy="evict_last")
        w0, w1, w2, w3 = e2m1x8_bf16x2(q.to(gl.uint32, bitcast=True), scale_bf16x2(sc))
        # join appends its axis last: this order flattens to the k pairs w0, w1, w2, w3
        p = gl.reshape(gl.join(gl.join(w0, w2), gl.join(w1, w3)), [KW, BM, 64])
        a = gl.convert_layout(_kperm(gl.reshape(_to_bf16(p), [KW, BM, 128]), KW, BM), al)
        b = gl.convert_layout(gl.permute(_kperm(xa, KW, T_C), (0, 2, 1)), bl)
        acc = mma_v2(a, b, acc)
    # e2m1x8_bf16x2 leaves a factor 2^-8 in the decoded weight
    res = gl.sum(acc, axis=0) * (gl.load(g_ptr) * 256.0)
    Lo: gl.constexpr = gl.SliceLayout(0, mma)
    om = gl.arange(0, BM, layout=gl.SliceLayout(1, Lo))
    ot = gl.arange(0, T_C, layout=gl.SliceLayout(0, Lo))
    gl.store(out_ptr + ot[None, :] * stride_o + (row0 + om)[:, None], res.to(out_ptr.dtype.element_ty), mask=(ot < T)[None, :])


@functools.cache
def _mma_layouts(kw: int, t_c: int, bm: int):
    warps = [[1 << i, 0, 0] for i in range(kw.bit_length() - 1)]
    lanes = [[0, 0, 32], [0, 0, 64], [0, 1, 0], [0, 2, 0], [0, 4, 0]]
    # codes [KW, BM, 16 words]: lane (gid, lid) loads words 4 lid .. 4 lid + 3 of rows gid, gid + 8, gid + 16, ...
    codes = gl.DistributedLinearLayout([[0, 0, 1], [0, 0, 2], [0, 8, 0]] + [[0, 16 << i, 0] for i in range(bm.bit_length() - 5)],
                                       [[0, 0, 4], [0, 0, 8], [0, 1, 0], [0, 2, 0], [0, 4, 0]], warps, [], [kw, bm, 16])
    # activations [KW, T_C, 128]: lane (gid, lid) loads k 32 lid .. 32 lid + 31 of tokens gid, gid + 8
    acts = gl.DistributedLinearLayout([[0, 0, 1 << i] for i in range(5)] + [[0, 8 << i, 0] for i in range(t_c.bit_length() - 4)],
                                      lanes, warps, [], [kw, t_c, 128])
    return codes, acts, gl.NVMMADistributedLayout([2, 0], [kw, 1, 1], [1, 16, 8])


def gemv_fma(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, g: torch.Tensor, R: int, V: int, num_warps: int) -> torch.Tensor:
    """x [T, K] bf16 row-contiguous, weight [N, K/2] uint8, scale flat 128x4 e4m3, g fp32 -> [T, N]; see :func:`gemv_config` for the shape rules."""
    T, K = x.shape
    N = weight.shape[0]
    out = torch.empty(T, N, dtype=x.dtype, device=x.device)
    _gemv_fma_kernel[(N // (R * num_warps),)](x, weight.view(torch.int32), scale.view(torch.uint8), g, out, T, x.stride(0), out.stride(0),
                                              K=K, T_C=triton.next_power_of_2(T), R=R, V=V, num_warps=num_warps)
    return out


def gemv_mma(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, g: torch.Tensor, bm: int, num_warps: int) -> torch.Tensor:
    """Same contract as :func:`gemv_fma`; T <= 16."""
    T, K = x.shape
    N = weight.shape[0]
    t_c = max(8, triton.next_power_of_2(T))
    codes, acts, mma = _mma_layouts(num_warps, t_c, bm)
    out = torch.empty(T, N, dtype=x.dtype, device=x.device)
    _gemv_mma_kernel[(N // bm,)](x, weight.view(torch.int32), scale.view(torch.uint8), g, out, T, x.stride(0), out.stride(0),
                                 K=K, T_C=t_c, BM=bm, Lc=codes, Lx=acts, mma=mma, num_warps=num_warps)
    return out


def gemv_config(N: int, K: int, T: int, max_tokens: int = 16):
    """(kernel, tile) for a [T, K] x [N, K] product, or None when neither kernel covers it (T > max_tokens <= 16, or an unsupported shape)."""
    if T == 1 and K < 16384 or T == 2 and N >= 16384:
        words = K // 8
        if N % 128 == 0:
            # wide layers take more rows per CTA; narrow ones longer words per lane
            if N >= 16384 and words % 64 == 0:
                return gemv_fma, {"R": 4, "V": 2, "num_warps": 4}
            for V, warps in ((4, 4), (2, 8), (1, 8)):
                if words % (32 * V) == 0 and not (N >= 16384 and V == 4):
                    return gemv_fma, {"R": 2, "V": V, "num_warps": warps}
    if T > max_tokens or K % 128:
        return None
    tile = {"bm": 16, "num_warps": 8} if T <= 8 or N % 32 else {"bm": 32, "num_warps": 8}
    return (gemv_mma, tile) if N % tile["bm"] == 0 else None
