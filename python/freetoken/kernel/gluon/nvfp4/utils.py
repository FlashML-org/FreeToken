"""NVFP4 helpers shared by the Gluon kernels."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def bf16x2_mul(a, b):
    # sm_90+; sm_80-89 need fma.rn.bf16x2 with a -0.0 addend
    return gl.inline_asm_elementwise("mul.rn.bf16x2 $0, $1, $2;", "=r,r,r", [a, b], gl.uint32, True, 1)


@gluon.jit
def prmt(a, b, sel: gl.constexpr):
    return gl.inline_asm_elementwise(f"prmt.b32 $0, $1, $2, {sel};", "=r,r,r", [a, b], gl.uint32, True, 1)


@gluon.jit
def scale_bf16x2(s):
    """e4m3 scale bytes -> bf16x2 words holding scale * 2^118 twice."""
    # 2^118 * the 2^-126 of e2m1x8_bf16x2 leaves 2^-8 for the caller to undo; every step is exact in bf16
    s = (s.to(gl.float8e4nv, bitcast=True).to(gl.float32) * 3.3230699894622794e35).to(gl.bfloat16)
    s = s.to(gl.uint16, bitcast=True).to(gl.uint32)
    return s | (s << 16)


@gluon.jit
def e2m1x8_bf16x2(q, s2):
    """8 e2m1 codes in a uint32 -> 4 bf16x2 words in k order, each value times the bf16x2 scale s2 and 2^-126."""
    # each 16-bit half's top nibble lands in a bf16 as sign + exponent/mantissa bits, a value of e2m1 * 2^-126
    o0 = (q & 0x80008000) | ((q >> 6) & 0x01C001C0)
    q1 = q << 4
    o1 = (q1 & 0x80008000) | ((q1 >> 6) & 0x01C001C0)
    q2 = q << 8
    o2 = (q2 & 0x80008000) | ((q2 >> 6) & 0x01C001C0)
    q3 = q << 12
    o3 = (q3 & 0x80008000) | ((q3 >> 6) & 0x01C001C0)
    # o0..o3 hold the k pairs (3, 7), (2, 6), (1, 5), (0, 4)
    w0 = bf16x2_mul(prmt(o3, o2, 0x5410), s2)
    w1 = bf16x2_mul(prmt(o1, o0, 0x5410), s2)
    w2 = bf16x2_mul(prmt(o3, o2, 0x7632), s2)
    w3 = bf16x2_mul(prmt(o1, o0, 0x7632), s2)
    return w0, w1, w2, w3
