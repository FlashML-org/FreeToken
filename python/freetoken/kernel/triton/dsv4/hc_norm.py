"""Fused mHC pre-norm helpers (2026-08-28 tail-fusion pass).

Per hc_pre call the eager chain was: big fp32 cast -> square -> mean-reduce ->
rsqrt -> cublas gemv (dot_kernel + reduce_1Block) -> broadcast mul, i.e. ~7
launches x 90 calls/step. Here:

  * ``hc_rms_cast``  -- one kernel: bf16->fp32 copy (the gemv still needs fp32
    input) + sum-of-squares + rsqrt(mean+eps), one CTA per token (deterministic).
  * ``hc_mix_gemv``  -- [MIX, KDIM] fp32 gemv with the rsqrt scale folded into the
    epilogue; one CTA per (token, row), sequential k-loop (no atomics, so
    run-to-run deterministic like the cublas dot+reduce pair it replaces).

Numerics: identical inputs/ops in fp32; only reduction order differs from ATen /
cublas (epsilon-level). CUDA-graph safe: fixed shapes, no host sync."""
import torch
import triton
import triton.language as tl


@triton.jit
def _hc_rms_cast_kernel(
    x_ptr, xf_ptr, rs_ptr, D, eps,
    stride_xm, stride_fm,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, tl.cdiv(D, BLOCK)):
        offs = start * BLOCK + tl.arange(0, BLOCK)
        mask = offs < D
        v = tl.load(x_ptr + m * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(xf_ptr + m * stride_fm + offs, v, mask=mask)
        acc += v * v
    ssq = tl.sum(acc, axis=0)
    tl.store(rs_ptr + m, 1.0 / tl.sqrt(ssq / D + eps))


def hc_rms_cast(x: torch.Tensor, eps: float):
    """``x`` [M, D] (bf16/fp16/fp32) -> (``xf`` [M, D] fp32, ``rs`` [M] fp32)."""
    M, D = x.shape
    xf = torch.empty(M, D, dtype=torch.float32, device=x.device)
    rs = torch.empty(M, dtype=torch.float32, device=x.device)
    _hc_rms_cast_kernel[(M,)](
        x, xf, rs, D, eps, x.stride(0), xf.stride(0),
        BLOCK=1024, num_warps=4,
    )
    return xf, rs


@triton.jit
def _hc_mix_gemv_kernel(
    xf_ptr, w_ptr, rs_ptr, out_ptr, D,
    stride_fm, stride_wr, stride_om,
    MIX: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid // MIX
    r = pid - m * MIX
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, tl.cdiv(D, BLOCK)):
        offs = start * BLOCK + tl.arange(0, BLOCK)
        mask = offs < D
        xv = tl.load(xf_ptr + m * stride_fm + offs, mask=mask, other=0.0)
        wv = tl.load(w_ptr + r * stride_wr + offs, mask=mask, other=0.0)
        acc += xv * wv
    dot = tl.sum(acc, axis=0)
    rs = tl.load(rs_ptr + m)
    tl.store(out_ptr + m * stride_om + r, dot * rs)


def hc_mix_gemv(xf: torch.Tensor, w: torch.Tensor, rs: torch.Tensor) -> torch.Tensor:
    """``out[m, r] = rs[m] * dot(w[r], xf[m])``; ``w`` [MIX, D] fp32."""
    M, D = xf.shape
    MIX = w.shape[0]
    out = torch.empty(M, MIX, dtype=torch.float32, device=xf.device)
    _hc_mix_gemv_kernel[(M * MIX,)](
        xf, w, rs, out, D, xf.stride(0), w.stride(0), out.stride(0),
        MIX=MIX, BLOCK=2048, num_warps=8,
    )
    return out
