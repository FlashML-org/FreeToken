"""Fused KDA gate math: one elementwise kernel replaces the ~7-kernel fp32 chain
(float cast, +dt_bias, exp/sigmoid/mul for g, sigmoid for beta).

    g[t,h,j] = lower_bound * sigmoid(decay[h] * (raw[t, h*K+j] + dt_bias[h*K+j]))
    beta[t,h] = sigmoid(b_raw[t,h])

Same fp32 math as the eager chain (``decay = exp(A_log)`` precomputed once by the
caller); only numerics delta is libdevice-vs-ATen sigmoid rounding (<=1 ulp).
CUDA-graph safe: fixed shapes, no host sync."""
import torch
import triton
import triton.language as tl


@triton.jit
def _kda_gate_kernel(
    raw_ptr, braw_ptr, dtb_ptr, decay_ptr, g_ptr, beta_ptr,
    stride_bt,
    lower_bound,
    H: tl.constexpr, K: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per (token, head)
    t = pid // H
    h = pid % H
    offs = h * K + tl.arange(0, K)
    raw = tl.load(raw_ptr + t * H * K + offs).to(tl.float32)
    dtb = tl.load(dtb_ptr + offs)
    dec = tl.load(decay_ptr + h)
    g = lower_bound * tl.sigmoid(dec * (raw + dtb))
    tl.store(g_ptr + t * H * K + offs, g)
    braw = tl.load(braw_ptr + t * stride_bt + h).to(tl.float32)
    tl.store(beta_ptr + t * H + h, tl.sigmoid(braw))


def kda_gate(
    raw: torch.Tensor,      # [total, H*K] bf16, contiguous
    b_raw: torch.Tensor,    # [total, H] bf16 (row-strided slice ok)
    dt_bias: torch.Tensor,  # [H*K] fp32
    decay: torch.Tensor,    # [H] fp32 = exp(A_log)
    lower_bound: float,
):
    total, qkv = raw.shape
    H = b_raw.shape[1]
    K = qkv // H
    g = torch.empty(total, H, K, dtype=torch.float32, device=raw.device)
    beta = torch.empty(total, H, dtype=torch.float32, device=raw.device)
    _kda_gate_kernel[(total * H,)](
        raw, b_raw, dt_bias, decay, g, beta,
        b_raw.stride(0), lower_bound, H=H, K=K,
        num_warps=1,
    )
    return g, beta
