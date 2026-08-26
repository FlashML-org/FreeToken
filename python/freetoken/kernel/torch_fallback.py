"""Pure-torch fallbacks for the triton/flashinfer/sgl_kernel layers.

FreeToken's ``layers/*`` and ``models/*`` kernels are GPU-only (triton / flashinfer /
sgl_kernel). When those packages are absent (CPU-only build, no NVIDIA card), the engine
needs functionally-identical pure-torch implementations so it can still load and run a
model. These mirror the numerical contract of the upstream kernels (RMSNorm with eps,
fused add+residual RMSNorm, SwiGLU/silu_and_mul, GELU, and RoPE applied via a cos/sin
cache). They are correctness-focused, not speed-optimized.

Selected by ``layers/norm.py``, ``layers/activation.py`` and ``layers/rotary.py`` only when
the GPU kernel package is unavailable, so the CUDA path is byte-for-byte unchanged.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------- #
# RMSNorm family
# --------------------------------------------------------------------------- #
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor | None = None) -> torch.Tensor:
    # x: [..., hidden]; weight: [hidden]
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    out_t = (xf * weight.to(torch.float32)).to(orig_dtype)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t


def fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    # in-place: residual += x; x = rmsnorm(residual)  (match triton/flashinfer:
    # normalize the accumulated sum, not the local input)
    residual.copy_(residual + x)
    x.copy_(rmsnorm(residual, weight, eps))


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor | None = None) -> torch.Tensor:
    return rmsnorm(x, weight, eps, out=out)


def gemma_fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    fused_add_rmsnorm(x, residual, weight, eps)


# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #
def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    # last dim split in half: gate, up
    a, b = x.chunk(2, dim=-1)
    res = torch.nn.functional.silu(a) * b
    if out is not None:
        out.copy_(res)
        return out
    return res


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    res = torch.nn.functional.gelu(a) * b
    if out is not None:
        out.copy_(res)
        return out
    return res


def gelu_tanh_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    res = torch.nn.functional.gelu(b, approximate="tanh") * a
    if out is not None:
        out.copy_(res)
        return out
    return res


def swigluoai_and_mul(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(a) * b


# --------------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------------- #
def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    # cos_sin_cache layout: [max_pos, rotary_dim] = [cos_block | sin_block], where
    # each block has length rotary_dim//2. The cache WIDTH is the true rotary_dim
    # (NOT 2*rotary_dim): rotary.py builds it as torch.cat((cos, sin)) with cos/sin
    # each of length rotary_dim//2. rotary_dim may equal head_size (full rope) or be
    # < head_size (partial rope).
    rotary_dim = cos_sin_cache.shape[-1]

    def _rope(x: torch.Tensor) -> torch.Tensor:
        # x may arrive as:
        #   (a) [num_tokens, head_size]           single head, OR
        #   (b) [num_tokens, num_heads*head_size] multi-head concat (engine path), OR
        #   (c) [num_tokens, num_heads, head_size] already 3-D.
        # RoPE is per-head: rotate the first `rotary_dim` dims of EACH head.
        # Normalize everything to [num_tokens*num_heads, head_size] so the rotation
        # below always sees one head per row.
        orig_shape = x.shape
        if x.dim() == 3:
            num_tokens, num_heads, hs = x.shape
            x = x.reshape(num_tokens * num_heads, hs)
        elif x.dim() == 2:
            num_tokens, width = x.shape
            if width % rotary_dim == 0 and width > rotary_dim:
                # multi-head concat: (tokens, num_heads*head_size)
                num_heads = width // rotary_dim
                x = x.reshape(num_tokens * num_heads, rotary_dim)
            else:
                # single head: (tokens, head_size)
                x = x.reshape(num_tokens, rotary_dim)
        else:
            raise ValueError(f"unexpected query rank {x.dim()}")

        cos = cos_sin_cache[positions, : rotary_dim // 2]  # [num_tokens, rotary_dim//2]
        sin = cos_sin_cache[positions, rotary_dim // 2 :]  # [num_tokens, rotary_dim//2]
        # Expand per-token cos/sin across the flattened heads (each head at a given
        # token position uses the same position's cos/sin). Result stays 2-D
        # [num_rows, rotary_dim//2] so it broadcasts cleanly against the 2-D x below.
        num_tokens = orig_shape[0]
        reps = x.shape[0] // num_tokens
        cos = cos.repeat_interleave(reps, dim=0)  # [num_rows, rotary_dim//2]
        sin = sin.repeat_interleave(reps, dim=0)

        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]

        half = rotary_dim // 2
        if is_neox:
            # NeoX: rotate the pair (i, i+rotary_dim/2) within the rotary block.
            x1 = x_rot[..., :half]
            x2 = x_rot[..., half:]
            rot = torch.cat([-x2, x1], dim=-1)
        else:
            # GPT-J interleaved: rotate adjacent pairs (0,1),(2,3),...
            rot = torch.stack([-x_rot[..., 1::2], x_rot[..., 0::2]], dim=-1).flatten(-2, -1)
        # x_rot and cos/sin are both 2-D [num_rows, rotary_dim]; duplicate the
        # half-width cos/sin across both sides of each pair so the elementwise
        # multiply lines up with x_rot (full rotary_dim). NO unsqueeze -> no phantom dim.
        c_full = torch.cat([cos, cos], dim=-1)  # [num_rows, rotary_dim]
        s_full = torch.cat([sin, sin], dim=-1)
        x_rotated = x_rot * c_full + rot * s_full

        if x_pass.shape[-1] > 0:
            out = torch.cat([x_rotated, x_pass], dim=-1)
        else:
            out = x_rotated

        # restore original shape
        return out.reshape(orig_shape)

    query.copy_(_rope(query))
    key.copy_(_rope(key))
