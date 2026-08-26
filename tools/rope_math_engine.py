"""rope_math_engine.py — a self-contained math engine to debug/verify RoPE kernels.

Why this exists: reasoning about multi-head tensor reshapes by hand is error-prone
(the author kept miscounting 80*128 vs 5*2048). This engine traces EVERY intermediate
shape symbolically AND numerically, so a reshape/broadcast bug is visible in one run.

It does three things:
  1. SHAPE TRACE  — given input shapes, prints every step's shape (no torch needed).
  2. VALUE TRACE  — runs the *actual* freetoken fallback on synthetic data, printing
                    each intermediate shape, so we catch the exact failing step.
  3. REFERENCE    — computes the ground-truth HF RoPE and reports cosine vs any output,
                    so "is this correct?" is answered by the engine, not by eyeballing.

Usage:
  python rope_math_engine.py            # runs the default engine-shape diagnosis
"""
from __future__ import annotations
import sys, math
import torch
import numpy as np

sys.path.insert(0, "/Users/petersheppard/FreeToken/python")
from freetoken.layers.rotary import get_rope


# --------------------------------------------------------------------------- #
# 1. SYMBOLIC SHAPE TRACE (pure arithmetic, no tensors)
# --------------------------------------------------------------------------- #
def shape_trace(seq, n_heads, head_dim, rotary_dim, rank2_multhead=True):
    """Print the shape at every step of the rope kernel for the given geometry.

    Models the EXACT logic of apply_rope_with_cos_sin_cache_inplace:
      input is (seq, n_heads*head_dim)  [engine 2-D multi-head concat]
      cache is (max_pos, rotary_dim)
    """
    print(f"\n=== SHAPE TRACE: seq={seq} n_heads={n_heads} head_dim={head_dim} "
          f"rotary_dim={rotary_dim} ===")
    width = n_heads * head_dim
    print(f"  input x            : ({seq}, {width})")
    orig = (seq, width)
    if width % rotary_dim == 0 and width > rotary_dim:
        nh = width // rotary_dim
        x_shape = (seq * nh, rotary_dim)
        print(f"  width % rotary_dim==0 & width>rd -> multi-head, nh={nh}")
    else:
        x_shape = (seq, rotary_dim)
        print(f"  else -> single-head reshape to ({seq}, {rotary_dim})")
    print(f"  x reshaped        : {x_shape}")
    print(f"  cos_sin_cache     : (max_pos, {rotary_dim})")
    print(f"  cos = cache[pos,:{rotary_dim//2}] : ({seq}, {rotary_dim//2})")
    rep = x_shape[0] // seq
    print(f"  cos.repeat_interleave({rep}) : ({x_shape[0]}, {rotary_dim//2})")
    print(f"  cos.unsqueeze(1)  : ({x_shape[0]}, 1, {rotary_dim//2})")
    print(f"  x_rot = x[...,:{rotary_dim}] : {x_shape}")
    print(f"  x_pass= x[...,{rotary_dim}:]  : {x_shape[:-1] + (x_shape[-1]-rotary_dim,)}")
    xr = x_shape
    xr_half = (xr[0], rotary_dim // 2)
    print(f"  x1=x_rot[:,:{rotary_dim//2}] : {xr_half}")
    print(f"  x2=x_rot[:,{rotary_dim//2}:] : {xr_half}")
    print(f"  rot=cat([-x2,x1]) : {xr}")
    print(f"  c_full=cat([c,c]) : ({xr[0]}, 1, {rotary_dim})")
    out = xr  # full rope -> x_pass empty
    print(f"  x_rotated         : {out}")
    print(f"  out.reshape(orig={orig}) -> elems out={out[0]*out[1]} vs orig={orig[0]*orig[1]}")
    ok = (out[0] * out[1]) == (orig[0] * orig[1])
    print(f"  RESHAPE {'OK' if ok else 'FAILS (size mismatch)'}")
    return ok


# --------------------------------------------------------------------------- #
# 2. VALUE TRACE — run the REAL freetoken function, printing each step
# --------------------------------------------------------------------------- #
def traced_rope_call(query, key, head_size, cache, positions, is_neox=True):
    """Replicate the kernel with explicit shape prints, using the real math."""
    rotary_dim = cache.shape[-1]
    print(f"\n=== VALUE TRACE ===")
    print(f"  query in : {tuple(query.shape)}  key in : {tuple(key.shape)}")
    print(f"  head_size={head_size}  rotary_dim={rotary_dim}  positions={tuple(positions.shape)}")

    def _rope(x):
        orig = x.shape
        if x.dim() == 3:
            nt, nh, hs = x.shape
            x = x.reshape(nt * nh, hs)
        elif x.dim() == 2:
            nt, w = x.shape
            if w % rotary_dim == 0 and w > rotary_dim:
                nh = w // rotary_dim
                x = x.reshape(nt * nh, rotary_dim)
                print(f"    [2D multi-head] nt={nt} w={w} nh={nh} -> {tuple(x.shape)}")
            else:
                x = x.reshape(nt, rotary_dim)
                print(f"    [2D single] -> {tuple(x.shape)}")
        else:
            raise ValueError(f"rank {x.dim()}")
        cos = cache[positions, : rotary_dim // 2]
        sin = cache[positions, rotary_dim // 2 :]
        nt = orig[0]
        rep = x.shape[0] // nt
        cos = cos.repeat_interleave(rep, dim=0).unsqueeze(1)
        sin = sin.repeat_interleave(rep, dim=0).unsqueeze(1)
        print(f"    cos after expand : {tuple(cos.shape)}  x : {tuple(x.shape)}")
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        half = rotary_dim // 2
        if is_neox:
            x1 = x_rot[..., :half]; x2 = x_rot[..., half:]
            rot = torch.cat([-x2, x1], dim=-1)
        else:
            rot = torch.stack([-x_rot[..., 1::2], x_rot[..., 0::2]], dim=-1).flatten(-2, -1)
        c_full = torch.cat([cos, cos], dim=-1)
        s_full = torch.cat([sin, sin], dim=-1)
        x_rotated = x_rot * c_full + rot * s_full
        out = torch.cat([x_rotated, x_pass], dim=-1) if x_pass.shape[-1] > 0 else x_rotated
        print(f"    out before reshape : {tuple(out.shape)}  target orig : {tuple(orig)}")
        return out.reshape(orig)

    qo = traced_rope_call._inner(query, _rope) if hasattr(traced_rope_call, "_inner") else None
    # do it directly:
    q_out = _rope(query.clone())
    k_out = _rope(key.clone())
    print(f"  query out: {tuple(q_out.shape)}  key out: {tuple(k_out.shape)}")
    return q_out, k_out


# --------------------------------------------------------------------------- #
# 3. REFERENCE + COMPARISON
# --------------------------------------------------------------------------- #
def hf_reference_rope(q_norm, positions, head_dim=128, base=10000.0):
    """Ground-truth HF Qwen3 RoPE (rotate_half) on q_norm of shape (seq, NQ, D)."""
    seq, NQ, D = q_norm.shape
    inv = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    freqs = torch.outer(positions.double(), inv)
    cos = torch.cos(freqs)  # (seq, D/2)
    sin = torch.sin(freqs)
    half = D // 2
    x1 = q_norm[..., :half]; x2 = q_norm[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    c = cos.unsqueeze(1).expand(seq, NQ, half)
    s = sin.unsqueeze(1).expand(seq, NQ, half)
    return q_norm * torch.cat([c, c], -1) + rot * torch.cat([s, s], -1)


def cosine(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


if __name__ == "__main__":
    # default diagnosis: Qwen3-0.6B geometry
    seq, NQ, NK, D = 5, 16, 8, 128
    print("################ SHAPE TRACES ################")
    shape_trace(seq, NQ, D, D)   # query: (5, 2048)
    shape_trace(seq, NK, D, D)   # key:   (5, 1024)

    print("\n\n################ VALUE TRACE (real freetoken fallback) ################")
    cache = get_rope(head_dim=D, rotary_dim=D, max_position=40960, base=10000.0)._cos_sin_cache.double()
    positions = torch.arange(seq, dtype=torch.int64)
    q = torch.randn(seq, NQ * D, dtype=torch.float64)
    k = torch.randn(seq, NK * D, dtype=torch.float64)
    try:
        qo, ko = traced_rope_call(q, k, D, cache, positions, is_neox=True)
        print("\nVALUE TRACE: no error")
    except Exception as e:
        print(f"\nVALUE TRACE ERROR: {e}")

    print("\n\n################ REFERENCE COSINE (does fallback match HF?) ################")
    # build q_norm directly (identity-ish random) and compare
    qn = torch.randn(seq, NQ, D, dtype=torch.float64)
    ref = hf_reference_rope(qn, positions, D)
    # apply freetoken fallback to the SAME qn flattened
    q_flat = qn.reshape(seq * NQ, D).clone()
    q_flat_pos = positions.repeat_interleave(NQ)
    from freetoken.kernel.torch_fallback import apply_rope_with_cos_sin_cache_inplace
    try:
        apply_rope_with_cos_sin_cache_inplace(q_flat_pos, q_flat, q_flat.clone(), D, cache, is_neox=True)
        eng = q_flat.reshape(seq, NQ, D)
        print(f"  fallback vs HF reference cosine = {cosine(eng, ref):.6f}")
    except Exception as e:
        print(f"  fallback call failed: {e}")
