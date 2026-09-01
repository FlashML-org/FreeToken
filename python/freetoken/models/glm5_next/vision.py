"""GLM-5.3-Flash vision tower (text-side port of HF ``Glm5NextVisionModel``).

Faithful eager translation -- the tower runs once per image at prefill, so plain
torch ops are fine. Attribute names mirror the checkpoint (``visual.blocks.N.attn.qkv``
etc.), and every weight stays BF16, exactly like the source ``model.visual.*`` tensors.

Contract (called by ``LLM.encode_images`` / the server's mm path):
    encode_images(pixel_values, grid_thw) -> [num_image_tokens, out_hidden] bf16
      pixel_values: [total_patches, in_ch * temporal_patch * patch * patch]  (HF processor)
      grid_thw:     [num_images, 3] long (t, h, w) patch grid per image
Tokens per image = t * (h/merge) * (w/merge); rows are in image order, matching the
``image_token_id`` placeholder runs the processor writes into the prompt."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, OPList


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * xf.to(dt)) if weight.dtype == dt else (weight.to(dt) * xf.to(dt))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_vision(q, k, cos, sin):
    # cos/sin arrive pre-unsqueezed fp32 (built once per forward, not per block)
    dt = q.dtype
    q, k = q.float(), k.float()
    qe = (q * cos) + (_rotate_half(q) * sin)
    ke = (k * cos) + (_rotate_half(k) * sin)
    return qe.to(dt), ke.to(dt)


class _VLinear(BaseOP):
    def __init__(self, in_f: int, out_f: int, bias: bool):
        self.weight = torch.empty(out_f, in_f, dtype=torch.bfloat16)
        self.bias = torch.empty(out_f, dtype=torch.bfloat16) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class _VNorm(BaseOP):
    """RMSNorm (weight only) in HF fp32 semantics."""

    def __init__(self, dim: int, eps: float):
        self.weight = torch.empty(dim, dtype=torch.bfloat16)
        self._eps = eps

    def forward(self, x):
        return _rms(x, self.weight, self._eps)


class _VLayerNorm(BaseOP):
    def __init__(self, dim: int):
        self.weight = torch.empty(dim, dtype=torch.bfloat16)
        self.bias = torch.empty(dim, dtype=torch.bfloat16)

    def forward(self, x):
        return F.layer_norm(x.float(), self.weight.shape, self.weight.float(),
                            self.bias.float()).to(x.dtype)


class _VAttention(BaseOP):
    def __init__(self, vc):
        self.qkv = _VLinear(vc["hidden_size"], vc["hidden_size"] * 3, vc["attention_bias"])
        self.proj = _VLinear(vc["hidden_size"], vc["hidden_size"], vc["attention_bias"])
        self.q_norm = _VNorm(vc["hidden_size"] // vc["num_heads"], vc["rms_norm_eps"])
        self.k_norm = _VNorm(vc["hidden_size"] // vc["num_heads"], vc["rms_norm_eps"])
        self._heads = vc["num_heads"]
        self._scale = (vc["hidden_size"] // vc["num_heads"]) ** -0.5

    def forward(self, x, bounds, cos, sin):
        S = x.shape[0]
        q, k, v = self.qkv.forward(x).reshape(S, 3, self._heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = self.q_norm.forward(q)
        k = self.k_norm.forward(k)
        q, k = _apply_rope_vision(q, k, cos, sin)
        # per-image full attention (bounds delimits images), exact sdpa per chunk
        outs = []
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            qi = q[a:b].transpose(0, 1)  # [H, L, D]
            ki = k[a:b].transpose(0, 1)
            vi = v[a:b].transpose(0, 1)
            oi = F.scaled_dot_product_attention(qi, ki, vi, scale=self._scale)
            outs.append(oi.transpose(0, 1))
        out = torch.cat(outs, dim=0).reshape(S, -1)
        return self.proj.forward(out)


class _VMLP(BaseOP):
    def __init__(self, vc, dim: int, inter: int, bias: bool):
        self.gate_proj = _VLinear(dim, inter, bias)
        self.up_proj = _VLinear(dim, inter, bias)
        self.down_proj = _VLinear(inter, dim, bias)
        self._limit = vc["swiglu_limit"]

    def forward(self, x):
        gate = self.gate_proj.forward(x).clamp(max=self._limit)
        up = self.up_proj.forward(x).clamp(min=-self._limit, max=self._limit)
        return self.down_proj.forward(F.silu(gate) * up)


class _VBlock(BaseOP):
    def __init__(self, vc):
        self.norm1 = _VNorm(vc["hidden_size"], vc["rms_norm_eps"])
        self.norm2 = _VNorm(vc["hidden_size"], vc["rms_norm_eps"])
        self.attn = _VAttention(vc)
        self.mlp = _VMLP(vc, vc["hidden_size"], vc["intermediate_size"], vc["attention_bias"])

    def forward(self, x, bounds, cos, sin):
        x = x + self.attn.forward(self.norm1.forward(x), bounds, cos, sin)
        x = x + self.mlp.forward(self.norm2.forward(x))
        return x


class _VConv3d(BaseOP):
    def __init__(self, out_c: int, in_c: int, k):
        self.weight = torch.empty(out_c, in_c, *k, dtype=torch.bfloat16)
        self.bias = torch.empty(out_c, dtype=torch.bfloat16)

    def forward(self, x):
        return F.conv3d(x, self.weight, self.bias, stride=self.weight.shape[2:])


class _VConv2d(BaseOP):
    def __init__(self, in_c: int, out_c: int, k: int):
        self.weight = torch.empty(out_c, in_c, k, k, dtype=torch.bfloat16)
        self.bias = torch.empty(out_c, dtype=torch.bfloat16)
        self._k = k

    def forward(self, x):
        return F.conv2d(x.to(self.weight.dtype), self.weight, self.bias, stride=self._k)


class _VPatchEmbed(BaseOP):
    """Checkpoint path ``visual.patch_embed.proj.{weight,bias}`` -> nested ``proj``."""

    def __init__(self, vc):
        k = (vc["temporal_patch_size"], vc["patch_size"], vc["patch_size"])
        self.proj = _VConv3d(vc["hidden_size"], vc["in_channels"], k)
        self._shape = (vc["in_channels"], vc["temporal_patch_size"], vc["patch_size"], vc["patch_size"])
        self._dim = vc["hidden_size"]

    def forward(self, x):
        x = x.view(-1, *self._shape).to(self.proj.weight.dtype)
        return self.proj.forward(x).view(-1, self._dim)


class _VMerger(BaseOP):
    def __init__(self, vc):
        dim, ctx = vc["out_hidden_size"], vc["projection_intermediate_size"]
        self.proj = _VLinear(dim, dim, False)
        self.post_projection_norm = _VLayerNorm(dim)
        self.gate_proj = _VLinear(dim, ctx, False)
        self.up_proj = _VLinear(dim, ctx, False)
        self.down_proj = _VLinear(ctx, dim, False)
        self._limit = vc["swiglu_limit"]

    def forward(self, x):
        x = self.proj.forward(x)
        x = F.gelu(self.post_projection_norm.forward(x))
        gate = self.gate_proj.forward(x).clamp(max=self._limit)
        up = self.up_proj.forward(x).clamp(min=-self._limit, max=self._limit)
        return self.down_proj.forward(F.silu(gate) * up)


class Glm5Vision(BaseOP):
    """The ``model.visual`` tower: patch embed -> 24 blocks -> merge/downsample -> merger."""

    def __init__(self, vc: dict):
        self._vc = vc
        self.patch_embed = _VPatchEmbed(vc)
        self.blocks = OPList([_VBlock(vc) for _ in range(vc["depth"])])
        self.post_layernorm = _VNorm(vc["hidden_size"], vc["rms_norm_eps"])
        # downsample: Conv2d(hidden -> out_hidden, k=s=merge)
        m = vc["spatial_merge_size"]
        self.downsample = _VConv2d(vc["hidden_size"], vc["out_hidden_size"], m)
        self.merger = _VMerger(vc)

    def _inv_freq_for(self, device) -> torch.Tensor:
        # Computed, not loaded -- and therefore built LAZILY on the runtime device.
        # The engine constructs models under a meta-device context; a tensor computed
        # in __init__ stays meta (no checkpoint key overwrites it) and dies at first
        # use ("Cannot copy out of meta tensor").
        inv = getattr(self, "_inv_freq_cache", None)
        if inv is None or inv.device != device:
            head_dim = self._vc["hidden_size"] // self._vc["num_heads"]
            rd = head_dim // 2
            inv = 1.0 / (
                10000.0 ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd)
            )
            self._inv_freq_cache = inv
        return inv

    def _position_ids(self, grid_thw: torch.Tensor) -> torch.Tensor:
        m = self._vc["spatial_merge_size"]
        device = grid_thw.device
        out = []
        for t, h, w in grid_thw.tolist():
            hp, wp = torch.meshgrid(torch.arange(h, device=device),
                                    torch.arange(w, device=device), indexing="ij")
            shape = (h // m, m, w // m, m)
            hp = hp.reshape(shape).transpose(1, 2).flatten()
            wp = wp.reshape(shape).transpose(1, 2).flatten()
            out.append(torch.stack([hp, wp], dim=-1).repeat(t, 1))
        return torch.cat(out, dim=0)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        vc = self._vc
        pos = self._position_ids(grid_thw)                        # [S, 2]
        # Attention segments follow the qwen2_vl/glm4v convention (HF
        # get_vision_cu_seqlens merge_temporal=False): each temporal unit is its
        # own segment -- t segments of h*w patches per video, one for an image.
        bounds = [0]
        for t, h, w in grid_thw.tolist():                         # one host sync total
            hw = h * w
            for _ in range(int(t)):
                bounds.append(bounds[-1] + hw)
        x = self.patch_embed.forward(pixel_values)
        inv = self._inv_freq_for(x.device)
        rot = (pos.unsqueeze(-1) * inv).flatten(1)                # [S, rd/2*2]
        emb = torch.cat((rot, rot), dim=-1)
        cos = emb.cos().unsqueeze(-2).float()                     # fp32 once, all 24 blocks
        sin = emb.sin().unsqueeze(-2).float()
        for blk in self.blocks.op_list:
            x = blk.forward(x, bounds, cos, sin)
        x = self.post_layernorm.forward(x)
        m = vc["spatial_merge_size"]
        x = x.view(-1, m, m, x.shape[-1]).permute(0, 3, 1, 2)
        x = self.downsample.forward(x).view(-1, vc["out_hidden_size"])
        return self.merger.forward(x)


__all__ = ["Glm5Vision"]
