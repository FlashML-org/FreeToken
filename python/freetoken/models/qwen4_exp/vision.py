"""Qwen3-VL-compatible vision tower used by Qwen3.8-Flash-Next."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, OPList

from .args import Qwen4VisionConfig


class _LayerNorm(BaseOP):
    def __init__(self, size: int, eps: float = 1e-6):
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class _Embedding(BaseOP):
    def __init__(self, count: int, width: int):
        self.weight = torch.empty(count, width)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.weight)


class _Conv3dPatchProjection(BaseOP):
    def __init__(self, config: Qwen4VisionConfig):
        self.weight = torch.empty(
            config.hidden_size,
            config.in_channels,
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        )
        self.bias = torch.empty(config.hidden_size)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # The processor already patchifies each Conv3d receptive field into one
        # row.  Flattening the kernel makes this exactly the stride==kernel Conv3d.
        return F.linear(pixels.to(self.weight.dtype), self.weight.flatten(1), self.bias)


class Qwen4VisionPatchEmbed(BaseOP):
    def __init__(self, config: Qwen4VisionConfig):
        self.proj = _Conv3dPatchProjection(config)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.proj.forward(pixels)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_vision_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_dtype, k_dtype = q.dtype, k.dtype
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    qf, kf = q.float(), k.float()
    q = qf * cos + _rotate_half(qf) * sin
    k = kf * cos + _rotate_half(kf) * sin
    return q.to(q_dtype), k.to(k_dtype)


class Qwen4VisionAttention(BaseOP):
    def __init__(self, config: Qwen4VisionConfig):
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = LinearReplicated(
            config.hidden_size, config.hidden_size * 3, has_bias=True
        )
        self.proj = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=True
        )

    def forward(
        self,
        hidden: torch.Tensor,
        segment_lengths: tuple[int, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        tokens = hidden.shape[0]
        q, k, v = (
            self.qkv.forward(hidden)
            .view(tokens, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q, k = _apply_vision_rope(q, k, cos, sin)
        outputs = []
        start = 0
        for length in segment_lengths:
            stop = start + length
            qs = q[start:stop].transpose(0, 1).unsqueeze(0)
            ks = k[start:stop].transpose(0, 1).unsqueeze(0)
            vs = v[start:stop].transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                qs, ks, vs, is_causal=False, scale=self.head_dim**-0.5
            )
            outputs.append(out.squeeze(0).transpose(0, 1))
            start = stop
        if start != tokens:
            raise ValueError(
                f"vision grid describes {start} patches but processor supplied {tokens}"
            )
        return self.proj.forward(torch.cat(outputs, dim=0).reshape(tokens, -1))


class Qwen4VisionMLP(BaseOP):
    def __init__(self, config: Qwen4VisionConfig):
        self.linear_fc1 = LinearReplicated(
            config.hidden_size, config.intermediate_size, has_bias=True
        )
        self.linear_fc2 = LinearReplicated(
            config.intermediate_size, config.hidden_size, has_bias=True
        )
        self.hidden_act = config.hidden_act

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.linear_fc1.forward(hidden)
        if self.hidden_act == "gelu_pytorch_tanh":
            projected = F.gelu(projected, approximate="tanh")
        elif self.hidden_act == "gelu":
            projected = F.gelu(projected)
        else:
            raise ValueError(f"Unsupported Qwen vision activation {self.hidden_act!r}")
        return self.linear_fc2.forward(projected)


class Qwen4VisionBlock(BaseOP):
    def __init__(self, config: Qwen4VisionConfig):
        self.norm1 = _LayerNorm(config.hidden_size)
        self.norm2 = _LayerNorm(config.hidden_size)
        self.attn = Qwen4VisionAttention(config)
        self.mlp = Qwen4VisionMLP(config)

    def forward(self, hidden, segment_lengths, cos, sin):
        hidden = hidden + self.attn.forward(
            self.norm1.forward(hidden), segment_lengths, cos, sin
        )
        return hidden + self.mlp.forward(self.norm2.forward(hidden))


class Qwen4VisionPatchMerger(BaseOP):
    def __init__(self, config: Qwen4VisionConfig, use_postshuffle_norm: bool = False):
        merged = config.hidden_size * config.spatial_merge_size**2
        self.norm = _LayerNorm(merged if use_postshuffle_norm else config.hidden_size)
        self.linear_fc1 = LinearReplicated(merged, merged, has_bias=True)
        self.linear_fc2 = LinearReplicated(merged, config.out_hidden_size, has_bias=True)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.merged = merged

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden = self.norm.forward(hidden.view(-1, self.merged))
        else:
            hidden = self.norm.forward(hidden).view(-1, self.merged)
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(hidden)))


class Qwen4VisionModel(BaseOP):
    def __init__(self, config: Qwen4VisionConfig):
        self.patch_embed = Qwen4VisionPatchEmbed(config)
        self.pos_embed = _Embedding(config.num_position_embeddings, config.hidden_size)
        self.blocks = OPList([Qwen4VisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen4VisionPatchMerger(config)
        self.deepstack_merger_list = OPList(
            [Qwen4VisionPatchMerger(config, True) for _ in config.deepstack_visual_indexes]
        )
        self.spatial_merge_size = config.spatial_merge_size
        self.num_grid_per_side = int(math.sqrt(config.num_position_embeddings))
        self.head_dim = config.hidden_size // config.num_heads
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        # The engine constructs the model under ``torch.device("meta")``. This
        # value is derived, not loaded from the checkpoint, so keep only its
        # shape here and materialize the tiny frequency vector on the real device.
        self._inv_dim = self.head_dim // 2

    def _position_data(self, grid_thw: torch.Tensor, dtype: torch.dtype):
        from transformers.vision_utils import (
            get_vision_interpolation_indices_and_weights,
            get_vision_position_ids,
        )

        indices, weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            mode="bilinear",
            align_corners=True,
            spatial_merge_size=self.spatial_merge_size,
        )
        pos = (self.pos_embed.forward(indices) * weights[:, :, None].to(dtype)).sum(1)
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        inv = 1.0 / (
            10000.0
            ** (
                torch.arange(
                    0,
                    self._inv_dim,
                    2,
                    dtype=torch.float32,
                    device=position_ids.device,
                )
                / self._inv_dim
            )
        )
        rotary = (position_ids.unsqueeze(-1).float() * inv).flatten(1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        return pos, rotary.cos().to(dtype), rotary.sin().to(dtype)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw = grid_thw.to(device=pixel_values.device, dtype=torch.long)
        hidden = self.patch_embed.forward(pixel_values)
        pos, cos, sin = self._position_data(grid_thw, hidden.dtype)
        hidden = hidden + pos.to(hidden.dtype)
        segment_lengths = tuple(
            int(h) * int(w)
            for t, h, w in grid_thw.detach().cpu().tolist()
            for _ in range(int(t))
        )
        deepstack = []
        for layer_id, block in enumerate(self.blocks.op_list):
            hidden = block.forward(hidden, segment_lengths, cos, sin)
            if layer_id in self.deepstack_visual_indexes:
                slot = self.deepstack_visual_indexes.index(layer_id)
                deepstack.append(self.deepstack_merger_list.op_list[slot].forward(hidden))
        merged = self.merger.forward(hidden)
        return torch.cat((merged, *deepstack), dim=-1) if deepstack else merged


__all__ = ["Qwen4VisionModel"]
