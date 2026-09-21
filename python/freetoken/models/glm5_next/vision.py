"""GLM-5.3-Flash vision tower: Conv3d patch embedding, pre-RMSNorm blocks with q/k norms and 2-D rope, a 2x2 Conv2d downsample and a clamped-SwiGLU merger into the text width."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LayerNorm, LinearReplicated, OPList, RMSNorm
from freetoken.models.weight_stream import BlockWeightStreamer

from .mlp import Glm5NextGatedMLP

if TYPE_CHECKING:
    from .config import VisionConfig


class _ConvParams(BaseOP):
    def __init__(self, out_ch: int, shape: tuple[int, ...]):
        self.weight = torch.empty(out_ch, *shape)
        self.bias = torch.empty(out_ch)

    forward = None


class Glm5NextVisionPatchEmbed(BaseOP):
    """Kept as a real Conv3d (kernel == stride): an equivalent unfolded matmul rounds differently."""

    def __init__(self, vc: VisionConfig):
        self._shape = (vc.in_channels, vc.temporal_patch_size, vc.patch_size, vc.patch_size)
        self.proj = _ConvParams(vc.hidden_size, self._shape)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = pixel_values.view(-1, *self._shape).to(self.proj.weight.dtype)
        return F.conv3d(x, self.proj.weight, self.proj.bias, stride=self._shape[1:]).view(x.shape[0], -1)


class Glm5NextVisionAttention(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.num_heads = vc.num_heads
        self.head_dim = vc.hidden_size // vc.num_heads
        self.qkv = LinearReplicated(vc.hidden_size, 3 * vc.hidden_size, has_bias=vc.attention_bias)
        self.proj = LinearReplicated(vc.hidden_size, vc.hidden_size, has_bias=vc.attention_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=vc.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=vc.rms_norm_eps)

    def forward(self, x: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        from freetoken.kernel.triton.rope import apply_rope_with_cos_sin_cache_inplace

        S, H, D = x.shape[0], self.num_heads, self.head_dim
        q, k, v = self.qkv.forward(x).view(S, 3, H * D).unbind(1)
        q = self.q_norm.forward(q.reshape(S * H, D)).view(S, H * D)
        k = self.k_norm.forward(k.reshape(S * H, D)).view(S, H * D)
        apply_rope_with_cos_sin_cache_inplace(positions, q, k, D, cache, is_neox=True)
        q, k, v = (t.view(S, H, D).transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        # bidirectional attention within each image; images never attend across
        outs = [
            F.scaled_dot_product_attention(qs, ks, vs)
            for qs, ks, vs in zip(torch.split(q, lengths, dim=2), torch.split(k, lengths, dim=2), torch.split(v, lengths, dim=2))
        ]
        o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
        return self.proj.forward(o[0].transpose(0, 1).reshape(S, H * D))


class Glm5NextVisionBlock(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.norm1 = RMSNorm(vc.hidden_size, eps=vc.rms_norm_eps)
        self.norm2 = RMSNorm(vc.hidden_size, eps=vc.rms_norm_eps)
        self.attn = Glm5NextVisionAttention(vc)
        self.mlp = Glm5NextGatedMLP(vc.hidden_size, vc.intermediate_size, vc.swiglu_limit, has_bias=vc.attention_bias)

    def forward(self, x: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        x = x + self.attn.forward(self.norm1.forward(x), cache, positions, lengths)
        return x + self.mlp.forward(self.norm2.forward(x))


class Glm5NextVisionPatchMerger(Glm5NextGatedMLP):
    """proj -> LayerNorm -> GELU in front of the clamped-SwiGLU MLP; the checkpoint stores both halves under one module."""

    def __init__(self, vc: VisionConfig):
        super().__init__(vc.out_hidden_size, vc.projection_intermediate_size, vc.swiglu_limit)
        self.proj = LinearReplicated(vc.out_hidden_size, vc.out_hidden_size, has_bias=False)
        # the reference builds a torch LayerNorm with the default eps
        self.post_projection_norm = LayerNorm(vc.out_hidden_size, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.gelu(self.post_projection_norm.forward(self.proj.forward(x))))


class Glm5NextVisionModel(BaseOP):
    """Pixels -> soft tokens [sum(h*w) / merge^2, out_hidden] in the text width, one grid_thw row per image."""

    def __init__(self, vc: VisionConfig):
        m = vc.spatial_merge_size
        self.patch_embed = Glm5NextVisionPatchEmbed(vc)
        self.blocks = OPList([Glm5NextVisionBlock(vc) for _ in range(vc.depth)])
        self.post_layernorm = RMSNorm(vc.hidden_size, eps=vc.rms_norm_eps)
        self.downsample = _ConvParams(vc.out_hidden_size, (vc.hidden_size, m, m))
        self.merger = Glm5NextVisionPatchMerger(vc)
        self._vc = vc
        self._inv_freq: torch.Tensor | None = None
        self._streamer: BlockWeightStreamer | None = None

    def place_weights(self, mode: str) -> None:
        """gpu: every tensor resident; host: block tensors in pinned banks streamed two blocks at a time (patch embedding, downsample and merger stay resident)."""
        if mode == "host" and self._streamer is None:
            self._streamer = BlockWeightStreamer(self.blocks.op_list, self.patch_embed.proj.weight.device)
        elif mode == "gpu" and self._streamer is not None:
            self._streamer.unstream()
            self._streamer = None
        elif mode not in ("gpu", "host"):
            raise ValueError(f"unknown vision weight placement {mode!r}")

    def _blocks(self):
        if self._streamer is None:
            return enumerate(self.blocks.op_list)
        return self._streamer.blocks(self.blocks.op_list)

    def _rope_table(self, grid: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        from transformers.vision_utils import get_vision_position_ids

        if self._inv_freq is None or self._inv_freq.device != device:
            # fp32 like the reference; head_dim/2 angles, one quarter of the head each for the row and the column index, pairs (i, i + head_dim/2) share an angle
            half = (self._vc.hidden_size // self._vc.num_heads) // 2
            self._inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, 2, dtype=torch.float32, device=device) / half))
        pos_ids = get_vision_position_ids(grid, self._vc.spatial_merge_size)
        freqs = (pos_ids.unsqueeze(-1) * self._inv_freq).flatten(1)
        # per-token rope rows [cos | sin] consumed by the NeoX kernel via positions=arange
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()
        return cache, torch.arange(pos_ids.shape[0], dtype=torch.int32, device=device)

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, grid_thw: List[List[int]]) -> torch.Tensor:
        from transformers.vision_utils import get_vision_cu_seqlens

        vc = self._vc
        device = self.patch_embed.proj.weight.device
        grid = torch.tensor(grid_thw, dtype=torch.long, device=device)
        x = self.patch_embed.forward(pixel_values.to(device))
        cache, positions = self._rope_table(grid, device)
        lengths = torch.diff(get_vision_cu_seqlens(grid)).tolist()
        for _, blk in self._blocks():
            x = blk.forward(x, cache, positions, lengths)
        x = self.post_layernorm.forward(x)
        m = vc.spatial_merge_size
        x = x.view(-1, m, m, vc.hidden_size).permute(0, 3, 1, 2)
        x = F.conv2d(x, self.downsample.weight, self.downsample.bias, stride=m).view(-1, vc.out_hidden_size)
        return self.merger.forward(x)


__all__ = ["Glm5NextVisionModel"]
