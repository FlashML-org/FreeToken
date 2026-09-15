"""MiniMax-M3 vision tower: Conv3d patch embedding, pre-LayerNorm CLIP blocks with a 3-axis rope, a GELU projector and a 2x2 patch-merge MLP into the text width."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LayerNorm, LinearReplicated, OPList
from freetoken.models.weight_stream import BlockWeightStreamer

if TYPE_CHECKING:
    from .config import VisionConfig


class _Conv3dWeight(BaseOP):
    def __init__(self, out_ch: int, shape: tuple[int, ...]):
        self.weight = torch.empty(out_ch, *shape)

    forward = None


class MiniMaxM3VisionEmbeddings(BaseOP):
    """Kept as a real Conv3d (kernel == stride): an equivalent unfolded matmul rounds differently."""

    def __init__(self, vc: VisionConfig):
        self._shape = (vc.num_channels, vc.temporal_patch_size, vc.patch_size, vc.patch_size)
        self.patch_embedding = _Conv3dWeight(vc.hidden_size, self._shape)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = pixel_values.view(-1, *self._shape).to(self.patch_embedding.weight.dtype)
        return F.conv3d(x, self.patch_embedding.weight, stride=self._shape[1:]).view(x.shape[0], -1)


def _rope_table(grid_thw: List[int], merge: int, head_dim: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token [cos | sin] rows over the rotated dims: the even part of the head split into three equal t/h/w bands, the remainder passes through."""
    t, h, w = grid_thw
    axis_dim = 2 * (((2 * (head_dim // 2)) // 3) // 2)
    inv_freq = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device) / axis_dim))
    hpos, wpos = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    # block-major over the merge blocks, the order the processor flattens patches in
    block = (h // merge, merge, w // merge, merge)
    hpos = hpos.reshape(block).transpose(1, 2).flatten().repeat(t)
    wpos = wpos.reshape(block).transpose(1, 2).flatten().repeat(t)
    tpos = torch.arange(t, device=device).repeat_interleave(h * w)
    coords = torch.stack([tpos, hpos, wpos], dim=-1).float()
    freqs = (coords.unsqueeze(-1) * inv_freq).flatten(1)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()
    return cache, torch.arange(coords.shape[0], dtype=torch.int32, device=device)


class MiniMaxM3VisionAttention(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.num_heads = vc.num_heads
        self.head_dim = vc.hidden_size // vc.num_heads
        self.qkv = LinearReplicated(vc.hidden_size, 3 * vc.hidden_size, has_bias=True)
        self.out_proj = LinearReplicated(vc.hidden_size, vc.hidden_size, has_bias=True)

    def forward(self, x: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.rope import apply_rope_with_cos_sin_cache_inplace

        S, H, D = x.shape[0], self.num_heads, self.head_dim
        q, k, v = self.qkv.forward(x).view(S, 3, H * D).unbind(1)
        apply_rope_with_cos_sin_cache_inplace(positions, q, k, D, cache, is_neox=True)
        # the fused SDPA kernels want a batch dim; 3-D inputs fall back to the O(S^2) math path
        q, k, v = (t.view(S, H, D).transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj.forward(o[0].transpose(0, 1).reshape(S, H * D))


class MiniMaxM3VisionMLP(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.fc1 = LinearReplicated(vc.hidden_size, vc.intermediate_size, has_bias=True)
        self.fc2 = LinearReplicated(vc.intermediate_size, vc.hidden_size, has_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2.forward(F.gelu(self.fc1.forward(x)))


class MiniMaxM3VisionEncoderLayer(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.self_attn = MiniMaxM3VisionAttention(vc)
        self.layer_norm1 = LayerNorm(vc.hidden_size, vc.layer_norm_eps)
        self.mlp = MiniMaxM3VisionMLP(vc)
        self.layer_norm2 = LayerNorm(vc.hidden_size, vc.layer_norm_eps)

    def forward(self, x: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn.forward(self.layer_norm1.forward(x), cache, positions)
        return x + self.mlp.forward(self.layer_norm2.forward(x))


class MiniMaxM3VisionEncoder(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.layers = OPList([MiniMaxM3VisionEncoderLayer(vc) for _ in range(vc.num_layers)])

    forward = None


class MiniMaxM3VisionTransformer(BaseOP):
    """Patches of one image -> [t * h * w, hidden]; the checkpoint has no post layernorm."""

    def __init__(self, vc: VisionConfig):
        self.embeddings = MiniMaxM3VisionEmbeddings(vc)
        self.pre_layrnorm = LayerNorm(vc.hidden_size, vc.layer_norm_eps)
        self.encoder = MiniMaxM3VisionEncoder(vc)
        self._vc = vc
        self._streamer: BlockWeightStreamer | None = None

    def place_weights(self, mode: str) -> None:
        """gpu: every tensor resident; host: encoder layer tensors in pinned banks streamed two layers at a time."""
        if mode == "host" and self._streamer is None:
            self._streamer = BlockWeightStreamer(self.encoder.layers.op_list, self.embeddings.patch_embedding.weight.device)
        elif mode == "gpu" and self._streamer is not None:
            self._streamer.unstream()
            self._streamer = None
        elif mode not in ("gpu", "host"):
            raise ValueError(f"unknown vision weight placement {mode!r}")

    def _layers(self):
        if self._streamer is None:
            return enumerate(self.encoder.layers.op_list)
        return self._streamer.blocks(self.encoder.layers.op_list)

    def forward(self, pixel_values: torch.Tensor, grid_thw: List[int]) -> torch.Tensor:
        vc = self._vc
        device = self.embeddings.patch_embedding.weight.device
        x = self.pre_layrnorm.forward(self.embeddings.forward(pixel_values.to(device)))
        cache, positions = _rope_table(grid_thw, vc.spatial_merge_size, vc.hidden_size // vc.num_heads, vc.rope_theta, device)
        for _, layer in self._layers():
            x = layer.forward(x, cache, positions)
        return x


class MiniMaxM3ProjectorMLP(BaseOP):
    def __init__(self, in_size: int, mid_size: int, out_size: int):
        self.linear_1 = LinearReplicated(in_size, mid_size, has_bias=True)
        self.linear_2 = LinearReplicated(mid_size, out_size, has_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2.forward(F.gelu(self.linear_1.forward(x)))


class MiniMaxM3VisionModel(BaseOP):
    """Pixels of one image -> soft tokens [t * h * w / merge^2, text_hidden]."""

    def __init__(self, vc: VisionConfig):
        self.vision_model = MiniMaxM3VisionTransformer(vc)
        self.multi_modal_projector = MiniMaxM3ProjectorMLP(vc.hidden_size, vc.projector_hidden_size, vc.text_hidden_size)
        self.patch_merge_mlp = MiniMaxM3ProjectorMLP(vc.spatial_merge_size**2 * vc.text_hidden_size, vc.projector_hidden_size, vc.text_hidden_size)
        self._merge = vc.spatial_merge_size**2

    def place_weights(self, mode: str) -> None:
        self.vision_model.place_weights(mode)

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, grid_thw: List[int]) -> torch.Tensor:
        x = self.multi_modal_projector.forward(self.vision_model.forward(pixel_values, grid_thw))
        return self.patch_merge_mlp.forward(x.view(x.shape[0] // self._merge, -1))


__all__ = ["MiniMaxM3VisionModel"]
