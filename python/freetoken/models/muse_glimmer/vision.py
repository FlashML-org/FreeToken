"""Muse-Glimmer vision tower: linear patch embedding plus a resampled learned position table, LayerNorm blocks with 2-D rope that attend inside 32x32-patch windows (full attention every fourth block), a 2x2 pixel shuffle, the GELU adapter, the projection into the text width and the weightless perception norm."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, GemmaRMSNorm, LayerNorm, LinearReplicated, OPList
from freetoken.models.weight_stream import BlockWeightStreamer

if TYPE_CHECKING:
    from .config import VisionConfig


class _Table(BaseOP):
    """A bare weight under the key of the checkpoint's nn.Embedding."""

    def __init__(self, rows: int, cols: int):
        self.weight = torch.empty(rows, cols)

    forward = None


class MuseGlimmerVisionPatchEmbedder(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.patch_embedding = LinearReplicated(vc.patch_dim, vc.hidden_size, has_bias=False)
        self.position_embedding_table = _Table(vc.pos_emb_side**2, vc.hidden_size)
        self._side = vc.pos_emb_side

    def forward(self, pixel_values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        from transformers.vision_utils import get_vision_interpolation_indices_and_weights

        x = self.patch_embedding.forward(pixel_values.to(self.patch_embedding.weight.dtype))
        # the reference resamples the table with grid_sample(align_corners=False, padding_mode="zeros")
        taps, weights = (
            t.to(x.device)
            for t in get_vision_interpolation_indices_and_weights(
                grid, self._side, mode="bilinear", align_corners=False, spatial_merge_size=1, padding="zeros"
            )
        )
        # one tap at a time: fp32 like the reference's weighted sum, without the [S, 4, hidden] gather
        table = self.position_embedding_table.weight
        pos = table[taps[:, 0]] * weights[:, 0, None]
        for t in range(1, taps.shape[1]):
            pos = pos + table[taps[:, t]] * weights[:, t, None]
        return x + pos.to(x.dtype)


class MuseGlimmerVisionAttention(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.num_heads = vc.num_heads
        self.head_dim = vc.hidden_size // vc.num_heads
        self.qkv = LinearReplicated(vc.hidden_size, 3 * vc.hidden_size, has_bias=True)
        self.proj = LinearReplicated(vc.hidden_size, vc.hidden_size, has_bias=True)

    def forward(self, x: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        from freetoken.kernel.triton.rope import apply_rope_with_cos_sin_cache_inplace

        S, H, D = x.shape[0], self.num_heads, self.head_dim
        q, k, v = self.qkv.forward(x).view(S, 3, H * D).unbind(1)
        apply_rope_with_cos_sin_cache_inplace(positions, q, k, D, cache, is_neox=True)
        q, k, v = (t.view(S, H, D).transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        # bidirectional attention within each segment: a window or a whole image
        outs = [
            F.scaled_dot_product_attention(qs, ks, vs)
            for qs, ks, vs in zip(torch.split(q, lengths, dim=2), torch.split(k, lengths, dim=2), torch.split(v, lengths, dim=2))
        ]
        o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
        return self.proj.forward(o[0].transpose(0, 1).reshape(S, H * D))


class MuseGlimmerVisionMLP(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.fc1 = LinearReplicated(vc.hidden_size, vc.intermediate_size, has_bias=True)
        self.fc2 = LinearReplicated(vc.intermediate_size, vc.hidden_size, has_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2.forward(F.gelu(self.fc1.forward(x)))


class MuseGlimmerVisionBlock(BaseOP):
    def __init__(self, vc: VisionConfig):
        # the reference builds the block norms with the torch default eps, not the config's
        self.norm1 = LayerNorm(vc.hidden_size, eps=1e-5)
        self.norm2 = LayerNorm(vc.hidden_size, eps=1e-5)
        self.attn = MuseGlimmerVisionAttention(vc)
        self.mlp = MuseGlimmerVisionMLP(vc)

    def forward(self, x: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        x = x + self.attn.forward(self.norm1.forward(x), cache, positions, lengths)
        return x + self.mlp.forward(self.norm2.forward(x))


class MuseGlimmerVisionAdapter(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.fc1 = LinearReplicated(vc.out_hidden_size, vc.projector_hidden_size, has_bias=False)
        self.fc2 = LinearReplicated(vc.projector_hidden_size, vc.projector_hidden_size, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.fc2.forward(F.gelu(self.fc1.forward(x))))


def _shuffle_index(grid: torch.Tensor, merge: int) -> torch.Tensor:
    """Row order that puts every merge x merge patch block of each image in merge**2 consecutive rows."""
    parts, offset = [], 0
    for t, h, w in grid.tolist():
        assert t == 1, "images only"
        block = torch.arange(h * w, device=grid.device).view(h // merge, merge, w // merge, merge)
        parts.append(block.permute(0, 2, 1, 3).reshape(-1) + offset)
        offset += h * w
    return torch.cat(parts)


class MuseGlimmerVisionModel(BaseOP):
    """Pixels -> [sum(h*w) / merge^2, text_hidden] soft tokens (the reference's get_image_features), one grid_thw row per image; adapter, projection and perception norm ride along so the whole image path is one module."""

    def __init__(self, vc: VisionConfig):
        self.patch_embedder = MuseGlimmerVisionPatchEmbedder(vc)
        self.ln_pre = LayerNorm(vc.hidden_size, eps=vc.layer_norm_eps)
        self.layers = OPList([MuseGlimmerVisionBlock(vc) for _ in range(vc.num_layers)])
        self.ln_post = LayerNorm(vc.hidden_size, eps=vc.layer_norm_eps)
        self.adapter = MuseGlimmerVisionAdapter(vc)
        self.projection = LinearReplicated(vc.projector_hidden_size, vc.text_hidden_size, has_bias=False)
        self.perception_emb_norm = GemmaRMSNorm(vc.text_hidden_size, eps=vc.text_rms_norm_eps, with_scale=False)
        self._vc = vc
        self._inv_freq: torch.Tensor | None = None
        self._streamer: BlockWeightStreamer | None = None

    def place_weights(self, mode: str) -> None:
        """gpu: every tensor resident; host: block tensors in pinned banks streamed two blocks at a time (embedder, adapter and projection stay resident)."""
        if mode == "host" and self._streamer is None:
            self._streamer = BlockWeightStreamer(self.layers.op_list, self.patch_embedder.patch_embedding.weight.device)
        elif mode == "gpu" and self._streamer is not None:
            self._streamer.unstream()
            self._streamer = None
        elif mode not in ("gpu", "host"):
            raise ValueError(f"unknown vision weight placement {mode!r}")

    def _layers(self):
        if self._streamer is None:
            return enumerate(self.layers.op_list)
        return self._streamer.blocks(self.layers.op_list)

    def _rope_table(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vc = self._vc
        device = position_ids.device
        if self._inv_freq is None or self._inv_freq.device != device:
            # the reference splits the head in two and gives each axis the frequency ladder of a head_dim/2 rope
            half = (vc.hidden_size // vc.num_heads) // 2
            self._inv_freq = 1.0 / (vc.rope_theta ** (torch.arange(0, half, 2, dtype=torch.float32, device=device) / half))
        freqs = (position_ids.float().unsqueeze(-1) * self._inv_freq).flatten(1)
        # per-token rope rows [cos | sin] consumed by the NeoX kernel via positions=arange
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()
        return cache, torch.arange(position_ids.shape[0], dtype=torch.int32, device=device)

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, grid_thw: List[List[int]]) -> torch.Tensor:
        from transformers.vision_utils import get_vision_cu_seqlens, get_vision_position_ids, get_vision_window_index

        vc = self._vc
        device = self.patch_embedder.patch_embedding.weight.device
        # the grid stays on the host: the index helpers loop in Python, so a device grid would sync per call
        grid = torch.tensor(grid_thw, dtype=torch.long)
        # window layers attend inside pos_emb_side x pos_emb_side patch tiles; tokens are reordered tile by tile for the whole stack
        window_index, cu_window = get_vision_window_index(
            grid, spatial_merge_size=1, window_size=vc.pos_emb_side * vc.patch_size, patch_size=vc.patch_size
        )
        x = self.ln_pre.forward(self.patch_embedder.forward(pixel_values.to(device), grid))[window_index.to(device)]
        # the reference feeds rope (w + 1, h + 1)
        position_ids = (get_vision_position_ids(grid, 1).flip(-1) + 1)[window_index].to(device)
        cache, positions = self._rope_table(position_ids)
        lengths = {
            "full_attention": torch.diff(get_vision_cu_seqlens(grid)).tolist(),
            "window_attention": torch.diff(cu_window).tolist(),
        }
        for i, layer in self._layers():
            x = layer.forward(x, cache, positions, lengths[vc.layer_types[i]])
        x = self.ln_post.forward(x[torch.argsort(window_index).to(device)])
        m = vc.merge_size
        x = x[_shuffle_index(grid, m).to(device)].view(-1, m * m, vc.hidden_size).permute(0, 2, 1).reshape(-1, vc.out_hidden_size)
        return self.perception_emb_norm.forward(self.projection.forward(self.adapter.forward(x)))


__all__ = ["MuseGlimmerVisionModel"]
