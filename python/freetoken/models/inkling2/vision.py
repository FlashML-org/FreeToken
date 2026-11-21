"""Inkling 2 vision tower: 20x20 patches embedded linearly plus a bilinearly resampled 32x32 position
table, a class token per image, pre-norm LayerScale blocks with axial 2-D rope (interleaved pairs,
width frequencies first) attending within each image, then every 2x2 patch block concatenated and
projected by a GELU MLP into the text width -- one soft token per 40x40 pixel cell."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LayerNorm, LinearReplicated, OPList
from freetoken.models.weight_stream import BlockWeightStreamer

if TYPE_CHECKING:
    from .config import VisionConfig


class Inkling2VisionBlock(BaseOP):
    def __init__(self, vc: VisionConfig):
        H = vc.hidden_size
        self.num_heads = vc.num_heads
        self.layernorm_before = LayerNorm(H, eps=vc.layer_norm_eps)
        self.layernorm_after = LayerNorm(H, eps=vc.layer_norm_eps)
        self.qkv = LinearReplicated(H, 3 * H, has_bias=True)
        self.out_proj = LinearReplicated(H, H, has_bias=True)
        self.fc1 = LinearReplicated(H, vc.intermediate_size, has_bias=True)
        self.fc2 = LinearReplicated(vc.intermediate_size, H, has_bias=True)
        self.lambda_1 = torch.empty(H)
        self.lambda_2 = torch.empty(H)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        S, Hd = x.shape
        H, D = self.num_heads, Hd // self.num_heads
        q, k, v = self.qkv.forward(self.layernorm_before.forward(x)).view(S, 3, H, D).unbind(1)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        # bidirectional attention within each image
        outs = [
            F.scaled_dot_product_attention(qs, ks, vs)
            for qs, ks, vs in zip(torch.split(q, lengths, dim=2), torch.split(k, lengths, dim=2), torch.split(v, lengths, dim=2))
        ]
        o = (outs[0] if len(outs) == 1 else torch.cat(outs, dim=2))[0].transpose(0, 1).reshape(S, Hd)
        x = x + self.lambda_1 * self.out_proj.forward(o)
        return x + self.lambda_2 * self.fc2.forward(F.gelu(self.fc1.forward(self.layernorm_after.forward(x))))


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each interleaved channel pair ``(x1, x2)`` by the row's angle, in fp32."""
    xf = x.float()
    pairs = xf.view(*xf.shape[:-1], -1, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    return (xf * cos[:, None] + rotated * sin[:, None]).to(x.dtype)


class Inkling2VisionModel(BaseOP):
    """Pixels ``[N, cell, cell, 3]`` (one 40x40 cell per soft token, raster order per image) and one
    ``(1, rows, cols)`` grid per image -> ``[N, text_hidden]``, the reference's get_image_features."""

    def __init__(self, vc: VisionConfig):
        H = vc.hidden_size
        self._vc = vc
        self.patch_embedding = LinearReplicated(3 * vc.patch_size**2, H, has_bias=False)
        self.position_embedding = torch.empty(vc.posemb_grid_size**2, H)
        self.class_embedding = torch.empty(H)
        self.cls_position_embedding = torch.empty(H)
        self.pre_layernorm = LayerNorm(H, eps=vc.layer_norm_eps)
        self.layers = OPList([Inkling2VisionBlock(vc) for _ in range(vc.num_layers)])
        self.merger_in = LinearReplicated(vc.merge_size**2 * H, vc.projection_hidden_dim, has_bias=False)
        self.merger_out = LinearReplicated(vc.projection_hidden_dim, vc.text_hidden_size, has_bias=False)
        self._inv_freq: torch.Tensor | None = None
        self._streamer: BlockWeightStreamer | None = None

    def place_weights(self, mode: str) -> None:
        """gpu: every tensor resident; host: block tensors in pinned banks streamed two blocks at a time."""
        if mode == "host" and self._streamer is None:
            self._streamer = BlockWeightStreamer(self.layers.op_list, self.patch_embedding.weight.device)
        elif mode == "gpu" and self._streamer is not None:
            self._streamer.unstream()
            self._streamer = None
        elif mode not in ("gpu", "host"):
            raise ValueError(f"unknown vision weight placement {mode!r}")

    def _layers(self):
        if self._streamer is None:
            return enumerate(self.layers.op_list)
        return self._streamer.blocks(self.layers.op_list)

    def _patch_positions(self, rows: int, cols: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Resampled position rows and the (row, col) of every patch, in cell-major order (each cell's 2x2 patches together)."""
        vc = self._vc
        m, side = vc.merge_size, vc.posemb_grid_size
        table = self.position_embedding.float().view(side, side, -1).permute(2, 0, 1)[None]
        grid = F.interpolate(table, size=(rows * m, cols * m), mode="bilinear", align_corners=False)[0]
        pos = grid.view(-1, rows, m, cols, m).permute(1, 3, 2, 4, 0).reshape(rows * cols * m * m, -1)
        r = torch.arange(rows * m, device=device).view(rows, m, 1, 1).expand(rows, m, cols, m)
        c = torch.arange(cols * m, device=device).view(1, 1, cols, m).expand(rows, m, cols, m)
        rc = torch.stack((r, c), dim=-1).permute(0, 2, 1, 3, 4).reshape(-1, 2)
        return pos, rc

    def _rope_tables(self, rc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vc = self._vc
        if self._inv_freq is None or self._inv_freq.device != rc.device:
            half = (vc.hidden_size // vc.num_heads) // 2
            self._inv_freq = 1.0 / (vc.rope_theta ** (torch.arange(0, half, 2, dtype=torch.float32, device=rc.device) / half))
        freqs = rc.float()[..., None] * self._inv_freq  # [S, 2 (row, col), half / 2]
        # width frequencies, then height, each repeated for its interleaved pair
        angles = torch.cat((freqs[:, 1], freqs[:, 0]), dim=-1).repeat_interleave(2, dim=-1)
        return angles.cos(), angles.sin()

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, grid_thw: List[List[int]]) -> torch.Tensor:
        vc = self._vc
        m, p = vc.merge_size, vc.patch_size
        device = self.patch_embedding.weight.device
        dtype = self.patch_embedding.weight.dtype
        pixels = pixel_values.to(device=device, dtype=dtype)
        patches = pixels.reshape(-1, m, p, m, p, 3).transpose(2, 3).reshape(-1, 3 * p * p)
        x = self.patch_embedding.forward(patches)
        cls = (self.class_embedding + self.cls_position_embedding).to(dtype)[None]
        seqs, rcs, lengths, start = [], [], [], 0
        for t, rows, cols in grid_thw:
            assert t == 1, "images only"
            n = rows * cols * m * m
            pos, rc = self._patch_positions(rows, cols, device)
            seqs += [cls, x[start : start + n] + pos.to(dtype)]
            # patches are 1-indexed; the class token sits at (0, 0)
            rcs += [rc.new_zeros(1, 2), rc + 1]
            lengths.append(n + 1)
            start += n
        x = self.pre_layernorm.forward(torch.cat(seqs))
        cos, sin = self._rope_tables(torch.cat(rcs))
        for _, layer in self._layers():
            x = layer.forward(x, cos, sin, lengths)
        keep = torch.ones(x.shape[0], dtype=torch.bool, device=device)
        keep[torch.tensor([sum(lengths[:i]) for i in range(len(lengths))], device=device)] = False
        cells = x[keep].reshape(-1, m * m * x.shape[-1])
        return self.merger_out.forward(F.gelu(self.merger_in.forward(cells)))


__all__ = ["Inkling2VisionModel"]
