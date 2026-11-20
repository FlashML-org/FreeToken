"""Inkling 2 vision tower: a pre-LayerNorm ViT over 20-pixel patches with a class token, a bilinearly resampled
32x32 position table, axial 2-D rope and layer scale, then a 2x2 patch merge MLP into the text width.

Attribute paths follow the checkpoint under ``model.visual`` (``backbone.layers.N.ls_1.gamma``, ``proj_in``), so
its tensors load by name. The math follows transformers' Inkling2VisionModel, including where it computes in fp32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LayerNorm, LinearReplicated, OPList
from freetoken.models.weight_stream import BlockWeightStreamer


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    num_channels: int
    patch_size: int
    merge_size: int
    posemb_grid_size: int
    layer_norm_eps: float
    rope_theta: float
    projection_hidden_dim: int
    projector_bias: bool
    text_hidden_size: int


def _first(section: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(section, name, None)
        if value is not None:
            return value
    return default


def parse_vision_config(hf_config: Any) -> VisionConfig | None:
    """None when the checkpoint carries no vision section. Reads both the raw config.json names
    (native_patch_size, adapter_merge_size, mlp_ratio) and the transformers config class names."""
    vc = getattr(hf_config, "vision_config", None)
    if vc is None:
        return None
    act = _first(vc, "hidden_act", default="gelu")
    if act != "gelu":
        raise NotImplementedError(f"inkling2 vision tower activation {act!r}; only gelu is implemented")
    rope = _first(vc, "rope_parameters", default=None) or {}
    hidden = int(vc.hidden_size)
    return VisionConfig(
        hidden_size=hidden,
        num_layers=int(vc.num_hidden_layers),
        num_heads=int(vc.num_attention_heads),
        intermediate_size=int(_first(vc, "intermediate_size", default=None) or hidden * float(_first(vc, "mlp_ratio", default=4.0))),
        num_channels=int(_first(vc, "num_channels", default=3)),
        patch_size=int(_first(vc, "patch_size", "native_patch_size")),
        merge_size=int(_first(vc, "spatial_merge_size", "adapter_merge_size")),
        posemb_grid_size=int(_first(vc, "posemb_grid_size", default=32)),
        layer_norm_eps=float(_first(vc, "layer_norm_eps", default=1e-5)),
        # the config class leaves rope_theta to RotaryEmbeddingConfigMixin.default_theta
        rope_theta=float(rope.get("rope_theta", 10000.0)),
        projection_hidden_dim=int(_first(vc, "projection_hidden_dim", default=4096)),
        projector_bias=bool(_first(vc, "projector_bias", default=False)),
        text_hidden_size=int(vc.text_hidden_size),
    )


def patch_grid(rows: int, cols: int, merge: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(row, col) of every patch, in the order the processor tiles them: tokens in raster order, each token's
    merge x merge patches in raster order inside it."""
    k = torch.arange(rows * cols * merge * merge)
    token, inner = k // (merge * merge), k % (merge * merge)
    row = (token // cols) * merge + inner // merge
    col = (token % cols) * merge + inner % merge
    return row, col


def _bilinear_taps(index: torch.Tensor, size: int, side: int) -> tuple[torch.Tensor, torch.Tensor]:
    """F.interpolate(bilinear, align_corners=False) of a side-long table to size, as 2 taps and weights per index."""
    src = (index.to(torch.float32) + 0.5) * side / size - 0.5
    floor = torch.floor(src)
    offsets = torch.arange(0, 2)
    taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)
    weights = (1 - (src[:, None] - floor[:, None] - offsets).abs()).clamp(min=0)
    return taps, weights


def position_taps(row: torch.Tensor, col: torch.Tensor, height: int, width: int, side: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[P, 4] indices into the flattened side x side table and their fp32 weights."""
    h_taps, h_weights = _bilinear_taps(row, height, side)
    w_taps, w_weights = _bilinear_taps(col, width, side)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices, weights


def _rotate_pairwise(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class _LayerScale(BaseOP):
    def __init__(self, size: int):
        self.gamma = torch.empty(size)

    forward = None


class Inkling2VisionEmbeddings(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.patch_embedding = LinearReplicated(vc.num_channels * vc.patch_size**2, vc.hidden_size, has_bias=False)
        self.position_embedding = torch.empty(vc.posemb_grid_size**2, vc.hidden_size)
        self.class_embedding = torch.empty(vc.hidden_size)
        self.cls_position_embedding = torch.empty(vc.hidden_size)
        self._vc = vc

    def forward(self, pixel_values: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        """[1 + patches, hidden]: the class token, then the patches in processor order."""
        vc = self._vc
        m, p = vc.merge_size, vc.patch_size
        weight = self.patch_embedding.weight
        # each token's tile -> its m x m patches, each flattened (row, col, channel) as the checkpoint's projection reads it
        x = pixel_values.to(weight.device, weight.dtype).reshape(-1, m, p, m, p, vc.num_channels).transpose(2, 3)
        patches = self.patch_embedding.forward(x.reshape(-1, vc.num_channels * p * p))
        row, col = patch_grid(rows, cols, m)
        indices, weights = position_taps(row, col, rows * m, cols * m, vc.posemb_grid_size)
        indices, weights = indices.to(weight.device), weights.to(weight.device)
        # bf16 table rows times fp32 weights, summed in fp32 and rounded once, as the reference does
        position = (self.position_embedding[indices] * weights[:, :, None]).sum(1)
        patches = patches + position.to(patches.dtype)
        cls = (self.class_embedding + self.cls_position_embedding).to(patches.dtype)
        return torch.cat([cls[None], patches])


class Inkling2VisionAttention(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.num_heads = vc.num_heads
        self.head_dim = vc.hidden_size // vc.num_heads
        self.q_proj = LinearReplicated(vc.hidden_size, vc.hidden_size, has_bias=True)
        self.k_proj = LinearReplicated(vc.hidden_size, vc.hidden_size, has_bias=True)
        self.v_proj = LinearReplicated(vc.hidden_size, vc.hidden_size, has_bias=True)
        self.out_proj = LinearReplicated(vc.hidden_size, vc.hidden_size, has_bias=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        S, H, D = x.shape[0], self.num_heads, self.head_dim
        q, k, v = (proj.forward(x).view(S, H, D) for proj in (self.q_proj, self.k_proj, self.v_proj))
        # rope on interleaved channel pairs, in fp32 and rounded back
        q = (q.float() * cos + _rotate_pairwise(q.float()) * sin).to(x.dtype)
        k = (k.float() * cos + _rotate_pairwise(k.float()) * sin).to(x.dtype)
        # the fused SDPA kernels want a batch dim; 3-D inputs fall back to the O(S^2) math path
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
        return self.out_proj.forward(o[0].transpose(0, 1).reshape(S, H * D))


class Inkling2VisionMLP(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.c_fc = LinearReplicated(vc.hidden_size, vc.intermediate_size, has_bias=True)
        self.c_proj = LinearReplicated(vc.intermediate_size, vc.hidden_size, has_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj.forward(F.gelu(self.c_fc.forward(x)))


class Inkling2VisionLayer(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.ln_1 = LayerNorm(vc.hidden_size, vc.layer_norm_eps)
        self.self_attn = Inkling2VisionAttention(vc)
        self.ls_1 = _LayerScale(vc.hidden_size)
        self.ln_2 = LayerNorm(vc.hidden_size, vc.layer_norm_eps)
        self.mlp = Inkling2VisionMLP(vc)
        self.ls_2 = _LayerScale(vc.hidden_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.ls_1.gamma * self.self_attn.forward(self.ln_1.forward(x), cos, sin)
        return x + self.ls_2.gamma * self.mlp.forward(self.ln_2.forward(x))


class Inkling2VisionBackbone(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.embeddings = Inkling2VisionEmbeddings(vc)
        self.ln_pre = LayerNorm(vc.hidden_size, vc.layer_norm_eps)
        self.layers = OPList([Inkling2VisionLayer(vc) for _ in range(vc.num_layers)])

    forward = None


class Inkling2VisionModel(BaseOP):
    """Tiles of one image ([rows * cols, tile, tile, 3], as the processor emits them) -> [rows * cols, text_hidden]."""

    def __init__(self, vc: VisionConfig):
        self.backbone = Inkling2VisionBackbone(vc)
        self.proj_in = LinearReplicated(vc.merge_size**2 * vc.hidden_size, vc.projection_hidden_dim, has_bias=vc.projector_bias)
        self.proj_out = LinearReplicated(vc.projection_hidden_dim, vc.text_hidden_size, has_bias=vc.projector_bias)
        head_dim = vc.hidden_size // vc.num_heads
        # one rope band per axis over head_dim / 2 channels, built on the CPU as the reference builds it
        band = torch.arange(0, head_dim // 2, 2, dtype=torch.float32, device="cpu") / (head_dim // 2)
        self._inv_freq = 1.0 / (vc.rope_theta ** band)
        self._vc = vc
        self._streamer: BlockWeightStreamer | None = None

    def place_weights(self, mode: str) -> None:
        """gpu: every tensor resident; host: the layer tensors in pinned banks streamed two layers at a time."""
        layers = self.backbone.layers.op_list
        if mode == "host" and self._streamer is None:
            self._streamer = BlockWeightStreamer(layers, self.proj_in.weight.device)
        elif mode == "gpu" and self._streamer is not None:
            self._streamer.unstream()
            self._streamer = None
        elif mode not in ("gpu", "host"):
            raise ValueError(f"unknown vision weight placement {mode!r}")

    def _layers(self):
        layers = self.backbone.layers.op_list
        return enumerate(layers) if self._streamer is None else self._streamer.blocks(layers)

    def _rope(self, rows: int, cols: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """[1 + patches, 1, head_dim] cos and sin: patches at their 1-based (row, col), the class token at (0, 0)."""
        row, col = patch_grid(rows, cols, self._vc.merge_size)
        pos = torch.cat([torch.zeros(1, 2, dtype=torch.long), torch.stack([row, col], -1) + 1]).to(device)
        freqs = pos[..., None].float() * self._inv_freq.to(device)
        cos, sin = (torch.cat([f[:, 1], f[:, 0]], -1).repeat_interleave(2, -1)[:, None] for f in (freqs.cos(), freqs.sin()))
        return cos, sin

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, grid_thw: List[int]) -> torch.Tensor:
        t, rows, cols = grid_thw
        if t != 1:
            raise ValueError("inkling2 vision takes images only (grid t == 1)")
        x = self.backbone.embeddings.forward(pixel_values, rows, cols)
        x = self.backbone.ln_pre.forward(x)
        cos, sin = self._rope(rows, cols, x.device)
        for _, layer in self._layers():
            x = layer.forward(x, cos, sin)
        # drop the class token; each run of merge^2 patches is one text token
        merged = x[1:].reshape(rows * cols, -1)
        return self.proj_out.forward(F.gelu(self.proj_in.forward(merged)))


__all__ = ["Inkling2VisionModel", "VisionConfig", "parse_vision_config", "patch_grid", "position_taps"]
