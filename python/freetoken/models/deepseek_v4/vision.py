"""DeepSeek-V4 vision tower, aligner and image-block layout.

Port of the reference ``inference/vision.py`` / ``inference/image_processor.py`` from
deepseek-ai/DeepSeek-V4-Flash-Vision-Exp. The tower, the aligner and the four sentinel
vectors are :class:`BaseOP` children of the model WRAPPER, so their unprefixed
state-dict keys (``vision.patch_embed.proj.weight``, ``aligner.w1.weight``,
``image_start``, ...) match the checkpoint tensor names verbatim with no renaming on
either side; every tensor is bf16 exactly as stored.

Image tokens occupy plain sequential positions: the 2-D structure lives inside the
tower (2-D RoPE over the patch grid) and in the N-layout of the token block the
:func:`build_image_block` / :func:`assemble_block` pair renders. All block embeddings,
sentinels included, come from the vision side, so the text embedding row of the
placeholder id is never read -- the span is overwritten before the first layer.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, OPList
from freetoken.models.weight_stream import BlockWeightStreamer

from .config import DSV4VisionConfig

# Sentinel token types inside an image block (reference inference/image_processor.py).
IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
# An image block is padded so its first row token lands on a compressor stride boundary.
COMPRESS_PAD_TO = 4


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class _Linear(BaseOP):
    """Linear in checkpoint-native layout: ``weight`` [out, in] plus an optional ``bias``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.weight = torch.empty(out_features, in_features, dtype=torch.bfloat16)
        self.bias = torch.empty(out_features, dtype=torch.bfloat16) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RMSNorm(BaseOP):
    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = torch.empty(dim, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(BaseOP):
    def __init__(self, patch_size: int, dim: int):
        self.proj = _Linear(3 * patch_size**2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj.forward(x.flatten(1))


class VisionAttention(BaseOP):
    def __init__(self, dim: int, n_heads: int):
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wqkv = _Linear(dim, 3 * dim)
        self.wo = _Linear(dim, dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv.forward(x).chunk(3, dim=-1))
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo.forward(o.transpose(0, 1).reshape(n, -1))


class VisionMLP(BaseOP):
    def __init__(self, dim: int, inter_dim: int):
        self.w1 = _Linear(dim, 2 * inter_dim, bias=False)
        self.w2 = _Linear(inter_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1.forward(x).chunk(2, dim=-1)
        return self.w2.forward(F.silu(gate) * up)


class VisionBlock(BaseOP):
    def __init__(self, dim: int, n_heads: int, inter_dim: int, eps: float):
        self.norm1 = RMSNorm(dim, eps)
        self.attn = VisionAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim, eps)
        self.mlp = VisionMLP(dim, inter_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward(self.norm1.forward(x), cos, sin)
        return x + self.mlp.forward(self.norm2.forward(x))


class DSV4VisionTower(BaseOP):
    """Native-resolution ViT: bidirectional attention inside one image, 2-D RoPE over its patch grid."""

    def __init__(self, vc: DSV4VisionConfig):
        self.rope_dim = vc.vision_dim // vc.vision_n_heads // 2
        self.rope_theta = vc.vision_rope_theta
        self.patch_embed = PatchEmbed(vc.vision_patch_size, vc.vision_dim)
        self.blocks = OPList(
            [
                VisionBlock(vc.vision_dim, vc.vision_n_heads, vc.vision_inter_dim, eps=1e-6)
                for _ in range(vc.vision_n_layers)
            ]
        )
        self.norm = RMSNorm(vc.vision_dim, 1e-6)
        self._streamer: BlockWeightStreamer | None = None

    def place_weights(self, mode: str) -> None:
        """gpu: every tensor resident; host: the block stack in pinned banks, two blocks streamed at a time (patch embed and final norm stay resident)."""
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

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        patches = patches.to(self.patch_embed.proj.weight.dtype)
        x = self.patch_embed.forward(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        cos, sin = cos.to(x.device), sin.to(x.device)
        for _, block in self._blocks():
            x = block.forward(x, cos, sin)
        return self.norm.forward(x)


class DSV4Aligner(BaseOP):
    """3x3/stride-3 merge of every 3x3 patch grid into one token, then a 2-layer erf-GELU MLP."""

    def __init__(self, vc: DSV4VisionConfig):
        self.downsample_ratio = vc.vision_downsample_ratio
        in_dim = vc.vision_dim * self.downsample_ratio**2
        self.w1 = _Linear(in_dim, vc.text_dim)
        self.w2 = _Linear(vc.text_dim, vc.text_dim)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        dtype = self.w1.weight.dtype
        x = x.to(dtype)
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2.forward(F.gelu(self.w1.forward(x)))


def grid_tokens(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    """LLM-token footprint of a resized image: the N-layout row/align padding included."""
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(max_w * patch_size * downsample_ratio / width, max_h * patch_size * downsample_ratio / height)
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    """Largest grid whose block still fits ``max_n_token`` once the alignment pads are added."""
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int):
    """The block's token types in their final order, plus the aligner-row order of its IMAGE slots.

    ``types`` [n_out] places every token of the block (lead pads, START, the row-interleaved
    IMAGE tokens, trailing pads, END). The reference streams them as ``vocab_size + types``;
    here the span carries the image's content pad id and the assembled embeddings are
    scattered over it. ``perm`` [G] maps aligner outputs (row-major grid) into the IMAGE slots.

    ``start_pos`` is the block's absolute position in the prompt: the lead-pad count aligns
    the first row token to a compressor stride boundary, so the layout -- and the block's
    length -- depend on where the image lands.
    """
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = torch.arange(rows * row_len).view(rows // 2, 2, row_len).transpose(1, 2).reshape(-1)
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(n_llm_h * n_llm_w).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, perm


def assemble_block(
    aligned: torch.Tensor,
    types: torch.Tensor,
    perm: torch.Tensor,
    sentinels: torch.Tensor,
) -> torch.Tensor:
    """Block embeddings [n_out, dim] from the aligner rows and the learned sentinels.

    ``sentinels`` [5, dim] is ``(start, pad, pad, newline, end)``, indexable by the type
    enum exactly like the reference ``merge_image_embeddings``.
    """
    block = sentinels[types.to(sentinels.device)]
    block[types == IMAGE] = aligned[perm.to(aligned.device)]
    return block


__all__ = [
    "COMPRESS_PAD_TO",
    "DSV4Aligner",
    "DSV4VisionTower",
    "IMAGE",
    "IMAGE_END",
    "IMAGE_NEW_LINE",
    "IMAGE_PAD",
    "IMAGE_START",
    "assemble_block",
    "build_image_block",
    "grid_tokens",
    "safe_resize",
    "solve_resize_ratio",
]
