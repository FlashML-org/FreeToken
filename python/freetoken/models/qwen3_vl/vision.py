"""Qwen VL vision tower: full-attention ViT blocks and a 2x2 patch merger; DeepStack taps ride along as extra output columns."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from freetoken.distributed import get_tp_info
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearRowParallel,
    OPList,
)
from freetoken.utils import div_even

from freetoken.models.weight_stream import BlockWeightStreamer

if TYPE_CHECKING:
    from freetoken.layers.quantization import QuantConfig
    from freetoken.message import MMItem
    from .config import VisionConfig


class VisionLayerNorm(BaseOP):
    def __init__(self, size: int, eps: float = 1e-6):
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)
        self._size = size
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self._size,), self.weight, self.bias, self._eps)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


_warned_rope_fallback = False


def _apply_vision_rope(
    q: torch.Tensor, k: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor, head_dim: int
) -> None:
    """In-place 2D rope through the NeoX text kernel: the [c,c]/[s,s] layout is the half-rotation layout and positions index per-token cache rows."""
    global _warned_rope_fallback
    S = q.shape[0]
    try:
        from freetoken.kernel.triton.rope import apply_rope_with_cos_sin_cache_inplace

        apply_rope_with_cos_sin_cache_inplace(
            positions, q.view(S, -1), k.view(S, -1), head_dim, cache, is_neox=True
        )
        return
    except ImportError:
        # only a missing triton falls back; a CUDA error must not re-run eager on a half-rotated q/k
        if not _warned_rope_fallback:
            _warned_rope_fallback = True
            from freetoken.utils import init_logger

            init_logger(__name__).warning("vision rope triton kernel unavailable; using eager")
    half = cache.shape[1] // 2
    cos = torch.cat((cache[:, :half], cache[:, :half]), dim=-1).unsqueeze(-2)
    sin = torch.cat((cache[:, half:], cache[:, half:]), dim=-1).unsqueeze(-2)
    q.copy_((q.float() * cos + _rotate_half(q.float()) * sin).to(q.dtype))
    k.copy_((k.float() * cos + _rotate_half(k.float()) * sin).to(k.dtype))


class VisionAttention(BaseOP):
    def __init__(self, vc: VisionConfig, *, quant_config: QuantConfig | None = None, prefix: str = ""):
        self.num_heads = div_even(vc.num_heads, get_tp_info().size)
        self.head_dim = vc.hidden_size // vc.num_heads
        self.qkv = LinearQKVMerged(
            vc.hidden_size, self.head_dim, vc.num_heads, vc.num_heads, has_bias=True,
            quant_config=quant_config, prefix=f"{prefix}.qkv",
        )
        self.proj = LinearOProj(
            vc.hidden_size, vc.hidden_size, has_bias=True, quant_config=quant_config, prefix=f"{prefix}.proj"
        )

    def attend(
        self, qkv: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor, lengths: List[int]
    ) -> torch.Tensor:
        """Rope + bidirectional attention within each image on the fused qkv projection; returns [S, heads * head_dim]."""
        S = qkv.shape[0]
        # q/k/v stay views into the fused projection: the rope kernel takes strides, SDPA needs only head_dim contiguous
        q, k, v = qkv.view(S, 3, self.num_heads, self.head_dim).unbind(1)
        _apply_vision_rope(q, k, cache, positions, self.head_dim)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        # bidirectional attention within each image; images never attend across
        outs = [
            F.scaled_dot_product_attention(qs, ks, vs)
            for qs, ks, vs in zip(
                torch.split(q, lengths, dim=2),
                torch.split(k, lengths, dim=2),
                torch.split(v, lengths, dim=2),
            )
        ]
        o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
        return o[0].transpose(0, 1).reshape(S, -1)


class VisionMLP(BaseOP):
    def __init__(self, vc: VisionConfig, *, quant_config: QuantConfig | None = None, prefix: str = ""):
        self.linear_fc1 = LinearColParallelMerged(
            vc.hidden_size, [vc.intermediate_size], has_bias=True, quant_config=quant_config, prefix=f"{prefix}.linear_fc1"
        )
        self.linear_fc2 = LinearRowParallel(
            vc.intermediate_size, vc.hidden_size, has_bias=True, quant_config=quant_config, prefix=f"{prefix}.linear_fc2"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= _MLP_ROWS:
            return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x), approximate="tanh"))
        # per-token op: row chunks bound the two [rows, intermediate] temporaries while the GEMMs stay large
        out = torch.empty_like(x)
        for lo in range(0, x.shape[0], _MLP_ROWS):
            rows = x[lo : lo + _MLP_ROWS]
            out[lo : lo + _MLP_ROWS] = self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(rows), approximate="tanh"))
        return out


# 4096 rows x 4304 x bf16 = 34 MiB per temporary while the GEMM still fills the GPU
# FREETOKEN_VIT_MLP_ROWS=0 disables the chunking; cuBLAS may round a row chunk one bf16 ulp differently from the whole image
_MLP_ROWS = int(os.environ.get("FREETOKEN_VIT_MLP_ROWS", 4096)) or 1 << 62
# 16384 rows x 1152 x fp32 = 72 MiB per interpolation temporary
_POS_ROWS = 16384


class VisionBlock(BaseOP):
    def __init__(self, vc: VisionConfig, *, quant_config: QuantConfig | None = None, prefix: str = ""):
        self.norm1 = VisionLayerNorm(vc.hidden_size)
        self.norm2 = VisionLayerNorm(vc.hidden_size)
        self.attn = VisionAttention(vc, quant_config=quant_config, prefix=f"{prefix}.attn")
        self.mlp = VisionMLP(vc, quant_config=quant_config, prefix=f"{prefix}.mlp")

    def forward(
        self, x: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor, lengths: List[int]
    ) -> torch.Tensor:
        x = x + self._attention(x, cache, positions, lengths)
        return x + self.mlp.forward(self.norm2.forward(x))

    def _attention(
        self, x: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor, lengths: List[int]
    ) -> torch.Tensor:
        # each temporary dies inside the next call: norm out in the qkv GEMM, qkv in attention, its output in proj
        o = self.attn.attend(self.attn.qkv.forward(self.norm1.forward(x)), cache, positions, lengths)
        return self.attn.proj.forward(o)


class _Conv3dParams(BaseOP):
    def __init__(self, out_ch: int, shape: tuple):
        self.weight = torch.empty(out_ch, *shape)
        self.bias = torch.empty(out_ch)

    forward = None


class _EmbeddingParams(BaseOP):
    def __init__(self, rows: int, dim: int):
        self.weight = torch.empty(rows, dim)

    forward = None


class VisionConv3dPatchEmbed(BaseOP):
    """Kept as a real Conv3d (kernel == stride): an equivalent unfolded matmul rounds differently."""

    def __init__(self, vc: VisionConfig):
        self._shape = (vc.in_channels, vc.temporal_patch_size, vc.patch_size, vc.patch_size)
        self.proj = _Conv3dParams(vc.hidden_size, self._shape)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = pixel_values.view(-1, *self._shape).to(self.proj.weight.dtype)
        out = F.conv3d(x, self.proj.weight, self.proj.bias, stride=self._shape[1:])
        return out.view(x.shape[0], -1)


class VisionPatchMerger(BaseOP):
    def __init__(
        self,
        vc: VisionConfig,
        use_postshuffle_norm: bool = False,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        merged = vc.hidden_size * vc.spatial_merge_size**2
        # the DeepStack mergers normalize the 2x2-merged vector, the final merger each patch
        self.norm = VisionLayerNorm(merged if use_postshuffle_norm else vc.hidden_size)
        self.linear_fc1 = LinearColParallelMerged(
            merged, [merged], has_bias=True, quant_config=quant_config, prefix=f"{prefix}.linear_fc1"
        )
        self.linear_fc2 = LinearRowParallel(
            merged, vc.out_hidden_size, has_bias=True, quant_config=quant_config, prefix=f"{prefix}.linear_fc2"
        )
        self._merged = merged
        self._postshuffle_norm = use_postshuffle_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._postshuffle_norm:
            x = self.norm.forward(x.view(-1, self._merged))
        else:
            x = self.norm.forward(x).view(-1, self._merged)
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x)))


class Qwen3VLVisionModel(BaseOP):
    """Pixels -> per-image soft tokens [sum(h*w)/merge^2, out_hidden x (1 + DeepStack levels)].

    Columns past out_hidden are the DeepStack side features in deepstack_visual_indexes order.
    """

    def __init__(self, vc: VisionConfig, *, quant_config: QuantConfig | None = None, prefix: str = "visual"):
        self.patch_embed = VisionConv3dPatchEmbed(vc)
        self.pos_embed = _EmbeddingParams(vc.num_position_embeddings, vc.hidden_size)
        self.blocks = OPList(
            [VisionBlock(vc, quant_config=quant_config, prefix=f"{prefix}.blocks.{i}") for i in range(vc.depth)]
        )
        self.merger = VisionPatchMerger(vc, quant_config=quant_config, prefix=f"{prefix}.merger")
        self.deepstack_merger_list = OPList(
            [
                VisionPatchMerger(
                    vc, use_postshuffle_norm=True, quant_config=quant_config, prefix=f"{prefix}.deepstack_merger_list.{k}"
                )
                for k in range(len(vc.deepstack_visual_indexes))
            ]
        )
        self._vc = vc
        self._num_grid_per_side = int(vc.num_position_embeddings**0.5)
        self._inv_dim = (vc.hidden_size // vc.num_heads) // 2
        self._inv_freq: torch.Tensor | None = None
        self._streamer: BlockWeightStreamer | None = None

    def place_weights(self, mode: str) -> None:
        """gpu: every tensor resident; host: block tensors in pinned banks streamed two blocks at a time (mergers and embeddings stay resident)."""
        if mode == "host" and self._streamer is None:
            self._streamer = BlockWeightStreamer(self.blocks.op_list, self.pos_embed.weight.device)
        elif mode == "gpu" and self._streamer is not None:
            self._streamer.unstream()
            self._streamer = None
        elif mode not in ("gpu", "host"):
            raise ValueError(f"unknown vision weight placement {mode!r}")

    def _blocks(self):
        if self._streamer is None:
            return enumerate(self.blocks.op_list)
        return self._streamer.blocks(self.blocks.op_list)

    def _add_pos_embed(self, x: torch.Tensor, grid: torch.Tensor) -> None:
        """x += bilinear resample of the position table, in row chunks so the fp32 temporaries stay small."""
        from transformers.vision_utils import get_vision_interpolation_indices_and_weights

        interp_idx, interp_w = get_vision_interpolation_indices_and_weights(
            grid,
            num_grid_per_side=self._num_grid_per_side,
            mode="bilinear",
            align_corners=True,
            spatial_merge_size=self._vc.spatial_merge_size,
        )
        table = self.pos_embed.weight
        for lo in range(0, x.shape[0], _POS_ROWS):
            idx, w = interp_idx[lo : lo + _POS_ROWS], interp_w[lo : lo + _POS_ROWS]
            # one corner at a time, multiply then add: the same per-element arithmetic as the fused [S, 4, hidden] .sum(1)
            acc = F.embedding(idx[:, 0], table).float() * w[:, 0, None]
            for k in range(1, idx.shape[1]):
                acc += F.embedding(idx[:, k], table).float() * w[:, k, None]
            x[lo : lo + _POS_ROWS].add_(acc.to(x.dtype))

    def _rope_table(self, grid: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        from transformers.vision_utils import get_vision_position_ids

        if self._inv_freq is None or self._inv_freq.device != device:
            # built lazily: the model itself is constructed on the meta device
            # kept in the model dtype (bf16) on purpose; fp32 here would change the rope numerics
            inv_dim = self._inv_dim
            self._inv_freq = (
                1.0 / (10000.0 ** (torch.arange(0, inv_dim, 2, dtype=torch.float32, device=device) / inv_dim))
            ).to(self.pos_embed.weight.dtype)
        pos_ids = get_vision_position_ids(grid, self._vc.spatial_merge_size)
        # long * bf16 -> bf16 freqs on purpose; fp32 freqs would round differently
        freqs = (pos_ids.unsqueeze(-1) * self._inv_freq).flatten(1)  # [S, head_dim/2]
        # per-token rope rows [cos | sin] consumed by the NeoX kernel via positions=arange
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.float32).contiguous()
        positions = torch.arange(pos_ids.shape[0], dtype=torch.int32, device=device)
        return cache, positions

    def _embed(self, pixel_values: torch.Tensor, grid_thw: List[List[int]]) -> tuple[torch.Tensor, tuple]:
        """Patch embeddings with the position table added, and the attention inputs every block takes."""
        from transformers.vision_utils import get_vision_cu_seqlens

        device = self.pos_embed.weight.device
        grid = torch.tensor(grid_thw, dtype=torch.long, device=device)
        x = self.patch_embed.forward(pixel_values.to(device))
        self._add_pos_embed(x, grid)
        cache, positions = self._rope_table(grid, device)
        return x, (cache, positions, torch.diff(get_vision_cu_seqlens(grid)).tolist())

    def _encode(
        self, pixel_values: torch.Tensor, grid_thw: List[List[int]], groups: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pixels through the patch embedding and the block stack; with ``groups`` each DeepStack tap's merged feature is copied into groups[1 + k]."""
        taps = self._vc.deepstack_visual_indexes
        x, attn = self._embed(pixel_values, grid_thw)
        for i, blk in self._blocks():
            x = blk.forward(x, *attn)
            if groups is not None and i in taps:
                k = taps.index(i)
                groups[1 + k].copy_(self.deepstack_merger_list.op_list[k].forward(x), non_blocking=True)
        return x

    @staticmethod
    def _unpark(groups: torch.Tensor, device: torch.device) -> torch.Tensor:
        """[rows, n * width] on the device from host slabs [n, rows, width]; group k fills columns k*width:(k+1)*width."""
        n, rows, width = groups.shape
        out = torch.empty(rows, n * width, dtype=groups.dtype, device=device)
        for k in range(n):
            out[:, k * width : (k + 1) * width].copy_(groups[k], non_blocking=True)
        return out

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, grid_thw: List[List[int]]) -> torch.Tensor:
        """Soft tokens [rows, width x (1 + taps)], rows = sum(h*w) / merge^2: the merger's columns first, then one DeepStack tap per range.

        Same numbers as ``forward_naive``.
        """
        vc = self._vc
        taps = vc.deepstack_visual_indexes
        if not taps:
            return self.merger.forward(self._encode(pixel_values, grid_thw))
        # the merger outputs leave the GPU as they are produced and come back after the last block, so only the block working set is resident while the blocks run
        rows = pixel_values.shape[0] // vc.spatial_merge_size**2
        groups = torch.empty(
            (1 + len(taps), rows, vc.out_hidden_size), dtype=self.pos_embed.weight.dtype, device="cpu",
            pin_memory=torch.cuda.is_available(),
        )
        x = self._encode(pixel_values, grid_thw, groups)
        groups[0].copy_(self.merger.forward(x), non_blocking=True)
        del x
        return self._unpark(groups, self.pos_embed.weight.device)

    @torch.inference_mode()
    def forward_naive(self, pixel_values: torch.Tensor, grid_thw: List[List[int]]) -> torch.Tensor:
        """Reference forward, written the plain way: the block stack, then the merger outputs side by side."""
        taps = self._vc.deepstack_visual_indexes
        x, attn = self._embed(pixel_values, grid_thw)
        features = []
        for i, blk in self._blocks():
            x = blk.forward(x, *attn)
            if i in taps:
                features.append(self.deepstack_merger_list.op_list[taps.index(i)].forward(x))
        return torch.cat([self.merger.forward(x)] + features, dim=1)


class QwenVLVisionMixin:
    """The engine hooks of a wrapper that owns a Qwen VL tower as ``self.visual``."""

    visual: Qwen3VLVisionModel

    def place_encoder_weights(self, mode: str) -> None:
        self.visual.place_weights(mode)

    def encode(self, item: MMItem) -> torch.Tensor:
        return self.visual.forward(item.feature, [item.grid_thw])


__all__ = ["Qwen3VLVisionModel", "QwenVLVisionMixin"]
