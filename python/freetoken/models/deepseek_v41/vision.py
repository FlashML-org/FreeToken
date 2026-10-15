"""DeepSeek ViT and 3x3 pixel-unshuffle aligner, with the shared encoder weight streamer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, OPList
from freetoken.models.weight_stream import BlockWeightStreamer


class VisionNorm(BaseOP):
    def __init__(self, dim: int):
        self.weight = torch.empty(dim, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6) * self.weight).to(x.dtype)


class VisionAttention(BaseOP):
    def __init__(self, vc):
        self.n_heads = vc.num_attention_heads
        self.head_dim = vc.hidden_size // self.n_heads
        self.wqkv = LinearReplicated(vc.hidden_size, 3 * vc.hidden_size, has_bias=True)
        self.wo = LinearReplicated(vc.hidden_size, vc.hidden_size, has_bias=True)

    def forward(self, x, cos, sin):
        n = x.shape[0]
        q, k, v = (t.reshape(n, self.n_heads, self.head_dim) for t in self.wqkv.forward(x).chunk(3, -1))
        def rotary(t):
            a, b = t.float().chunk(2, -1)
            return torch.cat((a * cos - b * sin, b * cos + a * sin), -1).to(t.dtype)
        q, k = rotary(q), rotary(k)
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v)
        return self.wo.forward(y[0].transpose(0, 1).reshape(n, -1))


class VisionMLP(BaseOP):
    def __init__(self, vc):
        self.w1 = LinearReplicated(vc.hidden_size, 2 * vc.intermediate_size, has_bias=False)
        self.w2 = LinearReplicated(vc.intermediate_size, vc.hidden_size, has_bias=False)

    def forward(self, x):
        gate, up = self.w1.forward(x).chunk(2, -1)
        return self.w2.forward(F.silu(gate) * up)


class VisionBlock(BaseOP):
    def __init__(self, vc):
        self.norm1 = VisionNorm(vc.hidden_size)
        self.attn = VisionAttention(vc)
        self.norm2 = VisionNorm(vc.hidden_size)
        self.mlp = VisionMLP(vc)

    def forward(self, x, cos, sin):
        x = x + self.attn.forward(self.norm1.forward(x), cos, sin)
        return x + self.mlp.forward(self.norm2.forward(x))


class PatchEmbed(BaseOP):
    def __init__(self, vc):
        self.proj = LinearReplicated(3 * vc.patch_size**2, vc.hidden_size, has_bias=True)

    def forward(self, x):
        return self.proj.forward(x.flatten(1))


class VisionTransformer(BaseOP):
    def __init__(self, vc):
        self.patch_embed = PatchEmbed(vc)
        self.blocks = OPList([VisionBlock(vc) for _ in range(vc.num_hidden_layers)])
        self.norm = VisionNorm(vc.hidden_size)
        self._vc = vc
        self._streamer = None

    def place_weights(self, mode: str) -> None:
        if mode == "host" and self._streamer is None:
            self._streamer = BlockWeightStreamer(self.blocks.op_list, self.patch_embed.proj.weight.device)
        elif mode == "gpu" and self._streamer is not None:
            self._streamer.unstream()
            self._streamer = None
        elif mode not in ("gpu", "host"):
            raise ValueError(f"unknown vision weight placement {mode!r}")

    def forward(self, patches, h: int, w: int):
        x = self.patch_embed.forward(patches)
        dim = self._vc.hidden_size // self._vc.num_attention_heads // 2
        inv = 1.0 / (self._vc.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim))
        rows = torch.arange(h, device=x.device)[:, None].expand(h, w)
        cols = torch.arange(w, device=x.device)[None, :].expand(h, w)
        angles = (torch.stack((rows, cols), -1).reshape(-1, 2, 1).float() * inv).flatten(1)
        cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
        blocks = enumerate(self.blocks.op_list) if self._streamer is None else self._streamer.blocks(self.blocks.op_list)
        for _, block in blocks:
            x = block.forward(x, cos, sin)
        return self.norm.forward(x)


class VisionAligner(BaseOP):
    def __init__(self, vc, dim: int):
        self.ratio = vc.downsample_ratio
        self.w1 = LinearReplicated(vc.hidden_size * self.ratio**2, dim, has_bias=True)
        self.w2 = LinearReplicated(dim, dim, has_bias=True)

    def forward(self, x, h: int, w: int):
        r = self.ratio
        x = x.view(h, w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -w % r, 0, -h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).T
        return self.w2.forward(F.gelu(self.w1.forward(x)))


class DeepseekV41Vision(BaseOP):
    def __init__(self, vc, dim: int, *, num_layers: int, num_experts: int):
        self.vision = VisionTransformer(vc)
        self.aligner = VisionAligner(vc, dim)
        self.image_start = torch.empty(dim)
        self.image_end = torch.empty(dim)
        self.image_newline = torch.empty(dim)
        # Keep all image-only parameters under visual, including FTW/text-only filtering.
        self.routing_bias = torch.empty(num_layers, num_experts, dtype=torch.float32)

    def place_weights(self, mode: str) -> None:
        self.vision.place_weights(mode)

    @torch.inference_mode()
    def forward(self, item):
        _, h, w = item.grid_thw
        patches = item.feature.to(device=self.image_start.device, dtype=self.image_start.dtype)
        x = self.aligner.forward(self.vision.forward(patches, h, w), h, w)
        r = self.aligner.ratio
        nh, nw = (h + r - 1) // r, (w + r - 1) // r
        out = x.new_empty(nh * (nw + 1) + 2, x.shape[-1])
        out[0], out[-1] = self.image_start, self.image_end
        rows = out[1:-1].view(nh, nw + 1, -1)
        rows[:, :nw] = x.view(nh, nw, -1)
        rows[:, nw] = self.image_newline
        return out
