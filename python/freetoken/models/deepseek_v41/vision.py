"""DeepSeek-V4.1 vision tower; upstream MIT attribution is in this package's NOTICE."""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.layers import BaseOP


class _ModuleBlockAdapter(BaseOP):
    """Expose native module parameters to the shared block streamer's tensor slots."""

    def __init__(self, module):
        self._module = module
        for name, parameter in module.named_parameters(recurse=False):
            setattr(self, name, parameter)
        for name, child in module.named_children():
            setattr(self, name, _ModuleBlockAdapter(child))

    def __setattr__(self, name, value):
        module = self.__dict__.get("_module")
        if module is not None and name in module._parameters:
            value = value if isinstance(value, nn.Parameter) else nn.Parameter(value, requires_grad=False)
            setattr(module, name, value)
        object.__setattr__(self, name, value)

    def forward(self, *args):
        return self._module(*args)


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim))
    hpos = torch.arange(n_h, device="cpu").unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device="cpu").unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj = nn.Linear(3 * args.vision_patch_size**2, args.vision_dim, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.vision_n_heads
        self.head_dim = args.vision_dim // args.vision_n_heads
        self.wqkv = nn.Linear(args.vision_dim, 3 * args.vision_dim, dtype=torch.bfloat16)
        self.wo = nn.Linear(args.vision_dim, args.vision_dim, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
        )
        return self.wo(o.squeeze(0).transpose(0, 1).reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(args.vision_dim, 2 * args.vision_inter_dim, bias=False, dtype=torch.bfloat16)
        self.w2 = nn.Linear(args.vision_inter_dim, args.vision_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1 = RMSNorm(args.vision_dim)
        self.attn = Attention(args)
        self.norm2 = RMSNorm(args.vision_dim)
        self.mlp = MLP(args)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, args):
        super().__init__()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.vision_n_layers)])
        self.norm = RMSNorm(args.vision_dim)
        self._streamer = None
        self._stream_blocks = None

    def place_weights(self, mode):
        if mode not in {"host", "gpu"}:
            raise ValueError(f"Unsupported vision weight placement: {mode}")
        if mode == "gpu" and self._streamer is not None:
            self._streamer.unstream()
            self._streamer, self._stream_blocks = None, None
        elif mode == "host" and self._streamer is None and self.blocks:
            from freetoken.models.weight_stream import BlockWeightStreamer

            device = self.patch_embed.proj.weight.device
            self._stream_blocks = [_ModuleBlockAdapter(block) for block in self.blocks]
            self._streamer = BlockWeightStreamer(self._stream_blocks, device)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        if patches.shape[0] != n_h * n_w:
            raise ValueError("vision patch count does not match its grid")
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        cos, sin = cos.to(x.device), sin.to(x.device)
        blocks = (self._streamer.blocks(self._stream_blocks) if self._streamer is not None
                  else enumerate(self.blocks))
        try:
            for _, block in blocks:
                x = block.forward(x, cos, sin)
        finally:
            if self._streamer is not None:
                blocks.close()
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        in_dim = args.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, args.dim, dtype=torch.bfloat16)
        self.w2 = nn.Linear(args.dim, args.dim, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


@torch.inference_mode()
def image_span_embeddings(transformer, patches, n_h, n_w, types):
    """Encode the full native span, including learned image delimiters and row separators."""
    from .image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START

    weight = transformer.vision.patch_embed.proj.weight
    types = torch.as_tensor(types, device=weight.device, dtype=torch.int64)
    if types.ndim != 1 or torch.any((types < IMAGE_START) | (types > IMAGE_END)):
        raise ValueError("invalid DeepSeek-V4.1 image token types")
    features = transformer.encode_image(patches.to(device=weight.device, dtype=weight.dtype), n_h, n_w)
    if features.shape != (int((types == IMAGE).sum()), transformer.image_start.numel()):
        raise ValueError("vision aligner output does not match image token span")
    span = torch.empty((types.numel(), features.shape[-1]), device=weight.device, dtype=features.dtype)
    span[types == IMAGE] = features
    for kind, name in ((IMAGE_START, "image_start"), (IMAGE_NEW_LINE, "image_newline"), (IMAGE_END, "image_end")):
        span[types == kind] = getattr(transformer, name).to(features.dtype)
    return span


@torch.inference_mode()
def merge_image_embeddings(transformer, batch, hidden: torch.Tensor):
    """Scatter canonical embedding rows and compatible native media into a prefill chunk."""
    if not batch.is_prefill:
        return hidden, None
    embeds = getattr(batch, "mm_embeds", None)
    legacy = any(getattr(req, "media", None) for req in batch.reqs)
    if embeds is None and not legacy:
        return hidden, None
    mask = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
    if embeds is not None:
        rows = batch.mm_rows
        if embeds.shape != (rows.numel(), hidden.shape[-1]):
            raise ValueError("DeepSeek-V4.1 multimodal embeddings have incompatible shape")
        hidden.index_copy_(0, rows, embeds.to(device=hidden.device, dtype=hidden.dtype))
        mask[rows] = True
    if not legacy:
        return hidden, mask
    offset = 0
    for req in batch.reqs:
        begin, end = req.cached_len, req.cached_len + req.extend_len
        for item in getattr(req, "media", None) or ():
            start = item["start"]
            types = item["types"]
            stop = start + types.numel()
            left, right = max(begin, start), min(end, stop)
            if left >= right:
                continue
            if "embeddings" not in item:
                span = image_span_embeddings(transformer, item["patches"], item["n_vit_h"], item["n_vit_w"], types)
                # The shared request object survives chunking; retain only the small CPU result.
                item["embeddings"] = span.cpu()
                item["patches"] = None
            dst = slice(offset + left - begin, offset + right - begin)
            hidden[dst] = item["embeddings"][left - start:right - start].to(hidden.device, hidden.dtype)
            mask[dst] = True
        offset += req.extend_len
    if offset != hidden.shape[0]:
        raise ValueError("prefill token count does not match request spans")
    return hidden, mask
