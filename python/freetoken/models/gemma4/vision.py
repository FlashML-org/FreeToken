from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, GemmaRMSNorm, LinearReplicated, OPList
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.models.gemma4.config import VisionConfig


logger = init_logger(__name__)


def _log_vision_stage(label: str, tensor: torch.Tensor) -> None:
    """Record a compact, opt-in fingerprint at one vision-model boundary.

    llama.cpp's mtmd debugger can expose intermediate vision graph values.  This
    matching summary lets an AMD FreeToken investigation identify the first
    divergent stage without dumping a full image embedding, which would both
    distort a timing run and create enormous artifacts.
    """
    if os.environ.get("FREETOKEN_GEMMA4_VISION_DEBUG") != "1":
        return
    values = tensor.detach().float()
    logger.info_rank0(
        "Gemma4 vision stage %s: shape=%s finite=%s mean=%.8f std=%.8f "
        "min=%.8f max=%.8f sum=%.8f first16=%s",
        label,
        tuple(values.shape),
        bool(torch.isfinite(values).all().item()),
        float(values.mean().item()),
        float(values.std(unbiased=False).item()),
        float(values.min().item()),
        float(values.max().item()),
        float(values.sum().item()),
        ",".join(f"{value:.8f}" for value in values.reshape(-1)[:16].cpu().tolist()),
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_multidim_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """2-D RoPE for the vision encoder.

    ``x`` is ``[B, P, N, head_dim]``; ``cos``/``sin`` are ``[B, P, head_dim]`` built by
    concatenating the x- and y-spatial rotations (each ``head_dim/2`` wide). The head
    dim is split into the two spatial halves and ``rotate_half`` is applied within each
    half independently, mirroring ``apply_multidimensional_rope`` in HF transformers.
    """
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    d = x.shape[-1] // 2
    out = []
    for k in range(2):
        sl = slice(k * d, (k + 1) * d)
        xp = x[..., sl]
        out.append(xp * cos[..., sl] + _rotate_half(xp) * sin[..., sl])
    return torch.cat(out, dim=-1)


class _VisionRotary:
    """On-the-fly 2-D rope cos/sin tables (no learnable params)."""

    def __init__(self, vc: VisionConfig):
        self.head_dim = vc.head_dim
        self.theta = vc.rope_theta
        self._inv_freq: torch.Tensor | None = None

    def _inv(self, device: torch.device) -> torch.Tensor:
        if self._inv_freq is None or self._inv_freq.device != device:
            spatial_dim = self.head_dim // 2
            self._inv_freq = 1.0 / (
                self.theta
                ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32, device=device) / spatial_dim)
            )
        return self._inv_freq

    def cos_sin(self, position_ids: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        inv = self._inv(position_ids.device)
        all_cos, all_sin = [], []
        for i in range(2):
            pos = position_ids[..., i].float()
            freqs = pos.unsqueeze(-1) * inv  # [B, P, spatial_dim/2]
            emb = torch.cat((freqs, freqs), dim=-1)
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        cos = torch.cat(all_cos, dim=-1).to(dtype)
        sin = torch.cat(all_sin, dim=-1).to(dtype)
        return cos, sin


class Gemma4VisionMLP(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.gate_proj = LinearReplicated(vc.hidden_size, vc.intermediate_size, has_bias=False)
        self.up_proj = LinearReplicated(vc.hidden_size, vc.intermediate_size, has_bias=False)
        self.down_proj = LinearReplicated(vc.intermediate_size, vc.hidden_size, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = F.gelu(self.gate_proj.forward(x), approximate="tanh")
        return self.down_proj.forward(act * self.up_proj.forward(x))


class Gemma4VisionAttention(BaseOP):
    """Bidirectional multi-head attention with per-head q/k/v RMSNorm and 2-D RoPE.

    Note: HF uses ``scaling=1.0`` (the usual ``1/sqrt(head_dim)`` factor is omitted).
    """

    def __init__(self, vc: VisionConfig):
        self.head_dim = vc.head_dim
        self.num_heads = vc.num_heads
        self.num_kv_heads = vc.num_kv_heads
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = LinearReplicated(vc.hidden_size, q_dim, has_bias=False)
        self.k_proj = LinearReplicated(vc.hidden_size, kv_dim, has_bias=False)
        self.v_proj = LinearReplicated(vc.hidden_size, kv_dim, has_bias=False)
        self.o_proj = LinearReplicated(q_dim, vc.hidden_size, has_bias=False)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=vc.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=vc.rms_norm_eps)
        self.v_norm = GemmaRMSNorm(self.head_dim, eps=vc.rms_norm_eps, with_scale=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        B, P, _ = x.shape
        q = self.q_norm.forward(self.q_proj.forward(x).view(B, P, self.num_heads, self.head_dim))
        k = self.k_norm.forward(self.k_proj.forward(x).view(B, P, self.num_kv_heads, self.head_dim))
        v = self.v_norm.forward(self.v_proj.forward(x).view(B, P, self.num_kv_heads, self.head_dim))

        q = _apply_multidim_rope(q, cos, sin).transpose(1, 2)
        k = _apply_multidim_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)

        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0)
        o = o.transpose(1, 2).reshape(B, P, self.num_heads * self.head_dim)
        return self.o_proj.forward(o)


class Gemma4VisionEncoderLayer(BaseOP):
    """SigLIP-style block with a 4-norm sandwich (norm placed *after* each sublayer)."""

    def __init__(self, vc: VisionConfig):
        eps = vc.rms_norm_eps
        H = vc.hidden_size
        self.self_attn = Gemma4VisionAttention(vc)
        self.mlp = Gemma4VisionMLP(vc)
        self.input_layernorm = GemmaRMSNorm(H, eps=eps)
        self.post_attention_layernorm = GemmaRMSNorm(H, eps=eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(H, eps=eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(H, eps=eps)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        h = self.input_layernorm.forward(x)
        h = self.self_attn.forward(h, cos, sin, attn_mask)
        h = self.post_attention_layernorm.forward(h)
        x = residual + h

        residual = x
        h = self.pre_feedforward_layernorm.forward(x)
        h = self.mlp.forward(h)
        h = self.post_feedforward_layernorm.forward(h)
        return residual + h


class Gemma4VisionEncoder(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.layers = OPList([Gemma4VisionEncoderLayer(vc) for _ in range(vc.num_layers)])
        self._rotary = _VisionRotary(vc)


class Gemma4VisionPatchEmbedder(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.input_proj = LinearReplicated(3 * vc.patch_size**2, vc.hidden_size, has_bias=False)
        self.position_embedding_table = torch.empty(2, vc.position_embedding_size, vc.hidden_size)

    def forward(
        self, pixel_values: torch.Tensor, position_ids: torch.Tensor, padding: torch.Tensor
    ) -> torch.Tensor:
        pixel_values = 2 * (pixel_values - 0.5)
        h = self.input_proj.forward(pixel_values.to(self.input_proj.weight.dtype))
        clamped = position_ids.clamp(min=0)
        x_emb = F.embedding(clamped[..., 0], self.position_embedding_table[0])
        y_emb = F.embedding(clamped[..., 1], self.position_embedding_table[1])
        pos = x_emb + y_emb
        pos = torch.where(padding.unsqueeze(-1), 0.0, pos)
        return h + pos


def _avg_pool_by_positions(
    hidden: torch.Tensor, position_ids: torch.Tensor, length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average ``k**2`` patches per output soft token, by 2-D patch position."""
    input_seq_len = hidden.shape[1]
    k = int((input_seq_len // length) ** 0.5)
    k_squared = k * k
    if k_squared * length != input_seq_len:
        raise ValueError(f"Cannot pool {hidden.shape} to {length}: {k}^2 * {length} != {input_seq_len}")
    clamped = position_ids.clamp(min=0)
    max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
    kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
    weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
    output = weights.transpose(1, 2) @ hidden.float()
    mask = torch.logical_not((weights == 0).all(dim=1))
    return output.to(hidden.dtype), mask


class Gemma4VisionModel(BaseOP):
    """Pixels -> soft tokens. Output is ``[num_valid_soft_tokens, hidden]`` (padding stripped)."""

    def __init__(self, vc: VisionConfig):
        self.patch_embedder = Gemma4VisionPatchEmbedder(vc)
        self.encoder = Gemma4VisionEncoder(vc)
        self._standardize = vc.standardize
        self._pooling_kernel_size = vc.pooling_kernel_size
        self._root_hidden = vc.hidden_size**0.5
        if vc.standardize:
            self.std_bias = torch.empty(vc.hidden_size)
            self.std_scale = torch.empty(vc.hidden_size)

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        k = self._pooling_kernel_size
        output_length = pixel_values.shape[-2] // (k * k)
        padding = (position_ids == -1).all(dim=-1)  # [B, P] True = padding patch

        h = self.patch_embedder.forward(pixel_values, position_ids, padding)
        _log_vision_stage("patch_embed", h)
        # Reference encoders operate on the unpadded patch grid. Record the
        # equivalent compact view as well, so padding-query values cannot hide
        # the first numerical mismatch during a parity investigation.
        _log_vision_stage("patch_embed_valid", h[~padding])
        cos, sin = self.encoder._rotary.cos_sin(position_ids, h.dtype)
        attn_mask = (~padding)[:, None, None, :]  # [B, 1, 1, P] True = attend
        for layer in self.encoder.layers.op_list:
            h = layer.forward(h, cos, sin, attn_mask)
        _log_vision_stage("encoder", h)
        _log_vision_stage("encoder_valid", h[~padding])

        h = h.masked_fill(padding.unsqueeze(-1), 0.0)
        pooled, mask = _avg_pool_by_positions(h, position_ids, output_length)
        pooled = pooled.float() * self._root_hidden
        _log_vision_stage("pooled_scaled", pooled)
        pooled = pooled[mask]  # [num_valid, hidden] fp32
        if self._standardize:
            pooled = (pooled - self.std_bias.float()) * self.std_scale.float()
        _log_vision_stage("standardized", pooled)
        return pooled.to(h.dtype)


class Gemma4MultimodalEmbedder(BaseOP):
    """Projects vision soft tokens into the text hidden space (scale-free pre-norm)."""

    def __init__(self, vc: VisionConfig):
        self.embedding_projection = LinearReplicated(
            vc.hidden_size, vc.text_hidden_size, has_bias=False
        )
        self.embedding_pre_projection_norm = GemmaRMSNorm(
            vc.hidden_size, eps=vc.rms_norm_eps, with_scale=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.embedding_pre_projection_norm.forward(x)
        _log_vision_stage("projector_norm", normalized)
        projected = self.embedding_projection.forward(normalized)
        _log_vision_stage("projector_output", projected)
        return projected


__all__ = ["Gemma4VisionModel", "Gemma4MultimodalEmbedder"]
