from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.layers.rotary import get_rope
from freetoken.utils import nvtx_annotate

from .df11_linear import LinearDF11

if TYPE_CHECKING:
    import torch

    from freetoken.models.config import ModelConfig


class Glm4MoeAttention(BaseOP):
    """GLM-4 MoE attention (single-device / TP=1).

    Matches HuggingFace ``Glm4MoeAttention``:
    - separate q/k/v projections with bias, bias-free o_proj;
    - per-head qk-norm (RMSNorm over ``head_dim``) applied before RoPE;
    - partial RoPE (NeoX half-rotation over the first ``rotary_dim`` dims).

    The projections are lossless DF11 (:class:`LinearDF11`): NVIDIA's NVFP4 recipe keeps
    every ``self_attn`` out of quantization (qkvo stay bf16), so we reproduce that exactly by
    storing the bf16 weights compressed (~10.7 bits/weight) and decoding them bit-for-bit in
    the forward. This fits a 32 GB VRAM target without the accuracy loss fp8 attention would add.
    From a GGUF file the projections are the file's packed ggml linears instead.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        head_dim = config.head_dim
        self.layer_id = layer_id
        self.num_qo_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.head_dim = head_dim

        gguf = config.quant is not None and config.quant.dialect == "gguf"

        def linear(in_features: int, out_features: int, has_bias: bool, name: str) -> BaseOP:
            if gguf:
                return LinearReplicated(in_features, out_features, has_bias, quant_config=config.quant, prefix=f"{prefix}.{name}")
            return LinearDF11(in_features, out_features, has_bias=has_bias)

        self.q_proj = linear(config.hidden_size, self.qo_attn_dim, config.has_attn_bias, "q_proj")
        self.k_proj = linear(config.hidden_size, self.kv_attn_dim, config.has_attn_bias, "k_proj")
        self.v_proj = linear(config.hidden_size, self.kv_attn_dim, config.has_attn_bias, "v_proj")
        self.o_proj = linear(self.qo_attn_dim, config.hidden_size, False, "o_proj")

        if config.use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=(
                tuple(config.rotary_config.scaling.items())
                if config.rotary_config.scaling
                else None
            ),
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        del x
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return self.o_proj.forward(o.view(-1, self.qo_attn_dim))


__all__ = ["Glm4MoeAttention"]
