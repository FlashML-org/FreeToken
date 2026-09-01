"""NoPE MLA attention and the GLM-5.3 compressed-indexer parameter layout."""

from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm


class _LayerNorm(BaseOP):
    def __init__(self, size: int, eps: float = 1e-6) -> None:
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x, (x.shape[-1],), self.weight, self.bias, self.eps
        )


class Glm5NextIndexer(BaseOP):
    """Indexer projections. The current DSA backend uses token keys for short-context
    identity and compatibility testing; the pool-compression tensors are retained so
    the checkpoint contract is exact while compressed selection is added upstream."""

    def __init__(self, config) -> None:
        args = config.glm5_args
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.wq_b = LinearReplicated(
            args.q_lora_rank, self.n_heads * self.head_dim, False
        )
        self.wk = LinearReplicated(config.hidden_size, self.head_dim, False)
        self.k_norm = _LayerNorm(self.head_dim)
        self.weights_proj = LinearReplicated(config.hidden_size, self.n_heads, False)
        self.index_kpool_compress_ape = torch.empty(args.index_kpool, self.head_dim)
        self.index_kpool_compress_gate = torch.empty(self.head_dim, config.hidden_size)

    def compute(self, x: torch.Tensor, q_resid: torch.Tensor):
        t = x.shape[0]
        q = self.wq_b.forward(q_resid).view(t, self.n_heads, self.head_dim)
        k = self.k_norm.forward(self.wk.forward(x))
        w = self.weights_proj.forward(x).float() * self.n_heads**-0.5
        return q, k, w


class Glm5NextAttention(BaseOP):
    def __init__(self, config, layer_id: int) -> None:
        args = config.glm5_args
        self.layer_id = layer_id
        self.num_heads = config.num_qo_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.q_a_proj = LinearReplicated(config.hidden_size, args.q_lora_rank, False)
        self.q_a_layernorm = RMSNorm(args.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = LinearReplicated(
            args.q_lora_rank, self.num_heads * self.qk_head_dim, False
        )
        self.kv_a_proj_with_mqa = LinearReplicated(
            config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = LinearReplicated(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            False,
        )
        self.o_proj = LinearReplicated(
            self.num_heads * self.v_head_dim, config.hidden_size, False
        )
        self.indexer = (
            Glm5NextIndexer(config) if args.indexer_types[layer_id] == "full" else None
        )
        self._w_uk = None
        self._w_uv = None

    def _kv_b(self):
        if self._w_uk is None:
            w = self.kv_b_proj.weight.view(
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank,
            )
            self._w_uk = w[:, : self.qk_nope_head_dim].contiguous()
            self._w_uv = w[:, self.qk_nope_head_dim :].transpose(1, 2).contiguous()
        return self._w_uk, self._w_uv

    def prepare_for_runtime(self) -> None:
        self._kv_b()
        self.kv_b_proj.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        t = x.shape[0]
        w_uk, w_uv = self._kv_b()
        q_resid = self.q_a_layernorm.forward(self.q_a_proj.forward(x))
        q = self.q_b_proj.forward(q_resid).view(t, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_proj_with_mqa.forward(x)
        c_kv, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm.forward(c_kv)
        q_absorbed = torch.bmm(q_nope.transpose(0, 1).contiguous(), w_uk).transpose(
            0, 1
        )
        indexer_qkw = (
            self.indexer.compute(x, q_resid)
            if self.indexer is not None
            and getattr(ctx.attn_backend, "dsa_enabled", False)
            else None
        )
        o_latent = ctx.attn_backend.mla_forward(
            q_absorbed.contiguous(),
            q_pe.contiguous(),
            c_kv.contiguous(),
            k_pe.contiguous(),
            self.layer_id,
            ctx.batch,
            indexer_qkw=indexer_qkw,
        )
        o = torch.bmm(o_latent.transpose(0, 1).contiguous(), w_uv).transpose(0, 1)
        return self.o_proj.forward(o.reshape(t, self.num_heads * self.v_head_dim))


__all__ = ["Glm5NextAttention"]
