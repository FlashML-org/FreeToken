"""QSA full-attention layer for Qwen3.8-Flash-Next (12 of 48 layers).

Gated GQA (24 q heads / 2 kv heads / head_dim 256, per-head zero-centered q/k norms, partial NeoX
rope over 64 dims, ``q_proj`` twice as wide for the output gate) plus the weights of the QSA
indexer (``index_qk_proj`` [640, 2560] = 4 index q heads x 128 then 1 index k head x 128, and the
two per-head index norms).

Model/backend split, same shape as MiniMax-M3's ``bsa_forward``: the layer owns the weights and
hands the backend the RAW index projections; the backend owns everything stateful (compressed key
slab, pending ring, scoring, top-k, expansion, sparse attend). The index k norm runs AFTER the
fp32 mean over each group of ``index_ratio`` raw keys, so it cannot be applied here -- both index
norm weights travel with the call (:class:`QSAIndexerInputs`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, GemmaPlusOneRMSNorm, LinearColParallelMerged, LinearReplicated
from freetoken.layers.rotary import get_rope
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


@dataclass(frozen=True)
class QSAIndexerInputs:
    """Everything the QSA backend needs from the indexer for one layer's forward.

    Frozen contract. ``q``/``k`` are the raw ``index_qk_proj`` slices: no norm, no rope. The
    backend applies, per HF ``Qwen4ExpTextQSAIndexer`` (modeling_qwen4_exp.py:611)::

        q_h    = rope64(rmsnorm(q_h) * (1 + q_norm_weight), pos = query position)
        kbar_b = rope64(rmsnorm(mean_fp32(k[4b:4b+4])) * (1 + k_norm_weight), pos = 4b)
        s_b    = sum_h relu(<q_h, kbar_b>) / sqrt(index_head_dim)

    and the pending ring stores ``k`` PRE-norm and PRE-rope, because a group's mean is only final
    once all ``index_ratio`` members exist. rope64 is ``get_rope(index_head_dim,
    config.rotary_config.rotary_dim, ...)`` -- the same frequencies as the main attention, a
    different ``head_size``, so the backend builds its own (cached) instance.
    """

    q: torch.Tensor  # [T, index_n_heads, index_head_dim]
    k: torch.Tensor  # [T, index_head_dim]
    q_norm_weight: torch.Tensor  # [index_head_dim], zero-centered: scale is (1 + w), fp32
    k_norm_weight: torch.Tensor  # [index_head_dim], zero-centered
    eps: float


class QSAAttentionBackend(Protocol):
    """The hook ``Qwen4ExpAttention`` calls; ``attention/qsa_sparse.py`` implements it.

    ``q`` is [T, num_qo_heads, head_dim] and ``k``/``v`` are [T, num_kv_heads*head_dim], all post
    norm+rope and in the model dtype; the return is [T, num_qo_heads, head_dim] (the layer applies
    the output gate and ``o_proj``). ``layer_id`` is the decoder layer id; the backend maps it to
    its own sparse-layer slot. Everything else -- KV store, per-request lengths, page rows -- comes
    from ``batch`` exactly as for ``BaseAttnBackend.forward``.
    """

    def qsa_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index: QSAIndexerInputs,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor: ...


class Qwen4ExpIndexer(BaseOP):
    """QSA indexer weights (checkpoint prefix ``self_attn.indexer``); the scoring lives in the backend."""

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        args = config.qwen4_args
        self.layer_id = layer_id
        self.num_heads = args.index_n_heads
        self.num_kv_heads = args.index_kv_heads
        self.head_dim = args.index_head_dim
        self.eps = config.rms_norm_eps
        self._split = [self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim]
        self.index_qk_proj = LinearReplicated(args.hidden_size, sum(self._split), has_bias=False)
        self.q_layernorm = GemmaPlusOneRMSNorm(self.head_dim, eps=self.eps)
        self.k_layernorm = GemmaPlusOneRMSNorm(self.head_dim, eps=self.eps)

    def forward(self, x: torch.Tensor) -> QSAIndexerInputs:
        q, k = self.index_qk_proj.forward(x).split(self._split, dim=-1)
        return QSAIndexerInputs(
            q=q.reshape(-1, self.num_heads, self.head_dim).contiguous(),
            k=k.reshape(-1, self.head_dim).contiguous(),
            q_norm_weight=self.q_layernorm.weight,
            k_norm_weight=self.k_layernorm.weight,
            eps=self.eps,
        )


class Qwen4ExpAttention(BaseOP):
    """Gated GQA with a QSA indexer::

        q, gate = chunk(q_proj(x).view(-1, num_q, 2*head_dim), 2, -1)
        q, k    = rope(q_norm(q), k_norm(k_proj(x)))          # first rotary_dim dims
        o       = backend.qsa_forward(q, k, v_proj(x), indexer(x), layer_id, batch)
        out     = o_proj(o * sigmoid(gate))

    q/k/v are one merged GEMM (``qkv_proj``, split ``[num_q*head_dim*2, kv, kv]``); the checkpoint
    ships ``q_proj``/``k_proj``/``v_proj`` separately, so the loader concatenates along dim 0.
    Other keys keep the checkpoint names: ``o_proj.weight``, ``q_norm.weight``, ``k_norm.weight``
    (both zero-centered, loaded RAW), ``indexer.*``.
    """

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = config.head_dim
        self.qo_attn_dim = self.num_q * self.head_dim
        self.kv_attn_dim = self.num_kv * self.head_dim
        self._qkv_split = [self.qo_attn_dim * 2, self.kv_attn_dim, self.kv_attn_dim]
        self.qkv_proj = LinearColParallelMerged(
            config.hidden_size, self._qkv_split, has_bias=False
        )
        self.o_proj = LinearReplicated(self.qo_attn_dim, config.hidden_size, has_bias=False)
        self.q_norm = GemmaPlusOneRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaPlusOneRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary = config.rotary_config
        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=rotary.rotary_dim,
            max_position=rotary.max_position,
            base=rotary.base,
            rope_scaling=tuple(rotary.scaling.items()) if rotary.scaling else None,
        )
        self.indexer = Qwen4ExpIndexer(config, layer_id)

    @nvtx_annotate("QSA")
    def forward(self, x: torch.Tensor, batch: Batch) -> torch.Tensor:
        qg, k, v = self.qkv_proj.forward(x).split(self._qkv_split, dim=-1)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()
        gate = qg[..., self.head_dim :].reshape(-1, self.qo_attn_dim)
        k = k.contiguous().view(-1, self.num_kv, self.head_dim)
        v = v.contiguous()
        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(k)
        q, k = self.rotary.forward(
            batch.positions, q.view(-1, self.qo_attn_dim), k.view(-1, self.kv_attn_dim)
        )
        index = self.indexer.forward(x)
        o = get_global_ctx().attn_backend.qsa_forward(
            q.view(-1, self.num_q, self.head_dim), k, v, index, self.layer_id, batch
        )
        gated = o.reshape(-1, self.qo_attn_dim) * torch.sigmoid(gate)
        return self.o_proj.forward(gated)


class TorchDenseQSAReference:
    """Dense oracle for :class:`QSAAttentionBackend` (fp32 math): attend to every visible token.

    QSA is exactly dense while a request sees at most ``index_budget + index_ratio - 1`` tokens
    (every complete block is selected), so this doubles as the equivalence oracle for the sparse backend. It
    keeps its own ``[slot, position]`` KV instead of a paged pool, so it needs no engine wiring;
    it is a test/reference object and is never registered as an attention backend.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_slots: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_kv = config.num_kv_heads
        self.head_dim = config.head_dim
        self.sm_scale = config.attn_sm_scale or self.head_dim**-0.5
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._shape = (num_slots, max_len, self.num_kv, self.head_dim)
        self._device = device
        self._dtype = dtype

    def _layer_cache(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_id not in self._cache:
            self._cache[layer_id] = tuple(
                torch.zeros(self._shape, device=self._device, dtype=self._dtype) for _ in range(2)
            )
        return self._cache[layer_id]

    def qsa_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index: QSAIndexerInputs,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        del index
        k_cache, v_cache = self._layer_cache(layer_id)
        k = k.view(-1, self.num_kv, self.head_dim)
        v = v.view(-1, self.num_kv, self.head_dim)
        out = torch.empty_like(q)
        offset = 0
        for r in batch.padded_reqs:
            n, slot, prefix = r.extend_len, r.table_idx, r.cached_len
            rows = slice(offset, offset + n)
            k_cache[slot, prefix : prefix + n] = k[rows]
            v_cache[slot, prefix : prefix + n] = v[rows]
            out[rows] = self._attend(
                q[rows], k_cache[slot, : prefix + n], v_cache[slot, : prefix + n], prefix
            )
            offset += n
        return out

    def _attend(
        self, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, prefix: int
    ) -> torch.Tensor:
        n, num_q, _ = q.shape
        total = keys.shape[0]
        rep = num_q // self.num_kv
        keys = keys.repeat_interleave(rep, dim=1).float()
        values = values.repeat_interleave(rep, dim=1).float()
        scores = torch.einsum("qhd,khd->hqk", q.float(), keys) * self.sm_scale
        visible = torch.arange(total, device=q.device) <= (
            prefix + torch.arange(n, device=q.device)
        ).unsqueeze(-1)
        scores = scores.masked_fill(~visible, float("-inf"))
        return torch.einsum("hqk,khd->qhd", scores.softmax(-1), values).to(q.dtype)


__all__ = [
    "QSAAttentionBackend",
    "QSAIndexerInputs",
    "Qwen4ExpAttention",
    "Qwen4ExpIndexer",
    "TorchDenseQSAReference",
]
