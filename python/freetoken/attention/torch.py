from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass
class TorchMetadata(BaseAttnMetadata):
    """Paged-cache gather metadata for auditable attention comparisons."""

    indices: torch.Tensor
    seqlens_q: list[int]
    seqlens_k: list[int]
    cached_lens: list[int]
    is_decode: bool
    cu_seqlens_q: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TorchAttentionBackend(BaseAttnBackend):
    """Eager PyTorch full attention used as independent correctness oracle."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self.num_q_heads = int(getattr(config, "num_qo_heads", 1))

    def _build_metadata(self, batch: Batch) -> TorchMetadata:
        ctx = get_global_ctx()
        reqs = batch.padded_reqs
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        indices = torch.cat([ctx.page_table[r.table_idx, : r.device_len] for r in reqs])
        is_decode = max(seqlens_q, default=0) == 1
        if is_decode:
            cu_seqlens_q = torch.arange(
                0, len(reqs) + 1, dtype=torch.int32, device=self.device
            )
        else:
            cu_seqlens_q = torch.tensor(
                [0] + seqlens_q, dtype=torch.int32, device=self.device
            ).cumsum_(0)
        return TorchMetadata(
            indices, seqlens_q, seqlens_k, cached_lens, is_decode, cu_seqlens_q
        )

    def prepare_metadata(self, batch: Batch) -> None:
        batch.attn_metadata = self._build_metadata(batch)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        k_raw = self.kvcache.k_cache(layer_id)
        v_raw = self.kvcache.v_cache(layer_id)
        kv_heads, head_dim = k_raw.shape[-2:]
        if head_dim != q.shape[-1]:
            raise ValueError(f"attention head_dim mismatch: cache={head_dim}, query={q.shape[-1]}")
        metadata = batch.attn_metadata
        if not isinstance(metadata, TorchMetadata):
            raise TypeError("TorchAttentionBackend requires TorchMetadata")
        k_all = k_raw.view(-1, kv_heads, head_dim)[metadata.indices]
        v_all = v_raw.view(-1, kv_heads, head_dim)[metadata.indices]
        spec = attn_spec or AttentionSpec()
        scale = spec.sm_scale if spec.sm_scale is not None else head_dim ** -0.5
        group = self.num_q_heads // kv_heads
        output = torch.empty(
            (q.shape[0], self.num_q_heads, head_dim), dtype=q.dtype, device=q.device
        )
        q_offset = k_offset = 0
        for lq, lk, cached in zip(metadata.seqlens_q, metadata.seqlens_k, metadata.cached_lens):
            qs = q[q_offset : q_offset + lq]
            ks = k_all[k_offset : k_offset + lk]
            vs = v_all[k_offset : k_offset + lk]
            if group > 1:
                ks = ks.repeat_interleave(group, dim=1)
                vs = vs.repeat_interleave(group, dim=1)
            scores = torch.einsum("qhd,khd->hqk", qs.float(), ks.float()) * scale
            rows = torch.arange(lq, device=q.device)
            cols = torch.arange(lk, device=q.device)
            mask = cols[None, :] > (cached + rows)[:, None]
            if spec.sliding_window is not None:
                mask |= cols[None, :] < (cached + rows)[:, None] - spec.sliding_window + 1
            probs = torch.softmax(scores.masked_fill(mask[None], float("-inf")), dim=-1)
            output[q_offset : q_offset + lq] = torch.einsum(
                "hqk,khd->qhd", probs, vs.float()
            ).to(q.dtype)
            q_offset += lq
            k_offset += lk
        return output

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        return None

    def prepare_for_capture(self, batch: Batch) -> None:
        return None

    def prepare_for_replay(self, batch: Batch) -> None:
        return None


__all__ = ["TorchAttentionBackend", "TorchMetadata"]
