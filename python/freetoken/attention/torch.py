from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import os
import torch

from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass
class TorchMetadata(BaseAttnMetadata):
    """Minimal contiguous-gather metadata for the pure-torch backend.

    ``indices`` maps every logical KV position (across all padded requests, in
    ``seqlens_k`` order) to its physical paged-cache slot, exactly like the triton
    backend's gather. The torch backend reads the SAME paged cache as triton, so a
    triton-vs-torch logit difference isolates the attention *compute* from the
    cache addressing.
    """

    indices: torch.Tensor
    seqlens_q: List[int]
    seqlens_k: List[int]
    cached_lens: List[int]
    is_decode: bool
    cu_seqlens_q: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TorchAttentionBackend(BaseAttnBackend):
    """Pure-PyTorch full-attention backend (no Triton/CUDA kernels).

    Serves as a numerically-explicit ground truth for debugging the hybrid
    qwen35moe model. It stores K/V into the same paged MHAKVCache as
    ``TritonAttentionBackend``, gathers the request's full K/V history via the
    identical ``indices`` page gather, and computes GQA softmax attention with
    PyTorch ops so every intermediate is auditable.

    Intended for correctness debugging / backend A-B comparison, not production
    serving. Registered as the ``"torch"`` attention backend (``AttnType.FULL``).
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self.num_q_heads = int(getattr(config, "num_qo_heads", 1))
        self.num_kv_heads = int(getattr(config, "num_kv_heads", 1))
        self.head_dim = int(getattr(config, "head_dim", 1))
        # Prefer the full-attention group spec head_dim (authoritative for kv heads).
        specs = getattr(config, "kv_cache_group_specs", lambda: ())()
        for spec in specs:
            name = getattr(spec, "attn_type", None)
            if name is not None and str(name) == "AttnType.FULL":
                self.head_dim = int(getattr(spec, "head_dim", self.head_dim))
                self.num_kv_heads = int(getattr(spec, "num_kv_heads", self.num_kv_heads))
                break
        # Debugging: contiguous (per-request) cache instead of the paged pool, to
        # isolate cache addressing from the attention compute.
        self._contig: dict[tuple[int, int], list] = {}
        self._use_contig = os.environ.get("FT_DEBUG_CONTIG_CACHE") == "1"

    def _build_metadata(self, batch: Batch) -> TorchMetadata:
        ctx = get_global_ctx()
        page_table = ctx.page_table
        reqs = batch.padded_reqs
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        is_decode = max(seqlens_q) == 1
        indices = torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs])
        if is_decode:
            cu_seqlens_q = torch.arange(0, len(reqs) + 1, dtype=torch.int32, device=self.device)
        else:
            cu_seqlens_q = torch.tensor(
                [0] + seqlens_q, dtype=torch.int32, device=self.device
            ).cumsum_(0)
        return TorchMetadata(
            indices=indices,
            seqlens_q=seqlens_q,
            seqlens_k=seqlens_k,
            cached_lens=cached_lens,
            is_decode=is_decode,
            cu_seqlens_q=cu_seqlens_q,
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
        if self._use_contig:
            return self._forward_contig(q, k, v, layer_id, batch, attn_spec)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        k_raw = self.kvcache.k_cache(layer_id)
        v_raw = self.kvcache.v_cache(layer_id)
        kv_heads, head_dim = k_raw.shape[-2], k_raw.shape[-1]
        assert head_dim == q.shape[-1], f"head_dim {head_dim} != {q.shape[-1]}"
        k_cache = k_raw.view(-1, kv_heads, head_dim)
        v_cache = v_raw.view(-1, kv_heads, head_dim)

        metadata = batch.attn_metadata
        assert isinstance(metadata, TorchMetadata)
        k_all = k_cache[metadata.indices]  # [total_kv, kv_heads, head_dim]
        v_all = v_cache[metadata.indices]  # [total_kv, kv_heads, head_dim]

        spec = attn_spec or AttentionSpec()
        scale = spec.sm_scale if spec.sm_scale is not None else head_dim ** -0.5
        group = self.num_q_heads // kv_heads

        num_q_tokens = q.shape[0]
        out = torch.empty((num_q_tokens, self.num_q_heads, head_dim), dtype=q.dtype, device=q.device)
        q_off = 0
        k_off = 0
        for lq, lk, cached in zip(
            metadata.seqlens_q, metadata.seqlens_k, metadata.cached_lens
        ):
            qs = q[q_off : q_off + lq]  # [lq, num_q, head_dim]
            ks = k_all[k_off : k_off + lk]  # [lk, kv_heads, head_dim]
            vs = v_all[k_off : k_off + lk]
            if group > 1:
                ks = ks.repeat_interleave(group, dim=1)  # [lk, num_q, head_dim]
                vs = vs.repeat_interleave(group, dim=1)
            # [num_q, lq, lk]
            scores = torch.einsum("qhd,khd->hqk", qs.float(), ks.float()) * scale
            # causal: query i (global cached+i) attends key col j <= cached+i
            if lq > 1 or lk > lq:
                rows = torch.arange(lq, device=scores.device)
                cols = torch.arange(lk, device=scores.device)
                masked = (cols[None, :] > (cached + rows)[:, None]).to(scores.device)
                scores = scores.masked_fill(masked[None, :, :], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            o = torch.einsum("hqk,khd->qhd", probs, vs.float()).to(q.dtype)  # [lq, num_q, head_dim]
            out[q_off : q_off + lq] = o
            q_off += lq
            k_off += lk

        return out

    def _forward_contig(self, q, k, v, layer_id, batch, attn_spec=None):
        """Contiguous (non-paged) attention: K/V accumulate per (layer, request) in
        a Python list keyed by logical position, so cache addressing is trivially
        correct. Isolates the paged-cache addressing from the attention compute."""
        metadata = self._build_metadata(batch)
        kv_heads = self.num_kv_heads
        head_dim = self.head_dim
        spec = attn_spec or AttentionSpec()
        scale = spec.sm_scale if spec.sm_scale is not None else head_dim ** -0.5
        group = self.num_q_heads // kv_heads
        # Store this forward's K/V rows per request (append in global position order).
        q_off = 0
        for i, lq in enumerate(metadata.seqlens_q):
            uid = batch.padded_reqs[i].uid
            buf = self._contig.setdefault((layer_id, uid), {"k": [], "v": []})
            kseg = k[q_off : q_off + lq].view(lq, kv_heads, head_dim)
            vseg = v[q_off : q_off + lq].view(lq, kv_heads, head_dim)
            for t in range(lq):
                buf["k"].append(kseg[t])
                buf["v"].append(vseg[t])
            q_off += lq
        # compute attention from the contiguous cache
        q_off = 0
        out = torch.empty(
            (q.shape[0], self.num_q_heads, head_dim), dtype=q.dtype, device=q.device
        )
        for i, lq in enumerate(metadata.seqlens_q):
            uid = batch.padded_reqs[i].uid
            buf = self._contig[(layer_id, uid)]
            ks = torch.stack(buf["k"])  # [acc, kv_heads, head_dim]
            vs = torch.stack(buf["v"])
            acc = ks.shape[0]
            cached = acc - lq
            qs = q[q_off : q_off + lq]
            if group > 1:
                ks = ks.repeat_interleave(group, dim=1)
                vs = vs.repeat_interleave(group, dim=1)
            scores = torch.einsum("qhd,khd->hqk", qs.float(), ks.float()) * scale
            rows = torch.arange(lq, device=scores.device)
            cols = torch.arange(acc, device=scores.device)
            masked = (cols[None, :] > (cached + rows)[:, None]).to(scores.device)
            scores = scores.masked_fill(masked[None, :, :], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            o = torch.einsum("hqk,khd->qhd", probs, vs.float()).to(q.dtype)
            out[q_off : q_off + lq] = o
            q_off += lq
        return out

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        # ROCm/graph capture is disabled for this debugging backend; no-op.
        return None

    def prepare_for_capture(self, batch: Batch) -> None:
        return None

    def prepare_for_replay(self, batch: Batch) -> None:
        return None


__all__ = ["TorchAttentionBackend", "TorchMetadata"]
