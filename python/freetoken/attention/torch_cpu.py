"""Pure-torch CPU attention backend.

FreeToken's upstream attention backends (triton / flash-attn / flashinfer / sgl-kernel)
all require a CUDA GPU. This module adds a **CPU-only** backend so the engine can serve
models on machines with no NVIDIA card — the OR-switch device fallback (``device_switch``)
already routes the engine to ``cpu``; this backend supplies the attention math for that path.

It implements the same ``BaseAttnBackend`` contract as the other backends:
  - ``prepare_metadata`` builds per-request paged KV spans (start/end into the flat page
    table ``indices`` tensor) plus the usual indptr / q_positions,
  - ``forward`` stores K/V into the paged cache, gathers each request's K/V contiguously
    via its span, and runs causal attention with
    ``torch.nn.functional.scaled_dot_product_attention`` (real math, not a stub).
    GQA is handled by SDPA's head broadcast.

The CPU path runs eager (no CUDA graphs), so the capture/replay hooks are no-ops.

This backend is correctness-focused, not speed-optimized; it exists to make the cpu-device
branch *logically executable* (a real generation on CPU). It is selected automatically when
``auto`` runs without CUDA, and can be forced with ``--attention-backend torch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass
class TorchCpuMetadata(BaseAttnMetadata):
    indptr: torch.Tensor
    indptr_q: torch.Tensor
    indices: torch.Tensor
    q_to_req: torch.Tensor
    q_positions: torch.Tensor
    is_decode: bool
    prefix_lens: torch.Tensor
    max_q_len: int
    # (start, end) into ``indices`` for each request's KV pages. Built in prepare_metadata.
    req_kv_spans: List[torch.Tensor] = field(default_factory=list)

    def get_last_indices(self, bs: int) -> torch.Tensor:
        # Index the LAST QUERY row per request (logits tensor holds query rows,
        # i.e. the tokens actually forwarded this step = extend_len, NOT the full
        # KV length which includes cached prefix). For a single unchunked prefill
        # extend_len == device_len so this equals indptr-1; for a chunked-prefill
        # continuation extend_len (1) < device_len (21) and using indptr (device_len)
        # would index out of bounds.
        return self.indptr_q[1 : 1 + bs] - 1


class TorchCPUAttentionBackend(BaseAttnBackend):
    """Faithful paged causal attention on CPU using torch SDPA."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device

    # -- capture/replay hooks: CPU runs eager, so these are no-ops ------------
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        return None

    def prepare_for_capture(self, batch: Batch) -> None:
        return None

    def prepare_for_replay(self, batch: Batch) -> None:
        return None

    def reset_capture(self) -> None:
        return None

    # -- metadata ------------------------------------------------------------
    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        device = self.device
        ctx = get_global_ctx()
        page_table = ctx.page_table
        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        num_query_tokens = sum(seqlens_q)
        is_decode = max(seqlens_q) == 1
        prefix_lens = torch.tensor(cached_lens, dtype=torch.int32, device=device)

        indptr = torch.tensor([0] + seqlens_k, dtype=torch.int32, device=device).cumsum_(0)
        # Query-side indptr (cumulative extend_len). The logits/query tensor holds only
        # the tokens forwarded THIS step (extend_len per request, not device_len which
        # includes the cached prefix). get_last_indices indexes this so the lm_head slices
        # the correct (last query) row even for chunked-prefill continuations.
        indptr_q = torch.tensor([0] + seqlens_q, dtype=torch.int32, device=device).cumsum_(0)
        if is_decode:
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):
            cu_seqlens_q = indptr
        else:
            cu_seqlens_q = torch.tensor(
                [0] + seqlens_q, dtype=torch.int32, device=device
            ).cumsum_(0)

        # Build per-request KV spans into the flat page-table indices tensor.
        req_kv_spans: List[torch.Tensor] = []
        flat_indices_parts = []
        offset = 0
        for req in reqs:
            kv_len = req.device_len
            flat_indices_parts.append(page_table[req.table_idx, :kv_len])
            req_kv_spans.append(
                torch.tensor([offset, offset + kv_len], dtype=torch.int32, device=device)
            )
            offset += kv_len
        indices = torch.cat(flat_indices_parts) if flat_indices_parts else torch.empty(0, dtype=torch.int32, device=device)

        q_to_req = torch.empty(num_query_tokens, dtype=torch.int32, device=device)
        o = 0
        for req_idx, q_len in enumerate(seqlens_q):
            q_to_req[o : o + q_len].fill_(req_idx)
            o += q_len

        q_positions = getattr(batch, "positions", None)
        if q_positions is None:
            q_positions = torch.zeros(num_query_tokens, dtype=torch.int64, device=device)

        batch.attn_metadata = TorchCpuMetadata(
            indptr=indptr,
            indptr_q=indptr_q,
            indices=indices,
            q_to_req=q_to_req,
            q_positions=q_positions,
            is_decode=is_decode,
            prefix_lens=prefix_lens,
            max_q_len=max(seqlens_q),
            req_kv_spans=req_kv_spans,
        )

    # -- forward --------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, TorchCpuMetadata)

        # 1) store this step's K/V into the paged cache (same as the triton backend).
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        k_raw = self.kvcache.k_cache(layer_id)
        v_raw = self.kvcache.v_cache(layer_id)
        kv_heads, head_dim = k_raw.shape[-2], k_raw.shape[-1]
        assert head_dim == q.shape[-1]
        # Keep the [num_slots, kv_heads, head_dim] layout so slot-based indexing
        # (req_indices into dim 0) selects a whole token's KV block per row.
        k_cache = k_raw
        v_cache = v_raw

        spec = attn_spec or AttentionSpec()
        scale = spec.sm_scale if spec.sm_scale is not None else q.shape[-1] ** -0.5

        ctx = get_global_ctx()
        page_table = ctx.page_table
        page_size = ctx.page_size
        q_positions = metadata.q_positions
        num_q_heads = q.shape[1]
        out = torch.empty(q.shape[0], num_q_heads, head_dim, dtype=q.dtype, device=self.device)

        # 2) per-request causal attention (SDPA handles GQA broadcast + causal mask).
        #    The KV cache is *paged*: k_cache has shape [num_pages, page_size,
        #    kv_heads, head_dim] and the page table maps logical token position ->
        #    (page, intra-page offset). We gather with 2-D advanced indexing
        #    [page_idx, intra] so each logical position maps to exactly one KV row.
        page_size = ctx.page_size
        q_off = 0  # running offset into the packed q / q_positions tensors
        for req in batch.padded_reqs:
            q_len = req.extend_len
            kv_len = req.device_len  # full KV after this step's store_kv
            if kv_len == 0 or q_len == 0:
                q_off += q_len
                continue
            # `k_cache()`/`v_cache()` return the raw 4-D buffer view
            # (num_pages, page_size, kv_heads, head_dim). Map each logical
            # token to its (page, intra-page) slot explicitly: `pages` is the
            # page index per token, `intra` its offset within that page. This
            # 2-D advanced index yields [kv_len, kv_heads, head_dim]. (Indexing
            # with a single flat index instead leaves the page_size dim in
            # place and produces a wrong 4-D block.)
            pages = page_table[req.table_idx].reshape(-1)[:kv_len].to(torch.long)
            intra = torch.arange(kv_len, dtype=torch.long, device=self.device) % page_size
            k_req = k_cache[pages, intra]  # [kv_len, kv_heads, head_dim]
            v_req = v_cache[pages, intra]
            q_req = q[q_off : q_off + q_len]  # [q_len, num_q_heads, head_dim]
            q_pos = q_positions[q_off : q_off + q_len].to(torch.int64)
            q_off += q_len

            q_t = q_req.transpose(0, 1).to(torch.float32)  # [num_q_heads, q_len, dim]
            # GQA: repeat each KV head to match the query head count (SDPA needs
            # equal head dims; it does not broadcast Q heads to fewer KV heads).
            if num_q_heads != kv_heads:
                rep = num_q_heads // kv_heads
                k_t = (
                    k_req.transpose(0, 1).to(torch.float32).repeat_interleave(rep, dim=0)
                )  # [num_q_heads, kv_len, dim]
                v_t = (
                    v_req.transpose(0, 1).to(torch.float32).repeat_interleave(rep, dim=0)
                )
            else:
                k_t = k_req.transpose(0, 1).to(torch.float32)  # [kv_heads, kv_len, dim]
                v_t = v_req.transpose(0, 1).to(torch.float32)
            kv_pos = torch.arange(kv_len, dtype=torch.int64, device=self.device)
            causal = kv_pos[None, :] <= q_pos[:, None]  # [q_len, kv_len]
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=causal, scale=scale
            )
            out[q_off - q_len : q_off] = attn_out.transpose(0, 1).to(q.dtype)

        return out
