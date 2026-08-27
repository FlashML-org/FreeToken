"""Qwen compressed sparse-attention backend.

QSA keeps exact full-resolution K/V.  Its small four-head indexer scores one
compressed key for each four-token group, selects 512 groups, expands them to
2048 token rows, and appends the current incomplete group.  The final attention
therefore uses the model's original K/V values without approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

_SCORE_WORKSPACE_BYTES = 128 << 20


@dataclass
class QSAMetadata(BaseAttnMetadata):
    cu_seqlens_q: torch.Tensor
    cu_seqlens_q_host: tuple[int, ...]
    logical_positions: torch.Tensor
    last_indices: torch.Tensor
    compressed_rows: tuple[torch.Tensor, ...]

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


def _compact_expanded_selection(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    compress_ratio: int,
    token_budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand compressed blocks and append the visible incomplete-group tail."""
    if block_indices.is_cuda:
        from freetoken.kernel.triton.qsa import compact_qsa_blocks

        return compact_qsa_blocks(
            block_indices,
            query_positions,
            compress_ratio=compress_ratio,
            token_budget=token_budget,
        )
    rows = block_indices.shape[0]
    device = block_indices.device
    offsets = torch.arange(compress_ratio, device=device, dtype=torch.long)
    expanded = block_indices.long().unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        block_indices.long().unsqueeze(-1) >= 0,
        expanded,
        torch.full_like(expanded, -1),
    ).reshape(rows, -1)[:, :token_budget]
    positions = query_positions.to(device=device, dtype=torch.long)
    expanded = torch.where(
        (expanded >= 0) & (expanded <= positions.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=device, dtype=torch.long)
    visible = positions + 1
    tail_start = torch.div(visible, compress_ratio, rounding_mode="floor") * compress_ratio
    tail_count = visible - tail_start
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail = torch.where(
        tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1),
        tail,
        torch.full_like(tail, -1),
    )
    result = torch.full(
        (rows, token_budget + compress_ratio - 1),
        -1,
        dtype=torch.long,
        device=device,
    )
    result[:, :token_budget].copy_(expanded)
    # topk is sorted, so every finite block precedes its -inf/-1 padding.
    # Insert the incomplete tail directly after those expanded blocks instead
    # of sorting all 2,051 columns on every layer and decode step.
    block_counts = (block_indices >= 0).sum(dim=1)
    tail_columns = block_counts.unsqueeze(1) * compress_ratio + tail_offsets.unsqueeze(0)
    tail_live = tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)
    row_ids = torch.arange(rows, device=device).unsqueeze(1).expand_as(tail_columns)
    result[row_ids[tail_live], tail_columns[tail_live]] = tail[tail_live]
    counts = (block_counts * compress_ratio + tail_count).to(torch.int32)
    return result.to(torch.int32), counts


def select_qsa_logical_rows(
    index_q: torch.Tensor,
    compressed_keys: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    compress_ratio: int,
    token_budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score compressed keys and return exact logical token selections.

    Index scores are ``sum_h(relu(q_h dot k)) / sqrt(dim)``.  Row tiling
    bounds the FP32 score workspace without changing the per-row top-k result.
    """
    rows, heads, dim = index_q.shape
    blocks = compressed_keys.shape[0]
    block_budget = token_budget // compress_ratio
    output_blocks = torch.full(
        (rows, block_budget), -1, dtype=torch.int32, device=index_q.device
    )
    if rows == 0:
        return _compact_expanded_selection(
            output_blocks,
            query_positions,
            compress_ratio=compress_ratio,
            token_budget=token_budget,
        )
    if blocks:
        # One four-head dot tensor and one reduced logits matrix are live at once.
        bytes_per_row = max(blocks * torch.float32.itemsize * (heads + 1), 1)
        row_chunk = max(1, min(rows, _SCORE_WORKSPACE_BYTES // bytes_per_row))
        keys = compressed_keys[:, 0].transpose(0, 1)
        columns = torch.arange(blocks, device=index_q.device).unsqueeze(0)
        for start in range(0, rows, row_chunk):
            stop = min(start + row_chunk, rows)
            queries = index_q[start:stop].reshape((stop - start) * heads, dim)
            if queries.is_cuda and queries.dtype in (torch.bfloat16, torch.float16):
                # Tensor-core BF16/FP16 inputs with FP32 accumulation/output.
                # This avoids materializing a full FP32 copy of the long index
                # cache on every sparse layer and decode step.
                dots = torch.mm(queries, keys, out_dtype=torch.float32)
            else:
                dots = queries.float() @ keys.float()
            dots = dots.view(stop - start, heads, blocks)
            logits = torch.relu_(dots).sum(dim=1)
            logits.mul_(dim**-0.5)
            visible_blocks = torch.div(
                query_positions[start:stop].to(torch.long) + 1,
                compress_ratio,
                rounding_mode="floor",
            )
            logits.masked_fill_(columns >= visible_blocks.unsqueeze(1), -float("inf"))
            width = min(block_budget, blocks)
            if width:
                scores, picks = torch.topk(logits, width, dim=1)
                picks = torch.where(
                    torch.isfinite(scores), picks, torch.full_like(picks, -1)
                )
                output_blocks[start:stop, :width] = picks.to(torch.int32)
    return _compact_expanded_selection(
        output_blocks,
        query_positions,
        compress_ratio=compress_ratio,
        token_budget=token_budget,
    )


class QSAAttnBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from freetoken.kvcache.qsa_pool import QSAKVCache

        self.config = config
        self.args = config.qwen4_args
        self.kvcache = get_global_ctx().kv_cache
        if not isinstance(self.kvcache, QSAKVCache):
            raise TypeError(f"qsa backend needs QSAKVCache, got {type(self.kvcache).__name__}")
        self.device = self.kvcache.device
        self.compress_ratio = int(self.args.indexer_compress_ratio)
        self.token_budget = int(self.args.indexer_budget)
        if self.token_budget % self.compress_ratio:
            raise ValueError("QSA token budget must divide by its compression ratio")

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        lengths = [int(req.extend_len) for req in reqs]
        cu_host = [0]
        for length in lengths:
            cu_host.append(cu_host[-1] + length)
        cu = torch.tensor(cu_host, dtype=torch.int32, device=self.device)
        logical = torch.cat(
            [
                torch.arange(req.cached_len, req.device_len, device=self.device)
                for req in reqs
                if req.extend_len
            ],
            dim=0,
        ) if sum(lengths) else torch.empty(0, dtype=torch.int64, device=self.device)
        page_table = get_global_ctx().page_table
        compressed_rows: list[torch.Tensor] = []
        for req in reqs:
            complete_blocks = int(req.device_len) // self.compress_ratio
            if complete_blocks:
                starts = torch.arange(
                    complete_blocks, device=self.device, dtype=torch.long
                ).mul_(self.compress_ratio)
                full_rows = page_table[int(req.table_idx)].index_select(0, starts)
                rows = torch.div(
                    full_rows.to(torch.int64),
                    self.compress_ratio,
                    rounding_mode="floor",
                )
            else:
                rows = torch.empty(0, dtype=torch.int64, device=self.device)
            compressed_rows.append(rows)
        batch.attn_metadata = QSAMetadata(
            cu_seqlens_q=cu,
            cu_seqlens_q_host=tuple(cu_host),
            logical_positions=logical,
            last_indices=cu[1:] - 1,
            compressed_rows=tuple(compressed_rows),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Qwen4-Exp QSA layers call qsa_forward()")

    def _compress_current_keys(self, indexer, index_k, layer_id: int, batch: Batch) -> None:
        md = batch.attn_metadata
        assert isinstance(md, QSAMetadata)
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        if reqs:
            self.kvcache.ensure_pending_capacity(max(int(req.table_idx) for req in reqs) + 1)
        page_table = get_global_ctx().page_table
        pooled: list[torch.Tensor] = []
        rope_positions: list[torch.Tensor] = []
        compressed_rows: list[torch.Tensor] = []
        cu = md.cu_seqlens_q_host
        ratio = self.compress_ratio

        for req_id, req in enumerate(reqs):
            begin, stop = cu[req_id], cu[req_id + 1]
            if begin == stop:
                continue
            start, end = int(req.cached_len), int(req.device_len)
            request_row = int(req.table_idx)
            current = index_k[begin:stop]
            batch_rope = getattr(batch, "rope_positions", None)
            if batch_rope is None:
                current_rope = md.logical_positions[begin:stop].view(-1, 1).expand(-1, 3)
            else:
                current_rope = batch_rope[:, begin:stop].transpose(0, 1)
            if start == 0:
                self.kvcache.clear_pending(layer_id, request_row)

            first_end = ((start + ratio) // ratio) * ratio - 1
            for group_end in range(first_end, end, ratio):
                group_start = group_end - ratio + 1
                if group_start < start:
                    prior_pos = torch.arange(group_start, start, device=self.device)
                    prior = self.kvcache.pending_group(layer_id, request_row, prior_pos)
                    first_rope = self.kvcache.pending_rope_group(
                        layer_id, request_row, prior_pos[:1]
                    )[0]
                    current_part = current[: group_end - start + 1]
                    members = torch.cat((prior, current_part), dim=0)
                else:
                    lo = group_start - start
                    members = current[lo : lo + ratio]
                    first_rope = current_rope[lo]
                if members.shape[0] != ratio:
                    raise RuntimeError("QSA compression received an incomplete key group")
                pooled.append(members.float().mean(dim=0).to(index_k.dtype))
                rope_positions.append(first_rope)
                full_row = page_table[request_row, group_start].to(torch.int64)
                compressed_rows.append(torch.div(full_row, ratio, rounding_mode="floor"))

            # Only the newest occurrence of each modulo slot is needed.  This
            # avoids duplicate-index writes for a long prefill chunk.
            keep_start = max(start, end - ratio)
            keep_positions = torch.arange(keep_start, end, device=self.device)
            self.kvcache.store_pending(
                layer_id,
                request_row,
                keep_positions,
                current[keep_start - start :],
                current_rope[keep_start - start :],
            )

        if pooled:
            pooled_tensor = torch.stack(pooled)
            positions = torch.stack(rope_positions).transpose(0, 1).contiguous()
            normalized = indexer.normalize_compressed_keys(pooled_tensor, positions)
            rows = torch.stack(compressed_rows).to(torch.int64)
            self.kvcache.store_compressed_k(normalized, rows, layer_id)

    def _select_physical_rows(
        self, index_q: torch.Tensor, layer_id: int, batch: Batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        md = batch.attn_metadata
        assert isinstance(md, QSAMetadata)
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        page_table = get_global_ctx().page_table
        cu = md.cu_seqlens_q_host
        selections: list[torch.Tensor] = []
        counts: list[torch.Tensor] = []
        ratio = self.compress_ratio
        compressed_pool = self.kvcache.compressed_k_cache(layer_id)

        for req_id, req in enumerate(reqs):
            begin, stop = cu[req_id], cu[req_id + 1]
            if begin == stop:
                continue
            positions = md.logical_positions[begin:stop]
            compressed_rows = md.compressed_rows[req_id]
            if compressed_rows.numel():
                compressed_keys = compressed_pool.index_select(0, compressed_rows)
            else:
                compressed_keys = compressed_pool[:0]
            logical, live = select_qsa_logical_rows(
                index_q[begin:stop],
                compressed_keys,
                positions,
                compress_ratio=ratio,
                token_budget=self.token_budget,
            )
            safe = logical.clamp_min(0).long()
            physical = page_table[int(req.table_idx)].index_select(
                0, safe.reshape(-1)
            ).reshape_as(logical)
            physical = torch.where(logical >= 0, physical, torch.full_like(physical, -1))
            selections.append(physical.to(torch.int32))
            counts.append(live)
        if not selections:
            width = self.token_budget + ratio - 1
            return (
                torch.empty((0, width), dtype=torch.int32, device=self.device),
                torch.empty(0, dtype=torch.int32, device=self.device),
            )
        return torch.cat(selections, dim=0), torch.cat(counts, dim=0)

    def qsa_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        indexer,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        from freetoken.kernel.triton.qsa import qsa_sparse_gqa

        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        self._compress_current_keys(indexer, index_k, layer_id, batch)
        selected, counts = self._select_physical_rows(index_q, layer_id, batch)
        # k is flattened [N, kv_heads * head_dim], so take the pool geometry
        # directly instead of inferring a head count from the flattened input.
        k_raw = self.kvcache.k_cache(layer_id)
        v_raw = self.kvcache.v_cache(layer_id)
        k_rows = k_raw.view(-1, k_raw.shape[-2], k_raw.shape[-1])
        v_rows = v_raw.view(-1, v_raw.shape[-2], v_raw.shape[-1])
        return qsa_sparse_gqa(
            q,
            k_rows,
            v_rows,
            selected,
            counts,
            q.shape[-1] ** -0.5,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        # Qwen4-Exp disables CUDA graphs because PLE owns per-request recurrent state.
        return None

    def prepare_for_capture(self, batch: Batch) -> None:
        self.prepare_metadata(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.prepare_metadata(batch)


__all__ = [
    "QSAAttnBackend",
    "QSAMetadata",
    "select_qsa_logical_rows",
]
