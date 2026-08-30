"""Qwen3.8-Flash-Next QSA compressed-block sparse attention backend.

Serves ``AttnType.QSA`` over ``kvcache/qsa_pool.py``: paged GQA K/V for the 12 full-attention
layers, a compressed index-key slab holding one key per ``index_ratio`` tokens, and a
per-request pending ring for the group a forward leaves open. The 36 GDN layers never reach
this backend, and the model has no dense attention layer, so :meth:`forward` is not served --
the only entry point is :meth:`qsa_forward` (``models/qwen4_exp/attention.py``).

One QSA layer's forward, all ragged over ``[T, ...]`` metadata:

1. store K/V at ``batch.out_loc``;
2. pool each row's closing group (members at positions >= ``cached_len`` come from this
   forward's raw index keys, the older ones from the pending ring), zero-centered rmsnorm it
   and rope it at the group's first position, then scatter it into the slab row
   ``out_loc // index_ratio`` (rows whose group does not close land on the request's scratch
   row and are never read);
3. store this forward's last ``ring_capacity`` raw index keys per request into the ring;
4. norm+rope the indexer queries at their own positions;
5. score every COMPLETE visible block (``sum_h relu(<q_h, k_bar_b>) / sqrt(index_head_dim)``,
   clamped to ``kvlen // index_ratio`` -- slab rows are never cleared, so stale rows must stay
   unreachable), take the top ``index_budget // index_ratio`` blocks, expand them to token
   indices plus the causal tail of the open group;
6. attend to exactly those tokens.

Addressing: the engine pins ``page_size == 64`` (this backend's ``page_sizes``), so a group of
``index_ratio`` tokens never straddles a page and ``block_table[req, p] = page_table[req, p *
64] // 64`` names both the K/V page and, viewed as ``page_size // index_ratio`` compressed
rows, the block's slab page. Decode stages that table plus the live lengths and table_idx into
static buffers (``prepare_for_replay``) so the whole path is CUDA-graph capturable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List

import torch
from freetoken.core import Batch, get_global_ctx
from freetoken.utils import init_logger

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

_CPU_PINNED = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
# Block-score transient budget (vLLM's number): the fp32 [rows, n_blocks] logits tile is
# 256 KB per row at a 1M-token context, so a long prefill must be scored in row chunks.
_LOGITS_WORKSPACE_BYTES = 128 << 20


TORCH_TOPK_ENV = "FREETOKEN_QSA_TORCH_TOPK"


def _resolve_block_topk() -> Callable | None:
    """The in-repo Triton block top-k, or None to fall back on torch.topk."""
    if os.getenv(TORCH_TOPK_ENV, "0") == "1":
        logger.info(f"qsa_sparse block top-k: torch.topk ({TORCH_TOPK_ENV}=1)")
        return None
    try:
        from freetoken.kernel.triton.qsa import qsa_block_topk
    except Exception as exc:
        logger.info(f"qsa_sparse block top-k: torch.topk (triton unavailable: {exc})")
        return None
    logger.info("qsa_sparse block top-k: triton qsa_block_topk")
    return qsa_block_topk


@dataclass
class QSASparseMetadata(BaseAttnMetadata):
    # fmt: off
    is_decode:        bool
    last_indices:     torch.Tensor  # gpu
    qo_indptr_cpu:    torch.Tensor  # cpu pinned int32 [bs+1]
    kv_len_cpu:       torch.Tensor  # cpu pinned int32 [bs]
    # Ragged per-token / per-request addressing. Decode defers these to the static graph
    # buffers (prepare_for_replay) or to a lazy eager snapshot at the first QSA layer.
    token_to_req:     torch.Tensor | None = None  # [T] int32
    cu_seqlens:       torch.Tensor | None = None  # [bs+1] int32
    seq_lens:         torch.Tensor | None = None  # [bs] int32, device_len
    ring_slots:       torch.Tensor | None = None  # [bs] int32, Req.table_idx
    block_table:      torch.Tensor | None = None  # [bs, W//page_size] int32, physical page ids
    # Per-forward scatter plans, built once by the first QSA layer and reused by the rest.
    # positions is bound here (not in prepare_metadata) because a capture batch has none yet.
    cmp_rows:         torch.Tensor | None = None  # [T] int32, compressed slab destination
    ring_rows:        torch.Tensor | None = None  # [T] int32, flat ring row or -1
    positions:        torch.Tensor | None = None  # [T] int32, logical query positions
    # fmt: on

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class QSASparseAttnBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from freetoken.kvcache.qsa_pool import QSAKVCache

        args = config.qwen4_args
        assert args is not None, "qsa_sparse backend needs ModelConfig.qwen4_args"
        self.head_dim = config.head_dim
        self.index_heads = args.index_n_heads
        self.token_topk = args.index_budget
        self.kvcache = get_global_ctx().kv_cache
        assert isinstance(self.kvcache, QSAKVCache), (
            f"qsa_sparse backend needs a QSA pool, got {type(self.kvcache).__name__}"
        )
        self.device = self.kvcache.device
        self.dtype = self.kvcache.dtype
        self.index_head_dim = self.kvcache.index_head_dim
        self.ratio = self.kvcache.index_ratio
        self.ring_capacity = self.kvcache.ring_capacity
        self.page_size = get_global_ctx().page_size
        assert self.page_size % self.ratio == 0, (
            f"QSA needs page_size ({self.page_size}) divisible by index_ratio ({self.ratio})"
        )
        self.cmp_page_size = self.page_size // self.ratio
        self.block_topk = self.token_topk // self.ratio
        self.select_width = self.token_topk + self.ratio - 1
        assert self.token_topk % self.ratio == 0, "QSA budget must be a whole number of blocks"
        # The sparse attend kernel bakes 1/sqrt(head_dim) into its exp2 scale.
        assert config.attn_sm_scale in (None, self.head_dim**-0.5), (
            "qsa_sparse serves the default 1/sqrt(head_dim) attention scale only"
        )
        # QSA layer -> index slab slot, in sparse-layer order (the pool's own convention).
        group = self._qsa_group(config)
        self._idx_slot = {lid: i for i, lid in enumerate(group.layer_ids)}
        self.rotary_config = group.rotary_config
        self._index_cos_sin: torch.Tensor | None = None

        self._block_topk_kernel = _resolve_block_topk()
        # decode staging (static buffers under CUDA graphs; eager decode snapshots per step)
        self._graph: dict[str, torch.Tensor] = {}
        self.capture_bs: List[int] = []

    @staticmethod
    def _qsa_group(config: ModelConfig):
        from freetoken.models.config import FullAttentionGroupConfig

        groups = [
            g
            for g in config.attention_groups
            if isinstance(g, FullAttentionGroupConfig) and g.index_ratio > 1
        ]
        assert len(groups) == 1, f"expected one QSA attention group, got {len(groups)}"
        return groups[0]

    # ----- slab views ---------------------------------------------------------------------
    def _cmp_pages(self, slot: int) -> torch.Tensor:
        """The compressed slab as ``[pages, page_size // ratio, 1, dim]``, the score kernel's
        paged layout. The scratch rows past ``cmp_scratch_base`` stay out of the view."""
        rows = self.kvcache.cmp_k_cache(slot)[: self.kvcache.cmp_scratch_base]
        return rows.view(-1, self.cmp_page_size, 1, self.index_head_dim)

    def _index_rope_cache(self) -> torch.Tensor:
        """cos/sin table of the indexer rope: same rotary_dim and frequencies as the main
        attention, ``head_size`` 128 instead of 256, so it is a separate get_rope instance.

        The table itself (not RotaryEmbedding.forward) because the indexer's norm+rope is one
        fused kernel and the compressed keys rope at their group's position, not the query's."""
        if self._index_cos_sin is None:
            from freetoken.layers.rotary import get_rope

            rotary = self.rotary_config
            with torch.device(self.device):
                rope = get_rope(
                    head_dim=self.index_head_dim,
                    rotary_dim=rotary.rotary_dim,
                    max_position=rotary.max_position,
                    base=rotary.base,
                    rope_scaling=tuple(rotary.scaling.items()) if rotary.scaling else None,
                )
            self._index_cos_sin = rope._cos_sin_cache.to(self.device)
        return self._index_cos_sin

    # ----- metadata -----------------------------------------------------------------------
    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        seqlens_q = [r.extend_len for r in reqs]
        seqlens_k = [r.device_len for r in reqs]
        is_decode = getattr(batch, "phase", None) == "decode"
        qo_indptr = torch.tensor([0] + seqlens_q, **_CPU_PINNED).cumsum_(0).to(torch.int32)
        kv_len = torch.tensor(seqlens_k, **_CPU_PINNED)
        last = (qo_indptr[1:].to(torch.int32) - 1).to(self.device, non_blocking=True)
        md = QSASparseMetadata(
            is_decode=is_decode,
            last_indices=last,
            qo_indptr_cpu=qo_indptr,
            kv_len_cpu=kv_len,
        )
        batch.attn_metadata = md
        if not is_decode:
            table_idx = torch.tensor([r.table_idx for r in reqs], **_CPU_PINNED)
            token_to_req = torch.repeat_interleave(
                torch.arange(len(reqs), dtype=torch.int32),
                torch.tensor(seqlens_q, dtype=torch.int32),
            ).pin_memory()
            md.cu_seqlens = qo_indptr.to(self.device, non_blocking=True)
            md.token_to_req = token_to_req.to(self.device, non_blocking=True)
            md.seq_lens = kv_len.to(self.device, non_blocking=True)
            md.ring_slots = table_idx.to(self.device, non_blocking=True)
            md.block_table = self._block_table(md.ring_slots.to(torch.int64))
        # Decode addressing is DEFERRED: a graph-bound step stages it into the static
        # buffers (prepare_for_replay), an eager step snapshots at the first QSA layer.

    def _block_base_view(self) -> torch.Tensor:
        """Every-``page_size``-th column of the page table: the per-page base slots. A strided
        VIEW, so gathering rows through it materializes only [bs, W/page_size]."""
        return get_global_ctx().page_table[:, :: self.page_size]

    def _block_table(self, table_idx: torch.Tensor) -> torch.Tensor:
        return (self._block_base_view().index_select(0, table_idx) // self.page_size).to(
            torch.int32
        )

    def _stage_decode(self, md: QSASparseMetadata, bs: int, table_idx: torch.Tensor) -> None:
        """Copy this step's addressing into the static graph buffers and point the metadata
        at them (restage-per-replay, m3/dsa precedent)."""
        self._graph["block_table"][:bs].copy_(
            self._block_base_view().index_select(0, table_idx) // self.page_size
        )
        self._graph["kvlen"][:bs].copy_(md.kv_len_cpu.to(self.device, non_blocking=True))
        self._graph["table_idx"][:bs].copy_(table_idx)
        md.block_table = self._graph["block_table"][:bs]
        md.seq_lens = self._graph["kvlen"][:bs]
        md.ring_slots = self._graph["table_idx"][:bs]
        md.token_to_req = self._graph["token_to_req"][:bs]
        md.cu_seqlens = self._graph["cu_seqlens"][: bs + 1]

    def _snapshot_decode(self, md: QSASparseMetadata, batch: Batch) -> None:
        """Eager decode (not graph-staged): this step's rows, once per forward. The live
        page-table row may mutate for the next batch while this one runs, so gather now."""
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        bs = len(reqs)
        table_idx = torch.tensor([r.table_idx for r in reqs], **_CPU_PINNED)
        md.ring_slots = table_idx.to(self.device, non_blocking=True)
        md.block_table = self._block_table(md.ring_slots.to(torch.int64))
        md.seq_lens = md.kv_len_cpu.to(self.device, non_blocking=True)
        md.token_to_req = torch.arange(bs, dtype=torch.int32, device=self.device)
        md.cu_seqlens = torch.arange(bs + 1, dtype=torch.int32, device=self.device)

    # ----- dense layers -------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "qsa_sparse serves QSA layers only (Qwen3.8-Flash-Next has no dense attention "
            "layer); the QSA layer calls qsa_forward"
        )

    # ----- QSA layers ---------------------------------------------------------------------
    def qsa_forward(
        self,
        q: torch.Tensor,  # [T, HQ, D]
        k: torch.Tensor,  # [T, KVH * D]
        v: torch.Tensor,  # [T, KVH * D]
        index,  # models.qwen4_exp.attention.QSAIndexerInputs
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        slot = self._idx_slot[layer_id]
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        if md.block_table is None:
            self._snapshot_decode(md, batch)
        if slot == 0 or md.cmp_rows is None:
            # Rebuilt at the first QSA layer of every forward, not cached on the metadata: a
            # capture batch runs its warmup and its capture through ONE metadata object, and a
            # cached plan would bake the warmup's addresses into the graph.
            self._plan_index_writes(md, batch)

        self._update_index_cache(index, md, slot)
        indices = self._select(index, md, slot)
        return qsa_sparse_paged_attention(
            q,
            self.kvcache.k_cache(layer_id),
            self.kvcache.v_cache(layer_id),
            indices,
            md.block_table,
            md.token_to_req,
            torch.empty_like(q),
        )

    def _plan_index_writes(self, md: QSASparseMetadata, batch: Batch) -> None:
        """Per-token slab row and ring row for this forward; the other QSA layers reuse it
        (it is layer-invariant). Pure device arithmetic: no host sync, graph-capturable."""
        md.positions = batch.positions
        out_loc = batch.out_loc.to(torch.int64)
        positions = batch.positions.to(torch.int64)
        rows = torch.arange(out_loc.numel(), device=self.device)
        req = md.token_to_req.to(torch.int64)
        slots = md.ring_slots.to(torch.int64).index_select(0, req)
        # out_loc % page_size == position % page_size and index_ratio divides page_size, so a
        # group closes exactly on out_loc % index_ratio == index_ratio - 1.
        closing = out_loc % self.ratio == self.ratio - 1
        scratch = self.kvcache.cmp_scratch_base + slots
        md.cmp_rows = torch.where(closing, out_loc // self.ratio, scratch).to(torch.int32)
        # Only the last ring_capacity rows of a request survive to the next forward; the rest
        # are masked off instead of dumped somewhere (vLLM rule).
        ends = md.cu_seqlens.to(torch.int64).index_select(0, req + 1)
        keep = rows >= ends - self.ring_capacity
        ring_row = slots * self.ring_capacity + positions % self.ring_capacity
        md.ring_rows = torch.where(keep, ring_row, torch.full_like(ring_row, -1)).to(
            torch.int32
        )

    def _update_index_cache(self, index, md: QSASparseMetadata, slot: int) -> None:
        """Compress each closing group into the slab, then refresh the pending ring."""
        from freetoken.kernel.triton.qsa import (
            qsa_compress_groups,
            qsa_index_norm_rope,
            qsa_store_rows,
        )

        rows = index.k.shape[0]
        ring = self.kvcache.pending_ring(slot)
        pooled = self._scratch("pooled", rows, self.index_head_dim, dtype=self.dtype)
        first = self._scratch("first_pos", rows, dtype=torch.int32)
        qsa_compress_groups(
            index.k,
            ring,
            md.ring_slots,
            md.token_to_req,
            md.cu_seqlens,
            md.positions,
            self.ratio,
            pooled,
            first,
        )
        qsa_index_norm_rope(
            pooled,
            first,
            self._index_rope_cache(),
            index.k_norm_weight,
            index.eps,
            self.kvcache.cmp_k_cache(slot),
            dest_rows=md.cmp_rows,
        )
        # After the compression read: the ring rows this forward overwrites are exactly the
        # ones a straddling group just consumed.
        qsa_store_rows(ring, md.ring_rows, index.k)

    def _select(self, index, md: QSASparseMetadata, slot: int) -> torch.Tensor:
        """Score complete visible blocks, take the top-k, expand them to token indices."""
        from freetoken.kernel.triton.qsa import (
            expand_qsa_block_indices,
            qsa_index_norm_rope,
            qsa_mqa_paged,
        )

        rows = index.q.shape[0]
        positions = md.positions
        q_index = self._scratch(
            "q_index", rows, self.index_heads, self.index_head_dim, dtype=self.dtype
        )
        qsa_index_norm_rope(
            index.q.view(-1, self.index_head_dim),
            positions,
            self._index_rope_cache(),
            index.q_norm_weight,
            index.eps,
            q_index.view(-1, self.index_head_dim),
            heads=self.index_heads,
        )
        cmp_pages = self._cmp_pages(slot)
        columns = md.block_table.shape[1] * self.cmp_page_size
        indices = self._scratch("indices", rows, self.select_width, dtype=torch.int32)
        rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            chunk = slice(start, end)
            logits = self._scratch("logits", end - start, columns, dtype=torch.float32)
            visible = self._scratch("visible", end - start, dtype=torch.int32)
            qsa_mqa_paged(
                q_index[chunk],
                cmp_pages,
                md.block_table,
                md.token_to_req[chunk],
                positions[chunk],
                md.seq_lens,
                self.ratio,
                logits,
                visible,
            )
            blocks = self._scratch("blocks", end - start, self.block_topk, dtype=torch.int32)
            self._top_blocks(logits, visible, blocks)
            expand_qsa_block_indices(
                blocks,
                positions[chunk],
                md.seq_lens,
                md.token_to_req[chunk],
                self.ratio,
                self.token_topk,
                indices[chunk],
            )
        return indices

    def _top_blocks(
        self,
        logits: torch.Tensor,
        visible: torch.Tensor,
        blocks: torch.Tensor,
    ) -> None:
        """Top ``block_topk`` complete blocks per row, row-relative, -1 padded."""
        assert blocks.shape == (logits.shape[0], self.block_topk), (
            f"qsa block top-k output must be [rows, {self.block_topk}], got {tuple(blocks.shape)}"
        )
        if self._block_topk_kernel is not None:
            scratch_width = self._topk_scratch_width(logits.shape[1])
            scratch = (
                self._scratch("topk_scratch", logits.shape[0], scratch_width, dtype=torch.int32)
                if scratch_width
                else None
            )
            self._block_topk_kernel(logits, visible, blocks, scratch)
            return
        # The score kernel only writes columns below visible_blocks; mask the rest so a
        # stale row cannot win a slot. Real block scores are relu sums, never -inf.
        columns = logits.shape[1]
        column = torch.arange(columns, dtype=torch.int32, device=logits.device)
        logits.masked_fill_(column.unsqueeze(0) >= visible.unsqueeze(1), -float("inf"))
        width = min(self.block_topk, columns)
        values, chosen = torch.topk(logits, width, dim=-1)
        blocks[:, :width] = torch.where(values > -float("inf"), chosen.to(torch.int32), -1)
        if width < self.block_topk:
            blocks[:, width:] = -1

    def _topk_scratch_width(self, columns: int) -> int:
        """int32 columns per row the block top-k wants as scratch, 0 when it wants none."""
        if self._block_topk_kernel is None:
            return 0
        from freetoken.kernel.triton.qsa import qsa_block_topk_scratch_width

        return qsa_block_topk_scratch_width(columns, self.block_topk)

    # ----- scratch ------------------------------------------------------------------------
    def _scratch(self, name: str, rows: int, *shape: int, dtype: torch.dtype) -> torch.Tensor:
        """A per-forward transient: the static decode buffer when it is wide enough (so a
        captured graph keeps one address), otherwise a fresh allocation."""
        buffer = self._graph.get(name)
        if buffer is not None and rows <= buffer.shape[0] and buffer.shape[1:] == shape:
            return buffer[:rows]
        return torch.empty((rows, *shape), dtype=dtype, device=self.device)

    # ----- CUDA graph (decode) --------------------------------------------------------------
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.capture_bs = sorted(bs_list)
        max_bs = max(bs_list)
        width = get_global_ctx().page_table.shape[1]
        pages = -(-width // self.page_size)
        columns = pages * self.cmp_page_size
        chunk = max(1, min(max_bs, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1)))
        topk_scratch = self._topk_scratch_width(columns)

        def empty(*shape: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=self.device)

        self._graph = {
            "block_table": torch.zeros((max_bs, pages), dtype=torch.int32, device=self.device),
            "kvlen": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "table_idx": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "token_to_req": torch.arange(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens": torch.arange(max_bs + 1, dtype=torch.int32, device=self.device),
            "logits": empty(chunk, columns, dtype=torch.float32),
            "visible": empty(max_bs, dtype=torch.int32),
            "blocks": empty(max_bs, self.block_topk, dtype=torch.int32),
            "indices": empty(max_bs, self.select_width, dtype=torch.int32),
            "pooled": empty(max_bs, self.index_head_dim, dtype=self.dtype),
            "first_pos": empty(max_bs, dtype=torch.int32),
            "q_index": empty(max_bs, self.index_heads, self.index_head_dim, dtype=self.dtype),
        }
        if topk_scratch:
            self._graph["topk_scratch"] = empty(chunk, topk_scratch, dtype=torch.int32)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.prepare_metadata(batch)
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        bs = batch.size
        dummy = torch.full(
            (bs,), batch.padded_reqs[0].table_idx, dtype=torch.int64, device=self.device
        )
        self._stage_decode(md, bs, dummy)

    def prepare_for_replay(self, batch: Batch) -> None:
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        assert batch.active_table_idx is not None, "decode batch is missing its page-table rows"
        self._stage_decode(md, batch.padded_size, batch.active_table_idx.to(torch.int64))

    def reset_capture(self) -> None:
        super().reset_capture()
        self._graph = {}


__all__ = ["QSASparseAttnBackend", "QSASparseMetadata"]
