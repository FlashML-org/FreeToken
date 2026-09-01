"""MiniMax-M3 block-sparse GQA attention backend.

Serves ``AttnType.BSA`` (one attention group over all layers: paged GQA K/V plus the
sparse layers' index-key slab, ``kvcache/bsa_pool.py``). Two entry points:

* dense leading layers call the generic :meth:`forward` -- delegated to a wrapped
  FULL backend (fa/fi/triton, picked by availability; page-size agnostic, since every
  generic backend addresses the pool by per-token rows regardless of the allocator's
  page size). The wrapped backend owns its own metadata; this backend's metadata
  carries it and the delegation swaps ``batch.attn_metadata`` around each call.
* sparse layers call :meth:`bsa_forward` with this token's (index_q, index_k): the
  backend scatters K/V and the index keys into the pool, scores 128-token blocks per
  index head (== per KV head), selects the top-``topk_blocks`` (+ forced init/local)
  blocks per query, and attends only those (``kernel/triton/minimax_m3_sparse.py``).

Addressing: the engine pins ``page_size == 128`` (this backend's ``page_sizes``
declaration), so one KV page IS one sparse block and the page table's every-128th
column is the block's base row -- ``block_rows[req, blk] = page_table[t, blk*128]``.

* **decode** stages the per-request block-base rows + live lengths into static
  buffers (``prepare_for_replay``, dsa/dsv4 precedent) and runs score -> top-k ->
  attend entirely on device-read lengths with shape-fixed grids, so the whole path
  lives inside the captured CUDA graph.
* **prefill/extend** is eager: per request, in query chunks bounded by the fp32
  score transient's budget (the [chunk, H, n_blocks] block-score tile), score +
  causal top-k + attend. Short contexts are exact dense attention by construction
  (top-k covers every visible block when ``kv_len <= topk_blocks * 128``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import Batch, get_global_ctx
from freetoken.utils import init_logger

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

_CPU_PINNED = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
# Prefill scoring transient budget: the fp32 block-score tile is
# [num_index_heads, chunk, n_blocks], so the query chunk shrinks as the context
# grows. Worst case is bounded by the model's max position (floor 16 queries x
# 4 heads x 8192 blocks x 4 B = 2 MB per iteration), not open-ended.
_PREFILL_SCORE_BYTES = 256 << 20
_PREFILL_SCORE_CHUNK = 4096


def _pick_inner_backend(block_size: int) -> str:
    """FULL backend for the dense leading layers, resolved through the engine's
    own FULL auto-tree (one resolver, so the two trees cannot diverge), then
    filtered for page-size compatibility: this pool serves 128-token pages, so a
    candidate that pins other page sizes (trtllm: 16/32/64) is skipped.

    ``FREETOKEN_M3_INNER_BACKEND`` overrides (e.g. "triton" on a box with a
    broken flashinfer JIT toolchain); the override is validated so a
    page-incompatible or non-FULL pick fails at construction with a real
    message instead of mis-addressing the pool.
    """
    import os

    from freetoken.attention import attention_backend_info
    from freetoken.attention.base import AttnType

    def _page_ok(name: str) -> bool:
        for part in name.split(","):
            sizes = attention_backend_info(part).page_sizes
            if sizes is not None and block_size not in sizes:
                return False
        return True

    override = os.getenv("FREETOKEN_M3_INNER_BACKEND")
    if override:
        for part in override.split(","):
            try:
                info = attention_backend_info(part)
            except KeyError:
                raise ValueError(
                    f"FREETOKEN_M3_INNER_BACKEND={override!r}: unknown attention "
                    f"backend {part!r}"
                ) from None
            if AttnType.FULL not in info.supported_types:
                raise ValueError(
                    f"FREETOKEN_M3_INNER_BACKEND={override!r}: {part!r} does not "
                    "serve FULL attention (the dense leading layers)."
                )
        if not _page_ok(override):
            raise ValueError(
                f"FREETOKEN_M3_INNER_BACKEND={override!r} cannot address the "
                f"{block_size}-token pages this model's pool uses."
            )
        logger.info(f"m3_sparse dense-layer backend: {override} (env override)")
        return override

    from freetoken.engine.engine import _resolve_auto_attention_backend

    name = _resolve_auto_attention_backend(frozenset({AttnType.FULL}))
    if not _page_ok(name):
        # trtllm (the sm_100 first pick) pins 16/32/64-token pages; walk the rest
        # of the SAME tree (same arch gates, same requirement probes) with the
        # page filter applied.
        from freetoken.engine.engine import _backend_requirements_met
        from freetoken.utils.arch import is_sm90_family

        skipped = name
        for candidate, arch_ok in (("fa,fi", is_sm90_family()), ("fi", True), ("triton", True)):
            if arch_ok and _page_ok(candidate) and _backend_requirements_met(candidate):
                name = candidate
                break
        else:  # pragma: no cover - triton is unconditional
            name = "triton"
        logger.info(
            f"m3_sparse dense-layer backend: {name} (auto; skipped {skipped!r}: "
            f"it cannot address this pool's {block_size}-token pages)"
        )
    else:
        logger.info(f"m3_sparse dense-layer backend: {name} (auto)")
    return name


@dataclass
class M3SparseMetadata(BaseAttnMetadata):
    # fmt: off
    is_decode:      bool
    last_indices:   torch.Tensor  # gpu
    qo_indptr_cpu:  torch.Tensor  # cpu pinned int32 [bs+1] (prefill request slicing)
    kv_len_cpu:     torch.Tensor  # cpu pinned int32 [bs]
    inner:          BaseAttnMetadata  # the wrapped FULL backend's metadata
    # decode addressing: per-request block-base rows (page_table[:, ::128] snapshot)
    # + live lengths, device-read. None on prefill / until staged.
    block_rows:     torch.Tensor | None = None
    kvlen:          torch.Tensor | None = None
    # prefill addressing plan, built ONCE at the first sparse layer's forward and
    # reused by all 57 (the tensors are layer-invariant): per request, its
    # contiguous block-base rows plus per-query-chunk (slice, cu, seq, prefix,
    # chunk_len) tuples. None until built / on decode.
    prefill_plan:   list | None = None
    # fmt: on

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class M3SparseAttnBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from freetoken.kvcache.bsa_pool import BSAKVCache

        args = config.m3_args
        assert args is not None and args.use_sparse, (
            "m3_sparse backend needs ModelConfig.m3_args with sparse attention on "
            "(the dense ablation resolves to AttnType.FULL and a generic backend)"
        )
        self.config = config
        self.args = args
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.kv_attn_dim = self.num_kv_heads * self.head_dim
        self.block_size = args.block_size
        self.topk_blocks = args.topk_blocks
        self.sm_scale = config.attn_sm_scale or (config.head_dim**-0.5)
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        # The serving switch is the POOL TYPE (parse_config resolved it into the
        # attention-group spec): this backend requires the index-key slab.
        assert isinstance(self.kvcache, BSAKVCache), (
            f"m3_sparse backend needs a BSA pool, got {type(self.kvcache).__name__}"
        )
        # layer -> index-slab slot (sparse-layer order; matches the pool's slots).
        self._idx_slot = {lid: args.sparse_slot(lid) for lid in args.sparse_layer_ids}

        # Wrapped FULL backend for the dense leading layers.
        from freetoken.attention import create_attention_backend

        self._inner_name = _pick_inner_backend(self.block_size)
        self.inner = create_attention_backend(self._inner_name, config)

        # decode staging (static buffers under CUDA graphs; eager decode builds
        # per-forward tensors lazily instead)
        self._block_rows_buf: torch.Tensor | None = None
        self._kvlen_buf: torch.Tensor | None = None
        self.capture_bs: List[int] = []

    # ----- slab views -------------------------------------------------------------------
    def _k_rows(self, layer_id: int) -> torch.Tensor:
        cache = self.kvcache.k_cache(layer_id)  # [pages, page_size, kvh, d]
        return cache.view(-1, cache.shape[2], cache.shape[3])

    def _v_rows(self, layer_id: int) -> torch.Tensor:
        cache = self.kvcache.v_cache(layer_id)
        return cache.view(-1, cache.shape[2], cache.shape[3])

    # ----- metadata ---------------------------------------------------------------------
    def prepare_metadata(self, batch: Batch) -> None:
        # The wrapped backend builds ITS metadata first (it owns the dense layers'
        # addressing); ours wraps it and the per-call delegation swaps them.
        self.inner.prepare_metadata(batch)
        inner_md = batch.attn_metadata
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        seqlens_q = [r.extend_len for r in reqs]
        seqlens_k = [r.device_len for r in reqs]
        is_decode = getattr(batch, "phase", None) == "decode"
        qo_indptr = torch.tensor([0] + seqlens_q, **_CPU_PINNED).cumsum_(0).to(torch.int32)
        kv_len = torch.tensor(seqlens_k, **_CPU_PINNED)
        last = (qo_indptr[1:].to(torch.int32) - 1).to(self.device, non_blocking=True)
        batch.attn_metadata = M3SparseMetadata(
            is_decode=is_decode,
            last_indices=last,
            qo_indptr_cpu=qo_indptr,
            kv_len_cpu=kv_len,
            inner=inner_md,
        )
        # Decode addressing (block_rows/kvlen) is DEFERRED: a graph-bound step stages
        # it into the static buffers (prepare_for_replay) and an eager step snapshots
        # lazily at the first sparse layer's bsa_forward.

    # ----- dense layers (delegated) -----------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        md = batch.attn_metadata
        assert isinstance(md, M3SparseMetadata)
        batch.attn_metadata = md.inner
        try:
            # The wrapped backend stores K/V itself (its forward contract).
            return self.inner.forward(q, k, v, layer_id, batch, attn_spec=attn_spec)
        finally:
            batch.attn_metadata = md

    # ----- sparse layers ----------------------------------------------------------------
    def bsa_forward(
        self,
        q: torch.Tensor,  # [T, HQ, D]
        k: torch.Tensor,  # [T, KVH * D]
        v: torch.Tensor,  # [T, KVH * D]
        index_q: torch.Tensor,  # [T, Hi, Di]
        index_k: torch.Tensor,  # [T, Di]
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        md = batch.attn_metadata
        assert isinstance(md, M3SparseMetadata)
        slot = self._idx_slot[layer_id]
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        # Scatter index keys unconditionally: short prefills select every block
        # TODAY, but their keys must exist once decode's history passes the top-k.
        self.kvcache.store_index_k(index_k, batch.out_loc, slot)

        if md.is_decode:
            if md.block_rows is None:
                # Eager decode (not graph-staged): per-request block-base SNAPSHOT
                # (the live page-table row may mutate for the next batch while this
                # one runs) + live lengths, once per step at the first sparse layer.
                md.block_rows = self._decode_block_rows(batch)
                md.kvlen = md.kv_len_cpu.to(self.device, non_blocking=True)
            return self._decode(md, layer_id, slot, q, index_q)
        return self._prefill(md, layer_id, slot, q, index_q, batch)

    # ----- decode (CUDA-graph capturable, single code path) ------------------------------
    def _decode(self, md, layer_id: int, slot: int, q, index_q) -> torch.Tensor:
        from freetoken.kernel.triton.minimax_m3_sparse import (
            m3_index_decode,
            m3_sparse_attn_decode,
        )

        block_rows, kvlen = md.block_rows, md.kvlen
        topk_idx = m3_index_decode(
            index_q,
            self.kvcache.index_k_cache(slot),
            block_rows,
            kvlen,
            block_rows.shape[1],
            self.topk_blocks,
            self.args.init_blocks,
            self.args.local_blocks,
        )
        out = torch.empty_like(q)
        m3_sparse_attn_decode(
            q, self._k_rows(layer_id), self._v_rows(layer_id),
            topk_idx, block_rows, kvlen, self.sm_scale, out,
        )
        return out

    # ----- prefill / extend (eager) -------------------------------------------------------
    def _prefill_plan(self, md, batch: Batch) -> list:
        """Per-request addressing plan, built ONCE per forward (layer-invariant --
        rebuilding it per sparse layer cost ~3 H2D copies x chunks x 57 layers of
        stream-serializing overhead): ``[(block_rows [1, n_blocks], kv_len,
        [(slice, cu, seq, prefix, chunk_len), ...]), ...]``."""
        if md.prefill_plan is not None:
            return md.prefill_plan
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        page_table = get_global_ctx().page_table
        qo = md.qo_indptr_cpu.tolist()
        plan = []
        for i, r in enumerate(reqs):
            m = qo[i + 1] - qo[i]
            if m == 0:
                continue
            kv_len = r.device_len
            n_blocks = -(-kv_len // self.block_size)
            # Contiguous per-request block-base rows (the [::block_size] view is
            # strided; the kernels read unit-stride rows). Tiny: n_blocks ints.
            block_rows = (
                page_table[r.table_idx, : kv_len : self.block_size]
                .to(torch.int32)
                .contiguous()
                .view(1, -1)
            )
            # Bound the fp32 [H, chunk, n_blocks] score transient (worst case capped
            # by max_position: 16 x 4 x (1M/128) x 4 B = 2 MB per chunk iteration).
            per_q = 4 * self.args.num_index_heads * max(n_blocks, 1)
            chunk = max(16, min(_PREFILL_SCORE_CHUNK, _PREFILL_SCORE_BYTES // per_q))
            seq = torch.tensor([kv_len], dtype=torch.int32, device=self.device)
            chunks = []
            for s0 in range(0, m, chunk):
                s1 = min(s0 + chunk, m)
                chunks.append((
                    slice(qo[i] + s0, qo[i] + s1),
                    torch.tensor([0, s1 - s0], dtype=torch.int32, device=self.device),
                    seq,
                    torch.tensor([r.cached_len + s0], dtype=torch.int32, device=self.device),
                    s1 - s0,
                ))
            plan.append((block_rows, kv_len, chunks))
        md.prefill_plan = plan
        return plan

    def _prefill(self, md, layer_id: int, slot: int, q, index_q, batch: Batch) -> torch.Tensor:
        from freetoken.kernel.triton.minimax_m3_sparse import (
            m3_index_score_prefill,
            m3_index_topk_prefill,
            m3_sparse_attn_prefill,
        )

        k_rows = self._k_rows(layer_id)
        v_rows = self._v_rows(layer_id)
        ik_rows = self.kvcache.index_k_cache(slot)
        out = torch.empty_like(q)

        for block_rows, kv_len, chunks in self._prefill_plan(md, batch):
            for sl, cu, seq, prefix, chunk_len in chunks:
                score = m3_index_score_prefill(
                    index_q[sl], ik_rows, block_rows, cu, seq, prefix,
                    chunk_len, kv_len,
                )
                topk_idx = m3_index_topk_prefill(
                    score, cu, prefix, chunk_len,
                    self.topk_blocks, self.args.init_blocks, self.args.local_blocks,
                )
                del score
                m3_sparse_attn_prefill(
                    q[sl], k_rows, v_rows, topk_idx, block_rows,
                    cu, seq, prefix, chunk_len, self.sm_scale, out[sl],
                )
        return out

    # ----- CUDA graph (decode) ------------------------------------------------------------
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.inner.init_capture_graph(max_seq_len, bs_list)
        self.capture_bs = sorted(bs_list)
        max_bs = max(bs_list)
        width = get_global_ctx().page_table.shape[1]
        n_blocks = -(-width // self.block_size)
        self._block_rows_buf = torch.zeros(
            (max_bs, n_blocks), dtype=torch.int32, device=self.device
        )
        self._kvlen_buf = torch.zeros(max_bs, dtype=torch.int32, device=self.device)

    def _block_base_view(self) -> torch.Tensor:
        """Every-``block_size``-th column of the page table: the per-block base
        rows. A strided VIEW -- gathering rows through it materializes only the
        [bs, W/128] result, never the full-width [bs, W] table snapshot (2 MiB/req
        at a 512K-token table; the staging-width class this repo fixed for DSV4)."""
        return get_global_ctx().page_table[:, :: self.block_size]

    def _decode_block_rows(self, batch: Batch) -> torch.Tensor:
        """This decode step's per-request block-base rows [bs, W/128], gathered off
        the scheduler-staged ``active_table_idx`` (a device tensor -- no host loop)."""
        assert batch.active_table_idx is not None, "decode batch is missing its page-table rows"
        return (
            self._block_base_view()
            .index_select(0, batch.active_table_idx.to(torch.int64))
            .to(torch.int32)
        )

    def _stage_decode(self, batch: Batch, bs: int, table_idx: torch.Tensor) -> None:
        """Copy this step's addressing into the static graph buffers and point the
        metadata at them (restage-per-replay, dsa precedent)."""
        md = batch.attn_metadata
        self._block_rows_buf[:bs].copy_(self._block_base_view().index_select(0, table_idx))
        self._kvlen_buf[:bs].copy_(md.kv_len_cpu.to(self.device, non_blocking=True))
        md.block_rows = self._block_rows_buf[:bs]
        md.kvlen = self._kvlen_buf[:bs]

    def prepare_for_capture(self, batch: Batch) -> None:
        # The wrapped backend builds its per-bs graph machinery first (its
        # prepare_for_capture rebuilds batch.attn_metadata as its own), then this
        # backend re-wraps the metadata and stages the dummy request's rows for
        # every slot -- replays overwrite with the live rows.
        self.inner.prepare_for_capture(batch)
        inner_md = batch.attn_metadata
        self.prepare_metadata(batch)
        md = batch.attn_metadata
        assert isinstance(md, M3SparseMetadata)
        md.inner = inner_md
        bs = batch.size
        dummy = torch.full(
            (bs,), batch.padded_reqs[0].table_idx, dtype=torch.int64, device=self.device
        )
        self._stage_decode(batch, bs, dummy)

    def prepare_for_replay(self, batch: Batch) -> None:
        md = batch.attn_metadata
        assert isinstance(md, M3SparseMetadata)
        assert batch.active_table_idx is not None, "decode batch is missing its page-table rows"
        batch.attn_metadata = md.inner
        try:
            self.inner.prepare_for_replay(batch)
        finally:
            batch.attn_metadata = md
        self._stage_decode(batch, batch.padded_size, batch.active_table_idx.to(torch.int64))

    def reset_capture(self) -> None:
        super().reset_capture()
        self.inner.reset_capture()
        self._block_rows_buf = None
        self._kvlen_buf = None


__all__ = ["M3SparseAttnBackend", "M3SparseMetadata"]
