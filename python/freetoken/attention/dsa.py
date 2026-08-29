"""GLM-5.2 DSA attention backend -- all-Triton gathered-KV sparse MLA.

The paged pool stores the MLA latent as one ``kv_lora_rank + qk_rope_head_dim`` row
per token (``kvcache/dsa_pool.py``: ``MLAKVCache`` latent slab; ``DSAKVCache`` adds
the index-key slab). The model absorbs ``kv_b`` into Q and onto the output, so
attention is one gathered-KV kernel over latent rows (``glm_dsa_sparse``), for every
regime:

* **DSA decode**: full-indexer layers score the history (fused gather-in-kernel
  logits, live length read from device memory), select top-``index_topk`` rows
  (selection semantics shared with dsv4_indexer at ratio=1 -- see dsa_indexer.py),
  and IndexShare followers reuse the leader's selection. Stateless kernels, so the
  whole path (gather -> score -> top-k -> attend) lives inside the captured CUDA
  graph; the per-step addressing (padded row snapshot + live lengths) is staged into
  static buffers by ``prepare_for_replay``.
* **Dense** (prefill within ``index_topk``, and the whole ``FREETOKEN_GLM_DSA=0``
  ablation): the IDENTITY-SELECTION degenerate case of the same kernel -- top-
  ``min(topk, T)`` covers every live token, so every query shares the request's
  position-ordered row list (query-dim stride 0, zero materialization) and causality
  rides the device-side ``counts[q] = position + 1`` the kernel already reads. No
  scoring, no top-k, no plan; bit-comparable to the sparse path at short kv by
  construction.
* **DSA long prefill**: per-request causal top-k in query chunks, same kernel.

No flashinfer anywhere: no plan/wrapper/workspace, and the backend serves any arch
Triton does (DSV4 is the precedent that one gathered-KV kernel serves all contexts
in production). The model calls :meth:`mla_forward` directly (the ``q_nope``/``q_pe``
split does not fit the generic ``forward(q, k, v)`` contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata
from .dsa_indexer import DSAIndexerMixin

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

_CPU_PINNED = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
# Prefill scoring transient budget: the fp32 logits tile is [chunk, kv_len], so the
# query chunk shrinks as the context grows. Worst case is bounded by the model's max
# position (floor 16 x 1M positions x 4 B = 64 MB), not open-ended.
_PREFILL_SCORE_BYTES = 128 << 20
_PREFILL_SCORE_CHUNK = 512


@dataclass
class DSAMetadata(BaseAttnMetadata):
    # fmt: off
    is_decode:      bool
    last_indices:   torch.Tensor  # gpu
    qo_indptr_cpu:  torch.Tensor  # cpu pinned int32 [bs+1] (prefill request slicing)
    kv_len_cpu:     torch.Tensor  # cpu pinned int32 [bs]
    # decode addressing: per-request padded row snapshot (position order, int32) +
    # live lengths, device-read. None on prefill (the host loop reads the live table).
    rows:           torch.Tensor | None = None
    kvlen:          torch.Tensor | None = None
    # k-pool: every kpool-th row (each pool's first-token physical row), position order.
    pool_rows:      torch.Tensor | None = None
    # group leader layer -> (sel_rows, counts); only the LIVE leader is retained
    sel:            dict = field(default_factory=dict)
    # fmt: on

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class DSAAttnBackend(DSAIndexerMixin, BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from freetoken.kvcache.dsa_pool import DSAKVCache, MLAKVCache

        args = config.glm_dsa_args
        assert args is not None, "dsa backend needs ModelConfig.glm_dsa_args (MLA dims)"
        self.config = config
        self.num_heads = config.num_qo_heads
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.latent_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.sm_scale = config.attn_sm_scale or (args.qk_head_dim**-0.5)
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device

        # The serving switch is the POOL TYPE: parse_config resolves FREETOKEN_GLM_DSA
        # once into the attention-group spec, the factory builds DSAKVCache (index
        # slab) or MLAKVCache (dense ablation), and the backend follows the storage.
        assert isinstance(self.kvcache, MLAKVCache), (
            f"dsa backend needs an MLA latent pool, got {type(self.kvcache).__name__}"
        )
        self.dsa_enabled = isinstance(self.kvcache, DSAKVCache)
        self.index_topk = args.index_topk
        self.index_scale = args.index_head_dim**-0.5 if args.index_head_dim else 0.0
        # GLM-5.3 k-pool: score `index_kpool`-token pools, expand winners to raw tokens and
        # always append the in-progress tail pool. 1 = per-token scoring (GLM-5.2).
        self.index_kpool = int(getattr(args, "index_kpool", 1))
        # layer -> group leader (most recent "full" layer); leader -> pool slot.
        # Only built when DSA serves: the dense ablation never consults indexer_types,
        # so a checkpoint with a malformed list cannot crash the ablation.
        self._leader: Dict[int, int] = {}
        self._idx_slot: Dict[int, int] = {}
        if self.dsa_enabled:
            # Hybrid linear-attention models (GLM-5.3): only the MLA full-attention
            # layers participate in DSA -- linear (KDA) layers own no indexer, never
            # call mla_forward, and must not consume index slots. Non-hybrid GLM-5.2
            # has every layer in the mla group, so this is the identity there.
            _mla_ids = None
            for _g in getattr(config, "attention_groups", ()) or ():
                if getattr(_g, "mla", False):
                    _mla_ids = frozenset(_g.layer_ids)
            lead = None
            # Capped to the SERVED layer count (dev num_layers overrides must not
            # index slots past the pool the factory sized from the same cap).
            # Iterate the derived list (glm5 MTP appends layer 45 past num_layers);
            # the mla-group membership filter keeps the served set exact.
            _bound = max(config.num_layers, len(args.indexer_types)) if _mla_ids else config.num_layers
            for lid, kind in enumerate(args.indexer_types[:_bound]):
                if _mla_ids is not None and lid not in _mla_ids:
                    continue
                if kind == "full":
                    lead = lid
                    self._idx_slot[lid] = len(self._idx_slot)
                assert lead is not None, "indexer_types must start with a 'full' layer"
                self._leader[lid] = lead
        # decode staging (static buffers under CUDA graphs; eager decode builds
        # per-forward tensors in prepare_metadata instead)
        self._rows_buf: torch.Tensor | None = None
        self._kvlen_buf: torch.Tensor | None = None
        self.max_seq_len = 0
        self.capture_bs: List[int] = []

    def forward(self, q, k, v, layer_id, batch, attn_spec: AttentionSpec | None = None):
        raise NotImplementedError("MLA models use mla_forward(), not forward().")

    # ----- metadata -------------------------------------------------------------------
    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        seqlens_q = [r.extend_len for r in reqs]
        seqlens_k = [r.device_len for r in reqs]
        # Follow the BATCH PHASE, not a max(extend)==1 heuristic: a fully radix-hit
        # prompt arrives as a 1-token PREFILL batch, and the scheduler only stages
        # active_table_idx (which the decode path's addressing requires) for
        # phase == "decode". The prefill path handles extend_len == 1 fine.
        is_decode = getattr(batch, "phase", None) == "decode"
        qo_indptr = torch.tensor([0] + seqlens_q, **_CPU_PINNED).cumsum_(0).to(torch.int32)
        kv_len = torch.tensor(seqlens_k, **_CPU_PINNED)
        last = (qo_indptr[1:].to(torch.int32) - 1).to(self.device, non_blocking=True)
        md = DSAMetadata(
            is_decode=is_decode,
            last_indices=last,
            qo_indptr_cpu=qo_indptr,
            kv_len_cpu=kv_len,
        )
        # Decode addressing (rows/kvlen) is DEFERRED: a graph-bound step stages it
        # into the static buffers (prepare_for_replay -> _stage_decode) and an eager
        # step snapshots lazily at the first layer's mla_forward -- building it here
        # would duplicate the same gather on every replayed step.
        batch.attn_metadata = md

    # ----- attention ------------------------------------------------------------------
    def _attend(
        self, q_cat: torch.Tensor, layer_id: int, sel: torch.Tensor, cnt: torch.Tensor
    ) -> torch.Tensor:
        """Gathered-KV MLA over latent rows: q [b, m, H, 576] -> [b, m, H, 512]."""
        from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn

        return glm_dsa_sparse_attn(
            q_cat, self.kvcache.latent_rows(layer_id), sel, self.sm_scale,
            counts=cnt, d_v=self.kv_lora_rank,
        )

    def mla_forward(
        self, q_nope, q_pe, c_kv, k_rope, layer_id, batch, indexer_qkw=None
    ) -> torch.Tensor:
        """Store this forward's latent rows and attend over the paged latent history.

        ``q_nope`` [T, H, kv_lora_rank] (kv_b-absorbed), ``q_pe`` [T, H, rope_dim],
        ``c_kv`` [T, kv_lora_rank] / ``k_rope`` [T, rope_dim] (the pool scatters the
        two latent halves). ``indexer_qkw`` = (q [T, Hi, Di], k [T, Di], w [T, Hi])
        on full-indexer layers, None on shared layers. Returns [T, H, kv_lora_rank].
        """
        md = batch.attn_metadata
        assert isinstance(md, DSAMetadata)
        if md.is_decode and md.rows is None:
            # Eager decode (not graph-staged): per-request padded row SNAPSHOT (the
            # live page_table row may mutate for the next batch while this one runs)
            # + device-read live lengths, once per step at the first layer.
            md.rows = self._decode_rows(batch).to(torch.int32)
            md.kvlen = md.kv_len_cpu.to(self.device, non_blocking=True)
            if self.index_kpool > 1:
                md.pool_rows = md.rows[:, :: self.index_kpool].contiguous()
        self.kvcache.store_kv(c_kv, k_rope, batch.out_loc, layer_id)
        if self.dsa_enabled and indexer_qkw is not None:
            # Scatter index keys unconditionally: short prefills serve through the
            # identity path TODAY, but their keys must exist once decode passes topk.
            self.kvcache.store_index_k(indexer_qkw[1], batch.out_loc, self._idx_slot[layer_id])
            if self.index_kpool > 1:
                self.kvcache.store_gate(indexer_qkw[3], batch.out_loc, self._idx_slot[layer_id])
                self._update_pools(md, layer_id, batch, indexer_qkw[4])

        if md.is_decode:
            return self._decode(md, layer_id, q_nope, q_pe, indexer_qkw)
        return self._prefill(md, layer_id, q_nope, q_pe, batch, indexer_qkw)

    # ----- decode (CUDA-graph capturable, single code path) -----------------------------
    def _decode(self, md, layer_id, q_nope, q_pe, indexer_qkw) -> torch.Tensor:
        bs = q_nope.shape[0]
        rows, kvlen = md.rows, md.kvlen
        if not self.dsa_enabled:
            # Identity selection == dense attention: every query walks its request's
            # whole row list, bounded by the device-side live length.
            sel, cnt = rows.view(bs, 1, -1), kvlen.view(bs, 1)
        else:
            if indexer_qkw is not None:
                q_idx, w = indexer_qkw[0], indexer_qkw[2]
                kp = self.index_kpool
                if kp > 1:
                    # Pool-compressed scoring: candidates are complete kpool-token pools
                    # (pooled keys at each pool's first-token row), causal by construction
                    # (valid = kvlen // kp counts only complete pools). Winners expand back
                    # to raw token positions; the in-progress tail pool (< kp tokens) is
                    # ALWAYS selected, unscored (HF index_kpool_always_select_tail).
                    from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_decode_logits

                    kv_pools = kvlen // kp
                    sp = glm_dsa_decode_logits(
                        q_idx, (w * self.index_scale),
                        self.kvcache.pool_k_cache(self._idx_slot[layer_id]),
                        md.pool_rows, kv_pools,
                    )
                    p_sel = min(self.index_topk // kp, sp.shape[-1])
                    picks = self.indexer_select_decode(
                        sp.view(bs, 1, -1), valid=kv_pools, topk=p_sel, offset=0
                    )[:, 0]  # [bs, P] pool ids, -1 sentinel
                    ar = torch.arange(kp, device=picks.device)
                    tok = picks.unsqueeze(-1) * kp + ar  # [bs, P, kp]
                    tok = torch.where(picks.unsqueeze(-1) >= 0, tok, -1).view(bs, -1)
                    tail = (kv_pools * kp).unsqueeze(-1) + ar[: kp - 1]  # [bs, kp-1]
                    tail = torch.where(tail < kvlen.unsqueeze(-1), tail, -1)
                    pos = torch.cat([tail, tok], dim=-1)  # [bs, kp-1 + P*kp]
                    sel = self.dsa_map_rows(pos, rows).view(bs, 1, -1)
                    # -1 holes are masked inside the sparse kernel; bound by full width
                    # (static shape -> graph-safe).
                    cnt = torch.full(
                        (bs, 1), pos.shape[-1], dtype=torch.int32, device=pos.device
                    )
                else:
                    s = self.dsa_decode_scores(q_idx, w, self._idx_slot[layer_id], rows, kvlen)
                    k_sel = min(self.index_topk, s.shape[-1])
                    picks = self.indexer_select_decode(
                        s.view(bs, 1, -1), valid=kvlen, topk=k_sel, offset=0
                    )[:, 0]  # [bs, K] positions, -1 sentinel
                    sel = self.dsa_map_rows(picks, rows).view(bs, 1, -1)
                    cnt = torch.clamp(kvlen, max=k_sel).to(torch.int32).view(bs, 1)
                # Only the live group leader's selection is ever read again.
                md.sel.clear()
                md.sel[layer_id] = (sel, cnt)
            sel, cnt = md.sel[self._leader[layer_id]]
        q_cat = torch.cat([q_nope, q_pe], dim=-1).view(bs, 1, self.num_heads, self.latent_dim)
        o = self._attend(q_cat, layer_id, sel, cnt)
        return o.view(bs, self.num_heads, self.kv_lora_rank)

    # ----- prefill / extend (eager) ------------------------------------------------------
    def _select_prefill(
        self, slot: int, q_idx: torch.Tensor, w: torch.Tensor,
        rows: torch.Tensor, positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-request causal top-k: ([1, m, K] physical rows, [1, m] counts)."""
        if self.index_kpool > 1:
            return self._select_prefill_kpool(slot, q_idx, w, rows, positions)
        kv_len = rows.numel()
        k_all = self.kvcache.index_k_cache(slot).index_select(0, rows.long())
        k_sel = min(self.index_topk, kv_len)
        m = q_idx.shape[0]
        sel = torch.empty(m, k_sel, dtype=torch.int32, device=self.device)
        start_pos = int(positions[0])
        # Bound the fp32 [chunk, kv_len] logits transient (worst case is capped by the
        # model's max_position: floor 16 x 1M x 4 B = 64 MB, see _PREFILL_SCORE_BYTES).
        chunk = max(16, min(_PREFILL_SCORE_CHUNK, _PREFILL_SCORE_BYTES // max(kv_len * 4, 1)))
        for s0 in range(0, m, chunk):
            s1 = min(s0 + chunk, m)
            scores = self.dsa_prefill_logits(q_idx[s0:s1], k_all, w[s0:s1])
            # Shared selection semantics (dsv4_indexer): token-granular == ratio 1.
            picks = self.indexer_select_prefill(
                scores.unsqueeze(0), start_pos=start_pos + s0, seqlen=s1 - s0,
                ratio=1, topk=k_sel, offset=0,
            )[0]
            sel[s0:s1] = self.dsa_map_rows(picks, rows.view(1, -1).expand(s1 - s0, -1))
        cnt = torch.clamp(positions + 1, max=k_sel).to(torch.int32)
        return sel.view(1, m, k_sel), cnt.view(1, m)

    def _select_prefill_kpool(
        self, slot: int, q_idx: torch.Tensor, w: torch.Tensor,
        rows: torch.Tensor, positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool-compressed causal top-k for prefill: query at absolute position p scores
        complete pools j < (p+1)//kp (indexer_select_prefill's ratio semantics == the HF
        pool_end<=p rule), winners expand to raw tokens, and p's in-progress tail pool
        ((p+1)%kp trailing tokens) is always appended unscored."""
        kp = self.index_kpool
        kv_len = rows.numel()
        n_pools = kv_len // kp
        pool_rows = rows[: n_pools * kp : kp]
        k_pool = self.kvcache.pool_k_cache(slot).index_select(0, pool_rows.long())
        p_sel = max(1, min(self.index_topk // kp, max(n_pools, 1)))
        m = q_idx.shape[0]
        width = (kp - 1) + p_sel * kp
        sel = torch.empty(m, width, dtype=torch.int32, device=self.device)
        start_pos = int(positions[0])
        ar = torch.arange(kp, device=self.device)
        chunk = max(16, min(_PREFILL_SCORE_CHUNK, _PREFILL_SCORE_BYTES // max(n_pools * 4, 1)))
        for s0 in range(0, m, chunk):
            s1 = min(s0 + chunk, m)
            scores = self.dsa_prefill_logits(q_idx[s0:s1], k_pool, w[s0:s1])  # [c, n_pools]
            picks = self.indexer_select_prefill(
                scores.unsqueeze(0), start_pos=start_pos + s0, seqlen=s1 - s0,
                ratio=kp, topk=p_sel, offset=0,
            )[0]  # [c, p_sel] pool ids, -1 sentinel
            tok = picks.unsqueeze(-1) * kp + ar
            tok = torch.where(picks.unsqueeze(-1) >= 0, tok, -1).reshape(s1 - s0, -1)
            pos_q = positions[s0:s1]
            tail = (((pos_q + 1) // kp) * kp).unsqueeze(-1) + ar[: kp - 1]
            tail = torch.where(tail <= pos_q.unsqueeze(-1), tail, -1)
            pos = torch.cat([tail, tok], dim=-1)
            sel[s0:s1] = self.dsa_map_rows(pos, rows.view(1, -1).expand(s1 - s0, -1))
        cnt = torch.full((1, m), width, dtype=torch.int32, device=self.device)
        return sel.view(1, m, width), cnt

    def _update_pools(self, md, layer_id, batch, ape: torch.Tensor) -> None:
        """Refresh pooled keys after this forward's index-k/gate scatter.

        Pool key (HF get_pooled_states): softmax over the pool's kp tokens of
        (gate_scores + ape) PER CHANNEL, weighted average of the raw index keys.
        Prefill (eager): recompute every pool the new tokens touch, per request.
        Decode (graph-captured): unconditionally recompute each request's CURRENT pool
        from static-shaped gathers -- incomplete pools are never scored, and the
        overwrite converges to the exact value the step the pool completes."""
        kp = self.index_kpool
        slot = self._idx_slot[layer_id]
        kcache = self.kvcache.index_k_cache(slot)
        gcache = self.kvcache.gate_cache(slot)
        apef = ape.float().unsqueeze(0)  # [1, kp, D]
        if md.is_decode:
            rows, kvlen = md.rows, md.kvlen
            bs = rows.shape[0]
            ar = torch.arange(kp, device=rows.device)
            pcur = torch.clamp((kvlen - 1) // kp, min=0)
            tok = pcur.unsqueeze(-1) * kp + ar                       # [bs, kp]
            kv_valid = kvlen
            if getattr(md, "spec_shared_row", False):
                # Spec verify: the bs rows are SEQUENTIAL positions of one request on a
                # shared table row. Two rows with the same current pool scatter to the
                # same slab slot, and inter-row scatter order is undefined -- so give
                # every writer the max-kvlen key set (identical bytes). A pool is only
                # ever SCORED by rows for which it is complete (kvlen_row//kp), and for
                # those the full-key value is exact; the trailing partial pool is never
                # scored and reconverges on the next regular decode step.
                kv_valid = kvlen.max().expand(bs)
            valid = tok < kv_valid.unsqueeze(-1)
            rg = rows.gather(1, tok.clamp(max=rows.shape[1] - 1).long()).clamp(min=0)
            kk = kcache[rg.reshape(-1).long()].view(bs, kp, -1).float()
            gg = gcache[rg.reshape(-1).long()].view(bs, kp, -1).float()
            logits = torch.where(valid.unsqueeze(-1), gg + apef, float("-inf"))
            probs = torch.softmax(logits, dim=1)
            pk = (probs * kk).sum(dim=1).to(kcache.dtype)            # [bs, D]
            first = rows.gather(1, (pcur * kp).unsqueeze(-1).long()).squeeze(-1)
            self.kvcache.store_pool_k(pk, first.clamp(min=0).long(), slot)
            return
        page_table = get_global_ctx().page_table
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        for r in reqs:
            if r.extend_len == 0:
                continue
            cached = r.device_len - r.extend_len
            p0, p1 = cached // kp, r.device_len // kp
            if p1 <= p0:
                continue
            row = page_table[r.table_idx, p0 * kp : p1 * kp].long()
            n = p1 - p0
            kk = kcache[row].view(n, kp, -1).float()
            gg = gcache[row].view(n, kp, -1).float()
            probs = torch.softmax(gg + apef, dim=1)
            pk = (probs * kk).sum(dim=1).to(kcache.dtype)
            self.kvcache.store_pool_k(pk, row[::kp], slot)

    def _prefill(self, md, layer_id, q_nope, q_pe, batch, indexer_qkw) -> torch.Tensor:
        t = q_nope.shape[0]
        q_cat = torch.cat([q_nope, q_pe], dim=-1)  # [T, H, 576]
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        page_table = get_global_ctx().page_table
        qo = md.qo_indptr_cpu.tolist()
        sparse = self.dsa_enabled and int(md.kv_len_cpu.max()) > self.index_topk
        if sparse and indexer_qkw is not None:
            q_idx, w = indexer_qkw[0], indexer_qkw[2]
            md.sel.clear()  # one live group leader at a time
            md.sel[layer_id] = [
                self._select_prefill(
                    self._idx_slot[layer_id],
                    q_idx[qo[i] : qo[i + 1]], w[qo[i] : qo[i + 1]],
                    page_table[r.table_idx, : r.device_len],
                    batch.positions[qo[i] : qo[i + 1]],
                )
                for i, r in enumerate(reqs)
            ]
        o = q_cat.new_empty(t, self.num_heads, self.kv_lora_rank)
        for i, r in enumerate(reqs):
            m = qo[i + 1] - qo[i]
            if m == 0:
                continue
            if sparse:
                sel, cnt = md.sel[self._leader[layer_id]][i]
            else:
                # Identity selection == dense (exact: top-min(k, T) covers every live
                # token at kv <= index_topk, and the ablation attends everything).
                # One shared row list broadcast across queries (stride 0), causality
                # through per-query counts.
                sel = page_table[r.table_idx, : r.device_len].view(1, 1, -1).to(torch.int32)
                cnt = (batch.positions[qo[i] : qo[i + 1]] + 1).to(torch.int32).view(1, m)
            o[qo[i] : qo[i + 1]] = self._attend(
                q_cat[qo[i] : qo[i + 1]].view(1, m, self.num_heads, self.latent_dim),
                layer_id, sel, cnt,
            ).view(m, self.num_heads, self.kv_lora_rank)
        return o

    # ----- CUDA graph (decode) ----------------------------------------------------------
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.max_seq_len = max_seq_len
        self.capture_bs = sorted(bs_list)
        max_bs = max(bs_list)
        width = get_global_ctx().page_table.shape[1]
        self._rows_buf = torch.full((max_bs, width), -1, dtype=torch.int32, device=self.device)
        self._kvlen_buf = torch.zeros(max_bs, dtype=torch.int32, device=self.device)
        if self.index_kpool > 1:
            pw = (width + self.index_kpool - 1) // self.index_kpool
            self._pool_rows_buf = torch.full(
                (max_bs, pw), -1, dtype=torch.int32, device=self.device
            )

    def _decode_rows(self, batch: Batch) -> torch.Tensor:
        """This decode step's per-request page-table rows [bs, W], gathered off the
        scheduler-staged ``active_table_idx`` (a device tensor -- no host loop)."""
        assert batch.active_table_idx is not None, "decode batch is missing its page-table rows"
        return get_global_ctx().page_table.index_select(0, batch.active_table_idx.to(torch.int64))

    def _stage_decode(self, batch: Batch, bs: int, table_idx: torch.Tensor) -> None:
        """Copy this step's addressing into the static graph buffers and point the
        metadata at them (restage-per-replay, same shape as the generic backends)."""
        md = batch.attn_metadata
        self._rows_buf[:bs].copy_(get_global_ctx().page_table.index_select(0, table_idx))
        self._kvlen_buf[:bs].copy_(md.kv_len_cpu.to(self.device, non_blocking=True))
        md.rows = self._rows_buf[:bs]
        md.kvlen = self._kvlen_buf[:bs]
        if self.index_kpool > 1:
            self._pool_rows_buf[:bs].copy_(self._rows_buf[:bs, :: self.index_kpool])
            md.pool_rows = self._pool_rows_buf[:bs]

    def prepare_for_capture(self, batch: Batch) -> None:
        # The capture batch is all dummy rows with no scheduler-staged
        # active_table_idx; stage the dummy request's table row for every slot
        # (dsv4_sparse precedent) -- replays overwrite with the live rows.
        self.prepare_metadata(batch)
        bs = batch.size
        dummy = torch.full(
            (bs,), batch.padded_reqs[0].table_idx, dtype=torch.int64, device=self.device
        )
        self._stage_decode(batch, bs, dummy)

    def prepare_for_replay(self, batch: Batch) -> None:
        assert batch.active_table_idx is not None, "decode batch is missing its page-table rows"
        self._stage_decode(
            batch, batch.padded_size, batch.active_table_idx.to(torch.int64)
        )

    def reset_capture(self) -> None:
        super().reset_capture()
        self._rows_buf = None
        self._kvlen_buf = None


__all__ = ["DSAAttnBackend", "DSAMetadata"]
