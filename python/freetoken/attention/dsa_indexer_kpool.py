"""glm5_next (GLM-5.3-Flash) kpool DSA backend: pooled-indexer addressing.

Extends the GLM-5.2 DSA backend with the kpool compression scheme: the indexer K
cache is scored at POOL granularity -- every ``index_kpool`` (4) consecutive
tokens fold into one entry, a per-channel ``softmax(gate + APE)``-weighted sum of
their raw keys -- so indexer compute and top-k shrink by 4x. Selection picks
``index_topk // kpool`` pools, each pool expands back to its constituent token
rows, and the request's trailing incomplete pool ("tail", up to kpool-1 tokens)
is force-included (``index_kpool_always_select_tail``). Selection widths are
therefore ``(kpool - 1) + select_k * kpool`` with ``-1`` gather-only sentinels;
the sparse-MLA kernel masks them.

Slab convention (KpoolDSAKVCache): the index slab is a 1/kpool SHADOW of the KV
pages -- a pool's entry lives at row ``token_slot // kpool`` (well-defined because
the engine pins ``page_size % kpool == 0``, so a pool never straddles a page).
Raw keys + gate scores of the in-progress pool live in per-request tail rings
written at ``pos % kpool``; a pool that closes reads its older members from the
ring (so a chunk may start mid-pool), and a decode step whose pool does NOT close
scatters its (garbage) pooled candidate into the request's scratch row -- an
unconditional, CUDA-graph-safe write that scoring never reads (QSA precedent).

Faithfulness: pooled entries are stored bf16 (the reference Hadamard-rotates and
fp8-quantizes them -- a memory device whose rotation cancels in the score); the
pooling softmax runs fp32, matching the reference kernel. Selection is a plain
top-k over pool scores plus the tail: the reference does NOT force-include the
query's own (complete) pool -- the model is trained with that scheme, so neither
do we. Sharing note: IndexShare does not exist here; every DSA layer owns its
indexer slot and is its own leader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from .dsa import DSAAttnBackend, DSAMetadata, KpoolPlan

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models import ModelConfig


class Glm5NextDSABackend(DSAAttnBackend):
    def __init__(self, config: "ModelConfig") -> None:
        super().__init__(config)
        self._slots_buf: torch.Tensor | None = None  # static graph buffer for ring_slots
        if self.dsa_enabled:
            from freetoken.kvcache.dsa_pool import KpoolDSAKVCache

            assert isinstance(self.kvcache, KpoolDSAKVCache), (
                "glm5_next kpool indexer needs the KpoolDSAKVCache (tail rings); "
                f"the pool factory built {type(self.kvcache).__name__}"
            )

    # ----- CUDA-graph decode staging ------------------------------------------------------
    # The tail rings are keyed by Req.table_idx; a captured decode step must read it
    # from a static buffer restaged per replay (rows/kvlen precedent in the parent).
    def init_capture_graph(self, max_seq_len: int, bs_list) -> None:
        super().init_capture_graph(max_seq_len, bs_list)
        self._slots_buf = torch.zeros(max(bs_list), dtype=torch.int64, device=self.device)

    def _stage_decode(self, batch: "Batch", bs: int, table_idx: torch.Tensor) -> None:
        super()._stage_decode(batch, bs, table_idx)
        self._slots_buf[:bs].copy_(table_idx)
        batch.attn_metadata.ring_slots = self._slots_buf[:bs]

    def reset_capture(self) -> None:
        super().reset_capture()
        self._slots_buf = None

    # ----- model-family hooks -----------------------------------------------------------
    def _model_args(self, config: "ModelConfig"):
        args = config.glm5_args
        assert args is not None, "kpool dsa backend needs ModelConfig.glm5_args"
        self.kpool = args.index_kpool
        assert self.kpool > 1 and args.index_kpool_compress, (
            "Glm5NextDSABackend serves the kpool-compressed indexer; a kpool=1 "
            "checkpoint should run the plain DSA backend"
        )
        assert args.index_topk % self.kpool == 0
        self.select_k = args.index_topk // self.kpool
        assert args.index_kpool_always_select_tail, (
            "tail force-inclusion is baked into the selection layout"
        )
        return args

    def _build_index_slots(self, args, config: "ModelConfig") -> None:
        # No IndexShare: every DSA layer owns its indexer and is its own leader.
        for lid in args.dsa_layer_ids:
            if lid >= config.num_layers:
                continue
            self._idx_slot[lid] = len(self._idx_slot)
            self._leader[lid] = lid

    # ----- store: fused single path (prefill AND decode, CUDA-graph capturable) ----------
    def _plan_kpool_writes(self, md, batch: "Batch", slot: int):
        """Per-token slab/ring routing for this forward; layer-invariant, so the
        first DSA layer computes it and the rest reuse it (QSA _plan_index_writes
        shape). Pure device arithmetic: no host sync, graph-capturable.

        Rebuilt at the FIRST indexer slot of every forward, never trusted across
        forwards: a capture batch runs its warmup and its capture through ONE
        metadata object, and a cached plan would bake the warmup's (non-graph-pool)
        tensor addresses into the graph (QSA precedent)."""
        if slot != 0 and md.kpool_plan is not None:
            return md.kpool_plan
        kp = self.kpool
        out_loc = batch.out_loc.to(torch.int64)
        positions = batch.positions.to(torch.int64)
        t = out_loc.numel()
        if md.is_decode:
            # One token per request. ring_slots is the backend's STATIC buffer under
            # graphs (restaged per replay in _stage_decode); eager reads the
            # scheduler-staged active_table_idx. arange shapes are fixed per capture.
            ring_slots = (
                md.ring_slots if md.ring_slots is not None else batch.active_table_idx
            ).to(torch.int64)
            token_to_req = torch.arange(t, device=self.device, dtype=torch.int32)
            cu_seqlens = torch.arange(t + 1, device=self.device, dtype=torch.int32)
        else:
            reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
            ring_slots = torch.tensor(
                [r.table_idx for r in reqs], dtype=torch.int64, pin_memory=True
            ).to(self.device, non_blocking=True)
            cu_cpu = md.qo_indptr_cpu
            cu_seqlens = cu_cpu.to(self.device, non_blocking=True)
            token_to_req = torch.repeat_interleave(
                torch.arange(len(reqs), dtype=torch.int32),
                (cu_cpu[1:] - cu_cpu[:-1]).to(torch.int64),
            ).pin_memory().to(self.device, non_blocking=True)
        slots = ring_slots.index_select(0, token_to_req.to(torch.int64))
        # page_size % kpool == 0, so out_loc % kp == position % kp: a group closes
        # exactly on position % kp == kp - 1. Non-closing rows land in the request's
        # scratch row (never scored).
        closing = positions % kp == kp - 1
        cmp_rows = torch.where(
            closing, out_loc // kp, self.kvcache.cmp_scratch_base + slots
        ).to(torch.int32)
        # Ring refresh: only each request's last kp rows survive to the next forward
        # (one keeper per pos%kp residue -- deterministic, no write races).
        rows = torch.arange(t, device=self.device, dtype=torch.int64)
        ends = cu_seqlens.to(torch.int64).index_select(0, token_to_req.to(torch.int64) + 1)
        keep = rows >= ends - kp
        ring_row = slots * kp + positions % kp
        ring_rows = torch.where(keep, ring_row, torch.full_like(ring_row, -1)).to(
            torch.int32
        )
        md.kpool_plan = KpoolPlan(cmp_rows, ring_rows, ring_slots, token_to_req, cu_seqlens)
        return md.kpool_plan

    def _store_index(self, inputs, batch: "Batch", layer_id: int) -> None:
        """One fused kernel serves prefill and decode (layout and member
        resolution: module docstring); the ring refresh follows the read."""
        from freetoken.kernel.triton.kpool_compress import kpool_compress_store
        from freetoken.kernel.triton.qsa import qsa_store_rows

        md = batch.attn_metadata
        assert isinstance(md, DSAMetadata)
        assert inputs.gate is not None and inputs.ape is not None, (
            "kpool store needs DSAIndexerInputs.gate/ape from the model's indexer"
        )
        k, gate = inputs.k, inputs.gate.to(inputs.k.dtype)
        slot = self._idx_slot[layer_id]
        tail_k, tail_g = self.kvcache.tail_k(slot), self.kvcache.tail_gate(slot)
        plan = self._plan_kpool_writes(md, batch, slot)
        kpool_compress_store(
            k, gate,
            tail_k.view(-1, k.shape[-1]), tail_g.view(-1, k.shape[-1]),
            inputs.ape,
            plan.ring_slots, plan.token_to_req, plan.cu_seqlens, batch.positions,
            self.kvcache.index_k_cache(slot), plan.cmp_rows,
            self.kpool,
        )
        # After the compression read: the ring rows this forward overwrites are
        # exactly the ones a straddling group just consumed.
        qsa_store_rows(tail_k, plan.ring_rows, k)
        qsa_store_rows(tail_g, plan.ring_rows, gate)

    # ----- selection: pools -> token rows + tail ------------------------------------------
    def _expand_and_tail(
        self,
        picks: torch.Tensor,  # [B, m, select_k] pool ids, -1 sentinel
        rows: torch.Tensor,  # [B, W] or [B, m, W] position-ordered physical rows
        q_pos: torch.Tensor,  # [B, m] query positions (token-granular)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selected pools -> token rows, tail tokens appended FIRST (fixed kpool-1
        slots, -1 padded). Returns (sel [B, m, (kpool-1) + select_k*kpool] int32,
        cnt [B, m] int32)."""
        kp = self.kpool
        b, m, k_sel = picks.shape
        device = picks.device
        offs = torch.arange(kp, device=device)

        # History: pool pick p -> token positions p*kp + [0, kp).
        hist_pos = picks.unsqueeze(-1) * kp + offs  # [B, m, k_sel, kp]
        hist_pos = torch.where(
            picks.unsqueeze(-1) < 0, hist_pos.new_full((), -1), hist_pos
        ).view(b, m, k_sel * kp)

        # Tail: positions [n_pools*kp, q_pos] of each query's own request/step.
        n_pools = (q_pos + 1) // kp  # complete pools at this query
        tail_start = n_pools * kp
        toffs = torch.arange(kp - 1, device=device)
        tail_pos = tail_start.unsqueeze(-1) + toffs  # [B, m, kp-1]
        tail_pos = torch.where(tail_pos <= q_pos.unsqueeze(-1), tail_pos, tail_pos.new_full((), -1))

        pos = torch.cat([tail_pos, hist_pos], dim=-1)  # [B, m, width]
        rows_b = rows if rows.dim() == 3 else rows.unsqueeze(1).expand(b, m, -1)
        sel = rows_b.gather(-1, pos.clamp_min(0).long()).to(torch.int32)
        sel = torch.where(pos < 0, sel.new_full((), -1), sel)
        # Valid entries: the fixed tail slots + all expanded picked pools. -1
        # sentinels inside the bound are masked by the sparse kernel.
        # clamp_max (not minimum(new_tensor)): no H2D copy, CUDA-graph safe.
        cnt = ((kp - 1) + n_pools.clamp_max(k_sel) * kp).to(torch.int32)
        return sel, cnt

    def _decode(self, md, layer_id, q_nope, q_pe, inputs) -> torch.Tensor:
        bs = q_nope.shape[0]
        rows, kvlen = md.rows, md.kvlen
        if not self.dsa_enabled:
            return super()._decode(md, layer_id, q_nope, q_pe, inputs)
        if inputs is not None:
            q_idx, w = inputs.q, inputs.w
            kp = self.kpool
            n_pools = (kvlen // kp).to(torch.int32)
            # Pool p's shadow row = any member's token slot // kp (pools never
            # straddle pages): stride the position-ordered row snapshot down to
            # pool granularity, then divide into the shadow slab.
            pool_rows = (rows[:, kp - 1 :: kp] // kp).contiguous()
            s = self.dsa_decode_scores(q_idx, w, self._idx_slot[layer_id], pool_rows, n_pools)
            k_sel = min(self.select_k, s.shape[-1])
            picks = self.indexer_select_decode(
                s.view(bs, 1, -1), valid=n_pools, topk=k_sel, offset=0
            )  # [bs, 1, k_sel] pool ids
            sel, cnt = self._expand_and_tail(
                picks.transpose(0, 1),  # [1, bs, k_sel]
                rows.unsqueeze(0),  # [1, bs, W]
                (kvlen - 1).view(1, bs),
            )
            sel = sel.transpose(0, 1)  # [bs, 1, width]
            cnt = cnt.view(bs, 1)
            md.sel.clear()
            md.sel[layer_id] = (sel, cnt)
        sel, cnt = md.sel[self._leader[layer_id]]
        q_cat = torch.cat([q_nope, q_pe], dim=-1).view(bs, 1, self.num_heads, self.latent_dim)
        o = self._attend(q_cat, layer_id, sel, cnt)
        return o.view(bs, self.num_heads, self.kv_lora_rank)

    def _select_prefill(
        self, slot: int, q_idx: torch.Tensor, w: torch.Tensor,
        rows: torch.Tensor, positions: torch.Tensor, start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-request causal top-k at pool granularity, expanded to token rows.
        ``start_pos`` is the request's cached_len (host int, no device sync)."""
        kp = self.kpool
        kv_len = rows.numel()
        n_pools_total = kv_len // kp
        pool_rows = rows[kp - 1 :: kp] // kp  # shadow rows (token slot // kpool)
        k_pool = self.kvcache.index_k_cache(slot).index_select(0, pool_rows.long())
        k_sel = min(self.select_k, max(n_pools_total, 1))
        m = q_idx.shape[0]
        width = (kp - 1) + k_sel * kp
        if n_pools_total == 0:
            # No complete pool to score (a sub-kpool request riding a sparse batch):
            # the selection is the tail alone (select_prefill would return zero pick
            # columns and break the fixed width below).
            picks = torch.full((1, m, k_sel), -1, dtype=torch.int32, device=self.device)
            return self._expand_and_tail(picks, rows.view(1, -1), positions.view(1, -1))
        sel = torch.empty(m, width, dtype=torch.int32, device=self.device)
        cnt = torch.empty(m, dtype=torch.int32, device=self.device)
        chunk = 512
        for s0 in range(0, m, chunk):
            s1 = min(s0 + chunk, m)
            scores = self.dsa_prefill_logits(q_idx[s0:s1], k_pool, w[s0:s1])
            picks = self.indexer_select_prefill(
                scores.unsqueeze(0), start_pos=start_pos + s0, seqlen=s1 - s0,
                ratio=kp, topk=k_sel, offset=0,
            )  # [1, s1-s0, k_sel] pool ids
            sel_c, cnt_c = self._expand_and_tail(
                picks, rows.view(1, -1), positions[s0:s1].view(1, -1)
            )
            sel[s0:s1] = sel_c[0]
            cnt[s0:s1] = cnt_c[0]
        return sel.view(1, m, width), cnt.view(1, m)


__all__ = ["Glm5NextDSABackend"]
