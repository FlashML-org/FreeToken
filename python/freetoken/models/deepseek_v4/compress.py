"""Stateful KV compressors (CSA/HCA gated pooling) + the Lightning Indexer."""

from __future__ import annotations

import torch
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.compress import gated_pool
from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace, fp4_act_quant_inplace
from freetoken.kernel.triton.dsv4.hadamard import hadamard_transform

from .args import DeepseekV4Args
from .layers import Linear, RMSNorm
from .ops import apply_rotary_emb, apply_rotary_emb_decode


class Compressor(nn.Module):
    """Learned gated pooling of KV over ``compress_ratio`` tokens (CSA overlap when
    ratio==4). Stateful across decode steps. KV / state buffers are bound from the pool."""

    def __init__(self, args: DeepseekV4Args, compress_ratio: int, head_dim: int, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.max_batch_size = args.max_batch_size
        coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32), requires_grad=False)
        # checkpoint stores these bf16; matmul upcasts on-chip (halves footprint + read).
        self.wkv = Linear(self.dim, coff * self.head_dim, kind="bf16")
        self.wgate = Linear(self.dim, coff * self.head_dim, kind="bf16")
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        # Paged-KV binding (set on first forward). All addressing -- the compressed-KV pool view,
        # the per-window-page compress-state ring, the decode snapshot -- is reached through the
        # LIVE backend (``self.attn`` -> ctx.attn_backend) per access, so the module holds no
        # state a runtime rebuild would have to unbind. ``tier`` selects which pool lists back
        # this compressor (the attention compressor vs the indexer's own).
        self.layer_id: int | None = None
        self.ring_size = coff * compress_ratio  # state rows per window page
        self.coff = coff
        self.item_size = coff * self.head_dim   # the ring row's kv|score split point
        self.P: int = 128  # window-page size (set in bind)
        self.freqs_cis: torch.Tensor | None = None
        # Rolling state register held in fp32 (the carry); seeded by prefill, advanced by
        # decode. At bs=1 it mirrors the page's ring block; persisted to the ring on write.
        self._kv_state: torch.Tensor | None = None
        self._score_state: torch.Tensor | None = None

    # Paged addressing (pool views, compressed rows, the compress-state ring) belongs to the
    # attention backend; this module keeps its weights and the reduction math.
    @property
    def attn(self):
        return get_global_ctx().attn_backend

    @property
    def cmp_pool(self) -> torch.Tensor:
        return self.attn.compress_pool(self.layer_id, self.tier)

    @property
    def state_ring(self):
        return self.attn.compress_state_ring(self.layer_id, self.tier)

    def bind_paged(self, pool, layer_id: int, freqs_cis, device, tier: str) -> None:
        self.layer_id = layer_id
        self.tier = tier
        self.P = pool.P
        self.freqs_cis = freqs_cis
        self._device = device
        coff = self.coff
        self._kv_state = torch.zeros(
            1, coff * self.compress_ratio, coff * self.head_dim, dtype=torch.float32, device=device
        )
        self._score_state = torch.full(
            (1, coff * self.compress_ratio, coff * self.head_dim),
            float("-inf"), dtype=torch.float32, device=device,
        )

    def reset(self) -> None:
        # Reset the rolling carry register (prefill re-seeds it). bs=1 single stream.
        self._kv_state.zero_()
        self._score_state.fill_(float("-inf"))

    def _write_through_carry(self, window_slot: int) -> None:
        # Persist the bs=1 register carry (ks|ss, post-APE) to the paged state ring at the current
        # window page's block, for cross-request / radix carry-by-value. ADDITIVE: the register stays
        # the compute source, so this write does not touch parity. cat: kv half = ks, score = ss.
        self.attn.write_carry(
            self.layer_id, self.tier, window_slot, self.ring_size,
            torch.cat([self._kv_state[0], self._score_state[0]], dim=-1),
        )

    def _seed_carry_from_ring(self, window_slot: int) -> None:
        # Carry-by-value on a radix re-prefill: seed the register carry FROM the paged ring block of
        # the matched window TAIL page (covering [start_pos-128, start_pos)), retained by the radix.
        # Replaces the from-scratch seed at a hit; the block was written by the producing request's
        # page-boundary write-through, cat([kv_state, score_state]) split at item_size.
        block = self.attn.read_carry(  # [ring_size, 2*item]
            self.layer_id, self.tier, window_slot, self.ring_size
        )
        item = self.item_size
        self._kv_state[0].copy_(block[:, :item])
        self._score_state[0].copy_(block[:, item:])

    def _write_boundary_carries(self, kv, score, seqlen: int, window_slots: torch.Tensor) -> None:
        # Full-prefill boundary carries: positions [0, seqlen), window_slots indexed by pos.
        self._write_boundary_carries_range(kv, score, 0, seqlen, window_slots)

    def _write_boundary_carries_range(self, kv, score, lo: int, hi: int, window_slots) -> None:
        self.attn.write_boundary_carries(
            layer_id=self.layer_id, tier=self.tier, ratio=self.compress_ratio,
            overlap=self.overlap, ring_size=self.ring_size, ape=self.ape,
            kv=kv, score=score, lo=lo, hi=hi, window_slots=window_slots,
        )

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x, start_pos: int, window_slots: torch.Tensor, tail_window_slot=None, ti: int = 0):
        # Prefill (single-request, start_pos==0) or carry-aware re-prefill (start_pos>0, radix hit).
        # ``window_slots`` is translate(full_loc_map[ti, start_pos:start_pos+seqlen]) (the NEW
        # tokens' page-tail slots). ``tail_window_slot`` (start_pos>0 only) is the matched tail
        # page's window slot from which the boundary carry is read. ``ti`` is the prefill request's
        # table row (its cmp row).
        assert self.cmp_pool is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        # start_pos>0 (radix re-prefill): carry-aware extend of the new tokens.
        if start_pos > 0:
            return self.extend(x, start_pos, window_slots, int(tail_window_slot), ti)
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        kv_full, score_full = kv, score  # pre-split (for the page-boundary carry write-through)
        ks, ss = self._kv_state, self._score_state
        should_compress = seqlen >= ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        offset = ratio if overlap else 0
        if overlap and cutoff >= ratio:
            ks[:bsz, :ratio] = kv[:, cutoff - ratio: cutoff]
            ss[:bsz, :ratio] = score[:, cutoff - ratio: cutoff] + self.ape
        if remainder > 0:
            kv, ks[:bsz, offset: offset + remainder] = kv.split([cutoff, remainder], dim=1)
            ss[:bsz, offset: offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
            score = score[:, :cutoff]
        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape
        if overlap:
            kv = self.overlap_transform(kv, 0)
            score = self.overlap_transform(score, float("-inf"))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        # Persist the compressor carry to the paged state ring: write at EVERY 128-page boundary the
        # prefill crosses (so any future radix match boundary reads its carry by value), plus the
        # in-progress tail page (the final-page carry == the decode seed for the token after the
        # whole prompt). The register stays the bs=1 compute source, so this does not perturb parity.
        if seqlen > 0:
            self._write_boundary_carries(kv_full, score_full, seqlen, window_slots)
            if seqlen % self.P != 0:
                self._write_through_carry(int(window_slots[-1].item()))
        if not should_compress:
            return None
        kv = self.norm(kv.to(dtype))
        freqs_cis = self.freqs_cis[:cutoff:ratio]
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if self.rotate:
            kv = hadamard_transform(kv)
            fp4_act_quant_inplace(kv, 32)
        else:
            act_quant_fp8_inplace(kv[..., :-rd], 64)
        # Scatter compressed blocks [0, seqlen//ratio) to the paged cmp_pool: block b -> row
        # full_loc(b*ratio) // ratio (arithmetic; block b starts at abs position b*ratio).
        block_starts = torch.arange(0, cutoff, ratio, device=self._device)
        self.attn.scatter_compressed(
            self.layer_id, self.tier,
            self.attn.compress_rows_of(ti, block_starts, self.compress_ratio), kv[0],
        )
        return kv

    def extend(self, x, start_pos: int, window_slots: torch.Tensor, tail_window_slot: int, ti: int = 0):
        """Carry-aware compressor extend for new tokens [start_pos, start_pos+seqlen).

        ``start_pos`` is 128-aligned (== the radix match boundary, divisible by both ratios).
        The carry needed at ``start_pos`` lives in the matched window TAIL page's ring block;
        seed the register from it, then run the SAME reduction as the from-scratch prefill on the
        new tokens (the seeded ``ks[:ratio]`` overlap provides the first new block's previous-
        block overlap, exactly as the from-scratch overlap-seed does). Compressed blocks scatter
        to their arithmetic rows (full_loc // ratio). ``window_slots`` is the new tokens' window
        slots (translate(full_loc_map[ti, start_pos:start_pos+seqlen])) for the boundary
        write-through; ``tail_window_slot`` is the matched tail page slot at start_pos-1.
        """
        assert self.cmp_pool is not None
        assert start_pos % self.P == 0, "radix re-prefill boundary must be 128-aligned"
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        # Seed the register carry FROM the ring (the matched tail page covering
        # [start_pos-128, start_pos)). The producing request wrote this boundary carry by value.
        self._seed_carry_from_ring(tail_window_slot)

        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        kv_full, score_full = kv, score
        ks, ss = self._kv_state, self._score_state
        should_compress = (start_pos + seqlen) // ratio > start_pos // ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        offset = ratio if overlap else 0
        # The seeded ks[:ratio] holds the boundary's overlap source (ratio-4); for ratio-128 it
        # is empty. The from-scratch prefill overwrites ks[:ratio] with kv[cutoff-ratio:cutoff]
        # (the LAST full block of THIS call); but the extend's FIRST block needs the BOUNDARY
        # overlap (already seeded), and subsequent blocks get their overlap from overlap_transform
        # within this call. So we DO NOT overwrite ks[:ratio] before building the block tensor;
        # instead we prepend the seeded carry as a synthetic block-(-1) into the reduction.
        if overlap:
            # Build [n_new_blocks, ratio, 2d] for new tokens, then prepend the carry block so
            # overlap_transform feeds block-(-1)'s first-half into block-0's overlap slots.
            kv_blocks = kv[:, :cutoff].unflatten(1, (-1, ratio))      # [1, nb, ratio, 2d]
            score_blocks = score[:, :cutoff].unflatten(1, (-1, ratio)) + self.ape
            carry_kv = ks[:bsz, :ratio].unsqueeze(1)                  # [1, 1, ratio, 2d]
            carry_ss = ss[:bsz, :ratio].unsqueeze(1)                  # already +ape at write time
            kv_blocks = torch.cat([carry_kv, kv_blocks], dim=1)       # [1, nb+1, ratio, 2d]
            score_blocks = torch.cat([carry_ss, score_blocks], dim=1)
            kv_t = self.overlap_transform(kv_blocks, 0)
            score_t = self.overlap_transform(score_blocks, float("-inf"))
            kv_t = kv_t[:, 1:]      # drop the synthetic carry block (no valid output)
            score_t = score_t[:, 1:]
            reduced = (kv_t * score_t.softmax(dim=2)).sum(dim=2)      # [1, nb, 2d]
        else:
            # ratio-128 (non-overlap): no cross-block overlap; the boundary is block-aligned so
            # the first new block starts clean. Reduce the new tokens' full blocks directly.
            kv_blocks = kv[:, :cutoff].unflatten(1, (-1, ratio))
            score_blocks = score[:, :cutoff].unflatten(1, (-1, ratio)) + self.ape
            reduced = (kv_blocks * score_blocks.softmax(dim=2)).sum(dim=2)
        # Seed the tail carry register + persist page-boundary carries for the extended range.
        # The register seed for the NEXT decode token mirrors the from-scratch prefill: overlap
        # seed = last full block kv[cutoff-ratio:cutoff], plus the remainder tail.
        # cutoff == 0 (1..ratio-1 new tokens) completes no new block, so the last complete block
        # is still the one the ring seeded -- leave the seeded overlap half alone. Zeroing it here
        # would drop the next block's overlap source, and _write_through_carry below would then
        # persist the damaged register into the tail page's ring.
        if overlap and cutoff >= ratio:
            ks[:bsz, :ratio] = kv[:, cutoff - ratio: cutoff]
            ss[:bsz, :ratio] = score[:, cutoff - ratio: cutoff] + self.ape
        if overlap:
            ks[:bsz, ratio:].zero_(); ss[:bsz, ratio:].fill_(float("-inf"))
        else:
            ks[:bsz].zero_(); ss[:bsz].fill_(float("-inf"))
        if remainder > 0:
            ks[:bsz, offset: offset + remainder] = kv[:, cutoff:]
            ss[:bsz, offset: offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
        # Page-boundary carries across [start_pos, start_pos+seqlen); plus tail if unaligned.
        end = start_pos + seqlen
        self._write_boundary_carries_range(kv_full, score_full, start_pos, end, window_slots)
        if end % self.P != 0:
            self._write_through_carry(int(window_slots[-1].item()))
        if not should_compress:
            return None
        reduced = self.norm(reduced.to(dtype))
        # freqs per compressed block: position b*ratio for absolute block b. start_pos is ratio-
        # aligned, so the new FULL blocks span [start_pos, start_pos+cutoff) at stride ratio. Use
        # the block-aligned `cutoff` end (NOT start_pos+seqlen): a stepped slice over an unaligned
        # span ceils to one extra element when seqlen % ratio != 0 (mirrors the from-scratch
        # prefill's freqs_cis[:cutoff:ratio]).
        freqs_cis = self.freqs_cis[start_pos:start_pos + cutoff:ratio]
        assert freqs_cis.size(0) == reduced.size(1), f"{freqs_cis.shape=} {reduced.shape=}"
        apply_rotary_emb(reduced[..., -rd:], freqs_cis)
        if self.rotate:
            reduced = hadamard_transform(reduced)
            fp4_act_quant_inplace(reduced, 32)
        else:
            act_quant_fp8_inplace(reduced[..., :-rd], 64)
        # Scatter the new compressed blocks (abs positions [start_pos, start_pos+cutoff)) to their
        # arithmetic rows full_loc(b*ratio) // ratio.
        block_starts = torch.arange(start_pos, start_pos + cutoff, ratio, device=self._device)
        self.attn.scatter_compressed(
            self.layer_id, self.tier,
            self.attn.compress_rows_of(ti, block_starts, self.compress_ratio), reduced[0],
        )
        return reduced

    def decode_step(
        self, x: torch.Tensor, pos: torch.Tensor, prev_window_slots: torch.Tensor,
        window_slots: torch.Tensor, rows: torch.Tensor, ti: int | None = None,
    ) -> None:
        """Batched single-token compressor update (EAGER/graph; ``pos``/``prev_window_slots``/
        ``window_slots`` are GPU int tensors ``[B]``; ``rows`` is the LOCAL row index [B] into the
        decode snapshot).

        The rolling carry lives PER ROW in the paged state ring. Each row's whole ``ring_size``
        carry block is READ from ``prev_window_slots``' page (where the prior step / prefill wrote
        it), advanced, and WRITTEN BACK to ``window_slots``' page -- so the carry follows the request
        across 128-page boundaries (within a page prev/cur are the same block). Distinct window pages
        -> disjoint ring blocks, so co-tenant rows never contaminate each other."""
        assert self.cmp_pool is not None
        B = x.size(0)
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        item = self.item_size  # the ring's kv|score split point
        x = x.view(B, -1)
        kv = bf16_linear_fp32(x, self.wkv.weight).view(B, 1, item)
        score = bf16_linear_fp32(x, self.wgate.weight).view(B, 1, item)
        idx_mod = pos % ratio  # [B]
        score = score + self.ape.index_select(0, idx_mod).view(B, 1, item)
        should = ((pos + 1) % ratio == 0).view(B, 1, 1)

        # Read each row's rolling carry block from the PREVIOUS token's page (ks|ss split at item).
        block = self.attn.read_carry_blocks(  # [B, ring_size, 2*item]
            self.layer_id, self.tier, prev_window_slots, self.ring_size
        )
        ks = block[..., :item].clone()  # [B, ring_size, item]
        ss = block[..., item:].clone()

        if overlap:
            wslot = (ratio + idx_mod).view(B, 1)  # per-row B-half slot
            ks.scatter_(1, wslot.unsqueeze(-1).expand(B, 1, item), kv)
            ss.scatter_(1, wslot.unsqueeze(-1).expand(B, 1, item), score)
            kv_eff = torch.cat([ks[:, :ratio, :d], ks[:, ratio:, d:]], dim=1)
            score_eff = torch.cat([ss[:, :ratio, :d], ss[:, ratio:, d:]], dim=1)
            compressed = gated_pool(kv_eff, score_eff, dtype)
            ks[:, :ratio] = torch.where(should, ks[:, ratio:], ks[:, :ratio])
            ss[:, :ratio] = torch.where(should, ss[:, ratio:], ss[:, :ratio])
        else:
            ks.scatter_(1, idx_mod.view(B, 1, 1).expand(B, 1, item), kv)
            ss.scatter_(1, idx_mod.view(B, 1, 1).expand(B, 1, item), score)
            compressed = gated_pool(ks, ss, dtype)
        # Persist the advanced carry block to THIS token's page (per row, disjoint blocks); the
        # next step reads it from here. Within a page prev==cur, so this is a self-consistent roll.
        self.attn.write_carry_blocks(
            self.layer_id, self.tier, window_slots, self.ring_size, torch.cat([ks, ss], dim=-1)
        )

        compressed = self.norm(compressed)
        freqs_t = self.freqs_cis.index_select(0, (pos + 1 - ratio).clamp_min(0))  # [B, rd//2]
        apply_rotary_emb_decode(compressed[..., -rd:], freqs_t)
        if self.rotate:
            compressed = hadamard_transform(compressed)
            fp4_act_quant_inplace(compressed, 32)
        else:
            act_quant_fp8_inplace(compressed[..., :-rd], 64)
        # Scatter the just-completed compressed block to cmp_pool, per row, only where the block
        # finished (``should``). Rows that did not complete a block scatter to their OWN reserved
        # scratch row (``scratch_base + row``, a discarded write) so the masked per-row write is
        # graph-safe (no host sync, no -1 index) and collision-free (distinct dst per row). The
        # completed-block row is arithmetic: full_loc(pos) // ratio (every pos in the block shares
        # it). Read off the full-loc SNAPSHOT so a concurrent next-batch allocate cannot redirect
        # this write (overlap safety); real rows resolve to a live row, dummy rows to reserved 0.
        cmp_dst = self.attn.decode_compress_rows(
            rows, pos, ratio, self.layer_id, self.tier, should.view(B), ti=ti
        )
        self.attn.scatter_compressed(self.layer_id, self.tier, cmp_dst, compressed.view(B, -1))


class Indexer(nn.Module):
    """Lightning Indexer: scores compressed KV and returns top-k positions to attend."""

    def __init__(self, args: DeepseekV4Args, compress_ratio: int):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim, kind="fp8")
        self.weights_proj = Linear(self.dim, self.n_heads, kind="bf16")
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio
        self.compressor = Compressor(args, compress_ratio, self.head_dim, rotate=True)
        # The indexer's own compressor writes its compressed keys into the idx tier; scoring
        # gathers them by block index. Both go through the LIVE backend (self.attn ->
        # ctx.attn_backend) per access, so a runtime rebuild needs no model unbind.
        self.layer_id: int | None = None
        self.freqs_cis: torch.Tensor | None = None

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    @property
    def idx_pool(self) -> torch.Tensor:
        return self.attn.compress_pool(self.layer_id, "idx")

    def bind_paged(self, pool, layer_id: int, freqs_cis, device):
        # The indexer's compressor scatters compressed indexer keys into pool.idx_pool[L] at
        # arithmetic rows (full_loc // ratio); its OWN compress-state ring is pool.indexer_state_ring[L]
        # (distinct from the attention compressor's, so the two never collide on per-page state slots).
        self.layer_id = layer_id
        self.freqs_cis = freqs_cis
        self._device = device
        self.compressor.bind_paged(pool, layer_id, freqs_cis, device, tier="idx")

    def reset(self) -> None:
        self.compressor.reset()

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int, window_slots: torch.Tensor, ti: int = 0):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = hadamard_transform(q)
        fp4_act_quant_inplace(q, 32)
        self.compressor(x, start_pos, window_slots, ti=ti)  # scatters indexer keys to idx_pool
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        keys = self.attn.indexer_keys(ti, end_pos // ratio, ratio, self.layer_id, bsz)
        scores = self.attn.indexer_prefill_logits(q, keys, weights)
        return self.attn.indexer_select_prefill(
            scores, start_pos=start_pos, seqlen=seqlen, ratio=ratio,
            topk=self.index_topk, offset=offset,
        )

    def extend(self, x, qr, start_pos, offset, window_slots, tail_window_slot, ti: int = 0):
        """Carry-aware indexer for re-prefill new tokens [start_pos, end). Runs the indexer's
        compressor extend (writes new idx blocks), scores each new query over compressed indexer
        blocks [0, end//ratio) with a causal mask, returns top-k block indices offset by ``offset``."""
        bsz, seqlen, _ = x.size()
        end = start_pos + seqlen
        freqs_cis = self.freqs_cis[start_pos:end]
        ratio, rd = self.compress_ratio, self.rope_head_dim
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = hadamard_transform(q)
        fp4_act_quant_inplace(q, 32)
        self.compressor(x, start_pos, window_slots, tail_window_slot=tail_window_slot, ti=ti)  # writes idx_pool
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        keys = self.attn.indexer_keys(ti, end // ratio, ratio, self.layer_id, bsz)
        scores = self.attn.indexer_prefill_logits(q, keys, weights)
        return self.attn.indexer_select_prefill(
            scores, start_pos=start_pos, seqlen=seqlen, ratio=ratio,
            topk=self.index_topk, offset=offset,
        )

    def decode_step(
        self, x: torch.Tensor, qr: torch.Tensor, pos: torch.Tensor, offset: int,
        prev_window_slots: torch.Tensor, window_slots: torch.Tensor, rows: torch.Tensor,
        n_stage: int,
    ) -> torch.Tensor:
        """Batched single-token indexer (EAGER/graph). Scores the first ``n_stage`` compressed
        indexer blocks per row (gathered PER ROW from the paged idx_pool), masks blocks beyond
        each row's valid count (``(pos+1)//ratio``) to ``-inf``, and returns ``index_topk`` block
        indices offset by ``offset`` (``-1`` for picks past the valid count). ``pos``/
        ``window_slots`` are ``[B]`` GPU int tensors; ``rows`` is the LOCAL row index [B] into the
        decode full-loc SNAPSHOT (read instead of the live map -> overlap safe); ``n_stage`` is the
        fixed staging width (== max valid count in eager, a static capture width under graph)."""
        B = x.size(0)
        ratio, rd = self.compress_ratio, self.rope_head_dim
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb_decode(q[..., -rd:], self.freqs_cis.index_select(0, pos))  # per-row freqs
        q = hadamard_transform(q)
        fp4_act_quant_inplace(q, 32)
        # Advance the indexer's own compressor (writes this token's idx key to idx_pool per row).
        self.compressor.decode_step(x, pos, prev_window_slots, window_slots, rows)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        valid = (pos + 1) // ratio  # [B] per-row valid block count
        # Head-reduced scores in one pass: the kernel gathers each block's key off the full-loc
        # SNAPSHOT (not the live map -> overlap-safe replay) and reads its column bound from
        # ``valid`` in DEVICE memory, so a captured graph scores only the live blocks instead of
        # the staged width. Columns past the bound come back -inf for the top-k.
        index_score = self.attn.indexer_decode_scores(
            q.reshape(B, self.n_heads, self.head_dim),
            weights.reshape(B, self.n_heads),
            valid, n_stage, ratio, self.layer_id,
        ).view(B, 1, n_stage)
        return self.attn.indexer_select_decode(
            index_score, valid=valid, topk=self.index_topk, offset=offset
        )
