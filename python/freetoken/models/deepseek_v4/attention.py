"""Multi-head Latent Attention over the paged DSV4 pools (sliding window + compressed
top-k); owns the per-layer Compressor/Indexer instances."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator
from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
from freetoken.kernel.triton.dsv4.norm import rms_norm

from .args import DeepseekV4Args
from .compress import Compressor, Indexer
from .layers import Linear, RMSNorm, get_compress_topk_idxs, get_window_topk_idxs
from .ops import apply_rotary_emb, apply_rotary_emb_decode, get_freqs_cis
from .parallel import div_tp, tp_size


class Attention(nn.Module):
    """Multi-head Latent Attention with sliding window + optional KV compression.

    KV lives in the paged DSV4 pools (window / compressed / indexer), resolved per token via
    the manager's slot maps. Attention is a PAGED gather: the top-k is built as GLOBAL pool
    slots (window-first, then compressed) and the backend's paged kernel gathers KV straight from
    ``window_pool[L]`` / ``cmp_pool[L]`` inside the kernel -- no per-forward staging slab. Both
    prefill and decode use the same paged kernel."""

    def __init__(self, layer_id: int, args: DeepseekV4Args, compress_ratio: int | None = None):
        # ``compress_ratio`` overrides the per-layer table. The dSpark draft layers pass 0:
        # their checkpoint ships no compressor or indexer weights, because a draft block
        # attends only over the sliding window the target already filled.
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.window_size = args.window_size
        self.compress_ratio = (
            args.compress_ratios[layer_id] if compress_ratio is None else compress_ratio
        )
        self.eps = args.norm_eps
        # dSpark draft layers set this. A drafted BLOCK is proposed at once, so each of
        # its queries may see the whole block, not just the positions before it -- that
        # is what makes the proposal semi-autoregressive instead of five serial steps.
        # Expressed through the sparse top-k: the causal cutoff moves from "my own
        # position" to "the end of my block". Context positions precede the block, so
        # they stay visible either way and need no special case.
        self.non_causal = False

        # Head parallelism: a rank owns whole o_groups, and therefore whole heads. The
        # per-group head width (wo_a_k) is a property of one group, so it does NOT shard.
        self.tp_size = tp_size()
        self.n_heads = div_tp(args.n_heads, "n_heads")
        self.n_groups = div_tp(args.o_groups, "o_groups")
        self._comm = DistributedCommunicator() if self.tp_size > 1 else None

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32), requires_grad=False)
        # wq_a / wkv / kv_norm stay replicated: the MLA latent KV is shared by every head.
        self.wq_a = Linear(self.dim, self.q_lora_rank, kind="fp8")
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim, kind="fp8")
        self.wkv = Linear(self.dim, self.head_dim, kind="fp8")
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        # wo_a: dequantized to bf16 (reference runs a bf16 grouped-output einsum).
        wo_a_rows = self.n_groups * args.o_lora_rank
        wo_a_k = args.n_heads * self.head_dim // args.o_groups
        self.wo_a = nn.Parameter(torch.empty(wo_a_rows, wo_a_k, dtype=torch.bfloat16), requires_grad=False)
        # Row-parallel: each rank consumes its own groups and contributes a partial sum.
        self.wo_b = Linear(self.n_groups * args.o_lora_rank, self.dim, kind="fp8")
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            self.indexer = Indexer(args, self.compress_ratio) if self.compress_ratio == 4 else None
        else:
            self.compressor = None
            self.indexer = None

        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        self._freqs_params = (
            self.rope_head_dim, args.max_seq_len, original_seq_len,
            rope_theta, args.rope_factor, args.beta_fast, args.beta_slow,
        )
        # Bound on first forward (freqs/arange only; pool buffers are read off the LIVE pool).
        self.freqs_cis: torch.Tensor | None = None

    # KV addressing lives in the attention backend (ctx.attn_backend), pool buffers on the live
    # pool (ctx.kv_cache) -- both read per access, so the model holds NO state a runtime rebuild
    # would have to unbind. Capture bakes whatever the property returns at capture time (stable).
    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind(self, pool, device: torch.device) -> None:
        # pool is the DSV4PagedKVCache carrying the slot maps (set by the manager). The model holds
        # only the pool HANDLE; buffers + slot maps are read off it per access via @property, so a
        # runtime pool rebuild needs no per-buffer unbind.
        L = self.layer_id
        win = self.window_size
        self.P = pool.P
        self.freqs_cis = get_freqs_cis(*self._freqs_params, device)
        if self.compress_ratio:
            self.compressor.bind_paged(pool, L, self.freqs_cis, device, tier="attn")
            if self.indexer is not None:
                self.indexer.bind_paged(pool, L, self.freqs_cis, device)

    def reset(self) -> None:
        if self.compressor is not None:
            self.compressor.reset()
        if self.indexer is not None:
            self.indexer.reset()


    def _wo(self, o: torch.Tensor, bsz: int, seqlen: int) -> torch.Tensor:
        # Under TP, ``o`` holds only this rank's heads, so the grouped einsum and wo_b
        # produce a PARTIAL sum over the full output dim. One all-reduce completes it --
        # the single attention-side collective per block.
        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a).flatten(2)
        o = self.wo_b(o)
        if self._comm is not None:
            o = self._comm.all_reduce(o)
        return o

    def _prefill_segment(self, x_seg, qr_seg, kv_seg, ti: int, start_pos: int, n: int):
        """One request's prefill work inside a (possibly ragged) batch: persist its window KV,
        advance its compressor/indexer carry, and return its per-query candidate lists.

        ``start_pos == 0`` is a cold prompt (carry re-seeded from scratch); ``start_pos > 0`` is a
        radix hit, whose carry is seeded BY VALUE from the matched tail page's ring block. Returns
        ``(win_global [1, n, w], cmp_global [1, n, c] | None)`` at natural widths (cold: w =
        min(n, win); extend: w = win); the caller pads BOTH halves to the batch-uniform widths
        and concatenates.
        """
        win, ratio, device = self.window_size, self.compress_ratio, x_seg.device
        end = start_pos + n
        slots = self.attn.window_slots_of(ti, start_pos, end)
        self.attn.store_window(kv_seg, self.layer_id, slots)

        if self.non_causal and start_pos == 0:
            # A draft block always continues an existing context, so it never takes the
            # cold path. Refuse rather than silently emit a causal grid.
            raise ValueError(
                "dSpark non-causal attention needs a resumed segment (start_pos > 0); "
                "the drafter is not meant to run a cold prompt"
            )
        if start_pos == 0:
            win_cols = get_window_topk_idxs(win, 1, n, 0).to(device)
            # natural width min(n, win); the caller pads to the batch-uniform width
            win_global = self.attn.win_cols_to_global(win_cols, slots)
        else:
            # Target attention uses its ordinary 128-token causal window.  A DSpark
            # draft query instead sees 128 CONTEXT tokens plus every token in its
            # parallel query block (paper section 3.1; same sparse list as vLLM).
            w_lo = max(
                0,
                start_pos - win if self.non_causal else start_pos - win + 1,
            )
            ws_pool = self.attn.window_slots_of(ti, w_lo, end)
            # One implementation of the masking rule, shared with the tests that pin it.
            # Non-causal (dSpark draft layers) gives every query in the block the SAME
            # window, ending at the block's last position, so a drafted query sees the
            # whole block plus the context before it.
            from .dspark import window_cols_for_block

            win_cols = window_cols_for_block(
                start_pos, n, win, self.non_causal, device=device
            ).unsqueeze(0)
            win_global = self.attn.win_cols_to_global(win_cols, ws_pool)

        if not ratio:
            return win_global, None
        # Only the compressor/indexer read the matched tail page's slot; resolving it costs a
        # host sync (.item()), so do it after the ratio-0 early-out.
        tail_ws = (
            int(self.attn.window_slots_of(ti, start_pos - 1, start_pos).item())
            if start_pos > 0 else None
        )
        if start_pos == 0:
            self.reset()  # re-seed the compressor/indexer carry from scratch
            blocks = (
                self.indexer(x_seg, qr_seg, 0, 0, slots, ti) if self.indexer is not None
                else get_compress_topk_idxs(ratio, 1, n, 0, 0).to(device)
            )
            self.compressor(x_seg, 0, slots, ti=ti)
        elif (
            self.non_causal
            or getattr(get_global_ctx().batch, "speculative", False)
            or start_pos % self.P != 0
        ):
            # A speculative block starts wherever generation reached, which is mid-page
            # 127 times out of 128. The compressor's EXTEND path cannot serve that: it
            # reduces tokens with ``start_pos // ratio`` arithmetic and asserts
            # 128-alignment, because it seeds the carry from a matched tail PAGE.
            #
            # Its per-token path has no such assumption -- it reads the carry from the
            # previous token's slot, advances it, and writes it to this token's slot,
            # which is exactly what vLLM's save_partial_states kernel does (one program
            # per token, ape chosen by position % compress_ratio, no alignment anywhere).
            # So drive the compressor per token and leave attention and MoE batched:
            # the compressor is cheap next to the expert fetch this block exists to
            # amortize.
            self._advance_compressor_per_token(x_seg, start_pos, n, slots, tail_ws, ti)
            # Run the INDEXER here too. Falling back to _compress_topk_extend chooses
            # compressed blocks positionally, which silently replaces DeepSeek-V4's
            # learned sparse selection with a fixed one -- so the target attends to the
            # wrong blocks and its own logits are wrong. Since a speculative block is
            # unaligned 127 times out of 128, that was every verify.
            blocks = (
                self.indexer.extend_unaligned(
                    x_seg, qr_seg, start_pos, 0, slots, tail_ws, ti
                )
                if self.indexer is not None
                else self._compress_topk_extend(n, start_pos, end, 0, device, 1)
            )
        else:
            blocks = (
                self.indexer.extend(x_seg, qr_seg, start_pos, 0, slots, tail_ws, ti)
                if self.indexer is not None
                else self._compress_topk_extend(n, start_pos, end, 0, device, 1)
            )
            self.compressor(x_seg, start_pos, slots, tail_window_slot=tail_ws, ti=ti)
        return win_global, self.attn.blocks_to_global(blocks, ratio, ti=ti)

    def _advance_compressor_per_token(
        self, x_seg, start_pos: int, n: int, slots: torch.Tensor, tail_ws: int | None,
        ti: int,
    ) -> None:
        """Walk the compressor one token at a time across an unaligned segment.

        Each step reads the carry from the PREVIOUS token's window slot and writes the
        advanced carry to this token's -- the same movement a decode step makes, so no
        block-boundary arithmetic is involved and any ``start_pos`` is valid.
        """
        device = x_seg.device
        rows = torch.zeros(1, dtype=torch.int64, device=device)
        prev = torch.full((1,), tail_ws if tail_ws is not None else int(slots[0]),
                          dtype=slots.dtype, device=device)
        for t in range(n):
            cur = slots[t:t + 1]
            pos_t = torch.full((1,), start_pos + t, dtype=torch.int64, device=device)
            # ti, not the decode snapshot: this runs inside a prefill, where none exists.
            self.compressor.decode_step(x_seg[:, t], pos_t, prev, cur, rows, ti=ti)
            prev = cur

    def forward_ragged(self, x, segments, flat_positions):
        """Ragged batched prefill (cu_seqlens). ``x`` is [1, T, dim] -- the requests' NEW token
        streams concatenated (NO padding); ``segments`` is [(offset, n, table_idx, start_pos)]
        tiling [0, T); ``flat_positions`` [T] is each token's ABSOLUTE position, so a radix-hit
        request continues from its cached length.

        q / kv / o run FLAT over all T tokens (one GEMM, one rope each). The compressor + indexer
        carry is stateful (a bs=1 register + per-page ring), so they run PER REQUEST on the
        request's [offset, offset+n) slice with its own slot-map row -- cold segments re-seed the
        carry, radix-hit segments resume it from the matched tail page. The per-request top-k lists
        are padded to a UNIFORM width (window -> ``win`` cols at bs>1, the natural width at
        bs==1; compressed -> the batch max) with
        ``-1`` (the kernel masks ``-1`` to a no-op, so padding is bit-identical), then concatenated
        into ONE flat ``topk_idxs[1, T, win+max_c]`` for a SINGLE paged-attention launch
        (grid (T, head)). Each query gathers only its own request's slots, so requests are isolated.
        """
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        device = x.device
        _, T, _ = x.size()
        if len(segments) == 1:
            # single contiguous segment: a free slice view instead of a per-layer gather
            freqs = self.freqs_cis[segments[0][3]:segments[0][3] + T]
        else:
            freqs = self.freqs_cis.index_select(0, flat_positions)  # [T, rd//2] per-token rope

        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
        q = rms_norm(q, None, self.eps)
        apply_rotary_emb(q[..., -rd:], freqs)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs)
        act_quant_fp8_inplace(kv[..., :-rd], 64)

        # Per-request: persistent window write + window/compressed top-k + stateful compressor/
        # indexer carry. Each request owns disjoint window/cmp/idx slots + ring blocks.
        win_parts: list[torch.Tensor] = []
        cmp_parts: list[torch.Tensor | None] = []
        max_c = 0
        for off, n, ti, start_pos in segments:
            win_global, cmp_global = self._prefill_segment(
                x[:, off:off + n], qr[:, off:off + n], kv[0, off:off + n], ti, start_pos, n
            )
            win_parts.append(win_global)
            cmp_parts.append(cmp_global)
            if cmp_global is not None:
                max_c = max(max_c, cmp_global.shape[-1])

        # Uniform window width for the single flat launch. bs==1 keeps the segment's natural
        # width (a cold prompt shorter than the window yields < win columns) -- bit-identical
        # to the historical single-request path: pad columns are masked no-ops value-wise, but
        # they shift the kernel's window/compressed tile split, and that rounding-order change
        # is enough to flip a greedy near-tie. bs>1 pads every part to win (the historical
        # batched behavior).
        # Ordinary target segments keep the historical batch-uniform ``win`` width.
        # DSpark's non-causal query block is wider: trailing ``win`` context columns
        # PLUS all gamma query columns.  Labeling only the first 128 columns as window
        # in a multi-request draft misclassifies/drops the query-query suffix even
        # though the sparse indices themselves correctly contain all 133 columns.
        n_window = (
            max(part.shape[-1] for part in win_parts)
            if self.non_causal
            else (win if len(segments) > 1 else win_parts[0].shape[-1])
        )
        flat = []
        for i, (off, n, ti, _start) in enumerate(segments):
            wg = win_parts[i]
            if wg.shape[-1] < n_window:  # pad window picks to the uniform width
                wg = F.pad(wg, (0, n_window - wg.shape[-1]), value=-1)
            if ratio:
                cg = cmp_parts[i]
                if cg.shape[-1] < max_c:  # pad compressed picks to the uniform width
                    cg = F.pad(cg, (0, max_c - cg.shape[-1]), value=-1)
                flat.append(torch.cat([wg, cg], dim=-1))
            else:
                flat.append(wg)
        # [1, T, win(+max_c)]; a 1-element cat would copy the whole list per layer for nothing
        topk_idxs = (flat[0] if len(flat) == 1 else torch.cat(flat, dim=1)).int()

        o = self.attn.attend(
            q, self.layer_id, topk_idxs, n_window, self.attn_sink, self.softmax_scale,
            has_compression=bool(ratio),
        )
        apply_rotary_emb(o[..., -rd:], freqs, True)
        return self._wo(o, 1, T)

    def _compress_topk_extend(self, seqlen, start_pos, end, offset, device, bsz):
        # ratio-128 layers (no indexer): each new query at abs pos p attends compressed blocks
        # [0, (p+1)//ratio); pool index = block + offset, causal-masked to -1 beyond (p+1)//ratio.
        ratio = self.compress_ratio
        n_blocks = end // ratio
        blk = torch.arange(n_blocks, device=device).unsqueeze(0)         # [1, n_blocks]
        abs_p1 = (start_pos + torch.arange(seqlen, device=device) + 1).unsqueeze(1)  # [seqlen,1]
        valid = blk < (abs_p1 // ratio)
        out = torch.where(valid, blk + offset, -1)
        return out.unsqueeze(0).expand(bsz, -1, -1)

    def decode_step(
        self, x: torch.Tensor, pos: torch.Tensor, rows: torch.Tensor, cmp_stage_cap: int,
        wctx=None,
    ) -> torch.Tensor:
        """Batched single-token attention (EAGER/graph). Per row, builds the per-query top-k as
        GLOBAL pool slots -- the live 128-window ring slots (window-first) then the first
        ``n_cmp_stage`` compressed-block slots -- and hands them to the backend, whose kernel
        gathers KV straight from ``window_pool`` / ``cmp_pool`` (no staging slab).

        ``pos`` is ``[B]`` GPU int (per-row position). ``rows`` is ``[B]`` LOCAL row indices
        (arange B) into the backend's decode SNAPSHOT -- the captured graph reads
        the snapshot, never the live slot maps, so a concurrent next-batch allocate_paged cannot
        corrupt this replay (overlap safety). ``cmp_stage_cap`` is the max position any row reaches;
        the layer's fixed compressed top-k width is ``(cmp_stage_cap+1)//ratio`` (= max valid count
        over rows in eager; a static capture width = ``(max_seq-1+1)//ratio`` under graph). Picks
        past each row's valid count are masked to -1 (the kernel ignores -1), so co-tenant rows
        stay isolated."""
        B = x.size(0)
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        device = x.device
        n_cmp_stage = (cmp_stage_cap + 1) // ratio if ratio else 0
        # Layer-invariant window-ring slots are computed ONCE in the model decode loop and threaded
        # in via wctx (fall back to a fresh metadata compute when called standalone). freqs_t
        # stays per-layer (freqs_cis differs between compressed and non-compressed layers).
        if wctx is None:
            wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        window_slots, prev_window_slots, window_slots_topk = wctx
        freqs_t = self.freqs_cis.index_select(0, pos)  # [B, rd_pairs] (per-layer rope)

        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
        q = rms_norm(q, None, self.eps)
        apply_rotary_emb_decode(q[..., -rd:], freqs_t)  # per-row position freqs

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb_decode(kv[..., -rd:], freqs_t)
        act_quant_fp8_inplace(kv[..., :-rd], 64)
        # Persistent window write: each row's new token to its own window slot (swa_ratio=1).
        self.attn.store_window(kv.view(B, -1), self.layer_id, window_slots)
        # prev_window_slots (compressor rolling carry, read from the previous token's window page)
        # and window_slots_topk (the GLOBAL window-ring slots, window-first column order) are
        # layer-invariant -> taken from the hoisted wctx instead of recomputed here.
        n_window = win
        cmp_counts = None
        if self.compress_ratio:
            self.compressor.decode_step(x, pos, prev_window_slots, window_slots, rows)
            valid = (pos + 1) // ratio  # [B] per-row compressed block count
            if self.indexer is not None:
                # Indexer returns the selected BLOCK indices (offset 0) into [0, n_cmp_stage) or -1.
                blocks = self.indexer.decode_step(
                    x, qr, pos, 0, prev_window_slots, window_slots, rows, n_cmp_stage
                )  # [B, 1, k] block index or -1
            else:
                # Each valid block b is its own pick (column order = block order); invalid -> -1.
                blk = torch.arange(n_cmp_stage, device=device)
                col_valid = blk[None, :] < valid[:, None]  # [B, n_cmp_stage]
                blocks = torch.where(col_valid, blk[None, :], -1).view(B, 1, n_cmp_stage)
            # Block index -> GLOBAL cmp row, off the decode SNAPSHOT (rows are local).
            compress_idxs = self.attn.blocks_to_global(blocks, ratio, rows=rows)
            topk_idxs = torch.cat([window_slots_topk, compress_idxs], dim=-1)
            # Valid compressed picks are a PREFIX of the list: the ratio-128 layer emits blocks in
            # order, and the indexer's top-k sorts its -1 picks (score -inf) last. So one count per
            # row bounds the kernel's loop -- read from DEVICE memory, so the captured graph's work
            # tracks the live position instead of the staged width.
            cmp_counts = valid.clamp(max=compress_idxs.shape[-1]).to(torch.int32).view(B, 1)
        else:
            topk_idxs = window_slots_topk
        topk_idxs = topk_idxs.int()

        o = self.attn.attend(
            q, self.layer_id, topk_idxs, n_window, self.attn_sink, self.softmax_scale,
            cmp_counts=cmp_counts, has_compression=bool(ratio),
        )
        apply_rotary_emb_decode(o[..., -rd:], freqs_t, True)  # per-row position freqs (inverse)
        return self._wo(o, B, 1)

    def verify_block(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        rows: torch.Tensor,
        cmp_stage_cap: int,
        num_reqs: int,
        span: int,
        wctx,
    ) -> torch.Tensor:
        """Fixed-shape causal target verification using decode-addressed kernels.

        Large projections, MoE work and sparse attention remain batched over all
        verification rows. Stateful compressors advance in request-major token order
        so each token reads its predecessor's new carry, with no host-derived position
        or address embedded in the CUDA graph.
        """
        n = x.shape[1]
        if n != num_reqs * span:
            raise ValueError(f"verify block has {n} rows, expected {num_reqs} * {span}")
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        flat = x.reshape(n, self.dim)
        xq = flat.unsqueeze(1)
        freqs_t = self.freqs_cis.index_select(0, pos)

        qr = q = self.q_norm(self.wq_a(xq))
        q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
        q = rms_norm(q, None, self.eps)
        apply_rotary_emb_decode(q[..., -rd:], freqs_t)

        kv = self.kv_norm(self.wkv(xq))
        apply_rotary_emb_decode(kv[..., -rd:], freqs_t)
        act_quant_fp8_inplace(kv[..., :-rd], 64)

        window_slots, prev_window_slots, window_slots_topk = wctx
        self.attn.store_window(kv.view(n, -1), self.layer_id, window_slots)
        n_cmp_stage = (cmp_stage_cap + 1) // ratio if ratio else 0
        cmp_counts = None
        if ratio:
            # This order is also the carry-journal order consumed after acceptance.
            for r in range(num_reqs):
                base = r * span
                row = rows[base:base + 1]
                for s in range(span):
                    i = base + s
                    self.compressor.decode_step(
                        flat[i:i + 1], pos[i:i + 1],
                        prev_window_slots[i:i + 1], window_slots[i:i + 1], row,
                    )
            valid = (pos + 1) // ratio
            if self.indexer is not None:
                blocks = self.indexer.verify_block(
                    flat, qr.reshape(n, -1), pos,
                    prev_window_slots, window_slots, rows, n_cmp_stage,
                    num_reqs, span,
                )
            else:
                blk = torch.arange(n_cmp_stage, device=x.device)
                blocks = torch.where(
                    blk[None, :] < valid[:, None], blk[None, :], -1
                ).view(n, 1, n_cmp_stage)
            compress_idxs = self.attn.blocks_to_global(blocks, ratio, rows=rows)
            topk_idxs = torch.cat([window_slots_topk, compress_idxs], dim=-1)
            cmp_counts = valid.clamp(max=compress_idxs.shape[-1]).to(torch.int32).view(n, 1)
        else:
            topk_idxs = window_slots_topk

        o = self.attn.attend(
            q, self.layer_id, topk_idxs.int(), win, self.attn_sink,
            self.softmax_scale, cmp_counts=cmp_counts, has_compression=bool(ratio),
        )
        apply_rotary_emb_decode(o[..., -rd:], freqs_t, True)
        return self._wo(o, n, 1).view(1, n, self.dim)
