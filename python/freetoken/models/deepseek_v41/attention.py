"""DSV41 attention layer (reference ``Attention``): latent MLA over a sliding window plus, on
compressing layers, ``index_topk`` compressed positions -- one paged sparse-attention call.

Per layer mode (``LayerRole``):

* Full     -- ``publish_*`` compresses its input into the kv source's main KV (fp4) and index keys
              (fp4), then its indexer selects fresh Top-K rows and publishes them on the forward's
              ``SharedSelection``; the decoder's first Full layer also builds the candidate pool.
* Reindex  -- selects fresh Top-K rows with its own indexer over the shared keys (inside the
              candidate pool in the decoder) and publishes them.
* Reuse    -- attends over the latest published Top-K rows.
* Window   -- the sliding window only.

All modes compute their own query and sliding-window KV (fp8 into the layer's window ring). The
attention output is rotated back (K == V share one rotated latent) and projected through the
grouped low-rank ``wo_a`` (block-diagonal over ``o_groups``; fp8 payload kept, W8A16 at decode) and ``wo_b``.

Prefill runs ragged over a flat token stream tiled by ``PrefillSegment``s; the compressor / indexer
state is per request, so those run per segment. Decode is batched (one token per row) and
CUDA-graph safe: every gather reads the backend's snapshot, never the live page table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from freetoken.attention.dsv41_sparse import PrefillSegment
from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.fp8_linear import GEMV_MAX_M, dequant_block_fp8, grouped_w8a16_gemv
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.layers.quantization import QuantKind
from freetoken.layers.quantization.scheme import fp8_block_size

from .args import DeepseekV41Args, LayerRole, Mode
from .compressor import Compressor
from .indexer import Indexer
from .rope import apply_rotary_emb, apply_rotary_emb_decode, get_freqs_cis


class DSV41Attention(BaseOP):
    def __init__(self, args: DeepseekV41Args, role: LayerRole, *, quant_config=None, prefix: str = ""):
        self.args = args
        self.role = role
        self.layer_id = role.layer_id
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window = args.window_size
        self.softmax_scale = args.head_dim**-0.5

        self.wq_a = LinearReplicated(args.dim, args.q_lora_rank, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wq_a")
        self.q_norm = RMSNorm(args.q_lora_rank, args.norm_eps)
        self.wq_b = LinearReplicated(args.q_lora_rank, args.n_heads * args.head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wq_b")
        self.wkv = LinearReplicated(args.dim, args.head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wkv")
        self.kv_norm = RMSNorm(args.head_dim, args.norm_eps)
        # Block-diagonal over groups. The reference dequantizes it to bf16 and runs a grouped einsum
        # (the activation stays bf16); a quantized checkpoint keeps the e4m3 payload + e8m0 scale
        # here and dequantizes in-kernel -- the same math, half the bytes per decode step, and no
        # bf16 copy in VRAM. An unquantized checkpoint ships it bf16.
        wo_a_scheme = quant_config.scheme_for(f"{prefix}.wo_a") if quant_config is not None else None
        self._wo_a_block = fp8_block_size(wo_a_scheme) if wo_a_scheme is not None and wo_a_scheme.kind is QuantKind.FP8_BLOCK else None
        rows, cols = args.o_groups * args.o_lora_rank, args.n_heads * args.head_dim // args.o_groups
        if self._wo_a_block is not None:
            self.wo_a = torch.empty(rows, cols, dtype=torch.float8_e4m3fn)
            self.wo_a_scale = torch.empty(rows // self._wo_a_block, cols // self._wo_a_block, dtype=torch.float8_e8m0fnu)
        else:
            self.wo_a = torch.empty(rows, cols, dtype=torch.bfloat16)
        self.wo_b = LinearReplicated(args.o_groups * args.o_lora_rank, args.dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wo_b")
        self.attn_sink = torch.empty(args.n_heads, dtype=torch.float32)

        self.compressor = Compressor(args, self.layer_id, role.ratio, quant_config=quant_config, prefix=f"{prefix}.compressor") if role.mode is Mode.FULL else None
        self.indexer = Indexer(args, role, quant_config=quant_config, prefix=f"{prefix}.indexer") if role.has_indexer else None
        self._freqs_params = args.freqs_params(self.layer_id)
        self._freqs: torch.Tensor | None = None

    # addressing lives on the live backend, pool buffers on the live pool (both read per access)
    @property
    def attn(self):
        return get_global_ctx().attn_backend

    @property
    def freqs_key(self) -> tuple:
        """The RoPE table this layer rotates with (window layers: base theta; compressing layers: the
        compressed theta under YaRN). Layers with the same key share one table."""
        return self._freqs_params

    def bind(self, device: torch.device, tables: dict | None = None) -> None:
        """Attach the RoPE table; ``tables`` (key -> table) lets the model share one table per key
        instead of every layer holding its own ``[max_seq_len, rd // 2]`` copy."""
        if tables is not None and self.freqs_key in tables:
            self._freqs = tables[self.freqs_key]
            return
        rd, original_seq_len, theta, factor, beta_fast, beta_slow = self._freqs_params
        self._freqs = get_freqs_cis(rd, self.args.max_seq_len, original_seq_len, theta, factor, beta_fast, beta_slow, device)
        if tables is not None:
            tables[self.freqs_key] = self._freqs

    @property
    def freqs(self) -> torch.Tensor:
        assert self._freqs is not None, "attention layer is not bound (call bind first)"
        return self._freqs

    # ----- shared pieces -------------------------------------------------------------------
    def _q(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qr = self.q_norm.forward(self.wq_a.forward(x))
        q = self.wq_b.forward(qr).unflatten(-1, (self.n_heads, self.head_dim))
        return qr, q

    def _kv(self, x: torch.Tensor) -> torch.Tensor:
        return self.kv_norm.forward(self.wkv.forward(x))

    def _wo(self, o: torch.Tensor) -> torch.Tensor:
        """``wo_b(wo_a(o))``: up to ``GEMV_MAX_M`` rows the grouped W8A16 GEMV over the fp8 payload, else the
        reference's grouped einsum over the bf16 (or dequantized) ``wo_a``."""
        t = o.shape[0]
        o = o.reshape(t, self.n_groups, -1)
        if self._wo_a_block is not None and t <= GEMV_MAX_M:
            return self.wo_b.forward(grouped_w8a16_gemv(o, self.wo_a, self.wo_a_scale, block=self._wo_a_block))
        wo_a = self.wo_a if self._wo_a_block is None else dequant_block_fp8(self.wo_a, self.wo_a_scale, block=self._wo_a_block)
        return self.wo_b.forward(torch.einsum("tgd,grd->tgr", o, wo_a.view(self.n_groups, self.o_lora_rank, -1)).flatten(1))

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        key = f"{prefix}.wo_a" if prefix else "wo_a"
        scale_key = f"{key}_scale"
        if self._wo_a_block is None and scale_key in state_dict:
            # a bf16 layer (no fp8 scheme) fed a quantized checkpoint dequantizes wo_a on the way in
            from .weight import dequant_fp8_block

            state_dict[key] = dequant_fp8_block(state_dict[key], state_dict.pop(scale_key))
        elif self._wo_a_block is not None and scale_key not in state_dict:
            # an FTW conversion made before wo_a kept its fp8 payload carries a dequantized bf16
            # tensor and no scale: say so instead of failing on the missing key
            raise RuntimeError(
                f"{key}: the checkpoint has no {scale_key} for the fp8 wo_a layout; an FTW checkpoint "
                "converted by an earlier build stores wo_a dequantized -- reconvert it with `ft checkpoint`"
            )
        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)

    # ----- prefill -----------------------------------------------------------------------------
    def publish_prefill(self, x: torch.Tensor, segments: List[PrefillSegment]) -> None:
        """Full mode: compress this layer's input for every segment into the source's main KV and
        index keys. Runs on EVERY new token (under bounded replay the decoder source publishes for
        the whole prompt even though its attention runs on the prompt's last window only)."""
        assert self.compressor is not None and self.indexer is not None
        attn, ratio, src = self.attn, self.role.ratio, self.layer_id
        for seg in segments:
            slots = attn.window_slots_of(seg.table_idx, seg.start_pos, seg.end)
            tail = None
            if self.compressor.needs_tail_carry(seg.start_pos):
                tail = int(attn.window_slots_of(seg.table_idx, seg.start_pos - 1, seg.start_pos).item())
            latent, starts = self.compressor.forward_prefill(x[seg.offset : seg.offset + seg.n], seg.start_pos, slots, tail)
            if not latent.shape[0]:
                continue
            rows = attn.compressed_rows_of(seg.table_idx, starts, ratio)
            freqs = self.freqs.index_select(0, starts)
            attn.store_index(self.indexer.index_keys(latent, freqs), src, rows)
            apply_rotary_emb(latent[..., -self.rope_dim :], freqs)
            attn.store_main(latent, src, rows)

    def forward_prefill(self, x: torch.Tensor, segments: List[PrefillSegment], positions: torch.Tensor) -> torch.Tensor:
        """Ragged prefill over ``x [T, dim]`` tiled by ``segments`` (``positions [T]`` absolute)."""
        attn, role = self.attn, self.role
        freqs = self.freqs.index_select(0, positions)
        qr, q = self._q(x)
        apply_rotary_emb(q[..., -self.rope_dim :], freqs)
        kv = self._kv(x)
        apply_rotary_emb(kv[..., -self.rope_dim :], freqs)

        sel = attn.selection
        win_parts, cmp_parts, pools = [], [], []
        for seg in segments:
            lo, hi = seg.offset, seg.offset + seg.n
            attn.store_window(kv[lo:hi], self.layer_id, attn.layer_window_slots_of(self.layer_id, seg.table_idx, seg.start_pos, seg.end))
            win_parts.append(attn.window_topk_prefill(seg, self.layer_id))
            if not role.compresses:
                continue
            if role.has_indexer:
                locs = attn.locs_prefill(seg.table_idx, seg.end // role.ratio * role.ratio)
                cand = sel.candidates[:, lo:hi] if role.uses_candidates else None
                rows, pool = self.indexer.select_prefill(
                    x[lo:hi], qr[lo:hi], freqs[lo:hi], start_pos=seg.start_pos, ratio=role.ratio, locs=locs, candidates=cand,
                )
                cmp_parts.append(rows)
                if pool is not None:
                    pools.append(pool)
            else:
                assert sel.source == role.kv_source, "Reuse layer without a published selection for its source"
                cmp_parts.append(sel.topk_rows[:, lo:hi])
        if role.has_indexer:
            sel.source, sel.topk_rows = role.kv_source, torch.cat(cmp_parts, dim=1)
            if pools:
                sel.candidates = torch.cat(pools, dim=1)
        win = torch.cat(win_parts, dim=1)
        topk = torch.cat([win, torch.cat(cmp_parts, dim=1)], dim=-1) if role.compresses else win

        o = attn.attend(q.unsqueeze(0), self.layer_id, topk.to(torch.int32), self.window, self.attn_sink, self.softmax_scale)[0]
        apply_rotary_emb(o[..., -self.rope_dim :], freqs, inverse=True)
        return self._wo(o)

    # ----- decode ------------------------------------------------------------------------------
    def forward_decode(self, x: torch.Tensor, pos: torch.Tensor, rows: torch.Tensor, dctx: "DecodeStepContext", cmp_stage_cap: int) -> torch.Tensor:
        """Batched single-token attention. ``pos [B]`` device positions, ``rows [B]`` local rows into
        the decode snapshot, ``dctx`` the step's layer-invariant context (ring slots, window
        candidates, RoPE rows), ``cmp_stage_cap`` the static compressed staging bound (max position
        any row reaches; the capture width under a graph)."""
        attn, role = self.attn, self.role
        B = x.shape[0]
        freqs_t = dctx.freqs[self.freqs_key]
        qr, q = self._q(x)
        apply_rotary_emb_decode(q[..., -self.rope_dim :].unsqueeze(1), freqs_t)
        kv = self._kv(x)
        apply_rotary_emb_decode(kv[..., -self.rope_dim :].unsqueeze(1), freqs_t)
        # the shared ring context feeds the compressor's carry ring on every layer; a request-private
        # window layer stores and reads its own KV through its ring context instead
        window_slots, prev_window_slots = dctx.window_slots, dctx.prev_window_slots
        private = dctx.private is not None and attn.pool.is_private_window(self.layer_id)
        tier = dctx.private if private else dctx.shared
        attn.store_window(kv, self.layer_id, tier.slots if private else window_slots)

        cmp_counts = None
        if role.compresses:
            ratio, src = role.ratio, role.kv_source
            n_stage = (cmp_stage_cap + 1) // ratio
            sel = attn.selection
            if role.mode is Mode.FULL:
                latent, completed = self.compressor.forward_decode(x, pos, prev_window_slots, window_slots)
                dst = attn.decode_store_rows(rows, pos, ratio, src, completed)
                freqs_g = self.freqs.index_select(0, (pos + 1 - ratio).clamp_min(0))
                k = self.indexer.k_norm.forward(self.indexer.wk.forward(latent))
                apply_rotary_emb_decode(k[..., -self.rope_dim :].unsqueeze(1), freqs_g)
                attn.store_index(k, src, dst)
                apply_rotary_emb_decode(latent[..., -self.rope_dim :].unsqueeze(1), freqs_g)
                attn.store_main(latent, src, dst)
            if role.has_indexer:
                cand = sel.candidates if role.uses_candidates else None
                cmp_rows, counts, pool = self.indexer.select_decode(x, qr, freqs_t, pos=pos, ratio=ratio, locs=attn.snapshot(), T=n_stage, candidates=cand)
                sel.source, sel.topk_rows, sel.cmp_counts = src, cmp_rows, counts
                if pool is not None:
                    sel.candidates = pool
                # the step's [window | compressed] candidate buffer of this tier takes the new rows once;
                # the Reuse layers that follow read them in place (no per-layer cat / cast)
                tier.topk[..., self.window :].copy_(cmp_rows)
            else:
                assert sel.source == src, "Reuse layer without a published selection for its source"
                counts = sel.cmp_counts
            topk = tier.topk
            cmp_counts = counts
        else:
            topk = tier.window_topk

        o = attn.attend(q.view(B, 1, self.n_heads, self.head_dim), self.layer_id, topk, self.window, self.attn_sink, self.softmax_scale, cmp_counts=cmp_counts)
        apply_rotary_emb_decode(o[..., -self.rope_dim :], freqs_t, inverse=True)
        return self._wo(o.view(B, self.n_heads, self.head_dim))


@dataclass
class TierCandidates:
    """One window tier's per-step decode candidates: the request's store slot and the int32
    ``[B, 1, win]`` window candidates, plus the ``[B, 1, win + topk]`` ``[window | compressed]``
    buffer whose compressed half the index layers of that tier fill as they publish."""

    slots: torch.Tensor
    window_topk: torch.Tensor
    topk: torch.Tensor


@dataclass
class DecodeStepContext:
    """Everything a decode step resolves once for all layers (a capture records it once): the shared
    pool's ring slots (the compressor's carry ring keys on them), the shared / private tier
    candidates, and each RoPE table gathered at the step's positions."""

    window_slots: torch.Tensor
    prev_window_slots: torch.Tensor
    shared: TierCandidates
    private: TierCandidates | None
    freqs: dict

    @classmethod
    def build(cls, layers, md, pos: torch.Tensor, rows: torch.Tensor, window: int, topk: int) -> "DecodeStepContext":
        B = pos.shape[0]

        def tier(slots, window_topk):
            win32 = window_topk.to(torch.int32)
            buf = torch.empty((B, 1, window + topk), dtype=torch.int32, device=pos.device)
            buf[..., :window].copy_(window_topk)
            return TierCandidates(slots, win32, buf)

        window_slots, prev_window_slots, window_topk = md.window_ctx(pos, rows)
        private = None
        if get_global_ctx().kv_cache.private_window_layer_ids:
            private = tier(*md.private_window_ctx(pos, rows))
        tables: dict = {}
        for layer in layers:
            key = layer.freqs_key
            if key not in tables:
                tables[key] = layer.freqs.index_select(0, pos)
        return cls(window_slots, prev_window_slots, tier(window_slots, window_topk), private, tables)


__all__ = ["DSV41Attention", "DecodeStepContext", "TierCandidates"]
