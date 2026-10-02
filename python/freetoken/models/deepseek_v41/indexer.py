"""DSV41 Lightning Indexer (reference ``Indexer``): scores the compressed positions and keeps the
``index_topk`` best per query.

A Full-mode layer owns the index keys: ``k = k_norm(wk(latent))`` from its compressor's unrotated
latent, RoPE'd at the group position, fp4-quantized (ue8m0 per 32) into the kv source's index pool.
Every Full / Reindex layer scores its own fp4-baked queries ``wq_b(qr)`` against those keys,
``sum_h relu(q_h . k) * w_h`` with ``w = weights_proj(x) * softmax_scale * n_heads**-0.5``.

The Hierarchical Sparse Indexer (sec. 2.3.2): the candidate-source layer (the decoder's first Full
layer) additionally keeps the ``candidate_topk_blocks`` best ``candidate_block_size``-position blocks
as a candidate pool; the decoder's Reindex layers score only that pool. Logits are produced in query
blocks bounded by ``LOGITS_BUDGET_BYTES`` so a long-context prefill never materializes ``[T, 1M]``.
"""

from __future__ import annotations

import torch
from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.fp8_linear import fp4_act_quant_inplace
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm

from .args import DeepseekV41Args, LayerRole, Mode
from .rope import apply_rotary_emb, apply_rotary_emb_decode

LOGITS_BUDGET_BYTES = 512 << 20


class Indexer(BaseOP):
    def __init__(self, args: DeepseekV41Args, role: LayerRole, *, quant_config=None, prefix: str = ""):
        self.role = role
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_dim = args.rope_head_dim
        self.topk = args.index_topk
        self.weights_scale = args.index_softmax_scale * args.index_n_heads**-0.5
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.wq_b = LinearReplicated(args.q_lora_rank, self.n_heads * self.head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wq_b")
        self.weights_proj = LinearReplicated(args.dim, self.n_heads, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.weights_proj")
        if role.mode is Mode.FULL:
            self.wk = LinearReplicated(args.head_dim, self.head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wk")
            self.k_norm = RMSNorm(self.head_dim, args.norm_eps)

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    # ----- keys (Full mode) ----------------------------------------------------------------
    def index_keys(self, latent: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """``[G, index_head_dim]`` keys from the unrotated latents, rotated at their group positions. The
        fp4 quantization is the pool's packing."""
        k = self.k_norm.forward(self.wk.forward(latent))
        apply_rotary_emb(k[..., -self.rope_dim :], freqs)
        return k

    # ----- queries ---------------------------------------------------------------------------
    def _queries(self, x: torch.Tensor, qr: torch.Tensor, rotate) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.wq_b.forward(qr).unflatten(-1, (self.n_heads, self.head_dim))
        rotate(q[..., -self.rope_dim :])
        fp4_act_quant_inplace(q, 32)
        weights = self.weights_proj.forward(x) * self.weights_scale
        return q, weights

    # ----- selection -------------------------------------------------------------------------
    def select_prefill(
        self, x: torch.Tensor, qr: torch.Tensor, freqs: torch.Tensor, *, start_pos: int, ratio: int, locs: torch.Tensor,
        candidates: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Per-query picks for one request's ``n`` new tokens at ``[start_pos, start_pos + n)``.

        ``locs [1, W]`` are the request's live full locs for the positions its queries may see (the
        source's row of compressed position ``t`` is ``locs[t * ratio] // ratio``); ``candidates [1, n,
        NC]`` restricts a Reindex layer to the pool. Returns ``(rows [1, n, topk] int32 global main rows,
        -1 tail; candidates [1, n, NC] | None -- the pool this layer built as candidate source)``.
        """
        n = x.shape[0]
        T = locs.shape[1] // ratio
        if T == 0:  # a prompt shorter than one compression block: no source row exists yet
            rows = torch.full((1, n, self.topk), -1, dtype=torch.int32, device=x.device)
            pool = None
            if self.role.is_candidate_source:
                pool = torch.full((1, n, self.candidate_topk_blocks * self.candidate_block_size), -1, dtype=torch.int32, device=x.device)
            return rows, pool
        q, w = self._queries(x, qr, lambda t: apply_rotary_emb(t, freqs))
        live = ((start_pos + torch.arange(1, n + 1, device=x.device)) // ratio).to(torch.int32).view(1, n)
        width = candidates.shape[-1] if candidates is not None else T
        qb = max(1, LOGITS_BUDGET_BYTES // max(1, width * 4))
        picks, pools = [], []
        for s0 in range(0, n, qb):
            s1 = min(n, s0 + qb)
            cand = candidates[:, s0:s1] if candidates is not None else None
            logits = self.attn.indexer_logits(q[None, s0:s1], w[None, s0:s1], self.role.kv_source, locs, ratio, live[:, s0:s1], T=T, candidates=cand)
            if cand is not None:
                pos = self.attn.select_topk_in_candidates(logits, cand, self.topk)
            else:
                pos = self.attn.select_topk(logits, live[:, s0:s1], self.topk)
                if self.role.is_candidate_source:
                    pools.append(self.attn.select_candidate_blocks(logits, live[:, s0:s1], self.candidate_topk_blocks, self.candidate_block_size))
            picks.append(self.attn.positions_to_rows(pos, locs, ratio))
        rows = torch.cat(picks, dim=1)
        pool = torch.cat(pools, dim=1) if pools else None
        return rows.to(torch.int32), pool

    def select_decode(
        self, x: torch.Tensor, qr: torch.Tensor, freqs: torch.Tensor, *, pos: torch.Tensor, ratio: int, locs: torch.Tensor,
        T: int, candidates: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Batched one-token selection. ``locs [B, W]`` is the decode snapshot (graph-static width),
        ``T`` the staged compressed width, ``pos [B]``. Returns ``(rows [B, 1, topk] int32, counts [B, 1]
        int32 valid picks, candidates [B, 1, NC] | None)``."""
        B = x.shape[0]
        q, w = self._queries(x, qr, lambda t: apply_rotary_emb_decode(t.unsqueeze(1), freqs))
        live = ((pos + 1) // ratio).to(torch.int32).view(B, 1)
        logits = self.attn.indexer_logits(q.view(B, 1, self.n_heads, self.head_dim), w.view(B, 1, -1), self.role.kv_source, locs, ratio, live, T=T, candidates=candidates)
        pool = None
        if candidates is not None:
            picks = self.attn.select_topk_in_candidates(logits, candidates, self.topk)
        else:
            picks = self.attn.select_topk(logits, live, self.topk)
            if self.role.is_candidate_source:
                pool = self.attn.select_candidate_blocks(logits, live, self.candidate_topk_blocks, self.candidate_block_size)
        rows = self.attn.positions_to_rows(picks, locs, ratio)
        counts = (picks >= 0).sum(dim=-1, dtype=torch.int32)  # valid picks form a prefix
        return rows.to(torch.int32), counts, pool


__all__ = ["Indexer", "LOGITS_BUDGET_BYTES"]
