"""Lightning-Indexer addressing and block selection, mixed into the sparse-attention backend.

The indexer scores every compressed block a query may attend and returns the ``index_topk``
best BLOCK indices; the attention layer then maps those to global compressed rows. Both the
gather (which paged row holds block ``b``) and the selection (which blocks are causally live,
and what a pick past the live count means) are addressing, so they live here -- the module
keeps ``wq_b`` / ``weights_proj`` and hands the projections in, mirroring sglang's
``C4IndexerBackendMixin.forward_c4_indexer(..., c4_indexer=self, ...)``.

The ``-1`` a selection emits for a pick past the live count is a gather-only sentinel: the
sparse-attention kernel masks it, and it never reaches a store.
"""

from __future__ import annotations

import torch


class IndexerBackendMixin:
    def indexer_keys(
        self, ti: int, n_blocks: int, ratio: int, layer_id: int, bsz: int
    ) -> torch.Tensor:
        """Blocks ``[0, n_blocks)`` of this request's indexer keys as a dense
        ``[bsz, n_blocks, index_head_dim]`` slab, in ascending block order.

        Byte-exact ``index_select`` of the bf16 pool: block ``b`` lives at the arithmetic row
        ``full_loc(b * ratio) // ratio``, read off the request's LIVE full locs (prefill/extend
        only -- decode reads the snapshot inside the fused kernel instead).
        """
        block_starts = torch.arange(0, n_blocks * ratio, ratio, device=self.device)
        rows = self.compress_rows_of(ti, block_starts, ratio)
        return self.compress_pool(layer_id, "idx").index_select(0, rows).unsqueeze(0).expand(
            bsz, -1, -1
        )

    def indexer_prefill_logits(
        self, q: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Head-reduced scores ``[bsz, seqlen, n_blocks]`` over a dense key slab."""
        from freetoken.kernel.triton.dsv4.indexer import indexer_logits

        return indexer_logits(q, keys, weights)

    def indexer_decode_scores(
        self, q: torch.Tensor, weights: torch.Tensor, valid: torch.Tensor, n_stage: int,
        ratio: int, layer_id: int, rows: torch.Tensor,
    ) -> torch.Tensor:
        """Head-reduced scores ``[B, n_stage]`` for a decode step, gathering each block's key off
        the decode SNAPSHOT and bounding the work by the live block count read from device
        memory (so a captured graph tracks the position, not the staged width)."""
        from freetoken.kernel.triton.dsv4.indexer import indexer_decode_logits

        return indexer_decode_logits(
            q, weights, self.compress_pool(layer_id, "idx"), self.snapshot(),
            rows, valid, n_stage, ratio,
        )

    def indexer_select_prefill(
        self, scores: torch.Tensor, *, start_pos: int, seqlen: int, ratio: int, topk: int,
        offset: int,
    ) -> torch.Tensor:
        """Causal top-k over compressed blocks for a prefill/extend range.

        Query ``s`` sits at absolute position ``p = start_pos + s`` and may attend blocks
        ``[0, (p + 1) // ratio)``. Blocks past that score ``-inf`` so they lose the top-k; any
        that still get picked (a short live count) come back as ``-1``.
        """
        device = scores.device
        n_blocks = scores.shape[-1]
        live = ((start_pos + torch.arange(1, seqlen + 1, device=device)) // ratio).unsqueeze(1)
        blk = torch.arange(n_blocks, device=device).repeat(seqlen, 1)
        scores = scores + torch.where(blk >= live, float("-inf"), 0)
        picks = scores.topk(min(topk, n_blocks), dim=-1)[1]
        return torch.where(picks >= live, -1, picks + offset)

    def indexer_select_decode(
        self, scores: torch.Tensor, *, valid: torch.Tensor, topk: int, offset: int
    ) -> torch.Tensor:
        """Top-k over a decode step's staged scores. Columns past each row's live count already
        score ``-inf`` (the scoring kernel writes them), so they sort last -- which is what lets
        the attention layer bound the sparse kernel with a single per-row count."""
        n_stage = scores.shape[-1]
        picks = scores.topk(min(topk, n_stage), dim=-1)[1]
        return torch.where(picks >= valid[:, None, None], -1, picks + offset)


__all__ = ["IndexerBackendMixin"]
