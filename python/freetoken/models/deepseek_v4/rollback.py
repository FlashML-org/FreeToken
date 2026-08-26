"""Undo a rejected speculative block's effect on DSV4's stateful compressor carry.

Rejecting a speculated token is easy for a plain paged KV cache: stop advancing the
position and the slots are reused. DSV4 is not plain. Its compressors and Lightning
Indexer carry a rolling state per request, and a decode step READS that carry from the
previous token's window page, advances it, and WRITES it back to the current token's
page. Speculating N tokens therefore advances the carry N times.

When the rejected tokens fall in a LATER window page than the last accepted token, no
undo is needed: the next step reads from the accepted token's page, which was never
touched, and the abandoned pages are freed with their ring blocks.

The hazard is a rejection INSIDE the page the last accepted token lives in. There,
``prev`` and ``cur`` are the same ring block, so the speculative advance overwrote the
state the next real step must read. Nothing else in the engine can rebuild it: the
carry is a reduction over the tokens seen so far, not a function of the KV alone.

So snapshot exactly those blocks before speculating, and restore on rejection. The
snapshot is bounded and small -- one ring block per (layer, tier) per request, which
for DSV4-Flash is about 5.5 MiB per row across the whole model -- and it is taken only
for the pages a rollback could not otherwise reconstruct.
"""

from __future__ import annotations

import torch

# The carry rollback and the loop that drives it are imported together by the engine;
# keeping needs_rollback here (beside the snapshot it guards) means the decision and the
# mechanism cannot drift apart.


class CarrySnapshot:
    """The compressor/indexer carry blocks a speculative block is about to overwrite.

    Take one before the draft advances the carry; ``restore()`` puts the engine back to
    the state the last accepted token left behind. Restoring is idempotent, so it is
    safe to call on the accept path too rather than branching at the call site.
    """

    __slots__ = ("_backend", "_saved", "_window_slots")

    def __init__(self, backend, layers, window_slots: torch.Tensor):
        """``layers`` are the target's attention modules; ``window_slots`` is ``[B]``,
        each row's CURRENT window slot -- the page a speculative step would advance in
        place. Layers with no compressor contribute nothing."""
        self._backend = backend
        self._saved: list[tuple[int, str, int, torch.Tensor]] = []
        for attn in layers:
            for tier, module in (("attn", attn.compressor), ("idx", attn.indexer)):
                comp = _compressor_of(module, tier)
                if comp is None:
                    continue
                block = backend.read_carry_blocks(
                    attn.layer_id, tier, window_slots, comp.ring_size
                )
                # clone: read_carry_blocks may view the live ring, which is exactly the
                # memory the speculative step is about to overwrite.
                self._saved.append((attn.layer_id, tier, comp.ring_size, block.clone()))
        self._window_slots = window_slots

    def restore(self) -> None:
        """Put every saved carry block back. Call this when a block is rejected."""
        for layer_id, tier, ring_size, block in self._saved:
            self._backend.write_carry_blocks(
                layer_id, tier, self._window_slots, ring_size, block
            )

    def __len__(self) -> int:
        return len(self._saved)

    @property
    def bytes(self) -> int:
        return sum(b.numel() * b.element_size() for _, _, _, b in self._saved)


def _compressor_of(module, tier: str):
    """The Compressor that owns ``tier``'s ring, or None if this layer has none.

    A ratio-0 layer has neither; a ratio-128 layer has a compressor but no indexer; the
    dSpark draft layers have neither, which is why they are not passed in at all.
    """
    if module is None:
        return None
    return module if tier == "attn" else getattr(module, "compressor", None)


def needs_rollback(accepted_upto: int, block_positions: torch.Tensor, page_size: int) -> bool:
    """Would a rejection at ``accepted_upto`` leave a stale carry behind?

    Only when a rejected position shares a window page with the last accepted one. If
    the block crossed into a new page at or before the rejection, the abandoned pages
    carry their own ring blocks away with them and the surviving page was never touched.
    """
    if accepted_upto >= block_positions.numel():
        return False  # nothing rejected
    last_kept = block_positions[accepted_upto - 1] if accepted_upto else None
    first_dropped = block_positions[accepted_upto]
    if last_kept is None:
        return False  # the whole block was rejected; the carry never advanced past it
    return int(last_kept) // page_size == int(first_dropped) // page_size


__all__ = ["CarrySnapshot", "needs_rollback"]
