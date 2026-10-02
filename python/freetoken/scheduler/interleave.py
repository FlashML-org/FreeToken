"""Decode-interleave scheduling policy.

Background (measured on production, 2026-09-15)
-----------------------------------------------
``Scheduler._schedule_next_batch`` picks::

    batch = (prefill_manager.schedule_next_batch(budget)
             or decode_manager.schedule_next_batch())

Prefill wins unconditionally.  A long prompt is chunked at the window pool's
budget (384 tokens in production), so a 300k-token context is 797 consecutive
prefill steps -- and while they run, *decoding requests already in flight are
never scheduled*.  Measured on the live service:

    four long cold prefills: 680/797/506/250 chunks, 267/317/190/93 s
    prefill-burst share of a 15-minute window: 70%

so an in-flight request waits up to ~5 minutes for its next token.  The
scheduler already carries the marker for this::

    # TODO: support other policies: e.g. DECODE first

Reversing the order outright is wrong -- it starves prefill, and a request that
never prefills never starts.  What is needed is *interleaving*: let prefill keep
priority, but force a decode step every N prefill steps so in-flight requests
make progress.  The cost is small by construction: measured chunk cost is
0.388 s, and one decode step for a handful of requests is a fraction of that,
so every 8th step costs ~1% of prefill wall time while bounding the stall at
8 chunks (~3 s) instead of the whole burst (~300 s).

This module holds the *policy* only -- a counter and a decision -- so it can be
unit-tested without a Scheduler, an engine, or a GPU.
"""

from __future__ import annotations


class DecodeInterleavePolicy:
    """Decide whether to force a decode step between prefill steps.

    ``every`` is the number of consecutive prefill steps allowed before a decode
    step is forced.  ``None`` or <= 0 disables the policy, reproducing the
    historical prefill-first behaviour bit-for-bit.

    The counter only advances on prefill steps that were actually scheduled, so
    a scheduler with nothing to prefill never accumulates credit, and the first
    decode of a burst is not delayed.
    """

    def __init__(self, every: int | None = None) -> None:
        self.every = int(every) if every else None
        self._prefill_streak = 0

    @property
    def enabled(self) -> bool:
        return self.every is not None and self.every > 0

    def note_prefill(self) -> None:
        """Record that a prefill step was scheduled in this slot."""
        self._prefill_streak += 1

    def note_decode(self) -> None:
        """Record that a decode step was scheduled in this slot (streak resets)."""
        self._prefill_streak = 0

    def wants_decode(self) -> bool:
        """True when the streak has reached the threshold and a decode is due."""
        if not self.enabled:
            return False
        return self._prefill_streak >= self.every

    def reset(self) -> None:
        self._prefill_streak = 0
