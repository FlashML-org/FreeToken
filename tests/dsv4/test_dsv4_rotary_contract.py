"""Which rotary variant the draft-context write uses.

There are two, and they are not interchangeable:

* ``apply_rotary_emb`` takes ONE row of frequencies and broadcasts it across the
  sequence. It reshapes freqs_cis to ``[1, T, 1, D]``.
* ``apply_rotary_emb_decode`` takes one row of frequencies PER ROW of x.

``store_context_kv`` gathers frequencies by absolute position -- ``freqs_cis.index_select
(0, positions)`` -- which is T rows, so it needs the decode variant. Handing those to
``apply_rotary_emb`` fails on the view, which is the lucky outcome: at T == 1 the two
shapes coincide, so a single-token catch-up would pass and every longer one would crash.
Worse, had the view happened to be valid, every context token would have been rotated to
the phase of position zero, and the drafter would attend against phases the target never
used -- degrading acceptance with nothing raised.
"""

from __future__ import annotations

import pathlib

import pytest

SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python" / "freetoken" / "models" / "deepseek_v4" / "dspark.py"
).read_text()


class TestTheDecodeVariantIsUsed:
    def test_store_context_kv_uses_the_per_row_rotary(self):
        body = SRC[SRC.index("def store_context_kv") : SRC.index("def catch_up_context")]
        assert "apply_rotary_emb_decode(" in body, (
            "store_context_kv gathers per-position frequencies, so it must use the "
            "per-row rotary"
        )

    def test_it_does_not_use_the_broadcast_rotary(self):
        body = SRC[SRC.index("def store_context_kv") : SRC.index("def catch_up_context")]
        calls = [
            ln for ln in body.splitlines()
            if "apply_rotary_emb(" in ln and not ln.strip().startswith("#")
        ]
        assert not calls, (
            f"the broadcast rotary cannot take T rows of frequencies: {calls}"
        )

    def test_a_head_dim_is_added_before_the_call(self):
        # The decode rotary broadcasts each row's frequencies over an inner head dim, so
        # a [T, rd] slice has to become [T, 1, rd] first.
        body = SRC[SRC.index("def store_context_kv") : SRC.index("def catch_up_context")]
        assert "unsqueeze(1)" in body, (
            "the decode rotary needs a head dim to broadcast each row's freqs over"
        )


class TestTheContractsDiffer:
    """Pin the difference itself, so neither helper can quietly grow the other's shape."""

    def test_the_broadcast_variant_rejects_per_row_frequencies(self):
        torch = pytest.importorskip("torch")

        from freetoken.models.deepseek_v4.ops import apply_rotary_emb

        T, rd = 5, 64
        x = torch.zeros(T, 1, rd)
        per_row = torch.ones(T, rd // 2, dtype=torch.complex64)
        with pytest.raises(RuntimeError):
            apply_rotary_emb(x, per_row)

    def test_a_single_token_hides_the_difference(self):
        # Why this was not caught earlier: at T == 1 the two shapes coincide, so the
        # wrong variant works exactly until a block is wider than one token.
        torch = pytest.importorskip("torch")

        from freetoken.models.deepseek_v4.ops import apply_rotary_emb

        x = torch.zeros(1, 1, 64)
        apply_rotary_emb(x, torch.ones(1, 32, dtype=torch.complex64))  # no raise
