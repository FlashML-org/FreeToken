"""The aux tap and the positions it was computed at must travel together.

The drafter derives context KV from the target's hidden state and writes it at those
tokens' window slots. Which tokens is not a free choice: the tap belongs to a specific
forward, and the KV must land where THAT forward's tokens live.

The catch-up runs before the target's forward for the current step, so the newest tap is
the PREVIOUS forward's. Addressing it with the current batch's positions -- the block
about to be drafted -- stores hidden states derived from one set of tokens into the slots
of another. Nothing raises. Attention still runs, the shapes all match, and the only
symptom is a drafter whose proposals are rejected, which reads as a weak drafter rather
than a wiring fault. That is what these tests exist to prevent.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest

MODEL = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python" / "freetoken" / "models" / "deepseek_v4" / "model.py"
)
SRC = MODEL.read_text()


def _fn(name: str) -> str:
    start = SRC.index(f"    def {name}(")
    nxt = SRC.index("\n    def ", start + 10)
    return SRC[start:nxt]


class TestTheTapCarriesItsAddressing:
    def test_every_site_that_stores_the_tap_stores_positions(self):
        # The two forwards (ragged prefill, batched decode) each write the tap. A third
        # that writes only the tensor would leave last_aux_addressing stale, and the
        # drafter would silently address the wrong step.
        sites = SRC.count("self._last_aux_hidden = torch.cat(aux, dim=-1)")
        writes = SRC.count("self._last_aux_positions = ")
        assert writes >= sites, (
            f"{sites} sites store the tap but only {writes} store its positions"
        )

    def test_addressing_is_exposed_as_one_unit(self):
        from freetoken.models.deepseek_v4.model import Transformer

        assert hasattr(Transformer, "last_aux_addressing"), (
            "positions and rows must be readable together with the tap"
        )

    def test_addressing_returns_none_until_a_forward_has_run(self):
        from freetoken.models.deepseek_v4.model import Transformer

        t = Transformer.__new__(Transformer)
        t._last_aux_hidden = None
        t._last_aux_positions = None
        t._last_aux_rows = None
        assert t.last_aux_addressing() is None


class TestTheCatchUpUsesTheTapsOwnPositions:
    def test_it_does_not_read_the_batch_metadata(self):
        body = _fn("catch_up_draft_context")
        for forbidden in ("batch.attn_metadata", "md.segments", "batch.positions"):
            assert forbidden not in body, (
                f"catch_up_draft_context reads {forbidden}: the tap is the PREVIOUS "
                "forward's, so the current batch's positions address the wrong tokens"
            )

    def test_it_takes_addressing_from_the_transformer(self):
        assert "last_aux_addressing()" in _fn("catch_up_draft_context")

    def test_a_row_count_mismatch_is_an_error_not_a_silent_skip(self):
        # Truncating to the shorter of the two would write SOME rows at wrong positions,
        # which is the failure mode this whole file is about.
        body = _fn("catch_up_draft_context")
        assert "raise RuntimeError" in body


class TestScatteredSlotLookup:
    """A flat batch spans requests, so slots come from a gather, not a slice."""

    def test_the_backend_exposes_the_gather_form(self):
        from freetoken.attention.dsv4_sparse import DSV4SparseAttnBackend as M

        assert hasattr(M, "window_slots_at")
        params = list(inspect.signature(M.window_slots_at).parameters)
        assert params[1:] == ["rows", "positions"]

    def test_it_gathers_per_token_rather_than_slicing_one_request(self):
        torch = pytest.importorskip("torch")

        from freetoken.attention.dsv4_sparse import DSV4SparseAttnBackend as M

        class _Pool:
            def __init__(self):
                # full_loc_map[row, position] -> a distinct full slot per pair.
                self.full_loc_map = torch.arange(4 * 8).reshape(4, 8)

            def translate_full_to_window(self, x):
                return x  # identity: the gather itself is under test

        class _B(M):
            # `pool` is a read-only property on the real backend, so shadow it rather
            # than constructing one (which would need a live KV pool and a CUDA device).
            pool = property(lambda self: self._pool)

            def __init__(self):
                self._pool = _Pool()

        rows = torch.tensor([0, 2, 2, 3])
        pos = torch.tensor([1, 0, 5, 7])
        got = _B().window_slots_at(rows, pos)
        assert got.tolist() == [1, 16, 21, 31], (
            "each token's slot must come from its OWN (row, position) pair"
        )
