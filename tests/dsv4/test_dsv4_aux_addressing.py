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
    def test_prefill_and_decode_build_one_feature_bundle(self):
        assert SRC.count("DSparkTargetFeatures(") >= 2

    def test_addressing_is_exposed_as_one_unit(self):
        from freetoken.models.deepseek_v4.model import DSparkTargetFeatures

        assert set(DSparkTargetFeatures.__dataclass_fields__) == {
            "hidden", "positions", "table_rows"
        }

    def test_addressing_returns_none_until_a_forward_has_run(self):
        from freetoken.models.deepseek_v4.model import Transformer

        t = Transformer.__new__(Transformer)
        t._target_features = None
        assert t.target_features() is None

    def test_tap_reshape_reads_the_dimension_from_model_args(self):
        # Transformer has no direct ``dim`` field. This typo survives source-only
        # wiring tests and crashes only when the first real aux tap runs.
        assert "self.dim * len(aux)" not in SRC
        assert SRC.count("self.args.dim * len(aux)") == 3

    def test_cuda_graph_features_are_owned_per_captured_batch_size(self):
        graph_src = (
            pathlib.Path(__file__).resolve().parents[2]
            / "python" / "freetoken" / "engine" / "graph.py"
        ).read_text()
        assert "self._dspark_feature_map[bs] = features" in graph_src
        assert "self._dspark_feature_map.get(batch.padded_size)" in graph_src
        assert "self._spec_feature_map[key] = features" in graph_src
        assert "self._spec_carry_map[key]" in graph_src
        assert "shared_carry if shared_carry is not None" in graph_src


class TestTheCatchUpUsesTheTapsOwnPositions:
    def test_it_does_not_read_the_batch_metadata(self):
        body = _fn("catch_up_draft_context")
        for forbidden in ("batch.attn_metadata", "md.segments", "batch.positions"):
            assert forbidden not in body, (
                f"catch_up_draft_context reads {forbidden}: the tap is the PREVIOUS "
                "forward's, so the current batch's positions address the wrong tokens"
            )

    def test_it_takes_one_explicit_feature_bundle(self):
        body = _fn("catch_up_draft_context")
        for field in ("features.hidden", "features.positions", "features.table_rows"):
            assert field in body

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
