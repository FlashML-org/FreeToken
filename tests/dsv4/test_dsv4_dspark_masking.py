"""The dSpark block mask, and the invariants that keep speculation honest.

Two failure modes drive these tests, and neither one crashes:

* A block mask that is secretly causal. The drafter still runs, still produces five
  tokens, and acceptance still works -- it is just five serial steps wearing a block's
  clothing, so the whole feature buys nothing and nothing reports that.
* A mask that leaks. If a query can read a slot outside its own request or beyond the
  block, the drafter proposes from context the target never had, acceptance collapses,
  and the only symptom is a poor acceptance rate that looks like a bad model.

CPU-only: the masking rule is pure index arithmetic, extracted so it can be checked
without a GPU or a KV pool.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from freetoken.models.deepseek_v4.dspark import window_cols_for_block

WIN = 128


def _visible(cols: torch.Tensor, row: int) -> set[int]:
    """Columns row ``row`` may actually read (``-1`` is masked out)."""
    return {int(c) for c in cols[row] if int(c) >= 0}


class TestCausalBaseline:
    """The unchanged path must stay exactly as it was."""

    def test_a_row_never_sees_past_its_own_position(self):
        cols = window_cols_for_block(start_pos=500, n=4, window=WIN, non_causal=False)
        w_lo = 500 - WIN + 1
        for row in range(4):
            highest = max(_visible(cols, row)) + w_lo
            assert highest == 500 + row, "causal rows end at their own position"

    def test_each_row_sees_exactly_one_window(self):
        cols = window_cols_for_block(start_pos=500, n=3, window=WIN, non_causal=False)
        for row in range(3):
            assert len(_visible(cols, row)) == WIN


class TestNonCausalBlock:
    """The whole point: every drafted query sees the whole block."""

    def test_every_row_sees_the_blocks_last_position(self):
        n = 5
        start = 500
        cols = window_cols_for_block(start_pos=start, n=n, window=WIN, non_causal=True)
        w_lo = start - WIN
        last_col = (start + n - 1) - w_lo
        for row in range(n):
            assert last_col in _visible(cols, row), (
                f"row {row} cannot see the block's final position -- the mask is "
                "still causal, so the block is five serial steps in disguise"
            )

    def test_all_rows_see_an_identical_candidate_set(self):
        cols = window_cols_for_block(start_pos=800, n=5, window=WIN, non_causal=True)
        first = _visible(cols, 0)
        for row in range(1, 5):
            assert _visible(cols, row) == first

    def test_it_differs_from_the_causal_mask(self):
        # Guards against a refactor that quietly drops the non_causal branch.
        a = window_cols_for_block(start_pos=500, n=5, window=WIN, non_causal=True)
        b = window_cols_for_block(start_pos=500, n=5, window=WIN, non_causal=False)
        assert not torch.equal(a, b)

    def test_it_never_reaches_beyond_the_block(self):
        n, start = 5, 500
        cols = window_cols_for_block(start_pos=start, n=n, window=WIN, non_causal=True)
        w_lo = start - WIN
        highest = max(max(_visible(cols, r)) for r in range(n)) + w_lo
        assert highest == start + n - 1, "a query must not read past its own block"

    def test_the_shared_context_stays_visible(self):
        n, start = 5, 500
        cols = window_cols_for_block(start_pos=start, n=n, window=WIN, non_causal=True)
        w_lo = start - WIN
        lowest = min(min(_visible(cols, r)) for r in range(n)) + w_lo
        assert lowest < start, "the block must still attend over the context before it"

    @pytest.mark.parametrize("n", [1, 2, 5, 8])
    def test_the_candidate_width_is_context_window_plus_block(self, n):
        cols = window_cols_for_block(start_pos=900, n=n, window=WIN, non_causal=True)
        assert cols.shape == (n, WIN + n)

    def test_a_single_token_still_keeps_all_context_tokens(self):
        a = window_cols_for_block(start_pos=700, n=1, window=WIN, non_causal=True)
        b = window_cols_for_block(start_pos=700, n=1, window=WIN, non_causal=False)
        assert a.shape[-1] == WIN + 1
        assert b.shape[-1] == WIN

    def test_ragged_launcher_labels_the_full_non_causal_width(self):
        # The index helper can correctly build 133 columns while the attention call
        # still labels only 128 of them as window columns. That only fails at bs > 1.
        from freetoken.models.deepseek_v4.attention import Attention

        body = inspect.getsource(Attention.forward_ragged)
        assert "max(part.shape[-1] for part in win_parts)" in body
        assert "if self.non_causal" in body


class TestMaskBounds:
    """A column index that escapes its range would read another request's KV."""

    @pytest.mark.parametrize("non_causal", [True, False])
    @pytest.mark.parametrize("start_pos,n", [(1, 5), (127, 5), (128, 5), (5000, 5)])
    def test_columns_stay_inside_the_gathered_slot_range(self, non_causal, start_pos, n):
        cols = window_cols_for_block(start_pos, n, WIN, non_causal)
        w_lo = max(
            0,
            start_pos - WIN if non_causal else start_pos - WIN + 1,
        )
        width = (start_pos + n) - w_lo  # slots the caller gathered for this segment
        assert int(cols.max()) < width, "a column past the gathered slots reads foreign KV"
        assert int(cols.min()) >= -1, "only -1 encodes 'masked'"

    @pytest.mark.parametrize("non_causal", [True, False])
    def test_no_row_is_entirely_masked(self, non_causal):
        # A fully masked row would make its query attend to nothing at all.
        cols = window_cols_for_block(300, 5, WIN, non_causal)
        for row in range(5):
            assert _visible(cols, row), f"row {row} has no visible column"
