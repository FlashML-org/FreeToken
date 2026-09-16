"""dSpark acceptance and adaptive width.

These two functions decide what the model EMITS. A wrong acceptance rule does not
crash or slow anything down -- it silently changes the output distribution, which is
the one failure mode speculative decoding must never have. So the rule is pinned here
rather than left to the loop that calls it.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.deepseek_v4.dspark import accepted_prefix, draft_width


def _t(*xs: int) -> torch.Tensor:
    return torch.tensor(xs, dtype=torch.long)


class TestAcceptedPrefix:
    def test_a_fully_matching_block_is_wholly_accepted(self):
        n, bonus = accepted_prefix(_t(5, 6, 7), _t(5, 6, 7, 8))
        assert n == 3
        assert bonus == 8, "verification's extra position supplies the next token free"

    def test_acceptance_stops_at_the_first_disagreement(self):
        n, bonus = accepted_prefix(_t(5, 6, 7), _t(5, 99, 7))
        assert n == 1
        assert bonus == 99, "the target's own token replaces the rejected one"

    def test_a_later_match_after_a_rejection_does_not_count(self):
        # Position 2 agrees, but it was drafted from a token the target rejected, so
        # keeping it would emit a sequence the target never would have produced.
        n, _ = accepted_prefix(_t(1, 2, 3), _t(1, 99, 3))
        assert n == 1

    def test_a_wholly_rejected_block_still_advances_by_one(self):
        n, bonus = accepted_prefix(_t(4, 5), _t(9, 9))
        assert (n, bonus) == (0, 9), "a rejected block must not stall the request"

    def test_no_bonus_when_verification_covered_no_extra_position(self):
        n, bonus = accepted_prefix(_t(1, 2), _t(1, 2))
        assert (n, bonus) == (2, -1)


class TestDraftWidth:
    def test_full_width_when_every_position_is_confident(self):
        assert draft_width(torch.tensor([0.9, 0.8, 0.7]), 0.5, 5) == 3

    def test_cuts_at_the_first_doubtful_position(self):
        # Verification costs a full target pass over whatever width it covers, so a
        # tail the drafter itself doubts is work bought and thrown away.
        assert draft_width(torch.tensor([0.9, 0.8, 0.2, 0.9]), 0.5, 5) == 2

    def test_always_verifies_at_least_one_position(self):
        assert draft_width(torch.tensor([0.1, 0.1]), 0.5, 5) == 1

    def test_never_exceeds_the_block_size(self):
        assert draft_width(torch.ones(9), 0.5, 5) == 5

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_width_is_within_bounds_for_any_threshold(self, threshold):
        w = draft_width(torch.rand(5), threshold, 5)
        assert 1 <= w <= 5
