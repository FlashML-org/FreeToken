"""DSpark greedy prefix acceptance.

These two functions decide what the model EMITS. A wrong acceptance rule does not
crash or slow anything down -- it silently changes the output distribution, which is
the one failure mode speculative decoding must never have. So the rule is pinned here
rather than left to the loop that calls it.
"""

from __future__ import annotations

import torch

from freetoken.models.deepseek_v4.dspark import accepted_prefix


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
