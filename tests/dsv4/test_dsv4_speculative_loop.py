"""The dSpark speculative step, end to end over fakes.

The loop's whole job is ORDER. Every bug available here is silent:

* snapshot after drafting -> the carry to restore is already gone, rollback is a no-op,
  and the request continues from state that saw tokens it never emitted;
* choose the verify width after verifying -> the confidence head saves nothing and the
  adaptive feature is decorative;
* accept more than the agreed prefix -> the model emits text the target would not have,
  which is the one thing speculative decoding must never do.

None of those raise. So the loop is driven here against fakes that record what happened
and in which order.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.deepseek_v4.dspark import SpeculativeLoop

BLOCK = 5


class Recorder:
    """Fakes for draft/verify/snapshot that log the order they were called in."""

    def __init__(self, proposed, target, confidence=None):
        self.proposed = torch.tensor(proposed, dtype=torch.long)
        self.target = torch.tensor(target, dtype=torch.long)
        self.confidence = (
            torch.ones(len(proposed)) if confidence is None
            else torch.tensor(confidence, dtype=torch.float)
        )
        self.calls: list[str] = []
        self.verified_with: torch.Tensor | None = None
        self.restored = 0

    def snapshot(self):
        self.calls.append("snapshot")
        return self

    def restore(self):
        self.calls.append("restore")
        self.restored += 1

    def draft(self):
        self.calls.append("draft")
        return self.proposed, self.confidence

    def verify(self, tokens):
        self.calls.append("verify")
        self.verified_with = tokens.clone()
        return self.target[: tokens.numel() + 1]


def _positions(start=500, n=BLOCK):
    return torch.arange(start, start + n)


class TestOrdering:
    def test_the_carry_is_snapshotted_before_the_draft_advances_it(self):
        r = Recorder([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
        SpeculativeLoop(BLOCK).step(
            draft=r.draft, verify=r.verify, positions=_positions(), snapshot=r.snapshot
        )
        assert r.calls.index("snapshot") < r.calls.index("draft"), (
            "drafting advances the carry in place; a snapshot taken afterwards "
            "captures the state a rejection is supposed to undo"
        )

    def test_verification_happens_after_drafting(self):
        r = Recorder([1, 2], [1, 2, 3])
        SpeculativeLoop(BLOCK).step(draft=r.draft, verify=r.verify, positions=_positions(2))
        assert r.calls == ["draft", "verify"]


class TestVerifyWidth:
    def test_only_the_confident_prefix_is_verified(self):
        # Verification costs a full target pass over whatever width it covers.
        r = Recorder([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], confidence=[0.9, 0.9, 0.1, 0.9, 0.9])
        res = SpeculativeLoop(BLOCK, confidence_threshold=0.5).step(
            draft=r.draft, verify=r.verify, positions=_positions()
        )
        assert r.verified_with.numel() == 2, "the doubted tail must not be paid for"
        assert res.verified == 2
        assert res.drafted == BLOCK

    def test_full_width_when_the_drafter_is_confident(self):
        r = Recorder([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
        res = SpeculativeLoop(BLOCK).step(
            draft=r.draft, verify=r.verify, positions=_positions()
        )
        assert res.verified == BLOCK


class TestEmission:
    def test_a_fully_accepted_block_emits_every_token_plus_the_bonus(self):
        r = Recorder([1, 2, 3], [1, 2, 3, 99])
        res = SpeculativeLoop(3).step(draft=r.draft, verify=r.verify, positions=_positions(500, 3))
        assert res.tokens == [1, 2, 3, 99]
        assert res.accepted == 3

    def test_a_partial_block_emits_the_prefix_then_the_targets_own_token(self):
        r = Recorder([1, 2, 3], [1, 77, 3, 4])
        res = SpeculativeLoop(3).step(draft=r.draft, verify=r.verify, positions=_positions(500, 3))
        assert res.tokens == [1, 77], "the rejected position is replaced, not dropped"
        assert res.accepted == 1

    def test_a_fully_rejected_block_still_emits_one_token(self):
        r = Recorder([1, 2, 3], [55, 2, 3, 4])
        res = SpeculativeLoop(3).step(draft=r.draft, verify=r.verify, positions=_positions(500, 3))
        assert res.tokens == [55], "a rejected block must never stall the request"
        assert res.accepted == 0

    def test_a_match_after_a_rejection_is_not_emitted(self):
        # Position 2 agrees, but it was drafted from a token the target rejected.
        r = Recorder([1, 2, 3], [1, 77, 3, 4])
        res = SpeculativeLoop(3).step(draft=r.draft, verify=r.verify, positions=_positions(500, 3))
        assert 3 not in res.tokens

    def test_every_step_emits_at_least_one_token(self):
        # The liveness property: no combination of draft and target can produce a step
        # that advances the request by nothing.
        for target in ([9, 9, 9, 9], [1, 9, 9, 9], [1, 2, 9, 9], [1, 2, 3, 9]):
            r = Recorder([1, 2, 3], target)
            res = SpeculativeLoop(3).step(
                draft=r.draft, verify=r.verify, positions=_positions(500, 3)
            )
            assert len(res.tokens) >= 1, f"stalled on target {target}"


class TestRollback:
    def test_rollback_runs_when_a_rejection_strands_the_carry(self):
        r = Recorder([1, 2, 3], [1, 77, 3, 4])
        res = SpeculativeLoop(3).step(
            draft=r.draft, verify=r.verify,
            positions=torch.tensor([500, 501, 502]),  # one page
            snapshot=r.snapshot,
        )
        assert res.rolled_back is True
        assert r.restored == 1

    def test_no_rollback_when_the_block_was_fully_accepted(self):
        r = Recorder([1, 2, 3], [1, 2, 3, 4])
        res = SpeculativeLoop(3).step(
            draft=r.draft, verify=r.verify, positions=_positions(500, 3), snapshot=r.snapshot
        )
        assert res.rolled_back is False
        assert r.restored == 0

    def test_no_rollback_when_the_rejection_fell_in_a_later_page(self):
        # Last accepted on page 0, first rejected on page 1: the abandoned page takes
        # its ring block with it and the surviving page was never touched.
        r = Recorder([1, 2, 3], [1, 2, 88, 4])
        res = SpeculativeLoop(3).step(
            draft=r.draft, verify=r.verify,
            positions=torch.tensor([126, 127, 128]),
            snapshot=r.snapshot,
        )
        assert res.rolled_back is False
        assert r.restored == 0

    def test_the_loop_works_without_a_snapshot_at_all(self):
        r = Recorder([1, 2, 3], [1, 77, 3, 4])
        res = SpeculativeLoop(3).step(draft=r.draft, verify=r.verify, positions=_positions(500, 3))
        assert res.rolled_back is False
        assert res.tokens == [1, 77]


class TestInvariants:
    @pytest.mark.parametrize("seed", range(12))
    def test_random_blocks_never_break_the_contract(self, seed):
        """Over random agreement patterns: emitted tokens must be exactly the agreed
        prefix plus the target's own next token, and the step must always advance."""
        g = torch.Generator().manual_seed(seed)
        proposed = torch.randint(0, 20, (BLOCK,), generator=g)
        target = torch.randint(0, 20, (BLOCK + 1,), generator=g)
        r = Recorder(proposed.tolist(), target.tolist())
        res = SpeculativeLoop(BLOCK).step(
            draft=r.draft, verify=r.verify, positions=_positions()
        )
        assert 1 <= len(res.tokens) <= BLOCK + 1
        assert res.accepted <= res.verified <= res.drafted
        # Everything but the last emitted token must be an accepted proposal.
        assert res.tokens[: res.accepted] == proposed[: res.accepted].tolist()
        # And the emitted sequence must match what the target itself would produce.
        assert res.tokens == target[: len(res.tokens)].tolist(), (
            "speculation emitted a sequence the target would not have"
        )

    def test_block_size_must_be_positive(self):
        with pytest.raises(ValueError, match="block_size"):
            SpeculativeLoop(0)
