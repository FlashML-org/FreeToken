"""Speculative sampling must not change what the model would have written.

This is the only property that matters here, and the only one that cannot be checked by
inspection: over many draws, the emitted token has to be distributed exactly as the
TARGET's own distribution p, whatever the drafter q proposed. If it is not, speculation
is a silent quality change -- output that reads fine, is faster, and is not the model
the caller asked for.

So these tests are statistical. They drive the acceptance rule thousands of times with
a fixed seed and compare the empirical distribution against p.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.deepseek_v4.dspark import (
    rejection_accept,
    rejection_accept_device,
    sampling_probs,
)

VOCAB = 6


def _probs(*weights: float) -> torch.Tensor:
    t = torch.tensor(weights, dtype=torch.float)
    return t / t.sum()


def _empirical(q_row, p_rows, draws=20000, seed=0):
    """Run one drafted position many times; return the emitted-token distribution.

    The draft samples from q, exactly as the real drafter does, so this exercises the
    accept/reject path over the whole support rather than one fixed proposal.
    """
    g = torch.Generator().manual_seed(seed)
    counts = torch.zeros(VOCAB)
    for _ in range(draws):
        x = torch.multinomial(q_row, 1, generator=g)
        n_acc, tok = rejection_accept(x, q_row.unsqueeze(0), p_rows, generator=g)
        counts[int(x) if n_acc == 1 else tok] += 1
    return counts / counts.sum()


class TestDistributionIsPreserved:
    def test_emitted_tokens_follow_the_target_not_the_draft(self):
        q = _probs(5, 1, 1, 1, 1, 1)          # drafter loves token 0
        p = _probs(1, 1, 3, 1, 1, 1)          # target prefers token 2
        p_rows = torch.stack([p, p])
        got = _empirical(q, p_rows)
        assert torch.allclose(got, p, atol=0.02), (
            f"emitted {got.tolist()} but the target's distribution is {p.tolist()} -- "
            "speculation changed the output distribution"
        )

    def test_it_holds_when_draft_and_target_agree(self):
        p = _probs(3, 2, 1, 1, 1, 1)
        got = _empirical(p, torch.stack([p, p]))
        assert torch.allclose(got, p, atol=0.02)

    def test_it_holds_when_they_barely_overlap(self):
        # The hard case: almost every proposal is rejected, so nearly every emitted
        # token comes from the residual. If the residual is wrong, this is where it shows.
        q = _probs(10, 10, 1, 1, 1, 1)
        p = _probs(1, 1, 1, 1, 10, 10)
        got = _empirical(q, torch.stack([p, p]))
        assert torch.allclose(got, p, atol=0.03)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_random_pairs_preserve_the_target(self, seed):
        g = torch.Generator().manual_seed(seed)
        q = torch.softmax(torch.randn(VOCAB, generator=g), dim=-1)
        p = torch.softmax(torch.randn(VOCAB, generator=g), dim=-1)
        got = _empirical(q, torch.stack([p, p]), draws=20000, seed=seed)
        assert torch.allclose(got, p, atol=0.03)


class TestAcceptanceBehaviour:
    def test_a_proposal_the_target_is_certain_of_is_always_accepted(self):
        q = _probs(1, 1, 1, 1, 1, 1)
        p = torch.zeros(VOCAB)
        p[3] = 1.0
        g = torch.Generator().manual_seed(0)
        n_acc, _ = rejection_accept(
            torch.tensor([3]), q.unsqueeze(0), torch.stack([p, p]), generator=g
        )
        assert n_acc == 1, "p(x)/q(x) >= 1 must accept outright"

    def test_a_proposal_the_target_rules_out_is_always_rejected(self):
        q = _probs(1, 1, 1, 1, 1, 1)
        p = torch.zeros(VOCAB)
        p[3] = 1.0
        g = torch.Generator().manual_seed(0)
        n_acc, tok = rejection_accept(
            torch.tensor([0]), q.unsqueeze(0), torch.stack([p, p]), generator=g
        )
        assert (n_acc, tok) == (0, 3), "p(x) == 0 must reject and resample to p's mass"

    def test_acceptance_stops_at_the_first_rejection(self):
        # Position 1 is impossible under p, so position 2 must never be reached even
        # though the target would have agreed with it.
        p_bad = torch.zeros(VOCAB)
        p_bad[5] = 1.0
        p_ok = torch.zeros(VOCAB)
        p_ok[1] = 1.0
        q = _probs(1, 1, 1, 1, 1, 1)
        g = torch.Generator().manual_seed(0)
        n_acc, _ = rejection_accept(
            torch.tensor([1, 0, 1]),
            torch.stack([q, q, q]),
            torch.stack([p_ok, p_bad, p_ok, p_ok]),
            generator=g,
        )
        assert n_acc == 1

    def test_a_fully_accepted_block_draws_its_bonus_from_the_target(self):
        p = torch.zeros(VOCAB)
        p[4] = 1.0
        q = _probs(1, 1, 1, 1, 9, 1)
        g = torch.Generator().manual_seed(0)
        n_acc, tok = rejection_accept(
            torch.tensor([4, 4]), torch.stack([q, q]), torch.stack([p, p, p]), generator=g
        )
        assert (n_acc, tok) == (2, 4)

    def test_a_step_always_emits_at_least_one_token(self):
        g = torch.Generator().manual_seed(0)
        for seed in range(20):
            gg = torch.Generator().manual_seed(seed)
            q = torch.softmax(torch.randn(VOCAB, generator=gg), dim=-1)
            p = torch.softmax(torch.randn(VOCAB, generator=gg), dim=-1)
            x = torch.multinomial(q, 2, replacement=True, generator=gg)
            n_acc, tok = rejection_accept(
                x, torch.stack([q, q]), torch.stack([p, p, p]), generator=g
            )
            assert 0 <= n_acc <= 2
            assert 0 <= tok < VOCAB


class TestDegenerateInputs:
    def test_a_zero_probability_draft_token_is_accepted_rather_than_dividing_by_zero(self):
        q = torch.zeros(VOCAB)
        q[0] = 1.0
        p = _probs(1, 1, 1, 1, 1, 1)
        g = torch.Generator().manual_seed(0)
        n_acc, _ = rejection_accept(
            torch.tensor([3]), q.unsqueeze(0), torch.stack([p, p]), generator=g
        )
        assert n_acc == 1

    def test_a_residual_of_zero_falls_back_to_the_target(self):
        # p entirely inside q: the residual is all zeros, so there is nothing to prefer
        # and normalizing it would divide by zero.
        p = _probs(1, 1, 0, 0, 0, 0)
        q = _probs(2, 2, 1, 1, 1, 1)
        g = torch.Generator().manual_seed(0)
        n_acc, tok = rejection_accept(
            torch.tensor([2]), q.unsqueeze(0), torch.stack([p, p]), generator=g
        )
        assert n_acc == 0
        assert tok in (0, 1), "the fallback must still draw from the target's support"


class TestDeviceResidentAcceptance:
    def test_full_acceptance_and_bonus_stay_exact(self):
        q = torch.zeros(2, VOCAB)
        q[0, 1] = 1.0
        q[1, 2] = 1.0
        p = torch.zeros(3, VOCAB)
        p[0, 1] = 1.0
        p[1, 2] = 1.0
        p[2, 4] = 1.0
        got = rejection_accept_device(
            torch.tensor([1, 2]), q, p, generator=torch.Generator().manual_seed(0)
        )
        assert got == (2, 4)

    def test_rejection_recovers_from_the_target_residual(self):
        q = torch.zeros(1, VOCAB)
        q[0, 0] = 1.0
        p = torch.zeros(2, VOCAB)
        p[0, 3] = 1.0
        p[1, 4] = 1.0
        got = rejection_accept_device(
            torch.tensor([0]), q, p, generator=torch.Generator().manual_seed(0)
        )
        assert got == (0, 3)

    def test_zero_width_draws_the_target_bonus(self):
        p = torch.zeros(1, VOCAB)
        p[0, 5] = 1.0
        got = rejection_accept_device(
            torch.empty(0, dtype=torch.long),
            torch.empty(0, VOCAB),
            p,
            generator=torch.Generator().manual_seed(0),
        )
        assert got == (0, 5)

    def test_accepted_length_and_bonus_share_one_host_transfer(self):
        import inspect

        body = inspect.getsource(rejection_accept_device)
        assert 'torch.stack((n_acc_device, bonus_device.to(torch.int64))).to(' in body
        assert ".item()" not in body


def test_temperature_one_probabilities_equal_the_direct_softmax():
    logits = torch.tensor([[1.0, -2.0, 3.0, 0.5]])
    got = sampling_probs(logits, temperature=1.0, top_p=1.0, top_k=-1)
    assert torch.equal(got, torch.softmax(logits.float(), dim=-1))
