"""Rejection sampler and verify-row sampling: greedy prefix match, per-row sampling params and per-row penalties."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB = 8


def _sampler(device, vocab=VOCAB):
    from freetoken.engine.sample import Sampler

    return Sampler(device=device, vocab_size=vocab)


@cuda
def test_greedy_prefix_match_exact():
    from freetoken.spec.rejection import reject
    from freetoken.engine.sample import BatchSamplingArgs

    device = torch.device("cuda")
    k = 3
    # Target argmax sequence: [3, 1, 2, 0] over 4 positions.
    logits = torch.full((k + 1, VOCAB), -10.0, device=device)
    for pos, t in enumerate([3, 1, 2, 0]):
        logits[pos, t] = 10.0
    args = BatchSamplingArgs(temperatures=None)
    sampler = _sampler(device)

    cases = [
        ([3, 1, 2], 3, [3, 1, 2, 0]),   # all accepted + bonus
        ([3, 1, 7], 2, [3, 1, 2]),      # mismatch at pos 2 -> recovery = target argmax
        ([7, 1, 2], 0, [3]),            # first draft rejected
    ]
    for draft, want_emitted, want_tokens in cases:
        d = torch.tensor([draft], device=device)
        out, emitted = reject(logits, d, args, sampler)
        assert int(emitted[0]) == want_emitted
        assert out[0, : want_emitted + 1].tolist() == want_tokens


@cuda
def test_each_verify_row_counts_the_drafts_before_it_as_committed():
    """Row j penalizes the committed counts plus drafts d1..dj, frequency per occurrence and presence once per token; the counts stay as they were."""
    from freetoken.engine.sample import BatchSamplingArgs

    device = torch.device("cuda")
    torch.manual_seed(1)
    vocab, presence, frequency = 32, 0.7, 0.4
    drafts = torch.tensor([[5, 9, 5]], device=device)  # d3 repeats d1
    counts = torch.zeros(vocab, dtype=torch.int32, device=device)
    counts[9] = 2  # already committed, so presence must not apply to it again
    logits = torch.randn(drafts.shape[1] + 1, vocab, device=device)
    args = BatchSamplingArgs(temperatures=None, penalties=[(0, counts, presence, frequency)])

    sampler = _sampler(device, vocab)
    tokens = sampler.sample_verify(logits, args, drafts)
    sampled = BatchSamplingArgs(temperatures=torch.ones(1, device=device), penalties=args.penalties)
    probs = sampler.probs(logits, sampled, drafts)

    for j in range(drafts.shape[1] + 1):
        seen = counts.clone()
        for d in drafts[0, :j].tolist():
            seen[d] += 1
        expected = logits[j] - frequency * seen - presence * (seen > 0)
        assert int(tokens[j]) == int(expected.argmax()), f"row {j}"
        # the sparse-q verify reads the same penalized rows
        assert torch.allclose(probs[j], torch.softmax(expected, dim=-1), atol=1e-6), f"row {j}"
    assert counts[9] == 2 and counts.sum() == 2


@cuda
def test_each_verify_row_samples_with_its_requests_params():
    """Request i's sampling params cover its k+1 verify rows: a top_k=1 request and a greedy one give the argmax on every row, a full-vocab one does not."""
    from freetoken.engine.sample import BatchSamplingArgs

    device = torch.device("cuda")
    torch.manual_seed(2)
    vocab, k, trials = 32, 2, 16
    args = BatchSamplingArgs(
        temperatures=torch.ones(3, device=device),
        top_k=torch.tensor([1, vocab, vocab], dtype=torch.int32, device=device),
        greedy_mask=torch.tensor([False, False, True], device=device),
    )
    drafts = torch.zeros(3, k, dtype=torch.int64, device=device)
    off_argmax = []
    for _ in range(trials):
        logits = 0.1 * torch.randn(3 * (k + 1), vocab, device=device)
        tokens = _sampler(device, vocab).sample_verify(logits, args, drafts)
        off_argmax.append((tokens != logits.argmax(dim=-1)).view(3, k + 1))
    off_argmax = torch.stack(off_argmax)
    assert not off_argmax[:, 0].any() and not off_argmax[:, 2].any()
    assert off_argmax[:, 1].any(dim=0).all()


@cuda
def test_sparse_q_rejection_commits_the_target_distribution():
    """Drafts drawn from a q over K candidates per slot: the token committed at each position still follows the target p (Monte Carlo, k=2, V=6, K=3)."""
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.spec.rejection import sparse_q_rejection

    device = torch.device("cuda")
    torch.manual_seed(2)
    vocab, K, k, n = 6, 3, 2, 2_000_000
    p = torch.softmax(torch.randn(k + 1, vocab, device=device) * 1.5, dim=-1)
    cand = torch.tensor([[0, 2, 5], [1, 2, 3]], device=device)
    q = torch.softmax(torch.randn(k, K, device=device), dim=-1)
    picks = torch.stack([torch.multinomial(q[j], n, replacement=True) for j in range(k)], dim=1)
    drafts = cand.gather(1, picks.T).T
    # softmax(log p) at T=1 hands the verify exactly p on every row
    logits = p.log().repeat(n, 1)
    args = BatchSamplingArgs(temperatures=torch.ones(n, device=device))
    out, emitted = sparse_q_rejection(
        logits, drafts, cand.expand(n, k, K).contiguous(), q.expand(n, k, K).contiguous(), args, _sampler(device, vocab))

    def marginal(tokens):
        return torch.bincount(tokens, minlength=vocab).float() / tokens.numel()

    accepted_first, accepted_all = emitted >= 1, emitted == k
    assert int(accepted_first.sum()) > 100_000 and int(accepted_all.sum()) > 100_000
    assert (marginal(out[:, 0]) - p[0]).abs().max() < 5e-3
    assert (marginal(out[accepted_first, 1]) - p[1]).abs().max() < 5e-3
    assert (marginal(out[accepted_all, k]) - p[k]).abs().max() < 5e-3


@cuda
def test_sparse_q_rejection_rows_without_a_usable_q_take_the_one_hot_result():
    """A greedy row keeps the argmax prefix match where q would accept its first draft, and a row whose q has a NaN or a wrong total is verified as if its drafts were argmaxes where trusting q would reject every slot."""
    import math

    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.spec.rejection import sparse_q_rejection

    device = torch.device("cuda")
    torch.manual_seed(4)
    k, K = 3, 2
    # p is one-hot at [3, 1, 2, 0], except that row 0's first position splits 0.55 / 0.45 between tokens 3 and 7
    logits = torch.full((k + 1, VOCAB), -30.0, device=device)
    for pos, t in enumerate([3, 1, 2, 0]):
        logits[pos, t] = 30.0
    logits = logits.repeat(3, 1)
    logits[0, 3], logits[0, 7] = math.log(0.55), math.log(0.45)
    drafts = torch.tensor([[7, 1, 7], [3, 1, 7], [3, 1, 7]], device=device)
    cand = torch.tensor([[[7, 4], [1, 4], [7, 4]], [[3, 4], [1, 4], [7, 4]], [[3, 4], [1, 4], [7, 4]]], device=device)
    # row 0: q(7) = 0.01 << p(7) accepts for sure; row 2: q(d) = 1e6 >> p(d) rejects for sure
    q = torch.tensor([[[0.01, 0.99], [0.5, 0.5], [0.5, 0.5]], [[0.5, float("nan")], [0.5, 0.5], [0.5, 0.5]], [[1e6, 0.5]] * k], device=device)
    args = BatchSamplingArgs(temperatures=torch.ones(3, device=device), greedy_mask=torch.tensor([True, False, False], device=device))

    out, emitted = sparse_q_rejection(logits, drafts, cand, q, args, _sampler(device))

    assert emitted.tolist() == [0, 2, 2]
    assert int(out[0, 0]) == 3 and out[1:, :3].tolist() == [[3, 1, 2]] * 2
