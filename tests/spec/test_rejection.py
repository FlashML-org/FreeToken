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
        # the verify with draft probs reads the same penalized rows
        assert torch.allclose(probs[j], torch.softmax(expected, dim=-1), atol=1e-6), f"row {j}"
    assert counts[9] == 2 and counts.sum() == 2


@cuda
def test_each_verify_row_samples_with_its_requests_params():
    """Request i's sampling params cover its k+1 verify rows: a top_k=1 request and a greedy one give the argmax on every row, a sampled one does not; the draft-probs verify reads each row's softmax at its request's temperature, renormalized over its top-k and then its top-p."""
    from freetoken.engine.sample import BatchSamplingArgs

    device = torch.device("cuda")
    torch.manual_seed(2)
    vocab, k, trials, bs = 32, 2, 16, 4
    args = BatchSamplingArgs(
        temperatures=torch.tensor([1.0, 1.3, 1.0, 0.7], device=device),
        top_k=torch.tensor([1, vocab, vocab, 6], dtype=torch.int32, device=device),
        top_p=torch.tensor([1.0, 0.9, 1.0, 0.8], device=device),
        greedy_mask=torch.tensor([False, False, True, False], device=device),
    )
    drafts = torch.zeros(bs, k, dtype=torch.int64, device=device)
    off_argmax = []
    for _ in range(trials):
        logits = 0.1 * torch.randn(bs * (k + 1), vocab, device=device)
        tokens = _sampler(device, vocab).sample_verify(logits, args, drafts)
        off_argmax.append((tokens != logits.argmax(dim=-1)).view(bs, k + 1))
    off_argmax = torch.stack(off_argmax)
    assert not off_argmax[:, 0].any() and not off_argmax[:, 2].any()
    assert off_argmax[:, 1].any(dim=0).all()

    logits = torch.randn(bs * (k + 1), vocab, device=device)
    probs = _sampler(device, vocab).probs(logits, args, drafts)
    for row, i in enumerate(torch.arange(bs).repeat_interleave(k + 1).tolist()):
        want = torch.softmax(logits[row] / args.temperatures[i], dim=-1)
        want = torch.where(want >= want.topk(int(args.top_k[i])).values[-1], want, 0.0)
        # top-p keeps the smallest best-first set whose mass reaches top_p
        p_sorted, order = (want / want.sum()).sort(descending=True)
        p_sorted = torch.where(p_sorted.cumsum(0) - p_sorted < args.top_p[i], p_sorted, 0.0)
        want = torch.zeros_like(want).scatter(0, order, p_sorted / p_sorted.sum())
        if args.greedy_mask[i]:
            want = torch.nn.functional.one_hot(logits[row].argmax(), vocab).float()
        torch.testing.assert_close(probs[row], want, atol=1e-6, rtol=1e-5, msg=f"row {row}")


@cuda
def test_reject_with_draft_probs_commits_the_target_distribution():
    """Drafts drawn from a q over K candidates per slot, in a batch that mixes rows trusting q, rows whose q is unusable and greedy rows: the token committed at each position still follows the target p on both sampled kinds, and a greedy row commits the argmax prefix (Monte Carlo, k=2, V=6, K=3)."""
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.spec.rejection import reject_with_draft_probs

    device = torch.device("cuda")
    torch.manual_seed(2)
    vocab, K, k, n = 6, 3, 2, 6_000_000
    p = torch.softmax(torch.randn(k + 1, vocab, device=device) * 1.5, dim=-1)
    cand = torch.tensor([[0, 2, 5], [1, 2, 3]], device=device)
    q = torch.softmax(torch.randn(k, K, device=device), dim=-1)
    picks = torch.stack([torch.multinomial(q[j], n, replacement=True) for j in range(k)], dim=1)
    drafts = cand.gather(1, picks.T).T
    # rows cycle through: q trusted, q unusable (a NaN or a total of 2), greedy
    kind = torch.arange(n, device=device) % 3
    q_rows = q.expand(n, k, K).clone()
    unusable = (kind == 1).nonzero().squeeze(1)
    q_rows[unusable[::2], 0, 0] = float("nan")
    q_rows[unusable[1::2], 0] *= 2
    # softmax(log p) at T=1 hands the verify exactly p on every row
    logits = p.log().repeat(n, 1)
    args = BatchSamplingArgs(temperatures=torch.ones(n, device=device), greedy_mask=kind == 2)
    out, emitted = reject_with_draft_probs(logits, drafts, cand.expand(n, k, K).contiguous(), q_rows, args, _sampler(device, vocab))

    def marginal(tokens):
        return torch.bincount(tokens, minlength=vocab).float() / tokens.numel()

    for rows in (kind == 0, kind == 1):
        rows_out, rows_emitted = out[rows], emitted[rows]
        accepted_first = rows_emitted >= 1
        assert int(accepted_first.sum()) > 100_000
        assert (marginal(rows_out[:, 0]) - p[0]).abs().max() < 5e-3
        assert (marginal(rows_out[accepted_first, 1]) - p[1]).abs().max() < 5e-3
    # both kinds draw the bonus token from p itself; a row that does not trust q rarely accepts every draft
    accepted_all = (kind < 2) & (emitted == k)
    assert int(accepted_all.sum()) > 100_000
    assert (marginal(out[accepted_all, k]) - p[k]).abs().max() < 5e-3
    greedy = kind == 2
    argmax = logits[: k + 1].argmax(dim=-1)
    assert torch.equal(emitted[greedy], (drafts[greedy] == argmax[:k]).cumprod(dim=1).sum(dim=1))
    committed = torch.arange(k + 1, device=device) <= emitted[greedy].unsqueeze(1)
    assert torch.equal(out[greedy][committed], argmax.expand(int(greedy.sum()), -1)[committed])
