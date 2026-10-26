"""Rejection sampler: greedy exactness + losslessness of both stochastic paths.

The lossless claim is checked against the mathematical target: with one-hot draft
q, the token emitted at position 0 must be distributed EXACTLY as the target's
renormalized p_0 -- for any draft token, accepted or not. The flashinfer and
the Triton samplers are held to the same statistical bar.
"""

from types import SimpleNamespace

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB = 8


def _sampler(device):
    from freetoken.engine.sample import Sampler

    return Sampler(device=device, vocab_size=VOCAB)


def _args(device, temperature=1.0):
    from freetoken.engine.sample import BatchSamplingArgs

    return BatchSamplingArgs(
        temperatures=torch.full((1,), temperature, device=device)
    )


@cuda
def test_greedy_prefix_match_exact():
    from freetoken.spec.rejection import RejectionSampler
    from freetoken.engine.sample import BatchSamplingArgs

    device = torch.device("cuda")
    rs = RejectionSampler()
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
        out, emitted = rs(logits, d, args, sampler)
        assert int(emitted[0]) == want_emitted
        assert out[0, : want_emitted + 1].tolist() == want_tokens


@cuda
@pytest.mark.parametrize("path", ["flashinfer", "torch"])
@pytest.mark.parametrize("draft_id", [0, 5])
def test_first_position_marginal_is_lossless(monkeypatch, path, draft_id):
    """P(first emitted token = v) must equal p_0(v) for every v: accepted-draft mass
    p(d) lands on d, and the residual (p with d zeroed, renormalized) covers the
    rest -- the mixture is exactly p_0. draft_id=0 is the target argmax (usually
    accepted), draft_id=5 a low-probability token (usually rejected)."""
    if path == "flashinfer":
        from freetoken.kernel.backend import is_flashinfer_installed

        if not is_flashinfer_installed():
            pytest.skip("flashinfer not installed")
    else:
        from freetoken.kernel import backend

        monkeypatch.setattr(backend, "is_flashinfer_installed", lambda: False)
        import freetoken.spec.rejection as rej_mod  # noqa: F401  (module-local import path)

    from freetoken.spec.rejection import RejectionSampler

    device = torch.device("cuda")
    torch.manual_seed(0)
    rs = RejectionSampler()
    sampler = _sampler(device)
    args = _args(device, temperature=1.0)

    k = 2
    logits = torch.randn(k + 1, VOCAB, device=device)
    p = torch.softmax(logits.float(), dim=-1)  # temperature 1, no top-k / top-p
    draft = torch.tensor([[draft_id, 1]], device=device)

    n, counts = 20000, torch.zeros(VOCAB, device=device)
    batched_logits = logits.repeat(1, 1)  # bs=1 per call; loop for trials
    for _ in range(n):
        out, emitted = rs(batched_logits, draft, args, sampler)
        counts[out[0, 0]] += 1
    freq = counts / n
    tol = 4.0 / (n ** 0.5)  # ~4 sigma for a Bernoulli frequency
    assert torch.all((freq - p[0]).abs() < tol), (
        f"first-token marginal deviates: freq={freq.tolist()} p={p[0].tolist()}"
    )


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

    tokens = Sampler_(device, vocab).sample_verify(logits, args, drafts)

    for j in range(drafts.shape[1] + 1):
        seen = counts.clone()
        for d in drafts[0, :j].tolist():
            seen[d] += 1
        expected = logits[j] - frequency * seen - presence * (seen > 0)
        assert int(tokens[j]) == int(expected.argmax()), f"row {j}"
    assert counts[9] == 2 and counts.sum() == 2


def Sampler_(device, vocab):
    from freetoken.engine.sample import Sampler

    return Sampler(device=device, vocab_size=vocab)
