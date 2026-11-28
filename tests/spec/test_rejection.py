"""Rejection sampler and verify-row sampling: greedy prefix match, per-row sampling params and per-row penalties."""

import json
from contextlib import nullcontext

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


# a byte-level vocabulary small enough to walk by hand, with every byte the schema's keys need; 12 and 13 end a
# reply, 14 is a thinking close tag
_JSON_TOKENS = [b"{", b"}", b'"a"', b'"b"', b":", b",", b"1", b"23", b" ", b'"', b"x", b'x"', None, None, None, b"{}", b"a", b"b"]
_JSON_EOS, _JSON_CLOSE = (12, 13), 14
_JSON_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer", "minimum": 0, "maximum": 99}, "b": {"type": "string", "maxLength": 3}},
    "required": ["a"],
}


def _json_grammar():
    from freetoken.grammar.matcher import Vocab

    return Vocab(_JSON_TOKENS, len(_JSON_TOKENS), _JSON_EOS).grammar(json.dumps(_JSON_SCHEMA))


def _logits_after(prefix, salt=0):
    """The fake target's logits after ``prefix``: random, but a function of the prefix alone."""
    g = torch.Generator().manual_seed(hash((salt, *prefix)) % 2**31)
    return torch.randn(len(_JSON_TOKENS), generator=g) * 4


def _check_reply(tokens):
    """The reply's text against the schema, by json.loads and the schema's own rules."""
    assert tokens[-1] in _JSON_EOS
    doc = json.loads(b"".join(_JSON_TOKENS[t] for t in tokens[:-1]))
    assert set(doc) <= {"a", "b"} and isinstance(doc["a"], int) and 0 <= doc["a"] <= 99
    assert "b" not in doc or (isinstance(doc["b"], str) and len(doc["b"]) <= 3)


@pytest.mark.parametrize("salt", range(8))
def test_masked_verify_rows_commit_what_masked_decode_steps_commit(salt, monkeypatch):
    """Greedy, a json_schema request commits the same reply under k drafts as one token a step: row j is masked from
    the state its drafts d1..dj reach, and a draft the mask forbids (drafts here are the unmasked argmax) is rejected
    as a mismatch is. The reply parses and validates."""
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.grammar.matcher import GrammarPlan, Matcher
    from freetoken.spec.rejection import reject

    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda name: nullcontext())
    grammar, sampler, k = _json_grammar(), _sampler(torch.device("cpu"), len(_JSON_TOKENS)), 3
    one = Matcher(grammar)
    plain = []
    while not plain or plain[-1] not in _JSON_EOS:
        args = BatchSamplingArgs(temperatures=None, grammar=GrammarPlan([one], 1))
        token = int(sampler.sample(_logits_after(plain, salt).unsqueeze(0), args)[0])
        plain.append(token)
        one.advance([token])
        assert len(plain) < 200
    _check_reply(plain)

    spec, rejected = Matcher(grammar), 0
    committed = []
    while not committed or committed[-1] not in _JSON_EOS:
        drafts = []
        for _ in range(k):
            drafts.append(int(_logits_after(committed + drafts, salt + 100).argmax()))
        rows = torch.stack([_logits_after(committed + drafts[:j], salt) for j in range(k + 1)])
        args = BatchSamplingArgs(temperatures=None, grammar=GrammarPlan([spec], k + 1))
        out, emitted = reject(rows, torch.tensor([drafts]), args, sampler)
        step = out[0, : int(emitted) + 1].tolist()
        rejected += int(emitted) < k
        if any(t in _JSON_EOS for t in step):
            step = step[: next(i for i, t in enumerate(step) if t in _JSON_EOS) + 1]
        committed += step
        spec.advance(step)
        assert len(committed) < 200
    assert committed == plain and rejected


def test_a_draft_the_schema_forbids_is_rejected_where_the_logits_favor_it():
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.grammar.matcher import GrammarPlan, Matcher
    from freetoken.spec.rejection import reject

    grammar = _json_grammar()
    vocab, k = len(_JSON_TOKENS), 3
    # the logits favor '{', 'x', ':' then '}'; after '{' only a key or ' ' may follow
    logits = torch.full((k + 1, vocab), -10.0)
    for row, t in enumerate([0, 10, 4, 1]):
        logits[row, t] = 10.0
    logits[1, 2] = 5.0  # the best allowed token at row 1: '"a"'
    args = BatchSamplingArgs(temperatures=None, grammar=GrammarPlan([Matcher(grammar)], k + 1))
    out, emitted = reject(logits, torch.tensor([[0, 10, 4]]), args, _sampler(torch.device("cpu"), vocab))
    assert int(emitted) == 1 and out[0, :2].tolist() == [0, 2]
    # unconstrained, the same drafts all pass
    out, emitted = reject(logits, torch.tensor([[0, 10, 4]]), BatchSamplingArgs(temperatures=None), _sampler(torch.device("cpu"), vocab))
    assert int(emitted) == 3


def test_a_thinking_span_stays_free_until_its_close_tag_even_inside_the_drafts():
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.grammar.matcher import GrammarPlan, Matcher
    from freetoken.spec.rejection import reject

    grammar = _json_grammar()
    vocab, k = len(_JSON_TOKENS), 3
    # every row prefers 'x' (not JSON), then '{'
    logits = torch.full((k + 1, vocab), -10.0)
    logits[:, 10] = 10.0
    logits[:, 0] = 5.0
    sampler = _sampler(torch.device("cpu"), vocab)
    thinking = Matcher(grammar, close=(_JSON_CLOSE,), thinking=True)
    args = BatchSamplingArgs(temperatures=None, grammar=GrammarPlan([thinking], k + 1))
    # rows 0 and 1 think freely; the close tag drafted at slot 2 constrains row 2 (after it) and row 3
    out, emitted = reject(logits, torch.tensor([[10, _JSON_CLOSE, 0]]), args, sampler)
    assert out[0].tolist()[:2] == [10, 10] and int(emitted) == 1
    logits[1, _JSON_CLOSE] = 20.0
    out, emitted = reject(logits, torch.tensor([[10, _JSON_CLOSE, 0]]), args, sampler)
    # row 3 reads '{': a key or a space, not the 'x' its logits favor
    assert int(emitted) == 3 and out[0].tolist()[:3] == [10, _JSON_CLOSE, 0] and int(out[0, 3]) in (2, 8)
    thinking.advance([10, _JSON_CLOSE])
    assert thinking.key > 0


def test_unconstrained_rows_sample_from_the_unmasked_logits(monkeypatch):
    """No plan, or a plan whose requests are all free, hands back the very logits tensor: nothing is copied or masked;
    in a mixed batch only the constrained request's row changes."""
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.grammar.matcher import GrammarPlan, Matcher

    grammar = _json_grammar()
    logits = torch.zeros(2, len(_JSON_TOKENS))
    logits[:, 10] = 10.0
    assert GrammarPlan([None, None], 1).apply(logits) is logits
    assert GrammarPlan([Matcher(grammar, close=(_JSON_CLOSE,), thinking=True), None], 1).apply(logits) is logits
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda name: nullcontext())
    sampler = _sampler(torch.device("cpu"), len(_JSON_TOKENS))
    tokens = sampler.sample(logits, BatchSamplingArgs(temperatures=None, grammar=GrammarPlan([Matcher(grammar), None], 1)))
    assert int(tokens[0]) != 10 and int(tokens[1]) == 10
