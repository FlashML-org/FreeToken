"""Short-convolution windows against the whole-sequence reference (``reference.short_conv``, checked against
transformers #51050's ``Inkling2LoopShortConvolution`` / ``Inkling2LoopDynamicShortConvolution``).

A chunked prefill, decode steps and a verify committed at every accepted count must each reproduce the
convolution of the whole sequence, which only holds when the window carries exactly the last K - 1 inputs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .reference import short_conv as ref_conv

K, C, HEADS = 4, 64, 4


def _meta(lens, slots):
    from freetoken.models.inkling2_loop.conv import ConvMetadata, prefill_indices

    x_index, state_index, next_index = prefill_indices(lens, K, torch.device("cpu"))
    return ConvMetadata(torch.tensor(slots), K, x_index, state_index, next_index)


def _taps(conv1d_weight):
    return conv1d_weight[:, 0, :].flip(-1).t().contiguous()


def _streams(seed, lens, dynamic):
    g = torch.Generator().manual_seed(seed)
    xs = [torch.randn(n, C, generator=g).to(torch.bfloat16) for n in lens]
    if dynamic:
        static = torch.randn(K, C, generator=g)
        dyns = [torch.randn(n, HEADS, K, generator=g) for n in lens]
        refs = [ref_conv(x, static_weight=static, dynamic=d) for x, d in zip(xs, dyns)]
        return xs, static, dyns, refs
    w = torch.randn(C, 1, K, generator=g)
    return xs, _taps(w), None, [ref_conv(x, conv1d_weight=w) for x in xs]


def _conv(x, taps, state, meta, dyn, **kw):
    from freetoken.models.inkling2_loop.conv import short_conv

    return short_conv(x, taps, state, meta, dynamic=dyn, head_dim=C // HEADS, **kw)


def _close(got, ref):
    # outputs are bf16, as in the reference's layer
    return (got.float() - ref).abs().max().item() < 3e-2 * ref.abs().max().item()


@pytest.mark.parametrize("dynamic", [False, True])
def test_chunked_prefill_then_decode(dynamic):
    from freetoken.models.inkling2_loop.conv import ConvMetadata

    lens = [23, 1, 40]
    xs, taps, dyns, refs = _streams(0, lens, dynamic)
    state = torch.zeros(5, K - 1, C, dtype=torch.bfloat16)
    slots = [3, 1, 4]
    # prefill chunk boundaries per request; every chunk reads the window the previous one left, then decode
    bounds = [[0, 7, 8, 20], [0, 1], [0, 2, 33]]
    for step in range(max(len(b) for b in bounds) - 1):
        live = [i for i, b in enumerate(bounds) if step + 1 < len(b)]
        spans = [(bounds[i][step], bounds[i][step + 1]) for i in live]
        meta = _meta([e - s for s, e in spans], [slots[i] for i in live])
        x = torch.cat([xs[i][s:e] for i, (s, e) in zip(live, spans)])
        dyn = torch.cat([dyns[i][s:e] for i, (s, e) in zip(live, spans)]) if dynamic else None
        out = _conv(x, taps, state, meta, dyn)
        off = 0
        for i, (s, e) in zip(live, spans):
            assert _close(out[off : off + e - s], refs[i][s:e])
            off += e - s
    done = [b[-1] for b in bounds]
    while any(d < n for d, n in zip(done, lens)):
        live = [i for i in range(len(lens)) if done[i] < lens[i]]
        meta = ConvMetadata(torch.tensor([slots[i] for i in live]), K)
        x = torch.cat([xs[i][done[i] : done[i] + 1] for i in live])
        dyn = torch.cat([dyns[i][done[i] : done[i] + 1] for i in live]) if dynamic else None
        out = _conv(x, taps, state, meta, dyn)
        for row, i in enumerate(live):
            assert _close(out[row : row + 1], refs[i][done[i] : done[i] + 1])
            done[i] += 1
    for i, n in enumerate(lens):
        tail = xs[i][max(0, n - (K - 1)) :]
        assert torch.equal(state[slots[i], K - 1 - tail.shape[0] :], tail)


def test_verify_commit_matches_decode():
    from freetoken.models.inkling2_loop.conv import ConvMetadata, ConvWindowVerify

    R, n_prefix = 5, 9
    xs, taps, _, refs = _streams(1, [n_prefix + R], False)
    x = xs[0]
    spec = SimpleNamespace(name="w", shape=(K - 1, C), layer_ids=(0,))
    handler = ConvWindowVerify(spec, max_bs=2, rows_per_req=R, device=torch.device("cpu"), dtype=torch.bfloat16)
    for accepted in range(1, R + 1):
        state = torch.zeros(2, K - 1, C, dtype=torch.bfloat16)
        _conv(x[:n_prefix], taps, state, _meta([n_prefix], [1]), None)
        live = state.clone()
        meta = ConvMetadata(torch.tensor([1]), K, rows_per_req=R, verify=True)
        out = _conv(x[n_prefix:], taps, state, meta, None, verify_stash=handler.stash(0))
        assert torch.equal(state, live), "a verify forward must not move the live window"
        assert _close(out, refs[0][n_prefix:])
        pool = SimpleNamespace(slot_state=lambda name, layer_id: state)
        handler.commit(pool, torch.tensor([1]), torch.tensor([accepted]))
        end = n_prefix + accepted
        assert torch.equal(state[1], x[end - (K - 1) : end])
