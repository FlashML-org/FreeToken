"""End-to-end forward smoke on the tiny model: cold prefill, batched decode, a second request, chunked
prefill and bounded replay all produce finite logits of the right shape (numerics: test_reference_parity)."""

from __future__ import annotations

import pytest
import torch

from .common import VOCAB, requires_cuda, write_tiny_checkpoint

pytestmark = requires_cuda


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dsv41-tiny")
    write_tiny_checkpoint(str(folder))
    return str(folder)


def _tokens(n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, VOCAB, (n,), generator=g).tolist()


def test_prefill_and_decode(checkpoint):
    from .harness import TinyEngine

    eng = TinyEngine(checkpoint, max_seq_len=1024, max_running_req=2, swa_decoder_replay="exact")
    r0 = eng.new_request(0, _tokens(300, 1))
    logits = eng.prefill([r0])
    assert logits.shape == (1, VOCAB) and torch.isfinite(logits).all()
    eng.finish_prefill([r0])
    r1 = eng.new_request(1, _tokens(50, 2))
    assert torch.isfinite(eng.prefill([r1])).all()
    eng.finish_prefill([r1])
    for step in range(5):  # batched decode over two requests at different positions
        out = eng.decode([r0, r1], [10 + step, 20 + step])
        assert out.shape == (2, VOCAB) and torch.isfinite(out).all()


def test_head_preserves_fp32_logit_margin(checkpoint, monkeypatch):
    from .harness import TinyEngine

    eng = TinyEngine(checkpoint, max_seq_len=1024, max_running_req=1)
    assert eng.model.head.weight.dtype == torch.bfloat16
    hidden = torch.zeros(2, eng.args.dim, device=eng.device, dtype=torch.bfloat16)
    hidden[:, :2] = 1
    eng.model.head.weight.zero_()
    eng.model.head.weight[:2, 0] = 1
    eng.model.head.weight[1, 1] = 2**-10
    monkeypatch.setattr(eng.model.model, "prefill", lambda *args: hidden)
    monkeypatch.setattr(eng.model.model, "decode", lambda *args: hidden[:1])

    req = eng.new_request(0, [3, 4])
    prefill = eng.prefill([req])
    eng.finish_prefill([req])
    decode = eng.decode([req], [5])
    for logits in (prefill, decode):
        # Both values round to 1 in bf16, which would make argmax pick token 0.
        assert logits[0, :2].tolist() == [1.0, 1.0 + 2**-10]
        assert logits.argmax(-1).item() == 1


def test_chunked_prefill_matches_single_shot(checkpoint):
    """Two 128-aligned chunks must reproduce the single-shot prefill's last-token logits: the
    compressor carry, the window ring and the compressed rows are all resumed through the pool. The
    chunks run their projections at other row counts, so the match is to rounding, not bitwise."""
    from .harness import TinyEngine
    from .test_reference_parity import _compare

    eng = TinyEngine(checkpoint, max_seq_len=1024, max_running_req=2, swa_decoder_replay="exact")
    toks = _tokens(300, 3)
    whole = eng.new_request(0, toks)
    ref = eng.prefill([whole])
    chunked = eng.new_request(1, toks)
    chunked.device_len = 256  # first chunk [0, 256)
    eng.prefill([chunked])
    chunked.cached_len, chunked.device_len = 256, 300
    got = eng.prefill([chunked])
    _compare("chunked prefill", got, ref)


def test_bounded_replay_runs(checkpoint):
    from .harness import TinyEngine

    eng = TinyEngine(checkpoint, max_seq_len=1024, max_running_req=2, swa_decoder_replay="bounded")
    r0 = eng.new_request(0, _tokens(300, 4))
    logits = eng.prefill([r0])
    assert logits.shape == (1, VOCAB) and torch.isfinite(logits).all()
    eng.finish_prefill([r0])
    out = eng.decode([r0], [5])
    assert torch.isfinite(out).all()


def test_bounded_replay_is_exact_within_one_window(checkpoint):
    """A prompt no longer than the window replays every token with no floor: bounded == exact."""
    from .harness import TinyEngine
    from .test_reference_parity import _compare

    toks = _tokens(100, 5)
    exact = TinyEngine(checkpoint, max_seq_len=1024, max_running_req=2, swa_decoder_replay="exact")
    want = exact.prefill([exact.new_request(0, toks)])
    bounded = TinyEngine(checkpoint, max_seq_len=1024, max_running_req=2, swa_decoder_replay="bounded")
    got = bounded.prefill([bounded.new_request(0, toks)])
    _compare("bounded within one window", got, want)


def test_bounded_replay_extends_a_short_final_chunk_back_over_the_window(checkpoint):
    """A short final chunk (or prefix-hit suffix) re-runs the encoder from the previous window page so
    the decoder still replays the prompt's whole last window: same logits as the single-shot bounded
    prefill of the same prompt."""
    from .harness import TinyEngine
    from .test_reference_parity import _compare

    eng = TinyEngine(checkpoint, max_seq_len=1024, max_running_req=2, swa_decoder_replay="bounded")
    toks = _tokens(300, 6)
    whole = eng.new_request(0, toks)
    ref = eng.prefill([whole])
    chunked = eng.new_request(1, toks)
    chunked.device_len = 256
    eng.prefill([chunked])
    chunked.cached_len, chunked.device_len = 256, 300  # 44 new tokens: the replay needs [172, 300)
    got = eng.prefill([chunked])
    _compare("bounded short final chunk", got, ref)


def test_bounded_extension_never_rewrites_cached_history(checkpoint):
    """The recomputed history of a bounded-replay extension is read-only on the pools: cached window /
    main / index rows (and ring carries) of positions before the hit keep whatever they held."""
    from .harness import TinyEngine

    eng = TinyEngine(checkpoint, max_seq_len=1024, max_running_req=2, swa_decoder_replay="bounded")
    toks = _tokens(300, 7)
    req = eng.new_request(0, toks)
    req.device_len = 256
    eng.prefill([req])
    pool = eng.pool
    # poison the cached rows of positions [128, 256) on an encoder layer and its kv source
    win_slots = pool.translate_full_to_window(eng.page_table[0, 128:256].long())
    src, ratio = 2, 2
    main_rows = pool.cmp_rows(eng.page_table[0, 128:256:ratio].long(), ratio)
    pool.window_pool[3][win_slots] = 0xAB
    pool.main_pool[src][main_rows] = 0xCD
    pool.idx_pool[src][main_rows] = 0xEF
    ring = pool.state_ring[src].buffer.clone()
    req.cached_len, req.device_len = 256, 300  # 44 new tokens: the encoder recomputes from 128
    eng.prefill([req])
    assert (pool.window_pool[3][win_slots] == 0xAB).all()
    assert (pool.main_pool[src][main_rows] == 0xCD).all() and (pool.idx_pool[src][main_rows] == 0xEF).all()
    assert torch.equal(pool.state_ring[src].buffer[: (win_slots.max() // 128 + 1) * 2], ring[: (win_slots.max() // 128 + 1) * 2])
    # the new tokens' rows were written
    new_slots = pool.translate_full_to_window(eng.page_table[0, 256:300].long())
    assert not (pool.window_pool[3][new_slots] == 0).all()


def test_bounded_prefix_hit_matches_cold_prefill(checkpoint):
    """A bounded-mode prefix hit followed by a short suffix (the encoder recompute reads two windows of
    the shared history, both live by the cache contract) produces the cold prefill's logits."""
    from .harness import TinyEngine
    from .test_reference_parity import _compare

    eng = TinyEngine(checkpoint, max_seq_len=2048, max_running_req=2, swa_decoder_replay="bounded")
    prefix = _tokens(768, 11)
    donor = eng.new_request(0, prefix + _tokens(20, 12))
    cold = eng.prefill([donor])
    eng.finish_prefill([donor])
    hit = eng.new_request_on_prefix(1, donor, prefix + _tokens(20, 12))  # the same prompt over the shared pages
    assert hit.cached_len == 768
    got = eng.prefill([hit])
    _compare("bounded prefix hit", got, cold)


def test_shared_prefix_replays_do_not_disturb_each_other(checkpoint):
    """Two requests hitting the same 768-token prefix with different suffix lengths: B's replay writes
    only B's private decoder rings and reads the shared encoder history, so A's pending decode sees the
    logits it would have seen without B, and neither the shared pages nor A's rings change."""
    from .harness import TinyEngine

    prefix = _tokens(768, 13)
    a_toks, b_toks = prefix + _tokens(20, 14), prefix + _tokens(40, 15)

    def run(interleave: bool):
        eng = TinyEngine(checkpoint, max_seq_len=2048, max_running_req=2, swa_decoder_replay="bounded")
        a = eng.new_request(0, a_toks)
        eng.prefill([a])
        eng.finish_prefill([a])
        dec_layer, enc_layer = eng.args.decoder_start_layer, 2
        shared_slots = eng.pool.translate_full_to_window(eng.page_table[0, 640:768].long())
        enc_before = eng.pool.window_pool[enc_layer][shared_slots].clone()
        P = eng.args.window_size
        ring_before = eng.pool.window_pool[dec_layer][0 * P : 1 * P].clone()  # A's ring row
        if interleave:
            b = eng.new_request_on_prefix(1, a, b_toks)
            eng.prefill([b])
            eng.finish_prefill([b])
        assert torch.equal(eng.pool.window_pool[enc_layer][shared_slots], enc_before)
        assert torch.equal(eng.pool.window_pool[dec_layer][0 * P : 1 * P], ring_before)
        return eng.decode([a], [7])

    assert torch.equal(run(True), run(False))


def test_commit_dedup_onto_a_longer_prompts_pages_keeps_the_decoder_state(checkpoint):
    """The radix commit may replace a request's freshly written pages with an existing node's (the
    reviewer's donor case): D prefilled 256 tokens, so under bounded replay its decoder never computed
    positions [0, 128); A (exactly D's first 128 tokens) is repointed onto D's pages at commit. A's
    decoder KV lives in A's own ring, so its next decode is unchanged by the repoint (D's shared rows
    came from a longer prefill, so to rounding)."""
    from .harness import TinyEngine
    from .test_reference_parity import _compare

    donor_toks = _tokens(256, 21)
    a_toks = donor_toks[:128]

    def run(repoint: bool):
        eng = TinyEngine(checkpoint, max_seq_len=1024, max_running_req=2, swa_decoder_replay="bounded")
        d = eng.new_request(0, donor_toks)
        eng.prefill([d])
        eng.finish_prefill([d])
        a = eng.new_request(1, a_toks)
        eng.prefill([a])
        eng.finish_prefill([a])
        if repoint:  # what CacheManager.cache_req does when the tree already holds the prefix
            eng.page_table[1, :128] = eng.page_table[0, :128]
        return eng.decode([a], [3])

    _compare("decode after a commit repoint", run(True), run(False))


def test_two_engines_rebind_the_shared_context_on_every_forward(checkpoint):
    """``TinyEngine`` instances share the process-wide ``Context``; each forward must run against its
    own page table, pool and backend even after another engine was built (the model resolves its
    backend through the context, so a stale binding would silently read the other engine's pools)."""
    from freetoken.core import get_global_ctx

    from .harness import TinyEngine

    eng = TinyEngine(checkpoint, max_seq_len=512, max_running_req=1)
    a = eng.new_request(0, _tokens(64, 41))
    before = eng.prefill([a])
    eng2 = TinyEngine(checkpoint, max_seq_len=512, max_running_req=1)
    assert eng2.ctx is eng.ctx and get_global_ctx().attn_backend is eng2.backend
    b = eng2.new_request(0, _tokens(64, 42))
    eng2.prefill([b])
    # a forward on the first engine binds its own resources again and reproduces its result
    eng.finish_prefill([a])
    a2 = eng.new_request(0, _tokens(64, 41))
    assert torch.equal(eng.prefill([a2]), before) and get_global_ctx().attn_backend is eng.backend
    assert get_global_ctx().kv_cache is eng.pool and get_global_ctx().page_table is eng.page_table
