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


@pytest.mark.parametrize("cut", [128, 256])
def test_bounded_chunked_prefill_matches_unchunked(checkpoint, cut):
    """A chunked bounded prefill gives each chunk its part of the prompt's last window ``[172, 300)``:
    a first chunk that ends before it (cut 128) runs no decoder row and hands the head a placeholder;
    one that reaches into it (cut 256) leaves rows in the private ring for the last chunk, which is
    shorter than the window (44 tokens). Both match the unchunked prefill."""
    from .harness import TinyEngine
    from .test_reference_parity import _compare

    eng = TinyEngine(checkpoint, max_seq_len=1024, max_running_req=2, swa_decoder_replay="bounded")
    toks = _tokens(300, 6)
    ref = eng.prefill([eng.new_request(0, toks)])
    chunked = eng.new_request(1, toks)
    chunked.device_len = cut
    first = eng.prefill([chunked])
    assert first.shape == (1, VOCAB) and torch.isfinite(first).all()
    chunked.cached_len, chunked.device_len = cut, 300
    _compare(f"bounded chunks cut at {cut}", eng.prefill([chunked]), ref)


def test_bounded_batch_mixes_a_decoderless_chunk_with_a_whole_prompt(checkpoint):
    """One batch: a chunk that ends before its prompt's last window (no decoder row, a placeholder for
    the head) next to a whole prompt; the whole prompt's logits are those of its own prefill."""
    from .harness import TinyEngine
    from .test_reference_parity import _compare

    eng = TinyEngine(checkpoint, max_seq_len=2048, max_running_req=2, swa_decoder_replay="bounded")
    whole = _tokens(300, 8)
    want = eng.prefill([eng.new_request(0, whole)])
    chunk = eng.new_request(0, _tokens(600, 9))
    chunk.device_len = 256
    other = eng.new_request(1, whole)
    got = eng.prefill([chunk, other])
    assert got.shape == (2, VOCAB) and torch.isfinite(got).all()
    _compare("whole prompt next to a decoderless chunk", got[1:], want)


def test_bounded_prefix_hit_matches_cold_prefill(checkpoint):
    """A prefix hit that would leave fewer than a window of new tokens (768 cached + 20 new) is capped
    one window before the prompt end, so the request prefills ``[640, 788)`` -- the decoder's whole
    window among it -- and produces the cold prefill's logits."""
    from .harness import TinyEngine
    from .test_reference_parity import _compare

    eng = TinyEngine(checkpoint, max_seq_len=2048, max_running_req=2, swa_decoder_replay="bounded")
    prefix = _tokens(768, 11)
    donor = eng.new_request(0, prefix + _tokens(20, 12))
    cold = eng.prefill([donor])
    eng.finish_prefill([donor])
    hit = eng.new_request_on_prefix(1, donor, prefix + _tokens(20, 12))  # the same prompt over the shared pages
    assert hit.cached_len == 640 and hit.extend_len == 148
    got = eng.prefill([hit])
    _compare("bounded prefix hit", got, cold)


def test_shared_prefix_replays_do_not_disturb_each_other(checkpoint):
    """Two requests sharing a 768-token prefix with different suffix lengths: B's hit is capped at 640
    (a window before its prompt end), its decoder writes only B's private ring and it reads the shared
    encoder history, so A's pending decode sees the logits it would have seen without B, and neither
    the shared pages nor A's ring change."""
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
