"""EncoderCache ownership: rows claimed at admission, freed when the last claim is gathered."""

from __future__ import annotations

import torch

from freetoken.mm.encoder_cache import EncoderCache

CPU = torch.device("cpu")


def test_entry_lives_until_every_claim_is_gathered():
    cache = EncoderCache(storage="cpu")
    emb = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    cache.register(7, 1, 6)
    cache.register(7, 2, 6)
    assert not cache.has(7)  # claimed, not encoded yet
    cache.put(7, emb)
    assert cache.has(7)
    assert torch.equal(cache.get_slice(7, 2, 4, CPU), emb[2:4])
    cache.consume(7, 1, 4)
    cache.consume(7, 1, 2)
    assert cache.has(7)  # request 2 still has rows to gather
    cache.consume(7, 2, 6)
    assert not cache.has(7) and cache.stats() == (0, 0)


def test_repeated_image_in_one_request_adds_up():
    cache = EncoderCache(storage="cpu")
    cache.register(7, 1, 4)
    cache.register(7, 1, 4)
    cache.put(7, torch.ones(4, 2))
    cache.consume(7, 1, 4)  # the first occurrence is done
    assert cache.has(7)
    cache.consume(7, 1, 4)
    assert not cache.has(7)


def test_release_drops_a_claim_even_before_the_tensor_arrives():
    cache = EncoderCache(storage="cpu")
    cache.register(5, 1, 3)
    cache.register(5, 2, 3)
    cache.release(1, [5, 99])  # unknown hash: no-op
    assert 5 in cache._entries
    cache.release(2, [5])
    assert cache._entries == {} and cache.stats() == (0, 0)
