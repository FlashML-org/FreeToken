"""Engine._run_mm_encoder against a fake model: chunked gathers, a shared image, precomputed embeddings, orphan jobs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.engine import Engine
from freetoken.message.backend import MMItem
from freetoken.mm.encoder_cache import EncoderCache

H = 8


class _FakeVision:
    def __init__(self):
        self.calls = 0

    def encode(self, item: MMItem) -> torch.Tensor:
        self.calls += 1
        return torch.full((item.num_tokens, H), float(item.hash))


def _engine(cache: EncoderCache, model=None) -> SimpleNamespace:
    return SimpleNamespace(
        encoder_cache=cache, device=torch.device("cpu"), dtype=torch.float32,
        model=model or _FakeVision(),
    )


def _item(h: int, n_tokens: int, precomputed: torch.Tensor | None = None) -> MMItem:
    feature = None if precomputed is not None else torch.zeros(1)
    return MMItem(
        modality="image", hash=h, pad_value=0, offsets=[[0, n_tokens]],
        feature=feature, precomputed_embeddings=precomputed,
    )


def _batch(jobs, plan) -> SimpleNamespace:
    return SimpleNamespace(mm_encoder_jobs=jobs, mm_gather_plan=plan, mm_embeds=None, mm_rows=None)


def test_entry_lives_until_its_consumer_gathers_the_last_row():
    cache = EncoderCache(storage="cpu")
    eng = _engine(cache)
    item = _item(h=3, n_tokens=4)
    cache.register(3, 7, 4)  # admission claims the whole image
    # chunk 1 consumes rows [0, 2) of a 4-token image
    Engine._run_mm_encoder(eng, _batch([item], [(7, 3, 0, 2, 4, 0)]))
    assert cache._entries[3].remaining == {7: 2}
    assert item.feature is None
    # chunk 2 finishes the image; the entry dies with its last claim
    Engine._run_mm_encoder(eng, _batch([], [(7, 3, 2, 4, 4, 0)]))
    assert not cache.has(3)
    assert eng.model.calls == 1


def test_shared_image_is_encoded_once_and_sliced_per_request():
    cache = EncoderCache(storage="cpu")
    eng = _engine(cache)
    a, b = _item(h=5, n_tokens=4), _item(h=5, n_tokens=4)
    cache.register(5, 1, 4)
    cache.register(5, 2, 4)
    batch = _batch([a, b], [(1, 5, 0, 4, 4, 0), (2, 5, 0, 2, 4, 4)])
    Engine._run_mm_encoder(eng, batch)
    assert eng.model.calls == 1
    assert batch.mm_embeds.shape == (6, H)
    assert torch.equal(batch.mm_embeds, torch.full((6, H), 5.0))
    # request 1 consumed its image; request 2 still has rows to gather in its next chunk
    assert cache._entries[5].remaining == {2: 2}


def test_precomputed_embeddings_bypass_the_encoder():
    cache = EncoderCache(storage="cpu")

    class _NoVision:
        def encode(self, item):
            raise AssertionError("encoder must not run for precomputed embeddings")

    eng = _engine(cache, model=_NoVision())
    emb = torch.arange(2 * H, dtype=torch.float32).view(2, H)
    item = _item(h=4, n_tokens=2, precomputed=emb)
    cache.register(4, 1, 2)
    batch = _batch([item], [(1, 4, 0, 2, 2, 0)])
    Engine._run_mm_encoder(eng, batch)
    assert torch.equal(batch.mm_embeds, emb)
    assert item.precomputed_embeddings is None
    assert not cache.has(4)


def test_job_without_gather_row_fails_loudly():
    eng = _engine(EncoderCache(storage="cpu"))
    with pytest.raises(AssertionError, match="encoder jobs without gather rows"):
        Engine._run_mm_encoder(eng, _batch([_item(h=1, n_tokens=2)], []))
