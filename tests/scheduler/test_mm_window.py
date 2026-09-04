"""Chunked prefill of an image prompt: each chunk scatters only the placeholder rows it holds."""

from types import SimpleNamespace

import torch

from freetoken.scheduler.scheduler import _mm_embeds_window

IMG = 7


def test_window_rows_follow_the_placeholders_across_chunks():
    ids = torch.tensor([1, 2, IMG, IMG, IMG, 3, IMG, IMG, 4, 5])
    embeds = (
        torch.arange(5).unsqueeze(1).float()
    )  # one row per placeholder, prompt order
    chunks = [
        (0, 4),
        (4, 8),
        (8, 10),
    ]  # a cut inside the first image run, one after the second
    got = []
    for start, end in chunks:
        req = SimpleNamespace(input_ids=ids[:end], cached_len=start, mm_embeds=embeds)
        got.append(_mm_embeds_window(req, IMG))
    assert [g.shape[0] for g in got] == [2, 3, 0]
    assert torch.equal(torch.cat(got), embeds)
    whole = SimpleNamespace(input_ids=ids, cached_len=0, mm_embeds=embeds)
    assert torch.equal(_mm_embeds_window(whole, IMG), embeds)
