"""Sampling-history contracts used by GGUF/Qwen serving."""

import math

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.sample import Sampler, apply_penalties


@pytest.mark.parametrize("field", ["presence_penalty", "frequency_penalty"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 2.01, -2.01])
def test_sampling_params_reject_invalid_penalties(field, value):
    with pytest.raises(ValueError, match=field):
        SamplingParams(**{field: value})


def _req(params: SamplingParams, prompt=(10, 11, 10)) -> Req:
    return Req(
        input_ids=torch.tensor(prompt, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=8,
        uid=1,
        sampling_params=params,
        cache_handle=None,
    )


def test_req_tracks_prompt_boundary_and_penalties_ignore_prompt_tokens():
    req = _req(SamplingParams(presence_penalty=0.5, frequency_penalty=0.25))
    req.append_host(torch.tensor([10, 12, 12], dtype=torch.int32))
    logits = torch.zeros((1, 16), dtype=torch.float32)

    apply_penalties(logits, [req])

    # Token 10 occurs once in generated history: prompt occurrences do not count.
    assert req.prompt_len == 3
    assert logits[0, 10].item() == -0.75
    # Token 12 occurs twice in generated history.
    assert logits[0, 12].item() == -1.0
    # Prompt-only token 11 remains untouched.
    assert logits[0, 11].item() == 0.0


def test_sampler_keeps_greedy_path_when_penalties_are_disabled():
    req = _req(SamplingParams())
    args = Sampler(device=torch.device("cpu"), vocab_size=16).prepare(
        Batch([req], "decode")
    )
    assert args.temperatures is None
    assert not args.apply_penalties


def test_sampler_marks_penalized_greedy_batch(monkeypatch):
    # CPU test avoids allocating pinned host memory; GPU path uses same helper.
    monkeypatch.setattr(
        "freetoken.engine.sample.make_device_tensor",
        lambda data, dtype, device: torch.tensor(data, dtype=dtype, device=device),
    )
    req = _req(SamplingParams(presence_penalty=0.1))
    args = Sampler(device=torch.device("cpu"), vocab_size=16).prepare(
        Batch([req], "decode")
    )
    assert args.temperatures is not None
    assert args.apply_penalties
