"""Sampler semantic and preparation-cache gates."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.engine.sample import Sampler, apply_penalties


def test_penalties_ignore_prompt_and_count_generated_tokens():
    logits = torch.zeros(1, 8)
    req = SimpleNamespace(
        prompt_len=3,
        input_ids=torch.tensor([1, 2, 3, 4, 4, 6], dtype=torch.int32),
        sampling_params=SamplingParams(presence_penalty=1.0, frequency_penalty=0.5),
    )
    apply_penalties(logits, [req])
    assert logits[0, 1].item() == 0.0  # prompt token is excluded
    assert logits[0, 4].item() == -2.0  # presence + two generated occurrences
    assert logits[0, 6].item() == -1.5


def test_greedy_without_penalty_has_no_sampling_tensors():
    sampler = Sampler(torch.device("cpu"), vocab_size=8)
    req = SimpleNamespace(sampling_params=SamplingParams())
    args = sampler.prepare(SimpleNamespace(reqs=[req]))
    assert args.temperatures is None
    assert args.top_k is None and args.top_p is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm device")
def test_sampling_args_reused_for_stable_request_semantics():
    sampler = Sampler(torch.device("cuda"), vocab_size=8)
    params = SamplingParams(temperature=0.7, top_k=4, top_p=0.95)
    req = SimpleNamespace(sampling_params=params)
    first = sampler.prepare(SimpleNamespace(reqs=[req]))
    second = sampler.prepare(SimpleNamespace(reqs=[req]))
    assert first is second
    assert first.temperatures is not None
    assert first.top_k is not None and first.top_p is not None
