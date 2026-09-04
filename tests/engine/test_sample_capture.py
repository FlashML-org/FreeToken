"""Capture-safe sampler contract tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.engine.engine import DeviceTokenChain
from freetoken.engine.sample import BatchSamplingArgs, Sampler


def test_sample_into_reuses_int32_output():
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0], [8.0, 2.0, 1.0, 0.0]])
    out = torch.full((2,), -1, dtype=torch.int32)
    scratch = torch.empty(2, dtype=torch.int64)
    result = sampler.sample_into(logits, BatchSamplingArgs(temperatures=None), None, out, scratch)
    assert result.data_ptr() == out.data_ptr()
    assert result.tolist() == [1, 0]


def test_sample_into_device_preserves_stable_output_address():
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
    out = torch.full((1,), -1, dtype=torch.int32)
    scratch = torch.empty(1, dtype=torch.int64)
    result = sampler.sample_into_device(
        logits, BatchSamplingArgs(temperatures=None), None, out, scratch
    )
    assert result.data_ptr() == out.data_ptr()
    assert result.tolist() == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm device")
def test_device_token_chain_keeps_async_observation_sources_distinct():
    device = torch.device("cuda")
    stream = torch.cuda.Stream(device=device)
    chain = DeviceTokenChain(device, capacity=1)
    with torch.cuda.stream(stream):
        first = chain.publish(torch.tensor([17], device=device), stream)
        _, cpu0, event0 = chain.stage(first, stream)
        second = chain.publish(torch.tensor([23], device=device), stream)
        _, cpu1, event1 = chain.stage(second, stream)
    event0.synchronize()
    event1.synchronize()
    assert first.data_ptr() != second.data_ptr()
    assert cpu0.item() == 17 and cpu1.item() == 23


@pytest.mark.parametrize(
    "args",
    [
        BatchSamplingArgs(torch.ones(1)),
        BatchSamplingArgs(None, apply_penalties=True),
    ],
)
def test_sample_into_rejects_dynamic_modes(args):
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    with pytest.raises(ValueError, match="greedy sampling without penalties"):
        sampler.sample_into(torch.zeros(1, 4), args, None, torch.empty(1, dtype=torch.int32))


def test_sampler_cpu_greedy_penalty_path_has_no_nvtx_dependency():
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    req = SimpleNamespace(
        prompt_len=0,
        input_ids=torch.tensor([1], dtype=torch.int32),
        sampling_params=SamplingParams(presence_penalty=1.0),
    )
    args = BatchSamplingArgs(None, apply_penalties=True)
    assert sampler.sample(torch.zeros(1, 4), args, SimpleNamespace(reqs=[req])).item() == 0
