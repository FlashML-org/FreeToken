"""Sampling bounds must preserve full-vocabulary and mixed-batch semantics."""

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.sample import Sampler


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA or ROCm GPU is required"
)


def _batch(params):
    return Batch(reqs=[
        Req(input_ids=torch.tensor([0], dtype=torch.int32), table_idx=i,
            cached_len=0, output_len=4, uid=i, sampling_params=p, cache_handle=None)
        for i, p in enumerate(params)
    ], phase="decode")


@pytest.mark.parametrize("top_k", [-1, 0, 17, 18, 2**31 - 1, 2**31, 2**80])
@pytest.mark.parametrize("top_p", [1.0, 0.8])
def test_full_vocabulary_top_k_disables_filter(top_k, top_p):
    """Keeping every entry disables top-k without changing temperature or top-p."""
    sampler = Sampler(torch.device("cuda"), vocab_size=17)
    args = sampler.prepare(_batch([SamplingParams(temperature=0.7, top_k=top_k, top_p=top_p)]))
    assert args.top_k is None
    assert args.temperatures.tolist() == pytest.approx([0.7])
    if top_p == 1:
        assert args.top_p is None
    else:
        assert args.top_p.tolist() == pytest.approx([top_p])
    # One finite logit gives a deterministic oracle without depending on RNG state.
    logits = torch.full((1, 17), -float("inf"), device="cuda")
    logits[0, 8] = 0
    assert sampler.sample(logits, args).item() == 8


@pytest.mark.parametrize("top_k", [2**31, 2**80])
def test_large_top_k_preserves_other_rows_in_mixed_batch(top_k):
    """An unfiltered row must not change another row's top-k, top-p or greedy choice."""
    sampler = Sampler(torch.device("cuda"), vocab_size=17)
    args = sampler.prepare(_batch([
        SamplingParams(temperature=0.7, top_k=top_k),
        SamplingParams(temperature=0.7, top_k=3, top_p=0.8),
        SamplingParams(temperature=0, top_k=top_k),
        SamplingParams(temperature=0.7, top_k=1),
    ]))
    assert args.top_k.tolist() == [17, 3, 17, 17]
    assert args.top_p.tolist() == pytest.approx([1.0, 0.8, 1.0, 1.0])
    assert args.greedy_mask.tolist() == [False, False, True, True]
    logits = torch.arange(68, device="cuda", dtype=torch.float32).reshape(4, 17)
    logits[2:, 0] = logits[2:, -1]  # Greedy ties must retain argmax's first-index rule.
    tokens = sampler.sample(logits, args)
    assert 0 <= tokens[0].item() < 17
    assert tokens[1].item() in torch.topk(logits[1], 3).indices.tolist()
    torch.testing.assert_close(tokens[2:].long(), logits[2:].argmax(-1))
