from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.scheduler.scheduler import _make_rope_positions


def _req(start, end, positions=None, delta=0):
    return SimpleNamespace(
        cached_len=start,
        device_len=end,
        extend_len=end - start,
        mrope_position_ids=positions,
        mrope_position_delta=delta,
    )


def test_scheduler_packs_prompt_mrope_and_plain_text_positions():
    prompt = torch.tensor(
        [[0, 1, 2, 2, 2, 3], [0, 1, 2, 2, 3, 3], [0, 1, 2, 3, 2, 3]]
    )
    batch = SimpleNamespace(
        padded_reqs=[_req(2, 6, prompt, -2), _req(0, 2)]
    )
    actual = _make_rope_positions(batch, torch.device("cpu"))
    expected = torch.cat((prompt[:, 2:6], torch.tensor([[0, 1], [0, 1], [0, 1]])), dim=1)
    assert torch.equal(actual, expected)


def test_scheduler_uses_mrope_delta_for_generated_tokens():
    prompt = torch.arange(18, dtype=torch.int64).view(3, 6)
    batch = SimpleNamespace(padded_reqs=[_req(6, 8, prompt, -2)])
    actual = _make_rope_positions(batch, torch.device("cpu"))
    assert torch.equal(actual, torch.tensor([[4, 5], [4, 5], [4, 5]]))


def test_scheduler_skips_rope_tensor_for_text_only_batch():
    batch = SimpleNamespace(padded_reqs=[_req(3, 5)])
    assert _make_rope_positions(batch, torch.device("cpu")) is None
