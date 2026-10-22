"""Checks of the eager reference only. These do not execute the Triton kernel."""
from pathlib import Path
import runpy

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
reference = runpy.run_path(str(ROOT / 'FreeToken/tests/kernels/test_sigmoid_router.py'))['reference']


def test_selection_bias_is_not_output_weight():
    scores = torch.tensor([[0.8, 0.6, 0.3]])
    weights, ids = reference(torch.logit(scores), torch.tensor([-0.4, 0., 0.5]), 2, renormalize=True)
    assert ids.tolist() == [[2, 1]]
    torch.testing.assert_close(weights, torch.tensor([[1/3, 2/3]]))


def test_group_score_uses_two_best_experts():
    weights, ids = reference(torch.zeros(1, 6), torch.tensor([8., 0., 5., 4., 1., 0.]), 2,
                             renormalize=False, n_group=3, topk_group=1)
    assert ids.tolist() == [[2, 3]]
    torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5]]))


@pytest.mark.parametrize('normalize', [True, False])
def test_scaling(normalize):
    scores = torch.tensor([[0.2, 0.4, 0.7]])
    weights, ids = reference(torch.logit(scores), torch.zeros(3), 2,
                             renormalize=normalize, routed_scale=2.5)
    assert ids.tolist() == [[2, 1]]
    expected = torch.tensor([[0.7, 0.4]])
    if normalize:
        expected /= 1.1
    torch.testing.assert_close(weights, expected * 2.5)


@pytest.mark.parametrize('eps', [0.0, 1e-20])
def test_zero_sum_contract(eps):
    weights, _ = reference(torch.full((1, 4), -1000.), torch.arange(4).float(), 2,
                           renormalize=True, eps=eps)
    assert torch.isnan(weights).all() if eps == 0 else torch.equal(weights, torch.zeros_like(weights))


def test_empty_grouped_input():
    weights, ids = reference(torch.empty(0, 90), torch.zeros(90), 5,
                             renormalize=True, n_group=3, topk_group=2)
    assert weights.shape == ids.shape == (0, 5)


def test_strided_input():
    torch.manual_seed(42)
    x = torch.randn(7, 180)[:, ::2]
    b = torch.randn(180)[::2]
    actual = reference(x, b, 5, renormalize=True, n_group=3, topk_group=2)
    expected = reference(x.contiguous(), b.contiguous(), 5, renormalize=True, n_group=3, topk_group=2)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
