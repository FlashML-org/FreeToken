from __future__ import annotations

import pytest
import torch


def test_windows_index_fallback_matches_index_select(monkeypatch):
    from freetoken.kernel import index

    index._TORCH_FALLBACK_KEYS.clear()
    monkeypatch.setattr(index.sys, "platform", "win32")
    monkeypatch.setattr(
        index,
        "_jit_index_module",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no matching kernel")),
    )
    weights = torch.arange(30, dtype=torch.float32).view(10, 3)
    indices = torch.tensor([7, 2], dtype=torch.int32)

    with pytest.warns(RuntimeWarning, match="torch.index_select"):
        actual = index.indexing(weights, indices)

    torch.testing.assert_close(actual, weights[[7, 2]])
    assert (weights.shape[1] * weights.element_size(), 1) in index._TORCH_FALLBACK_KEYS


def test_windows_index_fallback_masks_remote_vocab_rows(monkeypatch):
    from freetoken.kernel import index

    index._TORCH_FALLBACK_KEYS.clear()
    monkeypatch.setattr(index.sys, "platform", "win32")
    monkeypatch.setattr(
        index,
        "_jit_index_module",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no matching kernel")),
    )
    weights = torch.arange(12, dtype=torch.float32).view(4, 3)
    indices = torch.tensor([4, 6, 9], dtype=torch.int64)

    with pytest.warns(RuntimeWarning):
        actual = index.indexing(weights, indices, vocab_range=(4, 4))

    torch.testing.assert_close(actual[0], weights[0])
    torch.testing.assert_close(actual[1], weights[2])
    torch.testing.assert_close(actual[2], torch.zeros(3))
