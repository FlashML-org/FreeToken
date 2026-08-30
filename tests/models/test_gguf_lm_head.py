"""Contract tests for the untied native-GGUF language-model head.

The head must slice hidden states before the vocabulary projection during prefill. With a
248k-token vocabulary, projecting every prompt position can allocate hundreds of MiB of
logits that are immediately discarded. Decode, in contrast, must preserve every row.
"""

from types import SimpleNamespace

import torch

from freetoken.layers.gguf import GGUFLMHead, GGUFLinear
from freetoken.models.gguf.dequant import GGML_F32


class _Metadata:
    def __init__(self, last_indices: list[int]):
        self.last_indices = torch.tensor(last_indices, dtype=torch.long)
        self.requested_batch_size = None

    def get_last_indices(self, batch_size: int) -> torch.Tensor:
        self.requested_batch_size = batch_size
        return self.last_indices


def _capture_linear_input(monkeypatch):
    seen = {}

    def fake_forward(self, x):
        seen["x"] = x.clone()
        return x

    monkeypatch.setattr(GGUFLinear, "forward", fake_forward)
    return seen


def test_prefill_projects_only_each_sequence_last_position(monkeypatch):
    import freetoken.core

    metadata = _Metadata([2, 4])
    batch = SimpleNamespace(is_prefill=True, size=2, attn_metadata=metadata)
    monkeypatch.setattr(freetoken.core, "get_global_ctx", lambda: SimpleNamespace(batch=batch))
    seen = _capture_linear_input(monkeypatch)

    head = GGUFLMHead(in_features=4, out_features=8, quant_type=GGML_F32)
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    output = head.forward(hidden)

    expected = hidden[[2, 4]].contiguous()
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(seen["x"], expected)
    assert metadata.requested_batch_size == 2
    assert seen["x"].is_contiguous()


def test_decode_projects_all_rows(monkeypatch):
    import freetoken.core

    metadata = _Metadata([0])
    batch = SimpleNamespace(is_prefill=False, size=1, attn_metadata=metadata)
    monkeypatch.setattr(freetoken.core, "get_global_ctx", lambda: SimpleNamespace(batch=batch))
    seen = _capture_linear_input(monkeypatch)

    head = GGUFLMHead(in_features=4, out_features=8, quant_type=GGML_F32)
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    output = head.forward(hidden)

    torch.testing.assert_close(output, hidden)
    torch.testing.assert_close(seen["x"], hidden)
    assert metadata.requested_batch_size is None
