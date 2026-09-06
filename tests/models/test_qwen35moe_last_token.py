"""Qwen3.5 GGUF head must sample final prefill rows, not row zero."""

from types import SimpleNamespace

import torch


def test_qwen35moe_gathers_prefill_last_rows(monkeypatch):
    import freetoken.models.qwen3_5_moe.model as qwen_model

    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    captured = {}

    class FakeModel:
        def forward(self, input_ids):
            assert input_ids.equal(torch.tensor([3, 4, 5], dtype=torch.int32))
            return hidden

    class FakeHead:
        def forward(self, x):
            captured["x"] = x
            return x

    class Metadata:
        def get_last_indices(self, batch_size):
            assert batch_size == 2
            return torch.tensor([1, 4], dtype=torch.long)

    batch = SimpleNamespace(
        input_ids=torch.tensor([3, 4, 5], dtype=torch.int32),
        is_prefill=True,
        size=2,
        attn_metadata=Metadata(),
    )
    model = object.__new__(qwen_model.Qwen3_5MoEForCausalLM)
    model.model = FakeModel()
    model.lm_head = FakeHead()
    monkeypatch.setattr(qwen_model, "get_global_ctx", lambda: SimpleNamespace(batch=batch))

    model.forward()

    assert torch.equal(captured["x"], hidden[[1, 4]])
