"""Regression coverage for Qwen GGUF's serialized Gated DeltaNet decay."""

import pytest
import torch

from freetoken.models.qwen3_5_moe.gguf import _restore_gdn_value_head_order, _ssm_a_to_a_log


def test_qwen_gguf_ssm_a_inverts_llama_cpp_negative_exponential():
    """The loader recovers FreeToken's A_log rather than exponentiating twice."""
    a_log = torch.tensor([-3.0, -1.25, 0.0, 2.5], dtype=torch.float32)
    serialized = -torch.exp(a_log)

    recovered = _ssm_a_to_a_log(serialized)

    torch.testing.assert_close(recovered, a_log)


@pytest.mark.parametrize(
    "invalid",
    [torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([float("nan")])],
)
def test_qwen_gguf_ssm_a_rejects_values_that_are_not_negative_finite_decay(invalid):
    """Malformed decay coefficients cannot silently corrupt recurrent execution."""
    with pytest.raises(ValueError, match="finite negative"):
        _ssm_a_to_a_log(invalid)


def test_qwen_gguf_gdn_value_heads_restore_grouped_llama_cpp_order():
    """Two GGUF groups become consecutive per-key-head values in FreeToken."""
    grouped = torch.tensor([[0, 1], [10, 11], [20, 21], [30, 31]])

    restored = _restore_gdn_value_head_order(grouped, num_key_heads=2)

    torch.testing.assert_close(restored, torch.tensor([[0, 1], [20, 21], [10, 11], [30, 31]]))


def test_qwen_gguf_gdn_value_heads_reject_invalid_key_head_partition():
    """A malformed GGUF head layout fails before it reaches the recurrent kernel."""
    with pytest.raises(ValueError, match="incompatible"):
        _restore_gdn_value_head_order(torch.zeros(3), num_key_heads=2)
