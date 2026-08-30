"""Regression coverage for Qwen GGUF's serialized Gated DeltaNet decay."""

import pytest
import torch

from freetoken.models.qwen3_5_moe.gguf import (
    _restore_gdn_value_head_input_blocks,
    _restore_gdn_value_head_order,
    _restore_gdn_value_head_rows,
    _ssm_a_to_a_log,
)


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


def test_qwen_gguf_gdn_value_head_rows_restore_complete_quantized_rows():
    """A grouped Q8 projection V suffix regains Qwen's per-key-head order."""
    grouped = torch.tensor(
        [[0, 0], [2, 2], [4, 4], [6, 6], [1, 1], [3, 3], [5, 5], [7, 7]],
        dtype=torch.uint8,
    )

    restored = _restore_gdn_value_head_rows(grouped, num_key_heads=4, head_dim=2)

    expected = torch.tensor(
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]],
        dtype=torch.uint8,
    )
    torch.testing.assert_close(restored, expected)


def test_qwen_gguf_gdn_output_restores_q8_blocks_without_dequantizing():
    """Q8_0 blocks move intact when restoring GDN output-projection columns."""
    grouped_order = (0, 2, 4, 6, 1, 3, 5, 7)
    grouped = torch.stack(
        [
            torch.cat([torch.full((34,), head, dtype=torch.uint8) for head in grouped_order]),
            torch.cat([torch.full((34,), head + 20, dtype=torch.uint8) for head in grouped_order]),
        ]
    )

    restored = _restore_gdn_value_head_input_blocks(grouped, num_key_heads=4, head_dim=32)

    expected = torch.stack(
        [
            torch.cat([torch.full((34,), head, dtype=torch.uint8) for head in range(8)]),
            torch.cat([torch.full((34,), head + 20, dtype=torch.uint8) for head in range(8)]),
        ]
    )
    torch.testing.assert_close(restored, expected)
