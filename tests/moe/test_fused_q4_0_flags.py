"""Contract tests for opt-in Q4_0 MoE HIP investigation flags.

These unit tests intentionally do not allocate a GPU tensor or invoke hipcc.
They protect the important public guarantee that FP32 intermediates are an
explicit experiment and cannot be enabled by an arbitrary environment value.
"""

from freetoken.moe import fused_q4_0


def test_q4_fp32_intermediate_is_disabled_without_the_exact_opt_in(monkeypatch):
    """Absent and loose truthy values preserve the dtype-stable shipping path."""
    monkeypatch.delenv("FREETOKEN_GGUF_MOE_FP32_INTERMEDIATE", raising=False)
    assert fused_q4_0._use_fp32_intermediate() is False

    monkeypatch.setenv("FREETOKEN_GGUF_MOE_FP32_INTERMEDIATE", "true")
    assert fused_q4_0._use_fp32_intermediate() is False


def test_q4_fp32_intermediate_requires_exact_one(monkeypatch):
    """Only the documented value enables the temporary FP32 HIP experiment."""
    monkeypatch.setenv("FREETOKEN_GGUF_MOE_FP32_INTERMEDIATE", "1")
    assert fused_q4_0._use_fp32_intermediate() is True
