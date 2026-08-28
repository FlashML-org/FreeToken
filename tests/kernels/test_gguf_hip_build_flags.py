"""Unit coverage for the HIP-only GGUF extension compiler configuration.

These tests do not invoke hipcc.  They verify the environment that is prepared
before PyTorch's extension builder computes its target-specific cache key.
"""

import os
from types import SimpleNamespace

from freetoken.kernel import gguf


def test_hip_gguf_flags_pin_the_active_gfx_target(monkeypatch):
    """A single-GPU HIP process derives gfx1151 when no target was configured."""
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.delenv("FREETOKEN_HIP_GGUF_FAST_MATH", raising=False)
    monkeypatch.setattr(gguf.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        gguf.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(gcnArchName="gfx1151:sramecc-:xnack-"),
    )

    assert gguf._hip_gguf_cflags() == ["-O3"]
    assert gguf._hip_target_arch() == "gfx1151"
    assert os.environ["PYTORCH_ROCM_ARCH"] == "gfx1151"


def test_hip_gguf_fast_math_is_explicit_and_preserves_user_target(monkeypatch):
    """Fast math is opt-in and an explicit multi-target choice is never replaced."""
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1100;gfx1151")
    monkeypatch.setenv("FREETOKEN_HIP_GGUF_FAST_MATH", "true")

    assert gguf._hip_gguf_cflags() == ["-O3", "-ffast-math"]
    assert gguf._hip_target_arch() == "gfx1100"
    assert os.environ["PYTORCH_ROCM_ARCH"] == "gfx1100;gfx1151"
