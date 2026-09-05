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

    # The HIP build always records the reviewed one-row launch shape in the
    # compiler command, even when the caller did not set an experiment knob.
    # Keeping this explicit makes the extension cache key and build evidence
    # unambiguous for the default AMD serving path.
    assert gguf._hip_gguf_cflags() == ["-O3", "-DGGML_CUDA_MMV_Y=1"]
    assert gguf._hip_target_arch() == "gfx1151"
    assert os.environ["PYTORCH_ROCM_ARCH"] == "gfx1151"


def test_hip_gguf_flags_preserve_an_explicit_multi_target_choice(monkeypatch):
    """An explicit multi-target deployment choice is never replaced by auto-detection."""
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1100;gfx1151")

    # An explicit multi-target architecture choice must not remove the default
    # one-row launch definition from the recorded HIP compile flags.
    assert gguf._hip_gguf_cflags() == ["-O3", "-DGGML_CUDA_MMV_Y=1"]
    assert gguf._hip_target_arch() == "gfx1100"
    assert os.environ["PYTORCH_ROCM_ARCH"] == "gfx1100;gfx1151"
