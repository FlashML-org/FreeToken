"""Unit tests for explicit HIP GGUF build-shape selection.

These tests cover the host-side validation that protects the native GGUF
extension build.  They do not compile a HIP module, allocate a GPU tensor, or
need a model checkpoint.  Device execution and output parity belong to the
separate isolated candidate gate because they require the target AMD GPU.
"""

from __future__ import annotations

import pytest

from freetoken.kernel import gguf


def test_hip_gguf_flags_keep_the_one_row_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve the reviewed one-row MMV launch when no experiment is selected."""

    # Remove both knobs so the helper cannot inherit a developer-shell setting.
    monkeypatch.delenv("FREETOKEN_GGUF_MMV_Y", raising=False)
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    # Stub target discovery to keep this test independent of local GPU access.
    monkeypatch.setattr(gguf, "_hip_target_arch", lambda: "gfx1151")

    flags = gguf._hip_gguf_cflags()

    # The default must remain explicit in the compile command and target only
    # the active AMD architecture discovered by the helper.
    assert flags == ["-O3", "-DGGML_CUDA_MMV_Y=1"]
    assert gguf.os.environ["PYTORCH_ROCM_ARCH"] == "gfx1151"


def test_hip_gguf_flags_allow_the_reviewed_row_grouping_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow two, four, and eight rows while retaining the explicit default and reject path."""

    monkeypatch.setenv("FREETOKEN_GGUF_MMV_Y", "2")
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1151")

    assert gguf._hip_gguf_cflags() == ["-O3", "-DGGML_CUDA_MMV_Y=2"]

    # Four rows are a separately qualified RDNA4 experiment.  This assertion
    # proves the requested compile-time shape becomes part of the extension key.
    monkeypatch.setenv("FREETOKEN_GGUF_MMV_Y", "4")

    assert gguf._hip_gguf_cflags() == ["-O3", "-DGGML_CUDA_MMV_Y=4"]

    # Preserve the prior eight-row RDNA4 screen while adding the intermediate
    # geometry, so this candidate branch does not narrow test coverage.
    monkeypatch.setenv("FREETOKEN_GGUF_MMV_Y", "8")

    assert gguf._hip_gguf_cflags() == ["-O3", "-DGGML_CUDA_MMV_Y=8"]


def test_hip_gguf_flags_reject_an_unreviewed_row_grouping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed rather than compiling an arbitrary MMV workgroup shape."""

    monkeypatch.setenv("FREETOKEN_GGUF_MMV_Y", "3")

    with pytest.raises(RuntimeError, match="FREETOKEN_GGUF_MMV_Y must be 1, 2, 4, or 8"):
        gguf._hip_gguf_cflags()
