"""``e4m3_native()`` must not take the CUDA capability path on ROCm.

``torch.cuda.get_device_capability()`` still answers under HIP, but the tuple it returns
is a gfx number, not a CUDA compute capability: gfx1101 reports (11, 0) and gfx1201
reports higher still. Compared against the sm_89 fp8 boundary those are all ``>= (8, 9)``,
so the plain capability check silently selects the native fp8e4nv path on hardware that
has no such type -- every fp8 wrapper then passes fp8 pointers into kernels that cannot
take them, instead of the uint8 view + bf16 buffers the emulation path needs.

The guard for this lives in PR #132 and was absent from the RDNA4 integration branch,
which is how it went unnoticed: the existing e4m3 suite covers the emulation numerics
but never the dispatch decision, and on a CUDA box the ROCm branch is unreachable.

CPU-only.
"""

from __future__ import annotations

import pytest
import torch


def _fresh_native(monkeypatch, *, hip, capability=(9, 0), device_count=1) -> bool:
    """Re-evaluate ``e4m3_native()`` with the module cache cleared."""
    from freetoken.kernel.triton import e4m3_compat

    monkeypatch.setattr(e4m3_compat, "_native", None)
    monkeypatch.setattr(e4m3_compat, "FORCE_EMU", False)
    monkeypatch.setattr(e4m3_compat, "_env_force", lambda: False)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *a, **k: capability
    )
    return e4m3_compat.e4m3_native()


@pytest.mark.parametrize(
    "capability", [(11, 0), (12, 0), (9, 4)], ids=["gfx1101", "gfx1201-like", "gfx94x-like"]
)
def test_rocm_never_reports_native_e4m3(monkeypatch, capability):
    """Whatever gfx number HIP reports, the answer is False -- these tuples all clear
    the sm_89 threshold numerically, which is exactly the trap."""
    assert capability >= (8, 9), "test would be vacuous below the fp8 boundary"
    assert _fresh_native(monkeypatch, hip="7.14.60850", capability=capability) is False


def test_cuda_still_uses_the_capability_check(monkeypatch):
    """The ROCm guard must not disturb the NVIDIA path."""
    assert _fresh_native(monkeypatch, hip=None, capability=(9, 0)) is True
    assert _fresh_native(monkeypatch, hip=None, capability=(8, 6)) is False


def test_forced_emulation_still_wins_on_rocm(monkeypatch):
    """FORCE_EMU is checked first and stays authoritative."""
    from freetoken.kernel.triton import e4m3_compat

    monkeypatch.setattr(e4m3_compat, "_native", None)
    monkeypatch.setattr(e4m3_compat, "FORCE_EMU", True)
    monkeypatch.setattr(e4m3_compat, "_env_force", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "7.14.60850", raising=False)
    assert e4m3_compat.e4m3_native() is False
