import pytest

from freetoken.kernel.triton import e4m3_compat


PRE_FP8_ARCH = 80
NATIVE_FP8_ARCH = 89


def _patch_arch(monkeypatch, arch_box):
    """Point both capability sources at ``arch_box['arch']`` so the test exercises
    the real decision regardless of which source the implementation reads: the
    triton active target (``target_info.cuda_capability_geq``, used by the host
    twin and its compile-time twin) and ``torch.cuda.get_device_capability``."""
    monkeypatch.setattr(
        e4m3_compat.target_info,
        "cuda_capability_geq",
        lambda major, minor=0: arch_box["arch"] >= major * 10 + minor,
    )
    monkeypatch.setattr(
        e4m3_compat.torch.cuda,
        "get_device_capability",
        lambda device=None: (arch_box["arch"] // 10, arch_box["arch"] % 10),
    )


@pytest.mark.parametrize(
    ("arch", "expected"),
    [(PRE_FP8_ARCH, False), (NATIVE_FP8_ARCH, True)],
)
def test_e4m3_native_matches_active_target(monkeypatch, arch, expected):
    _patch_arch(monkeypatch, {"arch": arch})
    assert e4m3_compat.e4m3_native() is expected


def test_e4m3_native_tracks_worker_device_not_first_call(monkeypatch):
    """Regression: the host convention must follow the device the kernel actually
    compiles for, matching :func:`e4m3_native_cx`. On a heterogeneous host the
    first ``e4m3_native()`` call can happen before the worker binds its GPU (or
    while the non-native default device is current); a memoized snapshot taken
    then would pin a stale convention and force ``e4m3_kernel_view`` /
    ``e4m3_act_dtype`` to the wrong representation for the compiled kernel."""
    arch_box = {"arch": PRE_FP8_ARCH}
    _patch_arch(monkeypatch, arch_box)

    # First host decision, taken before the worker binds its native GPU.
    assert e4m3_compat.e4m3_native() is False

    # Worker binds its actual compute device (native fp8).
    arch_box["arch"] = NATIVE_FP8_ARCH

    # The host twin must now agree with the active compile target, not a stale
    # first-call cache.
    assert e4m3_compat.e4m3_native() is True


def test_e4m3_native_force_emu_short_circuits(monkeypatch):
    """FREETOKEN_FORCE_E4M3_EMU pins the emulated convention regardless of the
    device capability."""
    monkeypatch.setattr(e4m3_compat, "FORCE_EMU", True)
    monkeypatch.setattr(e4m3_compat, "_env_force", lambda: True)
    monkeypatch.setattr(
        e4m3_compat.target_info,
        "cuda_capability_geq",
        lambda major, minor=0: True,
    )
    assert e4m3_compat.e4m3_native() is False
