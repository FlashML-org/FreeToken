"""Version pairing between the runtime and the freetoken-kernel-cache wheel.

The comparator lives in freetoken.kernel.utils; this pins the matrix for the stamped
version scheme (runtime `0.1.1+g<sha>`, cache `0.1.1+cu130.g<sha>`) introduced by
scripts/build-release-wheels.sh."""

import os
import time

import pytest

from freetoken.kernel.utils import (
    _kernel_cache_version_ok,
    clear_stale_jit_lock,
    jit_cache_diagnostics,
    jit_cache_identity,
)


@pytest.mark.parametrize(
    ("cache", "runtime", "ok"),
    [
        # Unstamped pairs (the pre-stamp world) behave as before.
        ("0.1.1", "0.1.1", True),
        ("0.1.1+cu130", "0.1.1", True),
        ("0.2.0+cu130", "0.1.1", False),
        ("0.1.10+cu130", "0.1.1", False),  # prefix of the release string, not the release
        # Stamped pairs: same build passes...
        ("0.1.1+cu130.g3f01615c9", "0.1.1+g3f01615c9", True),
        # ...and a runtime/cache pair from two different builds is exactly the
        # mismatch this scheme exists to catch (bare release numbers agree!).
        ("0.1.1+cu130.gffc111e2e", "0.1.1+g3f01615c9", False),
        # One-sided stamps are tolerated (a dev build against a release wheel and
        # vice versa) -- only the release part is compared then.
        ("0.1.1+cu130", "0.1.1+g3f01615c9", True),
        ("0.1.1+cu130.g3f01615c9", "0.1.1", True),
        ("0.1.1", "0.1.1+g3f01615c9", True),
        # A `g...` token must be g+hex to count as a stamp; anything else is an
        # ordinary local segment and stays out of the comparison.
        ("0.1.1+cu130.gabcdefgh", "0.1.1+g3f01615c9", True),
        # The release part must still match even when stamps agree.
        ("0.2.0+cu130.g3f01615c9", "0.1.1+g3f01615c9", False),
        ("0.1.1+rocm.g3f01615c9", "0.1.1+cu130.g3f01615c9", False),
    ],
)
def test_kernel_cache_version_matrix(cache: str, runtime: str, ok: bool) -> None:
    assert _kernel_cache_version_ok(cache, runtime) is ok


def test_jit_cache_identity_changes_with_source_target_and_flags(tmp_path, monkeypatch):
    source = tmp_path / "kernel.cu"
    source.write_text("kernel-v1")
    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx1100")
    first = jit_cache_identity(source_paths=[str(source)], cuda_cflags=["--offload-arch=gfx1100"])

    source.write_text("kernel-v2")
    changed_source = jit_cache_identity(
        source_paths=[str(source)], cuda_cflags=["--offload-arch=gfx1100"]
    )
    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx1151")
    changed_target = jit_cache_identity(
        source_paths=[str(source)], cuda_cflags=["--offload-arch=gfx1151"]
    )
    changed_flags = jit_cache_identity(
        source_paths=[str(source)], cuda_cflags=["--offload-arch=gfx1151", "-DTEST=1"]
    )

    assert first["key"] != changed_source["key"]
    assert changed_source["key"] != changed_target["key"]
    assert changed_target["key"] != changed_flags["key"]
    assert first["source_sha256"] != changed_source["source_sha256"]
    assert changed_source["target"] == ["gfx1100"]
    assert changed_target["target"] == ["gfx1151"]


def test_stale_jit_lock_diagnostics_and_cleanup_are_scoped(tmp_path):
    build_dir = tmp_path / "jit"
    build_dir.mkdir()
    lock = build_dir / "lock"
    lock.write_text("stale")
    old = time.time() - 4 * 3600
    os.utime(lock, (old, old))

    diagnostics = jit_cache_diagnostics("test", str(build_dir))
    assert diagnostics["lock_exists"] is True
    assert diagnostics["stale"] is True
    assert clear_stale_jit_lock("test", str(build_dir)) is True
    assert not lock.exists()
    assert clear_stale_jit_lock("test", str(build_dir)) is False
