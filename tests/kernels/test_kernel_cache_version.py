"""Version pairing between the runtime and the freetoken-kernel-cache wheel.

The comparator lives in freetoken.kernel.utils; this pins the matrix for the stamped
version scheme (runtime `0.1.1+g<sha>`, cache `0.1.1+cu130.g<sha>`) introduced by
scripts/build-release-wheels.sh."""

from pathlib import Path

import pytest

from freetoken.kernel.utils import _kernel_cache_version_ok, _prebuilt_library_path


@pytest.mark.parametrize(
    ("platform", "suffix"),
    [("win32", ".dll"), ("linux", ".so")],
)
def test_prebuilt_library_path_uses_platform_suffix(platform: str, suffix: str) -> None:
    name = "freetoken__store_1024_128_1_false"
    assert _prebuilt_library_path(Path("cache"), name, platform=platform) == (
        Path("cache") / name / f"{name}{suffix}"
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
    ],
)
def test_kernel_cache_version_matrix(cache: str, runtime: str, ok: bool) -> None:
    assert _kernel_cache_version_ok(cache, runtime) is ok
