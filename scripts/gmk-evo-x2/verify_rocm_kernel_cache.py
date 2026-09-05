#!/usr/bin/env python3
"""Prove that a GMKtec EVO-X2 FreeToken C++ and HIP cache resolves without JIT.

The cache builder records successful compilation, but a file count alone cannot
prove that every shared object is loadable by the current Python, TVM FFI, ROCm
and FreeToken combination. This verifier sets the same strict environment used
by the isolated Qwen launcher, then asks every explicit AOT specification to
load itself. A missing object or ABI mismatch fails immediately because runtime
compilation remains disabled throughout the check.

This utility never starts a server, loads a model checkpoint, mutates a cache,
or contacts any non-GMKtec EVO-X2 endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Read the exact read-only cache directory to validate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Load every catalog module under strict no-JIT rules and emit metadata."""

    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    if not cache_dir.is_dir():
        raise SystemExit(f"cache directory does not exist: {cache_dir}")

    # Set these before importing the FreeToken builders because each spec calls
    # load_jit or load_aot internally. The loader first checks this directory
    # and raises if an exact shared object is absent.
    os.environ["FREETOKEN_KERNEL_CACHE_DIR"] = str(cache_dir)
    os.environ["FREETOKEN_DISABLE_JIT"] = "1"

    import torch

    from freetoken.kernel.aot import default_kernel_specs

    if torch.version.hip is None:
        raise SystemExit("expected a HIP-backed PyTorch runtime")

    specs = default_kernel_specs()
    loaded_names: list[str] = []
    # build() returns immediately from the prebuilt cache path. The otherwise
    # required build directory is never created because strict no-JIT makes a
    # cache miss an exception before tvm_ffi receives a compile request.
    for spec in specs:
        spec.build(cache_dir / "verification-build-never-used" / spec.name)
        loaded_names.append(spec.name)

    print(
        json.dumps(
            {
                "cache_dir": str(cache_dir),
                "device": torch.cuda.get_device_name(),
                "hip": torch.version.hip,
                "loaded_modules": len(loaded_names),
                "status": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
