#!/usr/bin/env python3
"""fetch_marlin_sm75.py — Downloads the Marlin WNA16 sm_75 kernel sources from
weicj/vLLM-2080Ti-Definitive at build time and generates the sm_75 instantiation
files via generate_kernels.py.

Called automatically by setup.py when building the marlin_sm75 extension.
Can also be run standalone: python scripts/fetch_marlin_sm75.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "weicj/vLLM-2080Ti-Definitive"
BRANCH = "vllm-2080ti-definitive-0.1.x"
BASE_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/csrc/moe/marlin_moe_wna16"

# Destination: python/freetoken/kernel/csrc/marlin_wna16/
DEST = Path(__file__).parent.parent / "python" / "freetoken" / "kernel" / "csrc" / "marlin_wna16"

FILES = [
    "kernel.h",
    "marlin_template.h",
    "ops.cu",
    "generate_kernels.py",
]

# Additional headers that marlin_template.h transitively includes from vLLM's
# csrc/quantization/marlin/ tree. We vendor minimal stubs for the symbols we
# actually use at sm_75 (no FP8, no NVFP4).
QUANTIZATION_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/csrc/quantization/marlin"
QUANTIZATION_FILES = [
    "marlin.cuh",
    "marlin_dtypes.cuh",
    "marlin_mma.h",
    "dequant.h",
]
SCALAR_TYPE_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/csrc/core/scalar_type.hpp"
REGISTRATION_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/csrc/core/registration.h"


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip]  {dest.name} (already exists)")
        return
    print(f"  [fetch] {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def generate_kernels(dest: Path, arch: str = "7.5") -> None:
    """Run generate_kernels.py to produce sm75_kernel_*.cu and kernel_selector.h."""
    script = dest / "generate_kernels.py"
    if not script.exists():
        raise FileNotFoundError(f"generate_kernels.py not found at {script}")
    print(f"  [gen]   generate_kernels.py for arch {arch}")
    result = subprocess.run(
        [sys.executable, str(script), arch],
        cwd=str(dest),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"generate_kernels.py failed:\n{result.stdout}\n{result.stderr}"
        )
    generated = sorted(dest.glob("sm75_kernel_*.cu"))
    print(f"  [gen]   generated {len(generated)} sm75 kernel files")
    selector = dest / "kernel_selector.h"
    if not selector.exists():
        raise FileNotFoundError("kernel_selector.h was not generated")


def main() -> None:
    print(f"Fetching Marlin WNA16 sm_75 sources into {DEST} ...")
    DEST.mkdir(parents=True, exist_ok=True)

    # Core Marlin files
    for fname in FILES:
        fetch(f"{BASE_URL}/{fname}", DEST / fname)

    # Transitive includes: quantization/marlin/
    quant_dir = DEST / "quantization" / "marlin"
    for fname in QUANTIZATION_FILES:
        fetch(f"{QUANTIZATION_BASE}/{fname}", quant_dir / fname)

    # core/scalar_type.hpp and core/registration.h
    core_dir = DEST / "core"
    fetch(SCALAR_TYPE_URL, core_dir / "scalar_type.hpp")
    fetch(REGISTRATION_URL, core_dir / "registration.h")

    # Generate sm75_kernel_*.cu + kernel_selector.h
    generate_kernels(DEST, arch="7.5")

    print("Done. Marlin sm_75 sources ready.")


if __name__ == "__main__":
    main()
