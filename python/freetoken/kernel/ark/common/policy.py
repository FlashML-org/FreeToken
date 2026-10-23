"""Fail-closed environment and measured-shape selection, separate for eager/Graph."""

import hashlib
import json
import os
import platform
import subprocess
from functools import lru_cache
from pathlib import Path

import torch

MODE = os.environ.get("FREETOKEN_ARK_POLICY", "validated")
if MODE not in ("native", "validated", "baseline"):
    raise ValueError("FREETOKEN_ARK_POLICY must be native, validated or baseline")
PROFILE_PATH = Path(os.environ.get("FREETOKEN_ARK_PROFILE", Path(__file__).with_name("validated_shapes.json")))
_profiles = json.loads(PROFILE_PATH.read_text()) if PROFILE_PATH.exists() else {}


@lru_cache(maxsize=1)
def current_environment():
    import triton

    if not torch.version.hip or not torch.cuda.is_available():
        return {}
    from .native import _compiler, source_digest

    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        compiler = subprocess.check_output([_compiler(), "--version"])
    except (OSError, subprocess.CalledProcessError):
        return {}
    package = Path(__file__).resolve().parents[3]
    baseline_paths = ("moe/fused_fp8_block.py", "kernel/triton/fp8_block_linear.py")
    baseline = hashlib.sha256(b"".join((package / path).read_bytes() for path in baseline_paths)).hexdigest()
    return {
        "torch": str(torch.__version__),
        "triton": triton.__version__,
        "hip": str(torch.version.hip),
        "gpu_name": p.name,
        "arch": p.gcnArchName.split(":")[0],
        "source_sha256": source_digest(),
        "device_count": torch.cuda.device_count(),
        "platform": platform.platform(),
        "compiler_sha256": hashlib.sha256(compiler).hexdigest(),
        "baseline_sha256": baseline,
    }


@lru_cache(maxsize=1)
def environment_matches():
    expected = _profiles.get("_environment", {})
    actual = current_environment()
    return bool(actual) and all(expected.get(k) == v for k, v in actual.items())


def key(*values):
    return ",".join(str(int(v)) for v in values)


def allowed(operator, shape_key, *, capturing=None):
    if os.environ.get("FREETOKEN_USE_ARK", "0") != "1" or MODE == "baseline":
        return False
    if MODE == "native":
        return True
    if not environment_matches():
        return False
    if capturing is None:
        capturing = torch.cuda.is_current_stream_capturing()
    return shape_key in _profiles.get(operator, {}).get("graph" if capturing else "eager", {})


def metadata_key(op, m):
    if op == "moe_decode_fp8":
        return key(m["M"], m["H"], m["I"], m["bank_rows"], m["top_k"])
    if op == "dense_fp8_decode":
        return key(m["M"], m["N"], m["K"])
    raise ValueError(f"unsupported ARK operator: {op}")
