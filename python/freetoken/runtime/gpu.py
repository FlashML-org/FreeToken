"""Detect NVIDIA vs AMD GPUs and keep AMD paths from falling into CUDA-only code.

Torch exposes both vendors through ``torch.cuda`` (ROCm reuses that namespace).
Callers must use this module instead of assuming ``torch.cuda`` means NVIDIA.

Designed to import without torch when possible; torch is loaded lazily.
"""

from __future__ import annotations

import functools
import os
from enum import Enum
from typing import Any, Iterable

# First-class Linux target for this fork (Radeon AI PRO R9700 / RX 9070 XT).
DEFAULT_GFX_ARCH = "gfx1201"

# Architectures the HIP JIT path knows how to request via ``--offload-arch``.
# Add a new gfx by appending it here and setting FREETOKEN_GFX_ARCH / TVM_FFI_ROCM_ARCH_LIST.
SUPPORTED_GFX_ARCHES: tuple[str, ...] = (
    "gfx1201",  # RDNA4: R9700, RX 9070 XT
    "gfx1200",  # RDNA4: RX 9060 XT
    "gfx1100",  # RDNA3: 7900 XTX / XT
    "gfx1101",  # RDNA3: 7800 XT / 7700 XT
    "gfx1102",  # RDNA3: 7600
    "gfx1030",  # RDNA2: 6900 / 6800
    "gfx1031",  # RDNA2: 6700 XT
    "gfx1151",  # Strix Halo (kept visible — not a desktop iGPU to hide)
)

# Raphael / Granite Ridge iGPU. Strix Halo (gfx1151) is intentionally not listed.
_IGPU_GFX = frozenset({"gfx1036", "gfx1103", "gfx1150"})

_TRUE = {"1", "true", "yes", "on"}

# gfx1201 MoE kernels have crashed under HIP graph replay (HIP 719 / driver TDR)
# on Windows ROCm (FlashML-org/FreeToken#82). gfx1200 replayed the same kernels.
# Default: disable graph replay for MoE on gfx1201 until a Linux box confirms it.
_GRAPH_REPLAY_UNSAFE_GFX = frozenset({"gfx1201"})


class Vendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    NONE = "none"
    # Default cu130 wheel on an AMD box — do not attempt CUDA kernels.
    CUDA_ON_AMD = "cuda_on_amd"


def _env_on(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE


def _torch():
    import torch

    return torch


@functools.cache
def vendor() -> Vendor:
    """Which GPU vendor this process should use.

    Order: explicit ``FREETOKEN_GPU_VENDOR``, then torch HIP/CUDA build, then
    a HIP runtime library on disk (so ``ft gpu`` works before torch is a ROCm
    wheel). Never treats ZLUDA as a vendor — that path is unsupported (#60).
    """
    forced = os.getenv("FREETOKEN_GPU_VENDOR", "").strip().lower()
    if forced in {"amd", "rocm", "hip"}:
        return Vendor.AMD
    if forced in {"nvidia", "cuda"}:
        return Vendor.NVIDIA
    if forced in {"none", "cpu"}:
        return Vendor.NONE

    try:
        torch = _torch()
        if getattr(torch.version, "hip", None):
            return Vendor.AMD
        # Wheel provenance only — do not call torch.cuda.is_available() here;
        # that initializes the runtime and makes HIP_VISIBLE_DEVICES mutation useless.
        if getattr(torch.version, "cuda", None):
            if _amd_hardware_present():
                return Vendor.CUDA_ON_AMD
            return Vendor.NVIDIA
    except Exception:
        pass

    if _amd_hardware_present():
        return Vendor.AMD
    return Vendor.NONE


def _amd_hardware_present() -> bool:
    """HIP runtime or an AMD PCI device (sysfs), without initializing torch.cuda."""
    if _hip_library_name() is not None:
        return True
    drm = "/sys/class/drm"
    try:
        for name in os.listdir(drm):
            path = os.path.join(drm, name, "device", "vendor")
            if not os.path.isfile(path):
                continue
            with open(path, encoding="ascii") as fh:
                if fh.read().strip().lower() == "0x1002":
                    return True
    except OSError:
        pass
    return False


def is_hip() -> bool:
    return vendor() == Vendor.AMD


def is_cuda() -> bool:
    return vendor() == Vendor.NVIDIA


def _looks_like_amd_device(index: int) -> bool:
    try:
        torch = _torch()
        name = torch.cuda.get_device_name(index).lower()
        props = torch.cuda.get_device_properties(index)
        arch = _gcn_from_props(props)
        if arch:
            return True
        return any(tok in name for tok in ("amd", "radeon", "instinct"))
    except Exception:
        return False


def _gcn_from_props(props: Any) -> str | None:
    raw = getattr(props, "gcnArchName", None) or getattr(props, "gcnArch", None)
    if not raw:
        return None
    return str(raw).split(":", 1)[0].strip() or None


def gcn_arch(index: int | None = None) -> str | None:
    """ISA name such as ``gfx1201``, or None when this is not an AMD device."""
    override = os.getenv("FREETOKEN_GFX_ARCH", "").strip()
    if override:
        return override.split(",", 1)[0]
    if not is_hip():
        return None
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            return None
        idx = 0 if index is None else index
        return _gcn_from_props(torch.cuda.get_device_properties(idx))
    except Exception:
        return None


def is_igpu(*, name: str = "", arch: str | None = None, total_bytes: int | None = None) -> bool:
    """True for a desktop APU iGPU that should not steal device 0 from a dGPU.

    Hides Granite Ridge / Raphael (Ryzen 9 9950X) ``AMD Radeon Graphics``.
    Does **not** hide Strix Halo / AI MAX — those are the intended GPU.
    """
    arch_l = (arch or "").split(":", 1)[0].lower()
    if arch_l in _IGPU_GFX:
        return True
    n = name.lower()
    if "granite ridge" in n or "raphael" in n:
        return True
    discrete_markers = ("rx ", "pro ", "instinct", "ai pro", "radeon ai", "w7900", "w7800")
    if "radeon graphics" in n and not any(m in n for m in discrete_markers):
        if total_bytes is None or total_bytes < 4 * (1 << 30):
            return True
    return False


def include_igpu() -> bool:
    return _env_on("FREETOKEN_INCLUDE_IGPU")


def hip_graph_replay_safe(arch: str | None = None) -> bool:
    """Whether HIP graph *replay* of fused MoE/GGUF kernels is believed safe.

    gfx1201 has a documented replay TDR on Windows ROCm (#82). Linux is unverified
    on this SKU; default conservative. Override with FREETOKEN_HIP_GRAPH_REPLAY=1.
    """
    if _env_on("FREETOKEN_HIP_GRAPH_REPLAY"):
        return True
    if os.getenv("FREETOKEN_HIP_GRAPH_REPLAY", "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    a = (arch or gcn_arch() or "").split(":", 1)[0]
    return a not in _GRAPH_REPLAY_UNSAFE_GFX


def nvidia_only_error(feature: str, *, hint: str = "--attention-backend triton") -> RuntimeError:
    """Loud failure instead of a CUDA symbol crash on AMD."""
    return RuntimeError(
        f"{feature} is NVIDIA-only and is not built for AMD/ROCm (native HIP; "
        f"ZLUDA is not supported — see FlashML-org/FreeToken#60). {hint}"
    )


def require_gpu() -> Vendor:
    """Fail with a clear message when no usable GPU backend is present."""
    v = vendor()
    if v is Vendor.CUDA_ON_AMD:
        raise RuntimeError(
            "This process loaded a CUDA PyTorch wheel on AMD hardware. Native HIP "
            "is required (ZLUDA is not supported — FlashML-org/FreeToken#60). "
            "Install a ROCm torch build and see docs/amd.md."
        )
    if v is Vendor.NONE:
        raise RuntimeError(
            "No usable GPU backend: this process has neither a CUDA torch+NVIDIA "
            "driver nor a HIP/ROCm torch+AMD driver. On AMD, install a ROCm PyTorch "
            "wheel (not the default cu130 extra) and see docs/amd.md. ZLUDA is not "
            "supported."
        )
    return v


def list_usable_devices() -> list[dict]:
    """Visible GPUs after optional iGPU filtering. Empty when torch has no CUDA/HIP."""
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            name = props.name
            arch = _gcn_from_props(props)
            total = int(props.total_memory)
            hidden = is_igpu(name=name, arch=arch, total_bytes=total) and not include_igpu()
            uuid = getattr(props, "uuid", None)
            out.append(
                {
                    "index": i,
                    "name": name,
                    "arch": arch,
                    "total_bytes": total,
                    "uuid": None if uuid is None else f"GPU-{uuid}",
                    "hidden_igpu": hidden,
                }
            )
        return out
    except Exception:
        return []


def usable_visible_indices() -> list[int]:
    return [d["index"] for d in list_usable_devices() if not d["hidden_igpu"]]


def default_visible_ordinal() -> int:
    """First non-hidden device, or 0 if nothing is listed."""
    usable = usable_visible_indices()
    return usable[0] if usable else 0


def _hip_library_name() -> str | None:
    import ctypes

    names: list[str] = []
    if os.name == "nt":
        names.extend(("amdhip64_7.dll", "amdhip64.dll"))
    else:
        names.extend(
            (
                "libamdhip64.so.7",
                "libamdhip64.so.6",
                "libamdhip64.so",
            )
        )
        rocm = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH")
        if rocm:
            names.insert(0, os.path.join(rocm, "lib", "libamdhip64.so"))
    for name in names:
        try:
            ctypes.CDLL(name)
            return name
        except OSError:
            continue
    return None


def load_hip_runtime():
    """ctypes handle to the HIP runtime, or None."""
    import ctypes

    name = _hip_library_name()
    if name is None:
        return None
    return ctypes.CDLL(name)


def _arch_from_name(name: str) -> str | None:
    n = name.lower()
    if "r9700" in n or "9070" in n or "ai pro r9700" in n:
        return "gfx1201"
    if "9060" in n:
        return "gfx1200"
    if "7900" in n or "w7900" in n:
        return "gfx1100"
    if "7800" in n or "7700" in n:
        return "gfx1101"
    if "7600" in n:
        return "gfx1102"
    if "6700" in n:
        return "gfx1031"
    if "6800" in n or "6900" in n:
        return "gfx1030"
    if "strix halo" in n or "ai max" in n:
        return "gfx1151"
    return None


def hip_enumerate_devices() -> list[dict]:
    """HIP devices via ctypes — does not initialize ``torch.cuda``."""
    import ctypes

    hip = load_hip_runtime()
    if hip is None:
        return []
    count = ctypes.c_int()
    if hip.hipGetDeviceCount(ctypes.byref(count)) != 0:
        return []
    out = []
    for i in range(count.value):
        buf = ctypes.create_string_buffer(256)
        if hip.hipDeviceGetName(buf, 256, ctypes.c_int(i)) != 0:
            name = f"AMD GPU {i}"
        else:
            name = buf.value.decode("utf-8", "replace")
        arch = _arch_from_name(name)
        hidden = is_igpu(name=name, arch=arch) and not include_igpu()
        out.append(
            {
                "index": i,
                "name": name,
                "arch": arch,
                "total_bytes": None,
                "uuid": None,
                "hidden_igpu": hidden,
            }
        )
    return out


def apply_amd_runtime_env(*, devices: Iterable[dict] | None = None) -> dict[str, str]:
    """Set HIP/Triton/tvm-ffi arch env vars for gfx1201 (or FREETOKEN_GFX_ARCH).

    Idempotent: does not overwrite a variable the user already set. Also hides
    Granite Ridge iGPUs from ``HIP_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES``
    when those are unset, using ctypes HIP so torch.cuda is not initialized.

    Returns the keys that were newly written.
    """
    written: dict[str, str] = {}

    def _set(key: str, value: str) -> None:
        if not os.environ.get(key):
            os.environ[key] = value
            written[key] = value

    if vendor() != Vendor.AMD:
        return written

    hip_devs = list(devices) if devices is not None else hip_enumerate_devices()
    arch = os.getenv("FREETOKEN_GFX_ARCH", "").strip() or None
    if arch is None:
        for d in hip_devs:
            if not d.get("hidden_igpu") and d.get("arch"):
                arch = d["arch"]
                break
    arch = arch or DEFAULT_GFX_ARCH

    _set("FREETOKEN_GFX_ARCH", arch)
    _set("TVM_FFI_ROCM_ARCH_LIST", arch)
    _set("TRITON_OVERRIDE_ARCH", arch)
    _set("PYTORCH_ROCM_ARCH", arch)
    _set("ROCM_SDK_TARGET_FAMILY", arch)

    if (
        not include_igpu()
        and not os.environ.get("CUDA_VISIBLE_DEVICES")
        and not os.environ.get("HIP_VISIBLE_DEVICES")
        and not os.environ.get("ROCR_VISIBLE_DEVICES")
        and hip_devs
    ):
        keep = [str(d["index"]) for d in hip_devs if not d.get("hidden_igpu")]
        if keep and len(keep) < len(hip_devs):
            joined = ",".join(keep)
            _set("HIP_VISIBLE_DEVICES", joined)
            _set("CUDA_VISIBLE_DEVICES", joined)

    return written


def describe() -> str:
    """Human-readable GPU backend summary (no invented tok/s)."""
    v = vendor()
    lines = [
        f"vendor: {v.value}",
        f"hip: {is_hip()}",
        f"cuda: {is_cuda()}",
        f"gcn_arch: {gcn_arch() or '-'}",
        f"hip_graph_replay_safe: {hip_graph_replay_safe()}",
        f"include_igpu: {include_igpu()}",
    ]
    devs = list_usable_devices()
    if not devs:
        lines.append("devices: (none visible — no AMD GPU in this process)")
    for d in devs:
        flag = " [hidden iGPU]" if d["hidden_igpu"] else ""
        gib = d["total_bytes"] / (1 << 30)
        lines.append(
            f"  gpu{d['index']}: {d['name']} arch={d['arch'] or '-'} "
            f"{gib:.1f} GiB{flag}"
        )
    lines.append("zluda: unsupported (native HIP only; FlashML-org/FreeToken#60)")
    return "\n".join(lines)
