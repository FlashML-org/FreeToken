from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass
from typing import Tuple


_GFX_ARCH_RE = re.compile(r"gfx\d+[a-z]?")


@dataclass(frozen=True)
class RocmArchCapability:
    """Normalized target capability record.

    ``status`` describes evidence level, not source-code intent. Until each target has a
    fresh served-request run, records stay ``compile-only`` and callers must not advertise
    them as supported serving targets.
    """

    target: str
    family: str
    wave_size: int
    generic_gguf_types: frozenset[str]
    graph_features: frozenset[str]
    status: str = "compile-only"

    @property
    def native_gguf_types(self) -> frozenset[str]:
        """Compatibility alias for callers written before generic routing was named."""
        return self.generic_gguf_types


@dataclass(frozen=True)
class CandidateCapability:
    """Exact native-candidate contract, separate from generic ROCm support."""

    target: str
    operations: frozenset[str]
    quant_types: frozenset[str]
    supported_shapes: frozenset[str]
    abi: str
    source_version: str
    self_test: str
    status: str = "unavailable"


_NATIVE_GGUF_TYPES = frozenset({"Q4_0", "Q4_K", "Q5_K", "Q6_K", "Q8_0"})
_ROCM_CAPABILITIES = {
    **{
        target: RocmArchCapability(
            target=target,
            family="rdna3",
            wave_size=32,
            generic_gguf_types=_NATIVE_GGUF_TYPES,
            graph_features=frozenset(),
        )
        for target in ("gfx1100", "gfx1101", "gfx1102", "gfx1103")
    },
    **{
        target: RocmArchCapability(
            target=target,
            family="rdna3.5",
            wave_size=32,
            generic_gguf_types=_NATIVE_GGUF_TYPES,
            graph_features=frozenset(),
        )
        for target in ("gfx1150", "gfx1151")
    },
    **{
        target: RocmArchCapability(
            target=target,
            family="rdna4",
            wave_size=32,
            generic_gguf_types=_NATIVE_GGUF_TYPES,
            graph_features=frozenset(),
        )
        for target in ("gfx1200", "gfx1201")
    },
}
ROCM_ARCHES = frozenset(_ROCM_CAPABILITIES)

_ROCM_CANDIDATES = {
    "gfx1100": CandidateCapability(
        target="gfx1100",
        operations=frozenset({"gguf_moe"}),
        quant_types=frozenset({"Q4_K", "Q5_K", "Q6_K", "Q8_0"}),
        supported_shapes=frozenset(),
        abi="moe-abi-v2",
        source_version="llama.cpp-b10434",
        self_test="required",
        status="compile-only",
    ),
}


def _gfx_arch_from(value: object) -> str | None:
    match = _GFX_ARCH_RE.search(str(value).lower())
    return match.group(0) if match else None


def rocm_arch_capability(value: object) -> RocmArchCapability:
    """Resolve exact supported target from an override or runtime ``gcnArchName``."""
    target = _gfx_arch_from(value)
    capability = _ROCM_CAPABILITIES.get(target)
    if capability is None:
        raise ValueError(
            f"ROCm target {value!r} is not in FreeToken target matrix; "
            f"expected one of {', '.join(sorted(ROCM_ARCHES))}"
        )
    return capability


def rocm_arch_matrix() -> dict[str, RocmArchCapability]:
    """Return immutable capability records keyed by normalized target."""
    return dict(_ROCM_CAPABILITIES)


def rocm_candidate_capability(value: object) -> CandidateCapability | None:
    """Return exact native-candidate metadata, without inferring family support."""
    target = _gfx_arch_from(value)
    if target is None:
        return None
    return _ROCM_CANDIDATES.get(target)


def rocm_candidate_matrix() -> dict[str, CandidateCapability]:
    """Return native candidates keyed by exact normalized target."""
    return dict(_ROCM_CANDIDATES)


def parse_rocm_arches(value: str, *, source: str = "ROCm architecture override") -> tuple[str, ...]:
    """Parse semicolon/comma/space-separated ROCm targets and reject unknown ones."""
    raw = [token for token in re.split(r"[;,\s]+", value.strip()) if token]
    arches: list[str] = []
    for token in raw:
        arch = _gfx_arch_from(token)
        if arch != token.lower() or arch not in ROCM_ARCHES:
            raise ValueError(
                f"{source} contains unsupported target {token!r}; "
                f"expected one of {', '.join(sorted(ROCM_ARCHES))}"
            )
        if arch not in arches:
            arches.append(arch)
    return tuple(arches)


@functools.cache
def is_rocm() -> bool:
    """True when torch is built for ROCm (AMD GPU) instead of CUDA."""
    try:
        import torch
    except Exception:
        return False
    return getattr(torch.version, "hip", None) is not None


def device_kind() -> str:
    """Return build backend: ``cuda``, ``rocm``, or ``cpu``."""
    try:
        import torch.version
    except Exception:
        return "cpu"
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "cpu"


def is_cuda() -> bool:
    """True when installed torch is a CUDA build."""
    return device_kind() == "cuda"


@functools.cache
def current_gpu_name() -> str | None:
    """Return current device name, or ``None`` when torch/device is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        return None


@functools.cache
def get_rocm_gfx_arch() -> str | None:
    """Return the current AMD GPU target (for example ``gfx1201``).

    Prefer the runtime device because build variables may contain multiple
    semicolon-separated targets. Environment variables remain useful for
    cross-compilation and systems where no GPU is currently visible; in that
    fallback mode the first target is returned. The result is process-cached
    for FreeToken's one-process-per-GPU execution model, so callers must select
    the intended device before the first call.
    """
    if not is_rocm():
        return None

    import torch

    override = os.getenv("FREETOKEN_ROCM_ARCH")
    if override:
        parse_rocm_arches(override, source="FREETOKEN_ROCM_ARCH")

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            for attr in ("gcnArchName", "arch"):
                arch = _gfx_arch_from(getattr(props, attr, ""))
                if arch:
                    return arch
        except (AttributeError, RuntimeError):
            pass

    for env_var in ("FREETOKEN_ROCM_ARCH", "PYTORCH_ROCM_ARCH", "HCC_AMDGPU_TARGET"):
        value = os.getenv(env_var, "")
        if value:
            arches = parse_rocm_arches(value, source=env_var)
            if arches:
                return arches[0]
    return None


@functools.cache
def _get_gfx_arch() -> int | None:
    """Return numeric current ROCm target for family-portable feature gates."""
    if not is_rocm():
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        for value in (getattr(props, "gcnArchName", None), current_gpu_name()):
            match = re.search(r"gfx(\d{3,4})", str(value or ""))
            if match:
                return int(match.group(1))
    except Exception:
        pass
    return None


def is_gfx_arch_ge(arch_int: int) -> bool:
    """True when current ROCm target is at least numeric ``arch_int``."""
    gfx = _get_gfx_arch()
    return gfx is not None and gfx >= arch_int


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    import torch
    import torch.version

    if is_rocm():
        return None
    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    """capability >= (major, minor). Open-ended: newer archs also pass. Only use this
    for family-portable features (e.g. PDL); arch-specific kernels (sm_90a/sm_100a
    cubins) need the closed is_smXX_family checks below."""
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


def _is_arch_family(major: int) -> bool:
    arch = _get_torch_cuda_version()
    return arch is not None and arch[0] == major


def is_sm90_family() -> bool:
    """Exactly major 9 (Hopper). For sm_90a-only kernels (e.g. FA3)."""
    return _is_arch_family(9)


def is_sm100_family() -> bool:
    """Exactly major 10 (datacenter Blackwell). For sm_100a/103a-only kernels
    (e.g. trtllm-gen) that consumer Blackwell (sm_120/121) cannot run."""
    return _is_arch_family(10)


def is_sm90_supported() -> bool:
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    return is_arch_supported(10, 0)
