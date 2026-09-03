"""--gpu for ft serve / bench bw / checkpoint: resolve entries to GPU UUIDs, bind by UUID at CUDA init.

Three device-id namespaces, converted explicitly:
- logical: position in the --gpu list == TP rank; each worker takes its own entry by rank and the id ends there.
- physical: NVML / nvidia-smi order, carried as a GPU UUID (_assigned_physical); not affected by CUDA_VISIBLE_DEVICES.
- visible: CUDA ordinal in this process (_assigned_visible), what torch.device("cuda", n) means.

The parent resolves --gpu entries to full UUIDs via NVML (resolve_gpu_uuids) and fails fast on a typo.
Each worker publishes its own entry (set_assigned_gpu / assign_gpu) and binds it when CUDA comes up (bind_assigned_gpu) by matching the UUID against CUDA's visible devices.
One process runs on one GPU. Binding is unconditional: a process that publishes nothing binds a default ordinal and records it, so assigned_visible_gpu() names that card in every case.
No process mutates CUDA_VISIBLE_DEVICES, and the UUID match holds under any CUDA_DEVICE_ORDER.

Stdlib only (torch is imported lazily); not under freetoken.utils, which imports transformers.
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

UUID_PREFIX = "GPU-"


def is_gpu_uuid(spec: str) -> bool:
    return spec[: len(UUID_PREFIX)].upper() == UUID_PREFIX


def is_gpu_index(spec: str) -> bool:
    # not str.isdigit(): that also accepts superscripts and other Unicode digits
    return spec.isascii() and spec.isdecimal()


def _canonical(entry: str) -> str:
    """A UUID in the exact form the driver matches (upper-case GPU- prefix), an index as-is."""
    if not (is_gpu_uuid(entry) or is_gpu_index(entry)):
        raise ValueError(
            f"{entry!r} is neither a GPU UUID (GPU-xxxx..., as `nvidia-smi -L` prints) "
            f"nor an nvidia-smi index"
        )
    return UUID_PREFIX + entry[len(UUID_PREFIX):] if is_gpu_uuid(entry) else entry


def parse_gpu_spec(value: str) -> tuple[str, ...]:
    """Split a --gpu value; ValueError on a bad entry, an empty value, or a mix of UUIDs and indices."""
    entries = tuple(_canonical(e.strip()) for e in value.split(",") if e.strip())
    if not entries:
        raise ValueError("--gpu needs at least one GPU")
    if len({is_gpu_uuid(e) for e in entries}) > 1:
        # the driver parses CUDA_VISIBLE_DEVICES as all-UUID or all-index
        raise ValueError("--gpu entries must be all UUIDs or all indices")
    return entries


def gpu_arg(value: str) -> tuple[str, ...]:
    """argparse type for a --gpu list."""
    try:
        return parse_gpu_spec(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def single_gpu_arg(value: str) -> str:
    """argparse type for a single-GPU --gpu."""
    entries = gpu_arg(value)
    if len(entries) != 1:
        raise argparse.ArgumentTypeError("takes exactly one GPU")
    return entries[0]


def _nvml_uuids() -> "list[str] | None":
    """Full GPU UUIDs in physical (nvidia-smi) order, or None when NVML is unavailable.

    Own ctypes loader instead of torch's _raw_device_uuid_nvml: that helper only knows the Linux library name, raises (not None) when the library is missing, and is private API.
    NVML exports are cdecl on every platform, so CDLL is right on Windows too (same as nvidia-ml-py).
    None on any failure -- no library, a stub library without the _v2 symbols, WSL, a dead device -- and callers fall back.
    """
    import ctypes

    if os.name == "nt":
        candidates = [
            "nvml.dll",
            os.path.join(os.environ.get("SystemRoot", r"C:\\Windows"), "System32", "nvml.dll"),
            os.path.join(os.environ.get("ProgramFiles", r"C:\\Program Files"), "NVIDIA Corporation", "NVSMI", "nvml.dll"),
        ]
    else:
        candidates = ["libnvidia-ml.so.1"]
    try:
        for name in candidates:
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            return None
        if lib.nvmlInit() != 0:
            return None
        try:
            count = ctypes.c_int()
            if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
                return None
            uuids = []
            for i in range(count.value):
                handle = ctypes.c_void_p()
                if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle)) != 0:
                    return None
                buf = ctypes.create_string_buffer(96)
                if lib.nvmlDeviceGetUUID(handle, buf, 96) != 0:
                    return None
                uuids.append(buf.value.decode("ascii", "replace"))
            return uuids
        finally:
            lib.nvmlShutdown()
    except (OSError, AttributeError):
        return None


def _match_uuid(spec: str, uuids: "list[str]", where: str) -> str:
    """The unique full UUID that ``spec`` prefixes, else ValueError."""
    hits = [u for u in uuids if u.upper().startswith(spec.upper())]
    if len(hits) != 1:
        raise ValueError(f"--gpu {spec}: not found or not a unique prefix {where}; run `nvidia-smi -L` to list GPUs")
    return hits[0]


def resolve_gpu_uuids(specs: Sequence[str]) -> "tuple[str, ...] | None":
    """--gpu entries -> full GPU UUIDs, one per TP rank; raises ValueError on a bad entry.

    A preset CUDA_VISIBLE_DEVICES is a quota to stay inside: an index counts within that list, a UUID must name one of its entries.
    Returns None when NVML is unavailable -- the worker then interprets the raw entries against CUDA's own enumeration (see bind_assigned_gpu).
    """
    specs = parse_gpu_spec(",".join(specs))
    if len({s.upper() for s in specs}) != len(specs):
        raise ValueError(f"--gpu {','.join(specs)}: the same GPU appears twice")
    uuids = _nvml_uuids()
    if uuids is None:
        return None
    preset_raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    preset = None if preset_raw is None else [e.strip() for e in preset_raw.split(",") if e.strip()]

    resolved: list[str] = []
    for spec in specs:
        if preset is None:
            if is_gpu_uuid(spec):
                resolved.append(_match_uuid(spec, uuids, "on this machine"))
            elif int(spec) < len(uuids):
                resolved.append(uuids[int(spec)])
            else:
                raise ValueError(f"--gpu {spec}: only {len(uuids)} GPU(s) on this machine; run `nvidia-smi -L` to list GPUs")
        else:
            entry = _preset_entry(spec, preset, preset_raw)
            # an integer entry is read in physical order, as under CUDA_DEVICE_ORDER=PCI_BUS_ID; a negative or MIG-form entry cannot name a whole GPU
            if is_gpu_uuid(entry):
                resolved.append(_match_uuid(entry, uuids, f"(from CUDA_VISIBLE_DEVICES={preset_raw!r})"))
            elif is_gpu_index(entry) and int(entry) < len(uuids):
                resolved.append(uuids[int(entry)])
            else:
                raise ValueError(
                    f"--gpu {spec}: cannot resolve CUDA_VISIBLE_DEVICES entry {entry!r} "
                    f"({len(uuids)} GPU(s) on this machine)"
                )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"--gpu {','.join(specs)}: the same GPU appears twice")
    return tuple(resolved)


def _preset_entry(spec: str, preset: "list[str]", preset_raw: str) -> str:
    """The CUDA_VISIBLE_DEVICES entry ``spec`` selects, else ValueError."""
    if not is_gpu_uuid(spec):
        idx = int(spec)
        if idx >= len(preset):
            raise ValueError(
                f"--gpu {spec}: only {len(preset)} GPU(s) are visible through "
                f"CUDA_VISIBLE_DEVICES={preset_raw!r} (indices count within that list)"
            )
        return preset[idx]
    if not all(is_gpu_uuid(p) for p in preset):
        raise ValueError(
            f"--gpu {spec}: CUDA_VISIBLE_DEVICES={preset_raw!r} lists GPUs by index; "
            f"give --gpu as an index into that list"
        )
    hits = [p for p in preset if p.upper().startswith(spec.upper()) or spec.upper().startswith(p.upper())]
    if len(hits) != 1:
        raise ValueError(
            f"--gpu {spec}: not one of the GPUs visible through CUDA_VISIBLE_DEVICES={preset_raw!r}"
        )
    return hits[0]


# The GPU this process was assigned, in whichever namespace it arrived in; bind_assigned_gpu fills in the visible one.
# Process-global on purpose: publishing is torch-free so a worker can do it before heavy imports, and kernel-compat checks (e4m3_native) need the device this process will use.
_assigned_physical: "str | None" = None
_assigned_visible: "int | None" = None


def set_assigned_gpu(target: str) -> None:
    """Publish this process's GPU before CUDA init; second call must agree.

    A UUID names a physical GPU and is converted at bind time; a bare index is already a visible ordinal (a preset CUDA_VISIBLE_DEVICES has narrowed to it).
    """
    global _assigned_physical, _assigned_visible
    physical = target if is_gpu_uuid(target) else None
    visible = None if physical is not None else int(target)
    current = (_assigned_physical, _assigned_visible)
    if current not in ((None, None), (physical, visible)):
        raise RuntimeError(f"set_assigned_gpu called twice: {current} then {target!r}")
    _assigned_physical, _assigned_visible = physical, visible


def assign_gpu(spec: "str | None") -> None:
    """Resolve one --gpu value and publish it for bind_assigned_gpu; no-op when the flag was not given."""
    if spec is None:
        return
    resolved = resolve_gpu_uuids([spec])
    set_assigned_gpu(resolved[0] if resolved else parse_gpu_spec(spec)[0])


def _visible_of_physical(uuid: str) -> int:
    """CUDA ordinal of the physical GPU ``uuid`` (or unique prefix) among this process's visible devices."""
    import torch

    seen: list[str] = []
    hits: list[int] = []
    for v in range(torch.cuda.device_count()):
        u = format_gpu_uuid(getattr(torch.cuda.get_device_properties(v), "uuid", None))
        seen.append(u or "?")
        if u is not None and u.upper().startswith(uuid.upper()):
            hits.append(v)
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise RuntimeError(f"--gpu {uuid}: not a unique prefix (visible: {', '.join(seen)})")
    raise RuntimeError(
        f"GPU {uuid} is not visible to CUDA in this process "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, "
        f"visible: {', '.join(seen) or 'none'})"
    )


def bind_assigned_gpu(default: int = 0):
    """torch.cuda.set_device this process's GPU and return the device.

    ``default`` is a visible ordinal, used and recorded when nothing was published, so the process always knows which card it runs on.
    A published UUID (or prefix) is matched against CUDA's own device list, so the result is right under any CUDA_DEVICE_ORDER.
    """
    global _assigned_visible
    import torch

    if _assigned_visible is None:
        _assigned_visible = default if _assigned_physical is None else _visible_of_physical(_assigned_physical)
    if not 0 <= _assigned_visible < torch.cuda.device_count():
        raise RuntimeError(
            f"cannot use CUDA device {_assigned_visible}: only {torch.cuda.device_count()} device(s) visible "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
        )
    device = torch.device("cuda", _assigned_visible)
    torch.cuda.set_device(device)
    return device


def assigned_visible_gpu() -> "int | None":
    """Visible ordinal this process is pinned to, or None before it publishes or binds a GPU (= the current device).

    Published-but-not-yet-bound still counts: compat checks in the window between publish and bind must judge the assigned card, not whatever the calling thread happens to sit on.
    """
    if _assigned_visible is not None:
        return _assigned_visible
    return None if _assigned_physical is None else _visible_of_physical(_assigned_physical)


def format_gpu_uuid(raw) -> str | None:
    """nvidia-smi form GPU-<uuid> from a uuid.UUID."""
    return None if raw is None else f"{UUID_PREFIX}{raw}"


def gpu_identity(index: int) -> dict:
    """{index, name, uuid, total_bytes} of visible device ``index``."""
    import torch

    props = torch.cuda.get_device_properties(index)
    return {
        "index": index,
        "name": props.name,
        "uuid": format_gpu_uuid(getattr(props, "uuid", None)),
        "total_bytes": int(props.total_memory),
    }
