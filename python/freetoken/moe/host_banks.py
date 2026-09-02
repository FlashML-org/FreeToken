"""Reusable pinned host-bank primitives shared by the fast expert-load paths.

Two ideas the parallel read of the original checkpoint and FTW (read a repacked
contiguous cache) paths both rely on:

* **pin-after-fill** -- allocate the bank as a *lazy* anonymous ``mmap`` (no pages
  resident, instant), fill it with real data, and only THEN ``cudaHostRegister`` it.
  Registering already-resident pages just page-locks them; registering a lazy mmap first
  faults+zero-fills every page (~137 GiB -> ~47 s for DSV4) and that zero-fill is then
  immediately overwritten by the read. So pin-after-fill removes a whole redundant pass.
* **chunked multi-threaded O_DIRECT** -- DMA straight from disk into the (page-aligned)
  bank, bypassing the page cache, with many concurrent ``preadv`` on one fd (scales to the
  device's queue-depth ceiling even for a single file).

The mmaps are held for the process lifetime (the banks live as long as the offload cache).
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import math
import mmap
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_BLK = 4096  # O_DIRECT alignment (page size)


class HostResidency(str, Enum):
    """Residency class of a host bank layer.

    Only PINNED (cudaHostRegister'd) memory can feed the GPU movement paths; LOCKED (mlock'd, no device address) and PAGEABLE layers must decode on the CPU executor.
    The non-pinned classes exist for hosts that cap CUDA pin quota (WSL/WDDM: ~half of RAM).
    """

    PINNED = "pinned"
    LOCKED = "locked"
    PAGEABLE = "pageable"


_DEFAULT_CHUNK = 8 << 20

# Hold the mmaps for the process lifetime; the offload cache reads from these banks forever.
_LIVE_BUFFERS: list[mmap.mmap] = []

def _env_born_pinned() -> bool | None:
    """``FREETOKEN_BANK_CUDA_ALLOC`` tri-state: unset -> ``None`` (default applies), else the parsed boolean."""
    v = os.environ.get("FREETOKEN_BANK_CUDA_ALLOC", "").strip().lower()
    if not v:
        return None
    return v in ("1", "true", "yes", "on")


def born_pinned_default() -> bool:
    """Whether PINNED serving banks use cudaHostAlloc instead of mmap + register-after-fill.

    Off by default: registered mmaps already read at the PCIe roofline and lazy mmaps commit pages only on fill. ``FREETOKEN_BANK_CUDA_ALLOC`` overrides."""
    env = _env_born_pinned()
    if env is not None:
        return env
    return False


# DeepSeek-V4-Flash-0731 ds_fp4 floor: 43 × 256 × 13_369_344. Used when the
# model config is not available yet. Not a benchmark.
DEFAULT_BANK_SHARE_NEED_BYTES = 43 * 256 * 13_369_344  # 147_169_738_752 ≈ 137.1 GiB

# amdgpu module parameter: KFD counts every GPU registration of host memory
# against ONE global "resident system memory" budget (about MemTotal - MemTotal/64).
# Two TP ranks registering the same shared pool are double-counted, so the second
# rank's pins fail partway through the load ("SVM mapping failed, exceeds resident
# system memory limit" in dmesg) unless the operator disables the limit.
_KFD_NO_SYSTEM_MEM_LIMIT = "/sys/module/amdgpu/parameters/no_system_mem_limit"


def _env_off(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("0", "false", "no", "off")


def bank_sharing_enabled() -> bool:
    """TP>1 maps one shmem expert pool unless ``FREETOKEN_BANK_SHARE=0``."""
    if _env_off("FREETOKEN_BANK_SHARE"):
        return False
    from freetoken.distributed import try_get_tp_info

    tp = try_get_tp_info()
    return tp is not None and tp.size > 1


def _meminfo_bytes(key: str) -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _read_sysfs(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def kfd_system_mem_limit_bytes() -> int | None:
    """KFD's global budget for GPU-registered host memory, or None when the
    ``amdgpu`` limit is disabled / not present. Mirrors the kernel's
    ``mem - (mem >> 6)`` sizing from MemTotal."""
    param = _read_sysfs(_KFD_NO_SYSTEM_MEM_LIMIT)
    if param is None or param.upper() in ("Y", "1"):
        return None
    total = _meminfo_bytes("MemTotal")
    if total is None:
        return None
    return total - (total >> 6)


def require_shared_pool_capacity(*, need_bytes: int, tp_size: int) -> None:
    """Fail before the load when the shmem pool cannot be resident, or when every
    TP rank cannot register it with its GPU. ``need_bytes <= 0`` skips (unit tests)."""
    if need_bytes <= 0:
        return
    need_gib = need_bytes / (1 << 30)
    avail = _meminfo_bytes("MemAvailable")
    if avail is not None and avail < need_bytes:
        raise RuntimeError(
            f"shared expert pool needs ~{need_gib:.1f} GiB of RAM (shmem, pinned for the "
            f"GPUs) but MemAvailable is {avail / (1 << 30):.1f} GiB. Free memory "
            f"(drop page cache, stop other model servers) or use a smaller model."
        )
    from freetoken.runtime.gpu import is_hip

    if tp_size <= 1 or not is_hip():
        return
    limit = kfd_system_mem_limit_bytes()
    if limit is None:
        return
    need_total = need_bytes * tp_size
    if need_total > limit:
        raise RuntimeError(
            f"TP={tp_size}: each rank registers the shared ~{need_gib:.1f} GiB expert pool "
            f"with its own GPU, and KFD charges every registration against one global "
            f"'resident system memory' budget (~{limit / (1 << 30):.1f} GiB on this host): "
            f"{tp_size} × {need_gib:.1f} = {need_total / (1 << 30):.1f} GiB. The second rank's "
            f"pins would fail partway through the load (dmesg: 'SVM mapping failed, exceeds "
            f"resident system memory limit'). The pages are shared, so this is double counting; "
            f"disable the limit:  echo Y | sudo tee {_KFD_NO_SYSTEM_MEM_LIMIT}  "
            f"(persist with 'options amdgpu no_system_mem_limit=1' in /etc/modprobe.d/amdgpu.conf). "
            f"FREETOKEN_BANK_SHARE=0 gives each rank a private pool instead ({tp_size}× RAM)."
        )


def prepare_shared_banks(*, need_bytes: int | None = None) -> str | None:
    """Enable the TP>1 shared expert pool and run its preflights.

    Returns a short description of the pool (``"memfd"``) or None when sharing is
    off. The pool is shmem (``memfd_create``): pages never go through filesystem
    writeback, which a disk-backed ``MAP_SHARED`` pool + ``hipHostRegister`` turned
    into a KFD evict/restore storm that wedged the load. ``/dev/shm`` is not used
    either: its mount is capped at ~50% of RAM.
    """
    from freetoken.distributed import try_get_tp_info

    tp = try_get_tp_info()
    if tp is None or tp.size <= 1:
        return None
    if _env_off("FREETOKEN_BANK_SHARE"):
        logger.warning(
            "FREETOKEN_BANK_SHARE=0: each TP rank allocates its own expert banks "
            "(~137 GiB × ranks). Dual-card DSV4 on ~192 GB RAM will likely OOM."
        )
        return None
    require_shared_pool_capacity(
        need_bytes=DEFAULT_BANK_SHARE_NEED_BYTES if need_bytes is None else need_bytes,
        tp_size=tp.size,
    )
    return "memfd"


def _share_create_rank() -> bool:
    from freetoken.distributed import try_get_tp_info

    tp = try_get_tp_info()
    return tp is None or tp.rank == 0


def _tp_cpu_broadcast(obj):
    """Broadcast a picklable object from TP rank 0 over the gloo CPU group.

    Never the default GPU world: a collective there on HIP makes torch guess the
    device from the global rank ("Guessing device ID based on global rank. This
    can cause a hang if rank to GPU mapping is heterogeneous"), and handing a few
    ``/proc`` paths across ranks needs no GPU at all.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "shared expert banks need torch.distributed initialized (TP>1) before the "
            "bank build; init_tp_process_group runs first in the engine"
        )
    from freetoken.distributed.info import try_get_tp_cpu_group

    group = try_get_tp_cpu_group()
    if group is None and dist.get_backend() != "gloo":
        raise RuntimeError(
            "no gloo CPU group registered for the shared-bank handoff and the default "
            "torch.distributed group is not gloo; init_tp_process_group registers one"
        )
    box = [obj]
    dist.broadcast_object_list(box, src=0, group=group)
    return box[0]


def _create_shared_bank(name: str, asize: int) -> tuple[int, mmap.mmap]:
    """shmem ``MAP_SHARED`` buffer of ``asize`` bytes: ``(fd, map)``.

    The fd stays open for the bank's lifetime so ``/proc/<pid>/fd/<fd>`` keeps
    resolving for the other TP ranks (same uid; Yama only gates ptrace *attach*,
    not this read-mode access).
    """
    if not hasattr(os, "memfd_create"):
        raise RuntimeError(
            "shared expert banks need Linux memfd_create; set FREETOKEN_BANK_SHARE=0 "
            "to give each TP rank a private pool on this platform"
        )
    fd = os.memfd_create(f"freetoken-bank-{name}", os.MFD_CLOEXEC)
    try:
        try:
            os.ftruncate(fd, asize)
        except OSError as exc:
            if exc.errno in (errno.ENOSPC, errno.ENOMEM):
                raise RuntimeError(
                    f"shared bank {name}: cannot size {asize / (1 << 30):.1f} GiB of shmem "
                    f"({os.strerror(exc.errno)}). The pool lives in RAM; free memory or "
                    f"use a smaller model."
                ) from exc
            raise
        return fd, mmap.mmap(fd, asize)
    except BaseException:
        os.close(fd)
        raise


def _open_shared_bank(path: str, asize: int) -> mmap.mmap:
    """Map rank 0's shmem bank (a ``/proc/<pid>/fd/<fd>`` path) ``MAP_SHARED``."""
    fd = os.open(path, os.O_RDWR)
    try:
        st = os.fstat(fd)
        if st.st_size < asize:
            raise RuntimeError(
                f"shared bank {path} is {st.st_size} bytes, need {asize} "
                f"(rank 0 sizes the pool before broadcasting its paths)"
            )
        return mmap.mmap(fd, asize)
    finally:
        os.close(fd)  # the mapping keeps the inode alive


class HostBank:
    """A page-aligned host buffer + its torch view, page-locked on demand: allocate -> fill -> ``pin()``/``lock()``.

    * ``"mmap"`` (default) -- lazy anonymous mmap; pages materialize on fill, then ``pin()`` registers or ``lock()`` OS-locks it.
    * ``"cuda"`` -- cudaHostAlloc, born pinned+mapped; ``pin()``/``lock()``/``release()`` are no-ops and it never takes LOCKED. See :func:`born_pinned_default`.
    * shared shmem (``share=True`` on rank 0, ``share_path=`` on the other ranks) --
      one ``memfd`` mapped ``MAP_SHARED`` by every TP rank so the expert pool is
      resident once. ``share_path`` is rank 0's ``/proc/<pid>/fd/<fd>`` for the bank.
      Never a disk file: writeback of a registered file mapping is a KFD
      evict/restore storm (see :func:`prepare_shared_banks`).

    The buffer is rounded up to the O_DIRECT block; ``tensor`` views exactly ``nbytes``. ``backing=None`` follows ``FREETOKEN_BANK_CUDA_ALLOC``."""

    __slots__ = ("tensor", "addr", "nbytes", "_buf", "_pinned", "_locked", "_share_path", "_share_fd")

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype,
                 *, backing: str | None = None,
                 share: bool = False,
                 share_path: str | os.PathLike[str] | None = None,
                 name: str = "bank"):
        if backing is None:
            plan = _requested_residency
            # a plan with non-pinned labels vetoes born-pinned: cudaHostAlloc spends the pin quota the plan exists to save
            born = _env_born_pinned() and (plan is None or not plan.has_unpinned)
            backing = "cuda" if born else "mmap"
        assert backing in ("mmap", "cuda"), backing
        if share and share_path is not None:
            raise ValueError("share=True creates the shmem bank; share_path= opens rank 0's; not both")
        if (share or share_path is not None) and backing == "cuda":
            raise ValueError("shared host banks require mmap backing, not cudaHostAlloc")
        elsize = torch.empty((), dtype=dtype).element_size()
        self.nbytes = math.prod(shape) * elsize
        asize = ((self.nbytes + _BLK - 1) // _BLK) * _BLK
        self._share_path: str | None = None
        self._share_fd: int | None = None
        if backing == "cuda":
            from freetoken.kernel.pinned import alloc_pinned_tensor

            # direct-IO readers need page alignment, but cudaHostAlloc only guarantees ~512 in practice
            # over-allocate one block and carve the aligned window; the numpy slice keeps the pinned storage alive via .base
            raw = alloc_pinned_tensor(asize + _BLK, dtype=torch.uint8)  # cudaMallocHost
            raw.zero_()  # keep the anonymous-mmap guarantee: unwritten regions stay zero
            off = (-raw.data_ptr()) % _BLK
            self._buf = raw.numpy()[off:off + asize]
            self.addr = raw.data_ptr() + off
            assert self.addr % _BLK == 0
            self._pinned = True  # born pinned+mapped; pin() is a no-op
        else:
            if share:
                self._share_fd, self._buf = _create_shared_bank(name, asize)
                self._share_path = f"/proc/{os.getpid()}/fd/{self._share_fd}"
            elif share_path is not None:
                self._share_path = str(share_path)
                self._buf = _open_shared_bank(self._share_path, asize)
            else:
                self._buf = mmap.mmap(-1, asize)  # lazy: address space only, no resident pages yet
            _LIVE_BUFFERS.append(self._buf)
            self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
            self._pinned = False
        self.tensor = torch.frombuffer(self._buf, dtype=dtype, count=self.nbytes // elsize).view(*shape)
        self._locked = False

    @property
    def share_path(self) -> str | None:
        """``/proc/<pid>/fd/<fd>`` other TP ranks open to map this bank; None if private."""
        return self._share_path

    @property
    def residency(self) -> HostResidency:
        if self._pinned:
            return HostResidency.PINNED
        if self._locked:
            return HostResidency.LOCKED
        return HostResidency.PAGEABLE

    def memoryview(self) -> memoryview:
        return memoryview(self._buf)

    def flush(self) -> None:
        """``msync`` the mapping (a no-op for shmem beyond the page cache; kept for tests / rank handoff)."""
        buf = self._buf
        if hasattr(buf, "flush"):
            buf.flush()

    def pin(self) -> None:
        """cudaHostRegister the (now-filled) buffer -- pin-after-fill.

        ``FREETOKEN_SKIP_BANK_PIN=1`` makes this a no-op for CPU-only tooling (the FTW converter); never set it when serving, the GPU paths need registered banks."""
        if self._pinned:
            return
        if os.environ.get("FREETOKEN_SKIP_BANK_PIN", "").strip().lower() in ("1", "true", "yes", "on"):
            return
        from freetoken.kernel.pinned import host_register

        try:
            host_register(self.addr, len(self._buf))
        except RuntimeError as exc:
            raise RuntimeError(
                f"cudaHostRegister failed for {len(self._buf) / 2**30:.1f} GiB"
            ) from exc
        self._pinned = True

    def release(self) -> None:
        """Drop the resident pages; the address space stays valid, the contents become undefined.

        For buffers that are done being read (the converter). No-op for born-pinned banks: registered pages cannot be dropped."""
        if self._pinned:
            return
        self._buf.madvise(mmap.MADV_DONTNEED)

    def lock(self) -> None:
        """mlock the (now-filled) buffer: resident without CUDA pin quota, but no device address -- only the CPU executor can serve a locked layer.

        Lock after fill, or the lazy mmap faults+zero-fills every page. A failed lock (RLIMIT_MEMLOCK) warns once and leaves the bank PAGEABLE, which every consumer treats the same."""
        if self._locked or self._pinned:  # cudaHostRegister already page-locks
            return
        global _os_lock_failed
        if _os_lock_failed:
            return  # the quota is exhausted for good; skip the syscall spam
        try:
            _os_lock(self.addr, len(self._buf))
        except (OSError, ImportError) as exc:
            _os_lock_failed = True
            logger.warning(f"bank lock failed; leaving this and later banks pageable: {exc}")
            return
        self._locked = True


_os_locked_total = 0  # bytes locked so far; the OS lock ceiling is a per-process quota
_os_lock_failed = False  # sticky: once over quota, later (bigger-total) locks fail too


def _os_lock(addr: int, nbytes: int) -> None:
    global _os_locked_total
    import resource

    # grow the soft RLIMIT_MEMLOCK (defaults to a few MiB); the hard limit needs privilege, past it mlock fails below
    want = _os_locked_total + nbytes + (256 << 20)
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != resource.RLIM_INFINITY and soft < want:
        new_soft = want if hard == resource.RLIM_INFINITY else min(want, hard)
        if new_soft > soft:
            try:
                resource.setrlimit(resource.RLIMIT_MEMLOCK, (new_soft, hard))
            except (OSError, ValueError):
                pass  # keep the old limit; mlock below reports the real ceiling
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes)):
        err = ctypes.get_errno()
        raise OSError(
            err,
            f"mlock({nbytes / 2**30:.1f} GiB): {os.strerror(err)} "
            f"(RLIMIT_MEMLOCK / `ulimit -l` caps OS-locked bytes; raise it or "
            f"shrink --moe-cpu-layers)",
        )
    _os_locked_total += nbytes


def alloc_banks(specs: dict[str, tuple[tuple[int, ...], torch.dtype]]) -> dict[str, HostBank]:
    """Allocate (lazy, unpinned) host banks from ``{name: (shape, dtype)}``.

    Same TP>1 shmem handoff as :func:`alloc_layer_banks`, one bank per name."""
    if not bank_sharing_enabled():
        return {name: HostBank(shape, dtype) for name, (shape, dtype) in specs.items()}
    keys = list(specs)
    if _share_create_rank():
        out = {name: HostBank(shape, dtype, share=True, name=name) for name, (shape, dtype) in specs.items()}
        _tp_cpu_broadcast([(name, out[name].share_path) for name in keys])
        return out
    handoff = _tp_cpu_broadcast(None)
    got = [name for name, _ in handoff]
    if got != keys:
        raise RuntimeError(
            f"shared expert banks: rank 0 built {got}, this rank expects {keys}; "
            f"the TP ranks disagree on the bank layout"
        )
    paths = dict(handoff)
    return {name: HostBank(shape, dtype, share_path=paths[name]) for name, (shape, dtype) in specs.items()}


def alloc_layer_banks(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]], num_layers: int
) -> dict[str, list[HostBank]]:
    """Allocate per-layer host banks: ``{name: ([num_experts, ...] row shape, dtype)}``
    -> one independently allocated (page-aligned, independently pin/lock-able)
    ``HostBank`` per layer per name.

    TP>1 (unless ``FREETOKEN_BANK_SHARE=0``): rank 0 creates every bank as shmem
    (``memfd``) and broadcasts their ``/proc`` paths over the gloo CPU group; the
    other ranks map the same inodes ``MAP_SHARED``, so the pool is resident once.
    """
    if not bank_sharing_enabled():
        return {
            name: [HostBank(shape, dtype) for _ in range(num_layers)]
            for name, (shape, dtype) in specs.items()
        }
    keys = [(name, layer) for name in specs for layer in range(num_layers)]
    if _share_create_rank():
        out = {
            name: [HostBank(shape, dtype, share=True, name=f"{name}.L{layer}") for layer in range(num_layers)]
            for name, (shape, dtype) in specs.items()
        }
        _tp_cpu_broadcast([(name, layer, out[name][layer].share_path) for name, layer in keys])
        return out
    handoff = _tp_cpu_broadcast(None)
    got = [(name, layer) for name, layer, _ in handoff]
    if got != keys:
        raise RuntimeError(
            f"shared expert banks: rank 0 built {len(got)} banks {got[:3]}..., this rank "
            f"expects {len(keys)} {keys[:3]}...; the TP ranks disagree on the bank layout"
        )
    paths = {(name, layer): path for name, layer, path in handoff}
    return {
        name: [HostBank(shape, dtype, share_path=paths[(name, layer)]) for layer in range(num_layers)]
        for name, (shape, dtype) in specs.items()
    }


class _ResidencyPlan:
    """Per-layer ``HostResidency`` labels, ambiently visible to the bank settle points.

    Installed by ``load_expert_banks`` around the provider dispatch so every loader honors --moe-cpu-layers without a new parameter in each signature. ``applied`` flips once a settle point consults the plan."""

    __slots__ = ("labels", "applied", "has_unpinned", "actual")

    def __init__(self, labels: list[str]):
        self.labels = list(labels)
        self.applied = False
        self.has_unpinned = any(r != HostResidency.PINNED.value for r in labels)
        self.actual: dict[int, str] = {}

    def residency_for(self, layer_id: int) -> str:
        self.applied = True
        return self.labels[layer_id]

    def record(self, layer_id: int, achieved: str) -> None:
        """One pageable bank downgrades the whole layer (a failed lock settles PAGEABLE)."""
        if self.actual.get(layer_id) != HostResidency.PAGEABLE.value:
            self.actual[layer_id] = achieved


_requested_residency: _ResidencyPlan | None = None


@contextlib.contextmanager
def requested_residency(labels: list[str] | None):
    """Install the ambient per-layer residency plan for the enclosed bank load (``None`` = no plan, everything pins)."""
    global _requested_residency
    if labels is None:
        yield None
        return
    plan = _ResidencyPlan(labels)
    prev, _requested_residency = _requested_residency, plan
    try:
        yield plan
    finally:
        _requested_residency = prev


def _settle(bank: HostBank, residency: str) -> None:
    """Route a filled bank to its residency class (PAGEABLE = leave the plain mmap)."""
    if residency == HostResidency.PINNED.value:
        bank.pin()
    elif residency == HostResidency.LOCKED.value:
        bank.lock()


def pin_banks(banks: dict[str, HostBank | list[HostBank]]) -> None:
    """Settle every bank after it has been filled -- pin-after-fill by default.
    List-valued entries are per-layer and honor the ambient :func:`requested_residency` plan; scalar banks always pin."""
    plan = _requested_residency
    for bank in banks.values():
        if isinstance(bank, list):
            for layer_id, layer_bank in enumerate(bank):
                residency = (
                    HostResidency.PINNED.value if plan is None
                    else plan.residency_for(layer_id)
                )
                _settle(layer_bank, residency)
                if plan is not None and residency == HostResidency.LOCKED.value:
                    plan.record(layer_id, layer_bank.residency.value)
        else:
            bank.pin()


class PinPipeline:
    """Settle (pin or lock) filled banks while other banks are still being read.

    cudaHostRegister is driver-serialized, so one background thread drains a queue and submitters never block: load time ~= max(read, settle).
    LOCKED banks mlock on the same thread (the quota bookkeeping in ``_os_lock`` is not thread-safe).
    A clean context-manager exit drains the queue and re-raises the first settle failure.
    """

    def __init__(self) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._exc: BaseException | None = None
        # the current device is thread-local: a fresh thread sits on device 0 and cudaHostRegister would build its context there -- carry the creator's (bound) device into the worker
        self._device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if self._device is not None:
            torch.cuda.set_device(self._device)
        while True:
            item = self._q.get()
            if item is None:
                return
            if self._exc is not None:
                continue  # drain without settling after a failure
            bank, residency, plan, layer_id = item
            try:
                _settle(bank, residency)
                if plan is not None and residency == HostResidency.LOCKED.value:
                    plan.record(layer_id, bank.residency.value)
            except BaseException as exc:  # surfaced by wait()/__exit__
                self._exc = exc

    def submit(self, bank: HostBank, residency: str = HostResidency.PINNED.value,
               plan=None, layer_id: int | None = None) -> None:
        self._q.put((bank, residency, plan, layer_id))

    def __call__(self, layer_id: int, banks: dict[str, HostBank]) -> None:
        """Layer-completion sink: queue every bank of the completed layer at its ambient :func:`requested_residency` label."""
        plan = _requested_residency
        residency = (
            HostResidency.PINNED.value if plan is None else plan.residency_for(layer_id)
        )
        for bank in banks.values():
            self.submit(bank, residency, plan, layer_id)

    def _join(self) -> None:
        self._q.put(None)
        self._thread.join()

    def wait(self) -> None:
        self._join()
        if self._exc is not None:
            raise self._exc

    def __enter__(self) -> "PinPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._join()  # no thread leak; the in-flight exception wins
            return
        self.wait()


class LayerCompletionTracker:
    """Fire a sink once per layer, when all of that layer's writes have landed.

    ``note(layer_id)`` is called after each write; at ``expected_per_layer``
    notes the layer's banks are handed to ``on_layer(layer_id, {name: bank})``
    exactly once. Thread-safe (shard-driven loaders write layers from many
    threads in arbitrary order).
    """

    def __init__(
        self,
        expected_per_layer: int,
        banks: dict[str, list],
        on_layer,
    ) -> None:
        assert expected_per_layer > 0
        self._expected = expected_per_layer
        self._banks = banks
        self._on_layer = on_layer
        self._counts: dict[int, int] = {}
        self._lock = threading.Lock()

    def note(self, layer_id: int) -> None:
        with self._lock:
            n = self._counts.get(layer_id, 0) + 1
            self._counts[layer_id] = n
            fire = n == self._expected
        if fire:
            self._on_layer(layer_id, {name: per[layer_id] for name, per in self._banks.items()})


def read_file_into(buf: memoryview | mmap.mmap, path: str, *, workers: int = 8,
                   chunk: int = _DEFAULT_CHUNK, drop_cache: bool = True) -> int:
    """Chunked multi-threaded O_DIRECT read of the whole file ``path`` into ``buf``
    (page-aligned). Returns the file size. The buffer must be >= the rounded-up file size."""
    size = os.path.getsize(path)
    if drop_cache:
        try:
            fd0 = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd0, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd0)
        except OSError:
            pass
    mv = buf if isinstance(buf, memoryview) else memoryview(buf)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    offs = list(range(0, size, chunk))

    def rd(o):
        want = min(chunk, len(mv) - o)
        want = min(want, ((size - o + _BLK - 1) // _BLK) * _BLK)
        os.preadv(fd, [mv[o:o + want]], o)

    try:
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        os.close(fd)
    return size


def _preadv_all(fd: int, dst: memoryview, offset: int, need: int) -> None:
    """preadv into ``dst`` until ``need`` bytes have landed; O_DIRECT may return a short count."""
    done = 0
    while done < need:
        if done % _BLK:  # a continuation read has to stay block-aligned on both sides
            raise OSError(f"unaligned short O_DIRECT read: {done} of {need} bytes at {offset}")
        got = os.preadv(fd, [dst[done:]], offset + done)
        if got <= 0:
            raise OSError(f"short O_DIRECT read: {done} of {need} bytes at {offset}")
        done += got


def read_range_into(buf: memoryview | mmap.mmap, path: str, *, file_offset: int, nbytes: int,
                    dest_offset: int = 0, workers: int = 8, chunk: int = _DEFAULT_CHUNK,
                    drop_cache: bool = True) -> int:
    """Chunked multi-threaded O_DIRECT read of ``path[file_offset : file_offset + nbytes]`` into ``buf`` at ``dest_offset``. Returns ``nbytes``.

    Byte-range counterpart of :func:`read_file_into`, for one tensor inside a shard. O_DIRECT needs the file offset AND the destination address block-aligned at the same time, which only holds when the two share their offset mod 4096 -- a safetensors data offset practically never lines up with the tensor's slot in the bank. Chunks that do line up DMA straight into ``buf``; the rest DMA into a page-aligned bounce (source window rounded out to whole blocks) and are copied into place, which also covers the unaligned head and tail.
    """
    mv = (buf if isinstance(buf, memoryview) else memoryview(buf)).cast("B")
    if dest_offset + nbytes > len(mv):
        raise ValueError(f"destination holds {len(mv)} bytes, need {dest_offset + nbytes}")
    base = ctypes.addressof(ctypes.c_char.from_buffer(mv))
    if drop_cache:
        try:
            fd0 = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd0, file_offset, nbytes, os.POSIX_FADV_DONTNEED)
            os.close(fd0)
        except OSError:
            pass
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    scratch = threading.local()

    def rd(i: int) -> None:
        n = min(chunk, nbytes - i)
        src, dst = file_offset + i, dest_offset + i
        if src % _BLK == 0 and (base + dst) % _BLK == 0 and n % _BLK == 0:
            _preadv_all(fd, mv[dst:dst + n], src, n)
            return
        head = src % _BLK
        span = ((head + n + _BLK - 1) // _BLK) * _BLK
        bounce = getattr(scratch, "buf", None)
        if bounce is None or len(bounce) < span:
            bounce = scratch.buf = mmap.mmap(-1, span)  # anonymous mmaps are page-aligned
        bmv = memoryview(bounce)
        _preadv_all(fd, bmv[:span], src - head, head + n)
        mv[dst:dst + n] = bmv[head:head + n]

    try:
        offs = list(range(0, nbytes, chunk))
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        os.close(fd)
    return nbytes


__all__ = [
    "HostBank",
    "HostResidency",
    "LayerCompletionTracker",
    "PinPipeline",
    "alloc_banks",
    "alloc_layer_banks",
    "bank_sharing_enabled",
    "born_pinned_default",
    "pin_banks",
    "DEFAULT_BANK_SHARE_NEED_BYTES",
    "kfd_system_mem_limit_bytes",
    "prepare_shared_banks",
    "read_file_into",
    "read_range_into",
    "requested_residency",
    "require_shared_pool_capacity",
]
