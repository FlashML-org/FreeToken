"""The startup banner: what this process is, printed before it takes 90 seconds to load.

A log that opens mid-way through loading experts answers none of the questions asked of
it afterwards. Nearly every confusing run in this engine's history was a configuration
question -- was dSpark on, how many ranks, which NUMA layout, which checkpoint -- and the
answer was derivable from the log only by knowing which incidental line to read. A run
with speculation off is distinguishable from one with it on by the expert-loader's
``layers=43`` versus ``layers=46``, which is not a thing anyone should have to know.

So the banner states the configuration outright, at the top, where a pasted log excerpt
will usually include it.
"""

from __future__ import annotations

import os
import platform
import sys

# The wordmark, from assets/freetoken-logo-dark.svg. 82 columns.
#
# Halved from a 163-column rendering by folding each 2x2 cell into one quadrant block
# glyph, which keeps the letterforms legible at half the width -- a plain every-other-
# column drop would have shredded the thin strokes. 163 columns overflowed narrower
# monitors, and a banner that wraps is worse than no banner.
#
# It prints in one colour. The glyphs interlock along the logo's slant, so the SVG's
# two-tone split ("ree" in off-white, the rest blue) cannot be reproduced by slicing
# columns without cutting through letters. Blue is the mark's dominant colour.
_ART = [
    "   ▜▘█████████▌                       ▗▄▄▄▄▄▄▄▄▄        ▄▄▄",
    "   ██████▀▀▀▀▀           ▄▄       ▄▄  █████████▌  ▄▖    ███        ▗▄         ▄",
    " ▗█▌▟███████▌  ██████▌▟█████▙▖ ▟█████▙▖  ███▘ ▗▟█████▙ ▐██▛▄███▀▗██████▖ ████████▖",
    "  ▗▄▄███████▌ ▗███▀▀▀███▙▄▟██▌▟██▙▄▟██▌ ▐███ ▗███▘ ▜██▌▟█████▀ ▐███▄▄███▗███▀▝███▌",
    " ▄▖▄▄███      ▐██▛   ███▀▀▀▀▀▘███▀▀▀▀▀▘ ▟██▌ ▐██▛  ███▘██████▖ ▟██▛▀▀▀▀▀▐██▌  ███",
    " ▝ ████▛      ███▘   ▜██████▛ ▜██████▛  ███▘ ▝███████▘▗██▛▝███▄▝██████▛ ███▘ ▐██▛",
    "   ▀▀▀▀▘      ▀▀▀     ▝▀▀▀▀    ▝▀▀▀▀    ▀▀▀    ▀▀▀▀▀  ▝▀▀▘ ▝▀▀▀  ▀▀▀▀   ▀▀▀  ▝▀▀▘",
]

# Shown instead when the terminal is too narrow for _ART. A 163-column banner that wraps
# is worse than no banner: every row folds and the logo becomes noise.
_ART_NARROW = [
    "   ══       ______              ______     __              ",
    "  ════     / ____/_______  ___ /_  __/___ / /_____  ____    ",
    " ═══      / /_  / ___/ _ \\/ _ \\ / / / __ \\/ //_/ _ \\/ __ \\  ",
    "  ══     / __/ / /  /  __/  __// / / /_/ / ,< /  __/ / / /  ",
    "   ═    /_/   /_/   \\___/\\___//_/  \\____/_/|_|\\___/_/ /_/   ",
]

_BLUE = "\033[38;2;88;166;240m"   # #58a6f0, the logo's blue
_WHITE = "\033[38;2;242;245;248m"  # #f2f5f8, the logo's off-white
_DIM = "\033[2m"
_OFF = "\033[0m"


def _use_colour(stream) -> bool:
    """Colour only where it will not become literal escape codes in a file.

    A banner is for humans, and this one is usually read in a redirected log. NO_COLOR
    is honoured because a startup banner is exactly the sort of thing people redirect
    into files and paste into issues.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _art(colour: bool) -> list[str]:
    """The wordmark, or the compact one when it will not fit.

    Width is only knowable for a terminal. A redirected log has no width, and the file
    holds long lines fine, so the full mark is the default there.
    """
    rows = _ART
    try:
        cols = os.get_terminal_size().columns
        if cols and cols < max(len(r) for r in _ART):
            rows = _ART_NARROW
    except OSError:
        pass  # not a terminal: a file takes the full mark
    if not colour:
        return list(rows)
    return [f"{_BLUE}{r}{_OFF}" for r in rows]


def _gpu_line() -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # One decimal on both, or a 15.6 GiB card reads as "16 GiB each, 62 GiB
        # total" and the arithmetic looks broken.
        return f"{n} x {name} ({gib:.1f} GiB each, {n * gib:.1f} GiB total)"
    except Exception:
        return None


def _cpu_line() -> str | None:
    """Physical cores, logical CPUs, and how many this process may actually use.

    The last number is the one that matters and the one nobody has: a cpuset or a
    taskset makes os.cpu_count() a lie, and the CPU MoE pool sizes itself from the
    permitted set, not the machine.
    """
    try:
        logical = os.cpu_count() or 0
        allowed = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else logical
        phys = None
        try:
            from freetoken.moe.cpu_executor import physical_core_cpus

            phys = len(physical_core_cpus())
        except Exception:
            pass
        desc = f"{phys} physical cores, {logical} logical" if phys else f"{logical} logical cpus"
        if allowed and allowed != logical:
            desc += f" ({allowed} permitted to this process)"
        return desc
    except Exception:
        return None


def _ram_line() -> str | None:
    """Total and available RAM.

    An offload MoE keeps its expert banks in host memory, so "available" is the number
    that decides whether a run fits -- and a SIGKILL'd rank can leave tens of GiB of
    pinned Shmem behind, which shows up here as a shortfall with no process to blame.
    """
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                info[k] = int(rest.split()[0])  # kB
        total = info.get("MemTotal", 0) / 1024**2
        avail = info.get("MemAvailable", 0) / 1024**2
        shmem = info.get("Shmem", 0) / 1024**2
        desc = f"{total:.0f} GiB total, {avail:.0f} GiB available"
        if shmem >= 1:
            desc += f", {shmem:.0f} GiB shmem"
        return desc
    except Exception:
        return None


def _numa_line(world_size: int) -> str | None:
    """The rank -> node map, which is otherwise only inferable from per-rank log lines."""
    try:
        from freetoken.utils import numa

        nodes = numa.numa_nodes()
        if len(nodes) < 2:
            return f"1 node ({len(nodes[0]) if nodes else '?'} cpus) -- nothing to spread"
        placed = []
        for r in range(world_size):
            p = numa.rank_placement(r, world_size)
            placed.append(str(p[0]) if p else "?")
        cpus = sum(len(n) for n in nodes)
        return f"{len(nodes)} nodes, {cpus} cpus | rank->node: {' '.join(placed)}"
    except Exception:
        return None


def format_banner(
    *,
    model_path: str | None = None,
    quant: str | None = None,
    tp_size: int = 1,
    dspark: bool = False,
    dspark_block: int | None = None,
    dspark_layers: int | None = None,
    moe_backend: str | None = None,
    host: str | None = None,
    port: int | None = None,
    colour: bool = False,
) -> str:
    from freetoken.version import __version__

    rows = _art(colour)
    d = _DIM if colour else ""
    off = _OFF if colour else ""

    # Facts worth a line each. Everything here has cost real debugging time to recover
    # from a log that did not say it.
    facts: list[tuple[str, str]] = [("version", __version__)]
    if model_path:
        facts.append(("model", os.path.basename(model_path.rstrip("/"))))
    if quant:
        facts.append(("experts", quant))
    if moe_backend:
        facts.append(("moe", moe_backend))
    facts.append(("parallel", f"TP={tp_size}"))
    gpu = _gpu_line()
    if gpu:
        facts.append(("gpu", gpu))
    cpu = _cpu_line()
    if cpu:
        facts.append(("cpu", cpu))
    ram = _ram_line()
    if ram:
        facts.append(("ram", ram))
    numa_desc = _numa_line(tp_size)
    if numa_desc:
        facts.append(("numa", numa_desc))
    if dspark:
        detail = "on"
        if dspark_block:
            detail += f" (block={dspark_block}"
            detail += f", {dspark_layers} draft layers)" if dspark_layers else ")"
        facts.append(("dspark", detail))
    else:
        facts.append(("dspark", "off -- plain autoregressive decode"))
    if host is not None and port is not None:
        facts.append(("serving", f"{host}:{port}"))
    facts.append(
        ("runtime", f"python {platform.python_version()} | {_torch_desc()}")
    )

    width = max(len(k) for k, _ in facts)
    lines = list(rows)
    lines.append("")
    for k, v in facts:
        lines.append(f"  {d}{k.rjust(width)}{off}  {v}")
    lines.append("")
    return "\n".join(lines)


def _torch_desc() -> str:
    """torch's build, and BOTH CUDA versions, because they legitimately differ.

    torch.version.cuda is the toolkit the wheel was COMPILED against, not the CUDA the
    installed driver provides. A cu130 wheel runs on a 13.3 driver through minor-version
    compatibility. Printing only the build version reads as "this box is on the wrong
    CUDA" to anyone who knows what their driver reports -- so print both, labelled.
    """
    try:
        import torch
    except Exception:
        return "torch unavailable"

    built = torch.version.cuda or "none"
    desc = f"torch {torch.__version__} | cuda {built} build"
    try:
        # Ask libcuda directly. torch._C._cuda_getDriverVersion does not exist in every
        # build (it is absent in 2.11.0+cu130), and cuDriverGetVersion is the same
        # number nvidia-smi reports, with no subprocess.
        import ctypes

        raw = ctypes.c_int(0)
        lib = ctypes.CDLL("libcuda.so.1")
        if lib.cuDriverGetVersion(ctypes.byref(raw)) != 0:
            raise OSError("cuDriverGetVersion failed")
        raw = raw.value  # packed 1000*major + 10*minor, e.g. 13030 -> 13.3
        if raw:
            desc += f" / {raw // 1000}.{(raw % 1000) // 10} driver"
    except Exception:
        pass
    return desc


def print_banner(stream=None, **kw) -> None:
    """Write the banner. Rank 0 only -- the caller decides that, not this."""
    stream = stream or sys.stderr
    try:
        stream.write(format_banner(colour=_use_colour(stream), **kw) + "\n")
        stream.flush()
    except Exception:
        # A banner must never be the reason a server fails to start.
        pass


__all__ = ["format_banner", "print_banner"]
