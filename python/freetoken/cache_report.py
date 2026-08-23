"""Rendering a server's cache geometry for a human: which pools the model has, how big each
one is, and what it costs in VRAM.

Shared by the two clients that report it -- ``ft ctl cache`` and the shell's ``/cache`` -- so
the same server document reads the same way whichever one you are holding. Everything here is
derived from ``GET /v1/cache/status``; nothing talks to a server and nothing imports torch or
prompt_toolkit, so ``ft ctl`` stays a dependency-light CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

# Every pool the rebuild endpoint knows about, and the unit a user types it in. Which of them
# a given model actually has is decided per server -- see CachePools.
CACHE_UNITS = {"moe": "slots", "kv": "tokens", "mamba": "slots", "swa": "tokens"}


def _int(source: dict, key: str) -> int:
    """Read an int out of a server document defensively: a missing key, a null, and a 0 all
    mean the same thing here -- "nothing to report" -- and none of them may raise."""
    try:
        return int((source or {}).get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class CachePools:
    """Which cache pools the served model actually has, read off its geometry.

    Drives what a client offers and accepts: a DSV4 has a window pool but no GDN state pool, a
    hybrid Qwen has the reverse, a dense model has no expert cache. Offering a target the model
    does not have is just a way to hand the user an error from the engine."""

    moe: bool = False
    kv: bool = True  # every model has a KV pool; the rebuild endpoint always takes num_pages
    mamba: bool = False
    swa: bool = False

    @property
    def targets(self) -> Tuple[str, ...]:
        return tuple(
            name
            for name, present in (
                ("moe", self.moe), ("kv", self.kv), ("mamba", self.mamba), ("swa", self.swa)
            )
            if present
        )

    @classmethod
    def from_geometry(cls, geometry: dict) -> "CachePools":
        unit = (geometry or {}).get("unit_bytes") or {}
        experts = _int(geometry, "num_experts") * _int(geometry, "num_moe_layers")
        return cls(
            moe=bool(experts or _int(geometry, "moe_cache_size") or _int(unit, "moe_per_expert")),
            mamba=bool(_int(geometry, "num_mamba_slots") or _int(unit, "mamba_per_slot")),
            # swa_page_size is the definitive signal: it is only non-zero for a window pool.
            swa=bool(_int(geometry, "swa_page_size") or _int(unit, "swa_per_token")),
        )


def format_percent(rate: float) -> str:
    percent = rate * 100
    if abs(percent - round(percent)) < 1e-9:
        return f"{percent:.0f}%"
    return f"{percent:.1f}%"


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1 << 30:
        return f"{num_bytes / (1 << 30):.1f} GiB"
    return f"{num_bytes / (1 << 20):.1f} MiB"


def format_tokens(pages: int, page_size: int) -> str:
    """``<tokens> tok`` for a paged pool, with the page arithmetic spelled out -- a rebuild
    takes tokens but allocates pages, so the decomposition is what makes a rounded-up request
    (and the pool's granularity) legible."""
    return f"{pages * page_size} tok ({pages} x {page_size})"


def cache_rate(cache_size: int, geometry: dict) -> float | None:
    """MoE residency: cached slots / the model's total routed experts (experts per layer x MoE
    layers, the same basis the engine sizes the cache against). None for a non-MoE model, or a
    server that reports no expert counts."""
    total = _int(geometry, "num_experts") * _int(geometry, "num_moe_layers")
    if total <= 0 or cache_size <= 0:
        return None
    return cache_size / total


def pages_for_tokens(tokens: int | None, page_size: int) -> int | None:
    """Round a token count up to whole pages. Pools are allocated in pages, so asking for 1000
    tokens of a 64-token page gets 16 pages (1024 tokens) -- never 15, which would be less than
    what was asked for."""
    if tokens is None:
        return None
    return max(1, -(-tokens // max(1, page_size)))


def pool_bytes(geometry: dict, pool: str, units: int | None = None) -> int:
    """VRAM a pool holds: its unit count x the engine's per-unit cost.

    This is the ALLOCATED size, not occupancy -- the pool is reserved up front, so this is what
    it costs whether or not anything is cached in it. The per-unit costs come from the ack the
    backend sends at load (compute_cache_status_meta), the same numbers the desktop's cache
    panel sizes its sliders from. 0 means "unknown": a pool the model does not have, or a
    server that reported no unit_bytes at all -- callers must not print a 0 as a real figure.

    ``units`` overrides the count from the geometry, for costing a size that is not live yet
    (a rebuild target, or the size a rebuild just returned)."""
    unit_bytes = (geometry or {}).get("unit_bytes") or {}
    if pool == "moe":
        count = _int(geometry, "moe_cache_size") if units is None else units
        return count * _int(unit_bytes, "moe_per_expert")
    if pool == "mamba":
        count = _int(geometry, "num_mamba_slots") if units is None else units
        return count * _int(unit_bytes, "mamba_per_slot")
    if pool == "kv":  # units are pages; the cost is per token
        pages = _int(geometry, "num_pages") if units is None else units
        return pages * max(1, _int(geometry, "page_size")) * _int(unit_bytes, "kv_per_token")
    if pool == "swa":
        pages = _int(geometry, "num_swa_pages") if units is None else units
        return pages * _int(geometry, "swa_page_size") * _int(unit_bytes, "swa_per_token")
    return 0


# Where each pool's {min, max} lives in the geometry's ``limits`` block. Conveniently, the
# server already denominates those in the unit a client types: tokens for the paged pools,
# slots for the others -- so the range reads as "what you may ask this pool to become".
_LIMIT_KEYS = {"moe": "moe_experts", "kv": "kv_tokens", "mamba": "mamba_slots", "swa": "swa_tokens"}


def format_range(geometry: dict, pool: str) -> str:
    """``min..max`` a rebuild accepts for a pool, in the same unit the size column shows.
    Empty when the server advertised no usable bounds (older engine, or a pool it treats as
    unknown) -- an empty range prints as an empty cell rather than a misleading ``0..0``."""
    limits = (geometry or {}).get("limits")
    item = limits.get(_LIMIT_KEYS.get(pool, "")) if isinstance(limits, dict) else None
    if not isinstance(item, dict):
        return ""
    low, high = _int(item, "min"), _int(item, "max")
    return f"{low}..{high}" if high > 0 else ""


def cache_status_rows(geometry: dict) -> List[Tuple[str, str, int, str]]:
    """``(pool, size, allocated_bytes, resize_range)`` for every pool the model has, in report
    order."""
    pools = CachePools.from_geometry(geometry)
    page_size = max(1, _int(geometry, "page_size"))
    swa_page_size = _int(geometry, "swa_page_size")
    rows: List[Tuple[str, str, int, str]] = []

    def _row(pool: str, detail: str) -> Tuple[str, str, int, str]:
        return (pool, detail, pool_bytes(geometry, pool), format_range(geometry, pool))

    if pools.moe:
        moe = _int(geometry, "moe_cache_size")
        detail = f"{moe} slots ({(geometry or {}).get('moe_cache_policy') or 'lru'}"
        rate = cache_rate(moe, geometry)
        detail += f", {format_percent(rate)})" if rate is not None else ")"
        rows.append(_row("moe", detail))
    kv_pages = _int(geometry, "num_pages")
    if kv_pages > 0:
        rows.append(_row("kv", format_tokens(kv_pages, page_size)))
    mamba = _int(geometry, "num_mamba_slots")
    if mamba > 0:  # hybrid (GDN) models only
        rows.append(_row("mamba", f"{mamba} slots"))
    swa_pages = _int(geometry, "num_swa_pages")
    if swa_page_size > 0 and swa_pages > 0:  # SWA (window pool) models only
        rows.append(_row("swa", format_tokens(swa_pages, swa_page_size)))
    return rows


def format_cache_status(doc: dict, *, prefix: str = "cache: ") -> str:
    """The served geometry as a table -- one row per pool the model has, carrying its size, the
    VRAM it holds, and the range a rebuild would accept -- under a header with the maintenance
    state and the engine's cache budget.

    Columns that nothing can be said about are dropped whole rather than filled with zeros: a
    server that reported no per-unit costs gets no vram column (a 0.0 GiB would be a lie, and
    the total then sums only what is known), one that advertised no limits gets no range."""
    geometry = (doc or {}).get("geometry") or {}
    rows = cache_status_rows(geometry)
    known = [size for _pool, _detail, size, _range in rows if size > 0]
    budget = _int(geometry, "cache_budget_bytes")

    header = f"{prefix}state={(doc or {}).get('state', 'serving')}"
    requested_prefill = _int(geometry, "requested_prefill_tokens")
    pool_prefill = _int(geometry, "pool_prefill_cap_tokens")
    effective_prefill = _int(geometry, "effective_prefill_tokens")
    if requested_prefill or effective_prefill:
        header += (
            f", prefill={effective_prefill} tok"
            f" (requested {requested_prefill}, pool cap {pool_prefill or 'none'}, "
            f"source {geometry.get('swa_capacity_source', 'none')}, "
            f"reason {geometry.get('prefill_limiting_reason', 'none')})"
        )
    if known:
        header += f", {format_bytes(sum(known))} allocated"
        if budget > 0:
            # Named for what it is: the ceiling the rebuild fit-check enforces, NOT a cap on
            # what is already allocated. An auto-sized pool can sit above it (and then every
            # rebuild is rejected), so calling it "of X" would read as an arithmetic error.
            header += f" (rebuild budget {format_bytes(budget)})"
    if not rows:
        return header

    with_vram = bool(known)
    with_range = any(resize for _pool, _detail, _size, resize in rows)
    table = [["pool", "size"] + (["vram"] if with_vram else []) + (
        ["resizable to"] if with_range else []
    )]
    for pool, detail, size, resize in rows:
        cells = [pool, detail]
        if with_vram:
            cells.append(format_bytes(size) if size > 0 else "")
        if with_range:
            cells.append(resize)
        table.append(cells)

    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    return "\n".join(
        [header]
        + ["  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in table]
    )
