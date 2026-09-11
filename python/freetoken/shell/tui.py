"""The interactive terminal chat: prompt loop, status bar, and the ``/`` commands.

Everything it knows about the model comes over HTTP from the server it attached to (see
:mod:`freetoken.shell.client`) -- the served model id, the thinking gears the family supports,
the cache geometry behind the status bar. No engine imports, no torch.
"""

from __future__ import annotations

import asyncio
import os
import contextlib
import re
import shutil
import signal
import time
from dataclasses import dataclass
from typing import Any, List, Tuple

from freetoken.cache_report import (
    CACHE_UNITS,
    CachePools,
    cache_rate,
    format_bytes,
    format_cache_status,
    format_percent,
    format_tokens,
    pages_for_tokens,
    pool_bytes,
)
from freetoken.env import ENV
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.styles import Style

from .client import (
    LOCAL_API_KEY,
    ContentDelta,
    ReasoningDelta,
    Sampling,
    ShellClient,
    ShellClientError,
    TurnDone,
)
from .render import (
    SHELL_FALLBACK_WIDTH,
    SHELL_STATUS_REFRESH_INTERVAL,
    ShellConsoleRenderer,
    ShellOutputBuffer,
    ShellStatusLine,
)

# How often the status bar re-reads /v1/stats while a turn is streaming. Matched to the bar's
# own repaint interval; the endpoint is a dict read on the server and is excluded from its
# access log (see server/access_log_filter.py), so polling it is free.
STATS_POLL_INTERVAL = SHELL_STATUS_REFRESH_INTERVAL
# Minimum gap between two "loading ..." lines while waiting for a server to come up. A phase
# change always prints immediately.
LOAD_PROGRESS_INTERVAL = 2.0


def _format_shell_model_label(model_id: str) -> str:
    name = str(model_id).rstrip("/").rsplit("/", 1)[-1] or str(model_id)
    lower_name = name.lower()
    if "qwen3.5" in lower_name:
        return "Qwen3.5"
    if "qwen3" in lower_name:
        return "Qwen3"
    if "gemma-4" in lower_name or "gemma4" in lower_name:
        return "Gemma4"
    return name


def _apply_think_command(
    arg: str, gears: Tuple[str, ...], current: str | None
) -> Tuple[str | None, str]:
    """Resolve a ``/think`` argument to ``(new_gear, message)``. ``new_gear`` is the gear to use
    going forward (unchanged on status/invalid input); ``message`` is the line to print."""
    if not gears:
        return current, "This model has no shell-controllable thinking."
    if arg in ("", "status"):
        return current, f"Thinking: {current} (available: {', '.join(gears)})"
    if arg == "toggle":
        idx = gears.index(current) if current in gears else -1
        nxt = gears[(idx + 1) % len(gears)]
        return nxt, f"Thinking: {nxt}"
    if arg in gears:
        return arg, f"Thinking: {arg}"
    return current, f"Usage: /think [{'|'.join(gears)}|toggle|status]"


def _cache_targets_hint(pools: CachePools) -> str:
    return " | ".join(f"{name} <N>" for name in pools.targets)


# A plausible value per target, so the generated example reads like something worth typing
# (a token count for the paged pools, a slot count for the others) whatever the model has.
_CACHE_EXAMPLES = {"moe": "4k", "kv": "128k", "mamba": "64", "swa": "32k"}


def _cache_usage(pools: CachePools) -> str:
    """The ``/cache`` usage line for THIS model: only its pools, with the unit each takes."""

    def _units_clause(unit: str, noun: str, qualifier: str = "") -> str:
        names = [n for n in pools.targets if CACHE_UNITS[n] == unit]
        if not names:
            return ""
        subject = (
            f"{names[0]} is a {noun}" if len(names) == 1 else f"{' and '.join(names)} are {noun}s"
        )
        return subject + qualifier

    clauses = [
        clause
        for clause in (
            _units_clause("tokens", "TOKEN count", " (rounded up to the pool's page size)"),
            _units_clause("slots", "slot count"),
        )
        if clause
    ]
    example = " ".join(f"{name} {_CACHE_EXAMPLES[name]}" for name in pools.targets)
    return (
        f"Usage: /cache [status | {_cache_targets_hint(pools)}]; combine targets, e.g. "
        f"/cache {example}. " + "; ".join(clauses) + ". N accepts a k/m suffix (k=1024, m=1024^2)."
    )


def _parse_count(token: str) -> int | None:
    """Parse a positive count with an optional ``k``/``m`` suffix (x1024 / x1024^2, case-
    insensitive -- these are cache sizes, so the binary units are the ones that land on page
    boundaries). Decimals are allowed when the result is whole (``1.5k`` -> 1536). Returns
    ``None`` for anything non-positive or malformed (``1.5`` with no suffix, ``2g``)."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*", token)
    if not match:
        return None
    multiplier = {"": 1, "k": 1024, "m": 1024 * 1024}[match.group(2).lower()]
    value = float(match.group(1)) * multiplier
    if value <= 0 or value != int(value):
        return None
    return int(value)


@dataclass(frozen=True)
class CacheCommand:
    """A parsed ``/cache`` invocation. ``kv``/``swa`` are token counts as typed -- the page
    conversion needs the server's geometry and happens at the call site."""

    action: str  # "status" | "rebuild" | "error"
    moe: int | None = None
    kv_tokens: int | None = None
    mamba: int | None = None
    swa_tokens: int | None = None
    error: str | None = None


def _cache_error(message: str) -> CacheCommand:
    return CacheCommand(action="error", error=message)


def _parse_cache_command(args: List[str], pools: CachePools) -> CacheCommand:
    """Parse ``/cache`` arguments against the pools this model actually has.

    No args (or ``status``) is a status query; one or more ``<target> <N>`` pairs is a rebuild.
    Also accepts the ``key=value`` spelling. ``N`` may carry a k/m suffix and must be positive.
    Units follow the pool: ``moe``/``mamba`` are slots, ``kv``/``swa`` are tokens (what the
    status line and the bar report), converted to whole pages against the live geometry."""
    if not args or args == ["status"]:
        return CacheCommand(action="status")
    usage = _cache_usage(pools)
    tokens: List[str] = []
    for a in args:
        tokens.extend(a.split("=", 1) if "=" in a else [a])
    values: dict[str, int] = {}
    i = 0
    while i < len(tokens):
        key = tokens[i].lower()
        if key not in pools.targets:
            known = (
                f"This model has no {key} pool. "
                if key in CACHE_UNITS
                else f"Unknown cache target {tokens[i]!r}. "
            )
            return _cache_error(known + usage)
        if i + 1 >= len(tokens):
            return _cache_error(f"Missing value for {key!r}. {usage}")
        val = _parse_count(tokens[i + 1])
        if val is None:
            return _cache_error(
                f"{key!r} expects a positive count (e.g. 512, 1.5k, 2M), got {tokens[i + 1]!r}"
            )
        values[key] = val
        i += 2
    if not values:
        return _cache_error(usage)
    return CacheCommand(
        action="rebuild",
        moe=values.get("moe"),
        kv_tokens=values.get("kv"),
        mamba=values.get("mamba"),
        swa_tokens=values.get("swa"),
    )


@dataclass
class ShellStats:
    cache_size: int = 0
    cache_policy: str = "lru"
    cache_rate: float | None = None
    model_label: str | None = None
    think_gear: str | None = None
    status: str = "idle"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    kv_used_pages: int = 0
    kv_total_pages: int = 0
    # KV page unit, so the status bar can render kv in tokens (pages x page_size).
    page_size: int = 1
    mamba_used_slots: int = 0
    mamba_total_slots: int = 0
    swa_used_tokens: int = 0
    swa_total_tokens: int = 0
    gpu_mem_bytes: int = 0
    started_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None

    def reset(self) -> None:
        self.status = "idle"
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.started_at = None
        self.first_token_at = None
        self.finished_at = None

    def mark_started(self, now: float | None = None) -> None:
        self.status = "prefill"
        self.started_at = time.time() if now is None else now
        self.first_token_at = None
        self.finished_at = None
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def set_prompt_tokens(self, count: int) -> None:
        # Prompt tokens arrive as a server-wide running total, so they can only grow within a
        # turn; a smaller reading means another client's request moved the total, or the
        # server restarted. Keep the larger value and let the turn's own usage settle it.
        self.prompt_tokens = max(self.prompt_tokens, count)

    def add_completion_tokens(self, count: int, now: float | None = None) -> None:
        if count <= 0:
            return
        current_time = time.time() if now is None else now
        if self.first_token_at is None:
            self.first_token_at = current_time
        self.status = "decode"
        self.completion_tokens += count

    def apply_usage(self, done: TurnDone) -> None:
        """Replace the live estimates with the turn's authoritative usage. Completion tokens
        were counted one-per-delta while streaming, which the server's own reasoning-parser
        buffering can make lag by a token or two."""
        if done.prompt_tokens > 0:
            self.prompt_tokens = done.prompt_tokens
        if done.completion_tokens > 0:
            self.completion_tokens = done.completion_tokens

    def apply_stats_doc(self, doc: dict) -> None:
        """Absorb a ``/v1/stats`` document. Every pool keeps last-known-value semantics: a
        model without that pool reports null and the segment stays off the bar."""
        kv = doc.get("kv")
        if isinstance(kv, dict) and int(kv.get("total_pages", 0) or 0) > 0:
            self.kv_used_pages = int(kv.get("used_pages", 0) or 0)
            self.kv_total_pages = int(kv["total_pages"])
            self.page_size = int(kv.get("page_size", 1) or 1)
        mamba = doc.get("mamba")
        if isinstance(mamba, dict) and int(mamba.get("total_slots", 0) or 0) > 0:
            self.mamba_used_slots = int(mamba.get("used_slots", 0) or 0)
            self.mamba_total_slots = int(mamba["total_slots"])
        swa = doc.get("swa")
        if isinstance(swa, dict) and int(swa.get("total_pages", 0) or 0) > 0:
            # /v1/stats denominates the window pool in its own pages; the bar shows tokens.
            swa_page_size = int(swa.get("page_size", 1) or 1)
            self.swa_used_tokens = int(swa.get("used_pages", 0) or 0) * swa_page_size
            self.swa_total_tokens = int(swa["total_pages"]) * swa_page_size
        vram = int(doc.get("vram_bytes", 0) or 0)
        if vram > 0:
            self.gpu_mem_bytes = vram

    def apply_geometry(self, geometry: dict) -> None:
        """Absorb the ``geometry`` block of ``/v1/cache/status``: the MoE slot cache and its
        residency rate against the model's total routed experts."""
        self.cache_size = int(geometry.get("moe_cache_size", 0) or 0)
        self.cache_policy = str(geometry.get("moe_cache_policy") or self.cache_policy)
        self.cache_rate = cache_rate(self.cache_size, geometry)
        page_size = int(geometry.get("page_size", 0) or 0)
        if page_size > 0:
            self.page_size = page_size

    def mark_finished(self, now: float | None = None) -> None:
        self.status = "done"
        self.finished_at = time.time() if now is None else now

    def tok_s(self, now: float | None = None) -> float:
        if self.first_token_at is None or self.completion_tokens == 0:
            return 0.0
        end_time = self.finished_at if self.finished_at is not None else (time.time() if now is None else now)
        elapsed = max(end_time - self.first_token_at, 1e-9)
        return self.completion_tokens / elapsed

    def format(self, now: float | None = None) -> str:
        prefix = f"[{self.status}]"
        if self.model_label:
            prefix += f" {self.model_label}"
            if self.think_gear is not None:
                prefix += f" ({self.think_gear})"
        cache_status = f"cache {self.cache_size} {self.cache_policy}"
        if self.cache_rate is not None:
            cache_status += f" {format_percent(self.cache_rate)}"
        token_status = f"↓{self.prompt_tokens} ↑{self.completion_tokens} {self.tok_s(now):.1f} tok/s"
        segments = [prefix, token_status, cache_status]
        if self.kv_total_pages > 0:
            kv_pct = format_percent(self.kv_used_pages / self.kv_total_pages)
            ps = self.page_size
            segments.append(f"kv {self.kv_used_pages * ps}/{self.kv_total_pages * ps} {kv_pct}")
        if self.mamba_total_slots > 0:  # hybrid (GDN) models only
            segments.append(f"mamba {self.mamba_used_slots}/{self.mamba_total_slots}")
        if self.swa_total_tokens > 0:  # SWA (window pool) models only
            swa_pct = format_percent(self.swa_used_tokens / self.swa_total_tokens)
            segments.append(f"swa {self.swa_used_tokens}/{self.swa_total_tokens} {swa_pct}")
        if self.gpu_mem_bytes > 0:
            segments.append(f"vram {self.gpu_mem_bytes / (1 << 30):.1f}GiB")
        return " | ".join(segments)


def _prompt_tokens_total(doc: dict) -> int:
    requests = doc.get("requests")
    if not isinstance(requests, dict):
        return 0
    return int(requests.get("prompt_tokens_total", 0) or 0)


async def _handle_cache_command(
    args: List[str], client: ShellClient, stats: ShellStats, renderer: ShellConsoleRenderer
) -> CachePools | None:
    """Shell ``/cache`` command: print the cache geometry, or resize any pool the model has at
    runtime (idle-only). Drives the same endpoints as ``ft ctl cache``.

    Returns the pools it read off the live geometry (None when the server was unreachable), so
    the caller can keep ``/help``'s command hints in sync with the served model."""
    try:
        # Fetched before parsing: the geometry says which pools this model has (so /cache only
        # offers those) and carries the page sizes the typed token counts convert against.
        doc = await client.cache_status()
    except ShellClientError as exc:
        renderer.write(f"{exc}\n")
        return None
    geometry = doc.get("geometry") or {}
    stats.apply_geometry(geometry)
    pools = CachePools.from_geometry(geometry)

    command = _parse_cache_command(args, pools)
    if command.action == "error":
        renderer.write(command.error + "\n")
        return pools
    if command.action == "status":
        renderer.write(format_cache_status(doc) + "\n")
        return pools

    page_size = max(1, int(geometry.get("page_size", 1) or 1))
    swa_page_size = int(geometry.get("swa_page_size", 0) or 0)
    num_pages = pages_for_tokens(command.kv_tokens, page_size)
    num_swa_pages = pages_for_tokens(command.swa_tokens, swa_page_size)

    def _target(pool: str, detail: str, units: int) -> str:
        # Cost the requested size up front: a rebuild that will not fit the engine's budget
        # says so as a rejection, and this is what makes the rejection legible.
        size = pool_bytes(geometry, pool, units=units)
        return f"{pool}={detail}" + (f" ({format_bytes(size)})" if size > 0 else "")

    target = ", ".join(
        part
        for part in (
            _target("moe", f"{command.moe} slots", command.moe) if command.moe is not None else "",
            _target("kv", format_tokens(num_pages, page_size), num_pages)
            if num_pages is not None else "",
            _target("mamba", f"{command.mamba} slots", command.mamba)
            if command.mamba is not None else "",
            _target("swa", format_tokens(num_swa_pages, swa_page_size), num_swa_pages)
            if num_swa_pages is not None else "",
        )
        if part
    )
    renderer.write(f"Rebuilding cache ({target}); serving pauses until done...\n")
    try:
        result = await client.cache_rebuild(
            moe_cache_size=command.moe,
            num_pages=num_pages,
            num_mamba_slots=command.mamba,
            num_swa_pages=num_swa_pages,
        )
    except ShellClientError as exc:
        renderer.write(f"{exc}\n")
        return pools
    status = result.get("status")
    if status != "ok":
        detail = f": {result['error']}" if result.get("error") else ""
        renderer.write(f"Cache rebuild {status}{detail}\n")
        return pools

    new_moe = result.get("moe_cache_size")
    if new_moe is not None:
        stats.cache_size = int(new_moe)
    # The rebuild clears the prefix cache and runs only when idle, so the new pools start
    # empty; reflect that immediately (the next poll keeps them live).
    new_pages = int(result.get("num_pages") or 0)
    if new_pages:
        stats.kv_used_pages = 0
        stats.kv_total_pages = new_pages
    new_mamba = int(result.get("mamba_slots") or 0)
    if new_mamba:
        stats.mamba_used_slots = 0
        stats.mamba_total_slots = new_mamba
    new_swa_pages = int(result.get("num_swa_pages") or 0)
    if new_swa_pages and swa_page_size > 0:
        stats.swa_used_tokens = 0
        stats.swa_total_tokens = new_swa_pages * swa_page_size
    elif stats.swa_total_tokens > 0:
        stats.swa_used_tokens = 0

    renderer.write("Cache rebuilt.\n")
    with contextlib.suppress(ShellClientError):
        # Re-read and report the geometry the engine actually landed on: it resolves the sizes
        # itself (and the residency rate follows the new slot count), so the fresh table is a
        # truer answer than echoing back what was asked for.
        doc = await client.cache_status()
        stats.apply_geometry(doc.get("geometry") or {})
        renderer.write(format_cache_status(doc) + "\n")
    return pools


def _help_text(think_gears: Tuple[str, ...], pools: CachePools) -> str:
    """``/help``, written against the served model: the thinking gears come from the server (a
    model with no controllable thinking says so rather than offering a command that no-ops), and
    ``/cache`` lists only the pools this model has."""
    think = (
        f"/think [{'|'.join(think_gears)}|toggle|status]"
        if think_gears
        else "/think"
    )
    think_help = (
        "switch the model's thinking gear"
        if think_gears
        else "(this model has no controllable thinking)"
    )
    rows = [
        ("/help", "show this message"),
        (think, think_help),
        (f"/cache [status | {_cache_targets_hint(pools)}]", ""),
        ("", "show or resize the cache pools; token targets are"),
        ("", "rounded up to the pool's page size"),
        ("/reset", "clear the conversation history"),
        ("/exit", "quit (Ctrl-D also works)"),
    ]
    width = max(len(name) for name, _ in rows if len(name) < 40)
    lines = ["Commands:"]
    for name, help_text in rows:
        if len(name) >= 40:  # too long to pair with its help on one line
            lines.append(f"  {name}")
            continue
        lines.append(f"  {name.ljust(width)}  {help_text}".rstrip())
    lines.append("")
    lines.append("Ctrl-C cancels the turn being generated; Esc+Enter inserts a newline.")
    return "\n".join(lines)


@contextlib.contextmanager
def _sigint_cancels(task: asyncio.Task):
    """Bind ^C to "cancel this turn" for as long as ``task`` runs.

    Per turn, not once for the session: prompt_toolkit claims SIGINT while it owns the terminal
    (to deliver it as a key) and calls ``remove_signal_handler`` on the way out, which drops
    whatever the loop had registered -- a handler installed once would be silently gone after
    the first prompt. Outside a turn the default handling is what we want anyway: ^C at the
    prompt reaches prompt_toolkit as a keystroke and quits the shell."""
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, task.cancel)
    except (NotImplementedError, RuntimeError):  # Windows, or not the main thread
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.remove_signal_handler(signal.SIGINT)


def _shell_sampling() -> Sampling:
    """Sampling for a shell turn. Unset knobs stay unset so the server fills them from the
    model's own generation_config.json (``--sampling-defaults model``)."""
    return Sampling(
        max_tokens=ENV.SHELL_MAX_TOKENS.value,
        temperature=ENV.SHELL_TEMPERATURE.value,
        top_p=ENV.SHELL_TOP_P.value,
        top_k=ENV.SHELL_TOP_K.value,
    )


def _format_load_progress(doc: dict) -> str:
    progress = doc.get("progress")
    phase = doc.get("phase") or "loading"
    if isinstance(progress, dict):
        done = int(progress.get("done_bytes", 0) or 0)
        total = int(progress.get("total_bytes", 0) or 0)
        if total > 0:
            return f"loading ({phase}): {done / (1 << 30):.1f}/{total / (1 << 30):.1f} GiB"
    return f"loading ({phase})..."


async def run_shell(
    origin: str, *, connect_grace: float = 0.0, api_key: str | None = None
) -> int:
    """Attach to the FreeToken server at ``origin`` and run the terminal chat.

    ``connect_grace`` is how long to keep retrying a refused connection before giving up --
    left at 0 when attaching to a server the user says is already running, raised when the
    caller just started one in this process (see ``server/api_server.py``).

    ``api_key`` is the server's ``--api-key`` when it has one: passed in by shell mode, read
    from ``FREETOKEN_API_KEY`` when attaching to a running server, else the local placeholder."""
    key = api_key or os.environ.get("FREETOKEN_API_KEY") or LOCAL_API_KEY
    client = ShellClient(origin, api_key=key)
    try:
        return await _run_shell(client, origin, connect_grace=connect_grace)
    finally:
        await client.aclose()


async def _run_shell(client: ShellClient, origin: str, *, connect_grace: float) -> int:
    write = ShellConsoleRenderer._write_stdout
    last_line = ""
    last_at = 0.0

    def _on_progress(doc: dict) -> None:
        """One line per phase, then at most one per LOAD_PROGRESS_INTERVAL -- a 60 GiB
        checkpoint moves ``done_bytes`` far too often to echo every reading."""
        nonlocal last_line, last_at
        line = _format_load_progress(doc)
        now = time.monotonic()
        new_phase = line.split(":", 1)[0] != last_line.split(":", 1)[0]
        if line == last_line or (not new_phase and now - last_at < LOAD_PROGRESS_INTERVAL):
            return
        last_line, last_at = line, now
        write(f"{line}\n")

    try:
        await client.wait_until_ready(on_progress=_on_progress, connect_grace=connect_grace)
        model_id = await client.model_id()
    except ShellClientError as exc:
        write(f"{exc}\n")
        return 1
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 130

    if not model_id:
        write(f"{origin} reports no served model\n")
        return 1

    # Best-effort: an unreadable status bar is not a reason to refuse to chat.
    cache_doc: dict = {}
    stats_doc: dict = {}
    with contextlib.suppress(ShellClientError):
        cache_doc = await client.cache_status()
        stats_doc = await client.stats()

    geometry = cache_doc.get("geometry") or {}
    reasoning = geometry.get("reasoning") or {}
    think_gears: Tuple[str, ...] = tuple(reasoning.get("gears") or ())
    think_kwargs: dict = reasoning.get("kwargs") or {}
    think_gear: str | None = reasoning.get("default")

    # Which pools this model has, so /help and /cache only ever offer targets it can resize.
    # Refreshed by every /cache (which re-reads the geometry), in case this first read failed.
    cache_pools = CachePools.from_geometry(geometry)

    stats = ShellStats(model_label=_format_shell_model_label(model_id), think_gear=think_gear)
    stats.apply_geometry(geometry)
    stats.apply_stats_doc(stats_doc)

    write(f"FreeToken shell -> {model_id} @ {origin}  (/help for commands, /exit to quit)\n")

    terminal_size = shutil.get_terminal_size((SHELL_FALLBACK_WIDTH, 24))
    status_line = ShellStatusLine(
        stats.format,
        display_width=terminal_size.columns,
        display_height=terminal_size.lines,
    )
    renderer = ShellConsoleRenderer(
        write=status_line.write_output,
        display_width=terminal_size.columns,
    )
    history: List[Tuple[str, str]] = []

    async def poll_stats(prompt_baseline: int) -> None:
        """Keep the status bar live while a turn streams: pool occupancy, VRAM, and the prompt
        tokens the server has admitted so far (chunked prefill reports them as it goes)."""
        while True:
            await asyncio.sleep(STATS_POLL_INTERVAL)
            try:
                doc = await client.stats()
            except ShellClientError:
                continue  # a blip in the status bar must never break the turn
            stats.apply_stats_doc(doc)
            stats.set_prompt_tokens(max(0, _prompt_tokens_total(doc) - prompt_baseline))
            status_line.force()

    async def run_turn(cmd: str) -> None:
        nonlocal history
        messages: List[dict] = []
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": cmd})

        try:
            prompt_baseline = _prompt_tokens_total(await client.stats())
        except ShellClientError:
            prompt_baseline = 0

        stats.think_gear = think_gear
        stats.mark_started()
        terminal_size = shutil.get_terminal_size((SHELL_FALLBACK_WIDTH, 24))
        renderer.display_width = terminal_size.columns
        status_line.display_width = terminal_size.columns
        status_line.display_height = terminal_size.lines
        output_buffer = ShellOutputBuffer(renderer)
        status_line.activate()
        poller = asyncio.create_task(poll_stats(prompt_baseline))

        answer: List[str] = []
        failure: str | None = None
        cancelled = False
        events = client.chat(
            messages,
            model=model_id,
            sampling=_shell_sampling(),
            chat_template_kwargs=think_kwargs.get(think_gear) if think_gear else None,
        )
        try:
            renderer.begin_turn(cmd)
            async for event in events:
                if isinstance(event, ReasoningDelta):
                    output_buffer.write_reasoning(event.text)
                    stats.add_completion_tokens(1)
                elif isinstance(event, ContentDelta):
                    answer.append(event.text)
                    output_buffer.write_content(event.text)
                    stats.add_completion_tokens(1)
                elif isinstance(event, TurnDone):
                    stats.apply_usage(event)
                status_line.maybe()
        except ShellClientError as exc:
            failure = str(exc)
        except asyncio.CancelledError:
            # ^C during a turn: closing the stream below disconnects, which the server turns
            # into an abort, so the engine stops decoding instead of finishing a dead turn.
            cancelled = True
        finally:
            poller.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await events.aclose()
            output_buffer.flush()
            stats.mark_finished()
            renderer.finish_turn()
            status_line.deactivate()

        if failure is not None:
            write(f"{failure}\n")
            return
        if cancelled:
            write("(cancelled)\n")
        # Keep what was actually shown, partial turn included, so a follow-up ("go on") sees
        # the same conversation the user does. The reasoning channel is never stored: it is
        # rendered separately and must not re-enter the next prompt.
        text = "".join(answer)
        if text or not cancelled:
            history.append((cmd, text))

    async def handle_command(cmd: str) -> None:
        nonlocal history, think_gear, cache_pools
        if cmd == "":
            return
        if cmd.startswith("/"):
            parts = cmd.split()
            slash = parts[0]
            if slash == "/exit":
                raise EOFError
            if slash in ("/help", "/?"):
                renderer.write(_help_text(think_gears, cache_pools) + "\n")
                return
            if slash == "/reset":
                history = []
                stats.reset()
                return
            if slash in ("/think", "/thinking"):
                arg = parts[1].lower() if len(parts) > 1 else "status"
                think_gear, message = _apply_think_command(arg, think_gears, think_gear)
                stats.think_gear = think_gear
                renderer.write(message + "\n")
                return
            if slash == "/cache":
                pools = await _handle_cache_command(parts[1:], client, stats, renderer)
                if pools is not None:
                    cache_pools = pools  # keep /help's hints on the served model
                return
            renderer.write(f"Unknown command: {cmd}. Try /help.\n")
            return

        turn_task = asyncio.create_task(run_turn(cmd))
        with _sigint_cancels(turn_task), contextlib.suppress(asyncio.CancelledError):
            # run_turn reports the cancellation and closes its stream itself; suppressing here
            # only covers the case where it re-raises after cleanup.
            await turn_task

    key_bindings = KeyBindings()

    @key_bindings.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @key_bindings.add("escape", "enter")
    def _insert_newline(event):
        event.current_buffer.insert_text("\n")

    @key_bindings.add("c-d")
    def _exit(event):
        event.app.exit(exception=EOFError())

    session = PromptSession(
        "$ ",
        multiline=True,
        key_bindings=key_bindings,
        erase_when_done=True,
        bottom_toolbar=lambda: stats.format(),
        style=Style.from_dict(
            {
                "bottom-toolbar": "reverse",
            }
        ),
    )

    try:
        while True:
            cmd = (await session.prompt_async()).strip()
            await handle_command(cmd)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        print("Exiting shell...")
    return 0
