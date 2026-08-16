"""Model-derived facts the server publishes to its clients.

These used to sit in the shell's TUI module, back when the shell ran inside the server
process and could read ``ServerArgs`` directly. The shell is an ordinary HTTP client now,
so they belong on the server side of the wire: ``/v1/cache/status`` is what hands them out
(``geometry.reasoning`` for the thinking gears, ``geometry.moe_*`` for the cache panel).
"""

from __future__ import annotations

import math
from typing import Any, Tuple


def think_spec(reasoning_parser: str | None) -> Tuple[Tuple[str, ...], str | None]:
    """Return ``(gears, default_gear)`` a client can offer for a model family, keyed by its
    configured reasoning parser. ``((), None)`` when the model has no controllable thinking.
    Verified per family against each model's chat template / encoder."""
    if reasoning_parser == "gpt_oss":
        return ("low", "medium", "high"), "medium"  # always-on, 3-level effort
    if reasoning_parser == "deepseekv32":
        return ("off", "on", "max"), "off"  # thinking on/off + a max-effort gear
    if reasoning_parser == "minimax":
        return ("on",), "on"  # template always opens a think block; no off path
    if reasoning_parser == "minimax_m3":
        # M3's template takes thinking_mode disabled/adaptive/enabled; adaptive
        # (the template's own default) lets the model decide per turn.
        return ("off", "adaptive", "on"), "adaptive"
    if reasoning_parser == "gemma4":
        return ("off", "on"), "off"  # gemma's template defaults thinking off
    if reasoning_parser in ("qwen3", "glm"):
        return ("off", "on"), "on"
    return (), None


def think_chat_template_kwargs(reasoning_parser: str | None, gear: str | None) -> dict:
    """The ``chat_template_kwargs`` that select ``gear`` for the model family."""
    if gear is None:
        return {}
    if reasoning_parser == "gpt_oss":
        return {"reasoning_effort": gear}
    if reasoning_parser == "deepseekv32":
        if gear == "max":
            return {"enable_thinking": True, "reasoning_effort": "max"}
        return {"enable_thinking": gear == "on"}
    if reasoning_parser == "minimax":
        return {}  # always thinks; its template reads no knob
    if reasoning_parser == "minimax_m3":
        mode = {"off": "disabled", "adaptive": "adaptive", "on": "enabled"}[gear]
        return {"thinking_mode": mode}
    return {"enable_thinking": gear == "on"}  # qwen3, glm, gemma4


def think_toggle_kwargs(reasoning_parser: str | None, enabled: bool) -> dict:
    """``chat_template_kwargs`` for a protocol-level thinking on/off toggle
    (Anthropic ``thinking.type``, Responses ``reasoning.effort``), routed through
    the same per-family mapping as the chat-completions gears -- a hardcoded
    ``enable_thinking`` is inert for templates that read a different knob (M3's
    ``thinking_mode``). A family without the requested direction returns ``{}``;
    with no configured parser the protocol-generic key is kept."""
    gears, _default = think_spec(reasoning_parser)
    if not gears:
        return {"enable_thinking": enabled}
    gear = "on" if enabled else "off"
    if gear not in gears:
        return {}
    return think_chat_template_kwargs(reasoning_parser, gear)


_THINKING_KWARG_KEYS = ("enable_thinking", "thinking", "thinking_mode", "reasoning_effort")
_DISABLE_EFFORTS = ("none", "off")


def effort_toggle_kwargs(
    reasoning_parser: str | None,
    effort: str | None,
    chat_template_kwargs: dict | None,
    thinking_type: str | None = None,
) -> dict:
    """Fold a protocol-level reasoning-effort request into the template kwargs.
    An explicit thinking-related key wins wholesale; unrelated extras ride along.
    Effort "none"/"off" (case-insensitive) disables thinking; any other or absent
    effort enables it, forwarded for templates that grade it (quantized against
    the checkpoint's probed vocabulary at render time). ``thinking_type`` is the
    DeepSeek-wire ``thinking: {"type": ...}`` toggle; when present it decides
    the on/off direction outright, "disabled" winning over any effort."""
    ctk = dict(chat_template_kwargs or {})
    if any(key in ctk for key in _THINKING_KWARG_KEYS):
        return ctk
    if isinstance(effort, str):
        effort = effort.strip().lower()
    disabled = effort in _DISABLE_EFFORTS
    if thinking_type == "disabled":
        disabled = True
    elif thinking_type == "enabled":
        disabled = False
    mapped = dict(think_toggle_kwargs(reasoning_parser, not disabled))
    if effort and not disabled and effort not in _DISABLE_EFFORTS:
        mapped.setdefault("reasoning_effort", effort)
    mapped.update(ctk)
    return mapped


def moe_total_experts(config: Any) -> int:
    """Total routed-expert slots the model has: experts per layer x MoE layers. Matches the
    engine's own basis (``Engine._resolve_auto_moe_cache_size``), so a residency rate derived
    from it agrees with the size the engine resolved -- ``num_moe_layers`` excludes the leading
    dense layers a model like DSV4 carries."""
    try:
        model_config = config.model_config
    except Exception:  # noqa: BLE001 -- dummy/absent config: report "unknown", never raise
        return 0
    return int(getattr(model_config, "num_moe_layers", 0) or 0) * int(
        getattr(model_config, "num_experts", 0) or 0
    )


def moe_cache_size(config: Any) -> int:
    """The configured MoE slot-cache size, resolving ``--moe-cache-rate`` to a slot count.
    Only a fallback for the reported geometry: the engine's actual allocation (from the
    ready ack, or a rebuild) wins wherever it is known."""
    cache_size = int(getattr(config, "moe_cache_size", 0) or 0)
    if cache_size > 0:
        return cache_size
    cache_rate = getattr(config, "moe_cache_rate", None)
    if cache_rate is None:
        return cache_size
    total_experts = moe_total_experts(config)
    if total_experts <= 0:
        return cache_size
    return math.ceil(total_experts * float(cache_rate))
