"""Model-derived facts the server publishes to its clients.

These used to sit in the shell's TUI module, back when the shell ran inside the server
process and could read ``ServerArgs`` directly. The shell is an ordinary HTTP client now,
so they belong on the server side of the wire: ``/v1/cache/status`` is what hands them out
(``geometry.reasoning`` for the thinking gears, ``geometry.moe_*`` for the cache panel).
"""

from __future__ import annotations

import math
from typing import Any, Tuple

from freetoken.tokenizer.effort import (
    EFFORT_SCALE,
    OPENAI_EFFORT_TRIPLE,
    THINKING_ADAPTIVE_KWARGS,
    THINKING_OFF_KWARGS,
    THINKING_ON_KWARGS,
    ThinkingProfile,
)


def thinking_toggle_kwargs(enabled: bool) -> dict:
    """``chat_template_kwargs`` for a protocol-level thinking on/off toggle
    (Anthropic ``thinking.type``, DeepSeek ``thinking``, Responses
    ``reasoning.effort``): every spelling the ecosystem's templates read,
    broadcast at once. A template picks the knob it knows and ignores the rest
    (Jinja never sees undeclared variables), so no per-family routing exists."""
    return dict(THINKING_ON_KWARGS if enabled else THINKING_OFF_KWARGS)


def derive_think_gears(
    profile: ThinkingProfile, parser_configured: bool
) -> Tuple[Tuple[str, ...], str | None, dict] | None:
    """``(gears, default_gear, kwargs_per_gear)`` for the /v1/cache/status
    ``geometry.reasoning`` block, derived from the checkpoint's probed thinking
    controls -- the checkpoint owns this knowledge; nothing here is keyed by
    model family. ``None`` when there is nothing controllable to offer.

    A template that grades effort without validating it gets the OpenAI triple
    (the only vocabulary such a template is known to understand); an always-on
    model with a reasoning parser but no observable knob shows a single "on"
    gear so clients can still label the state."""
    efforts = profile.efforts
    gears: list[str] = []
    kwargs: dict[str, dict] = {}
    if profile.toggleable:
        gears.append("off")
        kwargs["off"] = dict(THINKING_OFF_KWARGS)
    if profile.has_adaptive:
        gears.append("adaptive")
        kwargs["adaptive"] = dict(THINKING_ADAPTIVE_KWARGS)

    if efforts.consumes_effort:
        names = (
            [n for n in efforts.supported if n in EFFORT_SCALE]
            if efforts.validates
            else list(OPENAI_EFFORT_TRIPLE)
        )
        for name in sorted(names, key=lambda n: EFFORT_SCALE[n]):
            gears.append(name)
            gear_kwargs = dict(THINKING_ON_KWARGS) if profile.toggleable else {}
            gear_kwargs["reasoning_effort"] = name
            kwargs[name] = gear_kwargs
    elif profile.toggleable:
        gears.append("on")
        kwargs["on"] = dict(THINKING_ON_KWARGS)
    elif parser_configured:
        # Always-thinking family (minimax): no knob, but the state is real.
        gears.append("on")
        kwargs["on"] = {}

    if not gears:
        return None

    if profile.default_state == "off" and "off" in gears:
        default = "off"
    elif profile.default_state == "adaptive" and "adaptive" in gears:
        default = "adaptive"
    elif efforts.consumes_effort:
        default = efforts.default if efforts.default in gears else (
            "medium" if "medium" in gears else gears[-1]
        )
    else:
        default = "on" if "on" in gears else gears[-1]
    return tuple(gears), default, kwargs


_THINKING_KWARG_KEYS = ("enable_thinking", "thinking", "thinking_mode", "reasoning_effort")
_DISABLE_EFFORTS = ("none", "off")


def effort_toggle_kwargs(
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
    mapped = thinking_toggle_kwargs(not disabled)
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
