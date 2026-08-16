"""Reasoning-effort dialect handling.

Each checkpoint's template/encoder accepts only its own effort vocabulary and
hard-fails on the rest, while clients speak whatever dialect their provider
taught them. Named levels project onto the numeric scale vLLM and SGLang share,
and out-of-vocabulary values quantize to the nearest supported gear instead of
failing the request. The vocabulary is probed from the checkpoint's own
template, never from a static table: a parser-family registry cannot be keyed
correctly (Qwen3 and Qwen3.8 resolve to the same parser but only the latter
grades effort).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

#: Values must match vLLM's and SGLang's inkling table so the ecosystems agree
#: on what "medium" means relative to "xhigh".
EFFORT_SCALE: dict[str, float] = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "xhigh": 0.99,
    "max": 0.99,
}

KNOWN_REASONING_EFFORTS = tuple(EFFORT_SCALE)


@dataclass(frozen=True)
class EffortProfile:
    """What one checkpoint's template/encoder accepts, learned by probing it.

    ``default`` is the supported name whose rendering is byte-identical to
    passing no effort at all. ``consumes_effort`` False means no probe round
    ever changed its output or raised -- the template ignores the knob, so
    requests should not carry it.
    """

    supported: frozenset[str]
    default: str | None
    consumes_effort: bool


#: A gear farther than this on the scale misrepresents the request; drop the
#: value and let the template default apply. Keeps OpenAI's "medium" from
#: escalating to the DSV4 encoder's absolute-maximum "high" gear (0.2 away),
#: matching vLLM's DSV4 mapping, while "high" still reaches Qwen's "xhigh"
#: (0.09 away).
_MAX_QUANTIZE_DISTANCE = 0.15


def quantize_effort(value: Any, profile: EffortProfile) -> str | None:
    """Map a client's effort onto ``profile``; ``None`` means "send nothing".

    In-vocabulary values pass through untouched. Other named levels land on the
    nearest supported gear within ``_MAX_QUANTIZE_DISTANCE`` -- except "max",
    which is reachable only by its own name (vLLM's DSV4 rule: an extreme
    opt-in gear must never be entered by rounding). Everything else drops to
    the template default. With "max" excluded the remaining scale values are
    unique, so quantization is deterministic across processes.
    """
    if not profile.consumes_effort:
        return None
    if isinstance(value, str) and value in profile.supported:
        return value
    position = EFFORT_SCALE.get(value) if isinstance(value, str) else None
    if position is None:
        return None
    ranked = sorted(
        (name for name in profile.supported if name != "max"),
        key=lambda name: (abs(EFFORT_SCALE[name] - position), -EFFORT_SCALE[name]),
    )
    if ranked and abs(EFFORT_SCALE[ranked[0]] - position) <= _MAX_QUANTIZE_DISTANCE:
        return ranked[0]
    return None


#: One tool flips tool-conditional paths (the DSV4 encoder grades effort only
#: in thinking mode, which tools force).
_PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "No-op probe tool.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

#: (extra chat_template_kwargs, tools) per probe round: templates read effort
#: unconditionally, under tool-forced thinking, or only under an explicit
#: thinking opt-in -- one round per shape.
_PROBE_ROUNDS: tuple[tuple[dict[str, Any], list[dict[str, Any]] | None], ...] = (
    ({}, None),
    ({}, _PROBE_TOOLS),
    ({"enable_thinking": True}, None),
)


def probe_effort_profile(
    render: Callable[[dict[str, Any], list[dict[str, Any]] | None], Any],
) -> EffortProfile:
    """Learn a checkpoint's effort vocabulary by rendering probes through it.

    ``render(chat_template_kwargs, tools)`` returns a comparable rendering and
    raises on rejection. A round whose no-effort baseline raises is skipped:
    the template rejected the probe conversation shape, not the effort.
    """
    rejected: set[str] = set()
    diverged: set[str] = set()
    matches_baseline: dict[str, bool] = {name: True for name in KNOWN_REASONING_EFFORTS}
    ran_rounds = 0

    for base_kwargs, tools in _PROBE_ROUNDS:
        try:
            baseline = render(dict(base_kwargs), tools)
        except Exception:  # noqa: BLE001 -- template rejects the probe shape, not the effort
            continue
        ran_rounds += 1
        for name in KNOWN_REASONING_EFFORTS:
            try:
                rendering = render({**base_kwargs, "reasoning_effort": name}, tools)
            except Exception:  # noqa: BLE001 -- any raise means "not accepted"
                rejected.add(name)
                matches_baseline[name] = False
                continue
            if _renderings_differ(rendering, baseline):
                diverged.add(name)
                matches_baseline[name] = False

    if ran_rounds == 0:
        # Nothing learnable: sending no effort is the only safe rendering.
        return EffortProfile(supported=frozenset(), default=None, consumes_effort=False)

    supported = frozenset(name for name in KNOWN_REASONING_EFFORTS if name not in rejected)
    consumes = bool(rejected or diverged)
    default = None
    if consumes:
        defaults = [name for name in supported if matches_baseline[name]]
        if defaults:
            default = max(defaults, key=lambda name: EFFORT_SCALE[name])
    return EffortProfile(supported=supported, default=default, consumes_effort=consumes)


def _renderings_differ(a: Any, b: Any) -> bool:
    try:
        return bool(a != b)
    except Exception:  # noqa: BLE001 -- exotic tensor comparison; treat as divergence
        return True
