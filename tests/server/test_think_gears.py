"""derive_think_gears: the probed replacement for the per-family gear registry.

Each case fakes one model family's template behavior and asserts the derived
gears match (or improve on) what the deleted ``think_spec`` registry hardcoded.
"""
from __future__ import annotations

from freetoken.server.model_meta import derive_think_gears
from freetoken.tokenizer.effort import (
    EffortProfile,
    probe_effort_profile,
    probe_thinking_profile,
)


def profile_for(render):
    return probe_thinking_profile(render, probe_effort_profile(render))


def test_qwen3_style_on_off_toggle():
    # Old registry row: ("off", "on"), default "on".
    def render(kwargs, tools):
        return f"qwen3|think={kwargs.get('enable_thinking', True)}"

    gears, default, kwargs = derive_think_gears(profile_for(render), parser_configured=True)
    assert gears == ("off", "on") and default == "on"
    assert kwargs["off"]["enable_thinking"] is False
    assert kwargs["on"]["enable_thinking"] is True


def test_qwen38_style_graded_efforts():
    # The registry had no row for Qwen3.8's gears -- it showed off/on. Derived:
    # the template's real vocabulary, ascending, with the off toggle.
    def render(kwargs, tools):
        if kwargs.get("enable_thinking") is False:
            return "qwen38|off"
        effort = kwargs.get("reasoning_effort", "xhigh")
        if effort not in ("xhigh", "medium", "low"):
            raise ValueError(f"Unexpected reasoning effort {effort}")
        return f"qwen38|{effort}"

    gears, default, kwargs = derive_think_gears(profile_for(render), parser_configured=True)
    assert gears == ("off", "low", "medium", "xhigh") and default == "xhigh"
    assert kwargs["medium"]["reasoning_effort"] == "medium"
    assert kwargs["medium"]["enable_thinking"] is True
    assert kwargs["off"]["enable_thinking"] is False


def test_gemma4_style_default_off():
    # Old registry row: ("off", "on"), default "off".
    def render(kwargs, tools):
        return f"gemma|think={bool(kwargs.get('enable_thinking'))}"

    gears, default, _ = derive_think_gears(profile_for(render), parser_configured=True)
    assert gears == ("off", "on") and default == "off"


def test_gpt_oss_style_always_on_graded():
    # Old registry row: ("low", "medium", "high"), default "medium". The
    # template grades effort but never validates it, so the derived vocabulary
    # falls back to the OpenAI triple rather than every known name.
    def render(kwargs, tools):
        return f"harmony|{kwargs.get('reasoning_effort', 'medium')}"

    gears, default, kwargs = derive_think_gears(profile_for(render), parser_configured=True)
    assert gears == ("low", "medium", "high") and default == "medium"
    assert kwargs["high"] == {"reasoning_effort": "high"}  # no toggle: always on


def test_minimax_style_always_on_no_knob():
    # Old registry row: ("on",), default "on", kwargs {}.
    def render(kwargs, tools):
        return "minimax prompt"

    gears, default, kwargs = derive_think_gears(profile_for(render), parser_configured=True)
    assert gears == ("on",) and default == "on"
    assert kwargs["on"] == {}


def test_no_reasoning_parser_offers_nothing():
    # Old registry: unknown parser -> ((), None) -> reasoning block absent.
    def render(kwargs, tools):
        return "plain prompt"

    assert derive_think_gears(profile_for(render), parser_configured=False) is None


def test_dsv4_style_toggle_plus_efforts():
    # Old registry row: ("off", "on", "max"), default "off". Derived: the
    # encoder's full vocabulary replaces the curated "on" (its low gear renders
    # exactly what "on" did), keeping default off.
    def render(kwargs, tools):
        thinking = (
            bool(tools)
            or bool(kwargs.get("enable_thinking"))
            or kwargs.get("thinking_mode") == "enabled"
        )
        if not thinking:
            return "dsv4|chat"
        effort = kwargs.get("reasoning_effort") or "low"
        assert effort in ("low", "high", "max"), f"Invalid reasoning effort: {effort}"
        return f"dsv4|think|{effort}"

    gears, default, kwargs = derive_think_gears(profile_for(render), parser_configured=True)
    assert gears == ("off", "low", "high", "max") and default == "off"
    assert kwargs["max"]["reasoning_effort"] == "max"
    assert kwargs["max"]["enable_thinking"] is True
