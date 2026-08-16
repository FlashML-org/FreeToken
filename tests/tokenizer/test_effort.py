"""Unit tests for the reasoning-effort dialect layer (tokenizer/effort.py)."""
from __future__ import annotations

from freetoken.tokenizer.effort import (
    EFFORT_SCALE,
    EffortProfile,
    KNOWN_REASONING_EFFORTS,
    probe_effort_profile,
    quantize_effort,
)

QWEN38 = EffortProfile(
    supported=frozenset({"xhigh", "medium", "low"}), default="xhigh", consumes_effort=True
)
DSV4_OFFICIAL = EffortProfile(
    supported=frozenset({"low", "high", "max"}), default="low", consumes_effort=True
)
IGNORES = EffortProfile(supported=frozenset(KNOWN_REASONING_EFFORTS), default=None, consumes_effort=False)


def test_in_vocabulary_values_pass_through():
    for name in ("xhigh", "medium", "low"):
        assert quantize_effort(name, QWEN38) == name
    for name in ("low", "high", "max"):
        assert quantize_effort(name, DSV4_OFFICIAL) == name


def test_deepseek_dialect_high_lands_on_qwen_xhigh():
    # high (0.9) is nearer xhigh (0.99) than medium (0.7)
    assert quantize_effort("high", QWEN38) == "xhigh"


def test_openai_dialect_medium_drops_to_the_dsv4_default():
    # medium (0.7) is 0.2 from the nearest gear -- beyond the quantize
    # threshold, so nothing is sent and the encoder default (low) applies;
    # anything else would silently escalate OpenAI-default traffic to the
    # encoder's absolute-maximum "high" prompt. Matches vLLM's DSV4 mapping.
    assert quantize_effort("medium", DSV4_OFFICIAL) is None


def test_xhigh_lands_on_dsv4_high_never_max():
    # "max" is an extreme opt-in gear, reachable only by its own name
    # (vLLM's DSV4 rule) -- rounding must not enter it.
    assert quantize_effort("xhigh", DSV4_OFFICIAL) == "high"


def test_max_lands_on_qwen_xhigh():
    assert quantize_effort("max", QWEN38) == "xhigh"


def test_far_values_drop_instead_of_rounding():
    profile = EffortProfile(
        supported=frozenset({"low", "xhigh"}), default=None, consumes_effort=True
    )
    assert quantize_effort("minimal", profile) == "low"  # 0.1 away
    assert quantize_effort("medium", profile) is None  # 0.29 to the nearest gear


def test_unknown_and_non_string_values_drop_to_template_default():
    assert quantize_effort("banana", QWEN38) is None
    assert quantize_effort(3, QWEN38) is None
    assert quantize_effort(None, QWEN38) is None


def test_effort_ignoring_template_sends_nothing():
    assert quantize_effort("high", IGNORES) is None
    assert quantize_effort("xhigh", IGNORES) is None


# --------------------------------------------------------------------------- #
# Probe tests: fake render callables standing in for real templates/encoders.
# --------------------------------------------------------------------------- #
def _qwen38_render(kwargs, tools):
    # Validates unconditionally (enable_thinking undefined counts as on) and
    # renders a distinct effort preamble per gear, default xhigh.
    effort = kwargs.get("reasoning_effort", "xhigh")
    if effort not in ("xhigh", "medium", "low"):
        raise ValueError(f"Unexpected reasoning effort {effort}")
    preamble = {"xhigh": "think hard", "medium": "", "low": "think briefly"}[effort]
    return f"{preamble}|tools={bool(tools)}"


def test_probe_learns_the_qwen38_vocabulary():
    profile = probe_effort_profile(_qwen38_render)
    assert profile.supported == frozenset({"xhigh", "medium", "low"})
    assert profile.default == "xhigh"
    assert profile.consumes_effort
    assert profile.validates  # rejections observed -> the vocabulary is real


def _dsv4_render(kwargs, tools):
    # Grades effort only in thinking mode (tools force it); asserts on unknown
    # values there; "low" renders the empty preamble, matching the default.
    effort = kwargs.get("reasoning_effort") or "low"
    if not tools:
        return "chat prompt"
    assert effort in ("low", "high", "max"), f"Invalid reasoning effort: {effort}"
    preamble = {"low": "", "high": "absolute maximum", "max": "beyond maximum"}[effort]
    return f"{preamble}|thinking"


def test_probe_learns_the_dsv4_vocabulary_through_the_tools_round():
    profile = probe_effort_profile(_dsv4_render)
    assert profile.supported == frozenset({"low", "high", "max"})
    assert profile.default == "low"
    assert profile.consumes_effort
    assert profile.validates


def test_probe_marks_an_ignoring_template_as_not_consuming():
    profile = probe_effort_profile(lambda kwargs, tools: f"same|tools={bool(tools)}")
    assert not profile.consumes_effort
    assert not profile.validates
    assert profile.default is None


def test_probe_marks_an_interpolating_template_as_not_validating():
    # Grades effort (renders differ) but rejects nothing: consumes without a
    # trustworthy vocabulary.
    profile = probe_effort_profile(lambda kwargs, tools: f"p|{kwargs.get('reasoning_effort')}")
    assert profile.consumes_effort
    assert not profile.validates


def test_probe_skips_rounds_whose_baseline_fails():
    def render(kwargs, tools):
        if tools:  # template rejects tools outright -- round is uninformative
            raise RuntimeError("no tools supported")
        return _qwen38_render(kwargs, tools)

    profile = probe_effort_profile(render)
    assert profile.supported == frozenset({"xhigh", "medium", "low"})
    assert profile.consumes_effort
