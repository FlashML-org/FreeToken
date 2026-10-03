"""Unit tests for the shared thinking-mode resolver (tokenizer/tokenize.py)."""
from __future__ import annotations

from freetoken.tokenizer.tokenize import resolve_thinking_mode


def test_default_is_chat():
    assert resolve_thinking_mode({}, None) == "chat"
    assert resolve_thinking_mode(None, None) == "chat"


def test_tools_force_thinking():
    # dsv4 only emits well-formed tool calls in thinking mode.
    assert resolve_thinking_mode({}, [{"type": "function"}]) == "thinking"


def test_explicit_thinking_flags():
    assert resolve_thinking_mode({"thinking": True}, None) == "thinking"
    assert resolve_thinking_mode({"enable_thinking": True}, None) == "thinking"
    assert resolve_thinking_mode({"thinking_mode": "thinking"}, None) == "thinking"


def test_explicit_chat_mode():
    assert resolve_thinking_mode({"thinking_mode": "chat"}, None) == "chat"


def test_invalid_mode_falls_back_to_chat():
    assert resolve_thinking_mode({"thinking_mode": "bogus"}, None) == "chat"


def test_explicit_disable_overrides_tools():
    """An explicit enable_thinking=False (or thinking=False) must win even
    when tools are present — the caller knows their intent better than the
    tools-presumed-thinking heuristic."""
    assert resolve_thinking_mode({"enable_thinking": False}, [{"type": "function"}]) == "chat"
    assert resolve_thinking_mode({"thinking": False}, [{"type": "function"}]) == "chat"
    assert resolve_thinking_mode({"enable_thinking": False, "thinking": True}, [{"type": "function"}]) == "chat"
    assert resolve_thinking_mode({"thinking": False, "enable_thinking": True}, [{"type": "function"}]) == "chat"
