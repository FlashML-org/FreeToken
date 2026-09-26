"""Unit tests for the --default-thinking-mode server flag merge logic."""
from __future__ import annotations

from freetoken.server.openai_api import apply_default_thinking_mode


def test_auto_is_noop():
    assert apply_default_thinking_mode(None, "auto") is None
    assert apply_default_thinking_mode({"enable_thinking": True}, "auto") == {
        "enable_thinking": True
    }
    assert apply_default_thinking_mode(None, None) is None


def test_chat_fills_unset_kwargs():
    assert apply_default_thinking_mode(None, "chat") == {"enable_thinking": False}
    assert apply_default_thinking_mode({}, "chat") == {"enable_thinking": False}
    assert apply_default_thinking_mode({"other": 1}, "chat") == {
        "other": 1,
        "enable_thinking": False,
    }


def test_thinking_fills_unset_kwargs():
    assert apply_default_thinking_mode(None, "thinking") == {"enable_thinking": True}
    assert apply_default_thinking_mode({}, "thinking") == {"enable_thinking": True}


def test_explicit_request_value_wins():
    # Explicit per-request values override the server default, both directions.
    assert apply_default_thinking_mode({"enable_thinking": True}, "chat") == {
        "enable_thinking": True
    }
    assert apply_default_thinking_mode({"enable_thinking": False}, "thinking") == {
        "enable_thinking": False
    }
    assert apply_default_thinking_mode({"thinking": False}, "chat") == {"thinking": False}
    assert apply_default_thinking_mode({"thinking_mode": "thinking"}, "chat") == {
        "thinking_mode": "thinking"
    }


def test_original_dict_not_mutated():
    original = {"other": 1}
    result = apply_default_thinking_mode(original, "chat")
    assert result == {"other": 1, "enable_thinking": False}
    assert original == {"other": 1}


def test_unknown_mode_is_noop():
    assert apply_default_thinking_mode({"other": 1}, "bogus") == {"other": 1}