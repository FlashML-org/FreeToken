# SPDX-License-Identifier: Apache-2.0
# Renderer-probe approach adapted from vLLM PR #58772.
"""Preserve or fold late system instructions without hoisting the prompt head."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable


# Allow small role-framing differences, not a repeated environment header.
_INLINE_ROLE_SLACK_TOKENS = 8


class InlineSystemError(ValueError):
    """An input instruction cannot be placed without corrupting tool ordering."""


def has_inline_system(messages: list[dict]) -> bool:
    started = False
    for message in messages:
        if message.get("role") == "system":
            if started:
                return True
        else:
            started = True
    return False


def _add_text(message: dict, text: str, *, prepend: bool = False) -> dict:
    result = dict(message)
    content = result.get("content")
    def join(a, b):
        return "\n\n".join(part for part in (a, b) if part)
    if content is None or isinstance(content, str):
        result["content"] = join(text, content) if prepend else join(content, text)
    elif isinstance(content, list):
        parts = deepcopy(content)
        index = 0 if prepend else -1
        if parts and parts[index].get("type") == "text":
            previous = parts[index].get("text") or ""
            parts[index]["text"] = join(text, previous) if prepend else join(previous, text)
        else:
            parts.insert(0 if prepend else len(parts), {"type": "text", "text": text})
        result["content"] = parts
    else:
        raise InlineSystemError("inline system folding requires text or content blocks")
    return result


def _after_tool_results(messages: list[dict]) -> list[dict]:
    out, held = [], []
    pending_ids: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role == "system" and pending_ids:
            held.append(message)
            continue
        if role == "tool":
            out.append(message)
            pending_ids.discard(message.get("tool_call_id", ""))
            if not pending_ids:
                out.extend(held)
                held.clear()
            continue
        if pending_ids and held:
            raise InlineSystemError("inline system instruction interrupts incomplete tool results")
        out.append(message)
        pending_ids = {call.get("id", "") for call in message.get("tool_calls", [])} if role == "assistant" else set()
    if held:
        raise InlineSystemError("inline system instruction is waiting for missing tool results")
    return out


def normalize_inline_system(messages: list[dict], mode: str) -> list[dict]:
    if mode not in ("preserve", "fold"):
        raise InlineSystemError(f"invalid inline system mode: {mode}")
    if not has_inline_system(messages):
        return messages
    ordered = _after_tool_results(deepcopy(messages))
    # Without a stable leading system, some renderers attach tools to the first
    # late system message. Folding avoids moving tools when that message changes.
    if mode == "preserve" and ordered[0].get("role") == "system":
        return ordered
    out, pending = [], []
    started = False
    for message in ordered:
        role = message.get("role")
        if role == "system" and started:
            text = message.get("content") or ""
            if not isinstance(text, str):
                raise InlineSystemError("inline system instructions must be text")
            if text:
                if out and out[-1].get("role") in ("user", "tool") and not pending:
                    out[-1] = _add_text(out[-1], text)
                else:
                    pending.append(text)
            continue
        if pending:
            text = "\n\n".join(pending)
            pending.clear()
            if role == "user":
                message = _add_text(message, text, prepend=True)
            else:
                out.append({"role": "user", "content": text})
        out.append(message)
        started = started or role != "system"
    if pending:
        out.append({"role": "user", "content": "\n\n".join(pending)})
    return out


def _insertion(base: list[int], new: list[int]) -> tuple[int, list[int]] | None:
    start = 0
    while start < min(len(base), len(new)) and base[start] == new[start]:
        start += 1
    suffix = 0
    while suffix < min(len(base), len(new)) - start and base[-suffix-1] == new[-suffix-1]:
        suffix += 1
    if start + suffix != len(base):
        return None
    return start, new[start:len(new)-suffix]


def probe_inline_system(render: Callable[[list[dict]], str], tokenizer: Any) -> tuple[str, str]:
    """Conservatively require a distinct, prefix-stable system segment."""
    note, reasoning = "Echo inline instruction", "Foxtrot previous reasoning"
    system = {"role": "system", "content": "Alpha stable instructions"}
    user = {"role": "user", "content": "Bravo first question"}
    cases = [
        ("middle", [system, user, {"role": "assistant", "content": "Charlie first answer"},
                    {"role": "user", "content": "Delta second question"}], 3),
        ("trailing", [system, user], 2),
        ("tool result", [system, user,
            {"role": "assistant", "content": "", "reasoning_content": reasoning, "thinking": reasoning,
             "reasoning": reasoning, "tool_calls": [{"id": "probe_call", "type": "function",
                "function": {"name": "read", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "probe_call", "content": "Golf tool output"}], 4),
    ]
    max_system_tokens = None
    try:
        special_ids = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
        if not special_ids:
            return "fold", "renderer has no identifiable special tokens"
        for name, messages, index in cases:
            base_text = render(deepcopy(messages))
            base = tokenizer.encode(base_text, add_special_tokens=False)
            with_note = messages[:index] + [{"role": "system", "content": note}] + messages[index:]
            new_text = render(deepcopy(with_note))
            new = tokenizer.encode(new_text, add_special_tokens=False)
            found = _insertion(base, new)
            if found is None:
                return "fold", f"{name}: other prompt tokens change"
            position, segment = found
            if index == len(messages) and position >= len(base):
                return "fold", f"{name}: instruction follows the generation opening"
            if tokenizer.decode(segment).count(note) != 1 or not special_ids.intersection(segment):
                return "fold", f"{name}: no distinct role-marked instruction"
            if (reasoning in base_text) != (reasoning in new_text):
                return "fold", f"{name}: reasoning retention changes"
            if name == "middle":
                as_user = messages[:index] + [{"role": "user", "content": note}] + messages[index:]
                user_tokens = tokenizer.encode(render(deepcopy(as_user)), add_special_tokens=False)
                user_segment = _insertion(base, user_tokens)
                if user_segment is None:
                    return "fold", "middle: cannot isolate a user instruction"
                if user_segment[1] == segment:
                    return "fold", "system instructions render as ordinary user turns"
                max_system_tokens = len(user_segment[1]) + _INLINE_ROLE_SLACK_TOKENS
            if max_system_tokens is not None and len(segment) > max_system_tokens:
                return "fold", f"{name}: system insertion overhead ({len(segment)} > {max_system_tokens} tokens)"
    except Exception as exc:
        return "fold", f"probe could not render: {type(exc).__name__}"
    return "preserve", "renderer preserves distinct system turns and existing prompt tokens"
