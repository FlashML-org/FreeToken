from __future__ import annotations

import json
import os
from copy import deepcopy

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer.tokenize import TokenizeManager, _dsv4_arguments_str
from freetoken.tokenizer.inline_system import InlineSystemError, normalize_inline_system, probe_inline_system


def test_fold_keeps_leading_system_and_appends_text_without_mutation():
    messages = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "image"}},
                                     {"type": "text", "text": "question"}]},
        {"role": "system", "content": "instruction 1"},
        {"role": "system", "content": "instruction 2"},
    ]
    original = deepcopy(messages)
    result = normalize_inline_system(messages, "fold")
    assert result == [messages[0], {"role": "user", "content": [
        messages[1]["content"][0], {"type": "text", "text": "question\n\ninstruction 1\n\ninstruction 2"}]}]
    assert messages == original


@pytest.mark.parametrize("mode", ["preserve", "fold"])
def test_inline_system_waits_for_all_parallel_tool_results(mode):
    messages = [
        {"role": "system", "content": "stable"}, {"role": "user", "content": "question"},
        {"role": "assistant", "reasoning_content": "reason", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "system", "content": "before results"},
        {"role": "tool", "tool_call_id": "a", "content": "result a"},
        {"role": "system", "content": "between results"},
        {"role": "tool", "tool_call_id": "b", "content": "result b"},
        {"role": "system", "content": "after results"},
    ]
    original = deepcopy(messages)
    result = normalize_inline_system(messages, mode)
    assert [m["tool_call_id"] for m in result if m["role"] == "tool"] == ["a", "b"]
    assert result[2] == messages[2]
    assert [m["role"] for m in result[3:5]] == ["tool", "tool"]
    if mode == "fold":
        assert len(result) == 5
        assert result[-1]["content"] == "result b\n\nbefore results\n\nbetween results\n\nafter results"
    else:
        assert [m["content"] for m in result[5:]] == ["before results", "between results", "after results"]
    assert messages == original


@pytest.mark.parametrize("mode", ["preserve", "fold"])
def test_inline_system_does_not_hide_missing_tool_results(mode):
    messages = [{"role": "assistant", "tool_calls": [{"id": "missing"}]},
                {"role": "system", "content": "instruction"}]
    with pytest.raises(InlineSystemError, match="missing tool results"):
        normalize_inline_system(messages, mode)


def test_fold_after_assistant_prepends_next_user_or_supplies_a_user_turn():
    messages = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"},
                {"role": "system", "content": "instruction"}]
    assert normalize_inline_system(messages, "fold")[-1] == {"role": "user", "content": "instruction"}
    result = normalize_inline_system(messages + [{"role": "user", "content": "next"}], "fold")
    assert result == messages[:2] + [{"role": "user", "content": "instruction\n\nnext"}]


def test_preserve_keeps_instruction_role_and_only_folds_without_a_stable_system():
    messages = [{"role": "system", "content": "stable"}, {"role": "user", "content": "question"},
                {"role": "system", "content": "late"}]
    assert normalize_inline_system(messages, "preserve") == messages
    assert normalize_inline_system(messages[1:], "preserve") == [{"role": "user", "content": "question\n\nlate"}]


class ProbeTokenizer:
    vocabulary = {text: 10000+i for i, text in enumerate(
        ["<system>", "</system>", "<user>", "</user>", "<assistant>", "</assistant>",
         "<tool>", "</tool>", "<think>", "</think>"])}
    all_special_ids = list(vocabulary.values())

    def get_added_vocab(self):
        return self.vocabulary

    def encode(self, text, **kwargs):
        import re
        parts = re.split("(" + "|".join(re.escape(x) for x in self.vocabulary) + ")", text)
        return [token for part in parts for token in ([self.vocabulary[part]] if part in self.vocabulary else map(ord, part))]

    def decode(self, ids):
        inverse = {v: k for k, v in self.vocabulary.items()}
        return "".join(inverse[x] if x in inverse else chr(x) for x in ids)


@pytest.mark.parametrize("behavior,expected", [("preserve", "preserve"), ("reject", "fold"),
    ("drop", "fold"), ("bare", "fold"), ("user", "fold"), ("rewrite", "fold"),
    ("after_generation", "fold"), ("drop_reasoning", "fold")])
def test_renderer_probe_checks_more_than_template_acceptance(behavior, expected):
    def render(messages):
        inline = any(m["role"] == "system" for m in messages[1:])
        output, late = "", ""
        for i, message in enumerate(messages):
            role, content = message["role"], message.get("content", "")
            if role == "system" and i:
                if behavior == "reject":
                    raise ValueError("system first only")
                if behavior == "drop":
                    continue
                if behavior == "bare":
                    output += content
                    continue
                if behavior == "user":
                    role = "user"
                if behavior == "after_generation":
                    late += content
                    continue
            if behavior == "rewrite" and inline and i == 0:
                content = "changed head"
            reason = message.get("reasoning_content")
            if reason and not (behavior == "drop_reasoning" and inline):
                content = "<think>" + reason + "</think>" + content
            output += f"<{role}>{content}</{role}>"
        return output + "<assistant>" + late
    assert probe_inline_system(render, ProbeTokenizer())[0] == expected


@pytest.mark.parametrize("position", ["middle", "trailing", "tool result"])
@pytest.mark.parametrize("metadata", ["Reasoning strength: high. Valid recipients: self, user.", "tool schema " * 1000],
                         ids=["metadata", "tools"])
def test_inline_probe_rejects_repeated_system_metadata(position, metadata):
    def render(messages):
        output = ""
        for i, message in enumerate(messages):
            role = message["role"]
            content = message.get("content", "")
            if role == "system" and i:
                if i < len(messages)-1:
                    location = "middle"
                else:
                    location = "tool result" if messages[i-1]["role"] == "tool" else "trailing"
                if location == position:
                    content += metadata
            content = message.get("reasoning_content", "") + content
            output += f"<{role}>{content}</{role}>"
        return output + "<assistant>"

    mode, reason = probe_inline_system(render, ProbeTokenizer())
    assert mode == "fold"
    assert "overhead" in reason


def test_inline_probe_allows_small_role_framing_overhead():
    def render(messages):
        return "".join(f'<{m["role"]}>' + ("role: " if m["role"] == "system" else "")
                       + m.get("reasoning_content", "") + m.get("content", "")
                       + f'</{m["role"]}>' for m in messages) + "<assistant>"

    assert probe_inline_system(render, ProbeTokenizer())[0] == "preserve"


def test_inline_probe_folds_when_user_insertion_cannot_be_measured():
    def render(messages):
        rewrite = any(m["role"] == "user" and m.get("content") == "Echo inline instruction" for m in messages)
        return ("changed head" if rewrite else "stable head") + "".join(
            f'<{m["role"]}>' + m.get("reasoning_content", "") + m.get("content", "")
            + f'</{m["role"]}>' for m in messages) + "<assistant>"

    mode, reason = probe_inline_system(render, ProbeTokenizer())
    assert mode == "fold"
    assert "cannot isolate" in reason


def test_inline_probe_cache_is_scoped_to_tools_and_render_kwargs(monkeypatch):
    import freetoken.tokenizer.tokenize as module
    calls = []
    monkeypatch.setattr(module, "probe_inline_system", lambda *_: (calls.append(1) or ("fold", "test")))
    manager = TokenizeManager(FakeTokenizer())
    for _ in range(3):
        assert manager.inline_system_mode(None, {}) == "fold"
    manager.inline_system_mode([], {})
    manager.inline_system_mode(None, {"enable_thinking": True})
    assert len(calls) == 3


def test_inline_policy_is_opt_in_and_not_forwarded_to_templates():
    tokenizer = FakeTokenizer()
    manager = TokenizeManager(tokenizer)
    messages = [{"role": "user", "content": "question"}, {"role": "system", "content": "instruction"}]
    msg = TokenizeMsg(uid=1, text=messages, sampling_params=SamplingParams())
    manager.render_prompt(msg)
    assert tokenizer.chat_template_messages == messages
    msg.inline_system_policy = "fold"
    manager.render_prompt(msg)
    assert tokenizer.chat_template_messages == [{"role": "user", "content": "question\n\ninstruction"}]
    assert "inline_system_policy" not in tokenizer.chat_template_kwargs


@pytest.mark.needs_weights
@pytest.mark.parametrize("variable,expected_mode", [
    ("FREETOKEN_DSV4_MODEL", "fold"),
    ("FREETOKEN_INLINE_GLM_MODEL", "preserve"),
    ("FREETOKEN_INLINE_QWEN_MODEL", "fold"),
])
def test_native_model_inline_system_tool_history(variable, expected_mode):
    from transformers import AutoTokenizer
    from freetoken.server.anthropic_api import convert_anthropic_to_genspec
    from freetoken.server.anthropic_models import AnthropicMessagesRequest

    path = os.environ.get(variable)
    if not path:
        pytest.skip(f"set {variable} to a local checkpoint")
    manager = TokenizeManager(AutoTokenizer.from_pretrained(path, local_files_only=True))
    data = {
        "model": "local", "max_tokens": 64, "system": "Stable project instructions.",
        "thinking": {"type": "enabled"},
        "tools": [{"name": "read", "input_schema": {"type": "object", "properties": {}}}],
        "messages": [
            {"role": "user", "content": "Inspect source:\n" + "value = 1\n" * 512},
            {"role": "system", "content": "Keep all existing tests."},
        ],
    }
    ids, prompts = [], []
    for turn in (1, 2):
        data["messages"].extend([
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": f"Retained thought {turn}.", "signature": ""},
                {"type": "tool_use", "id": f"read_{turn}", "name": "read", "input": {}},
            ]},
            {"role": "system", "content": f"During tool call {turn}."},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": f"read_{turn}", "content": f"Source result {turn}."},
            ]},
            {"role": "system", "content": f"Remaining budget: {1000-turn}."},
        ])
        req = AnthropicMessagesRequest.model_validate(data)
        spec = convert_anthropic_to_genspec(req, {})
        assert manager.inline_system_mode(spec.template_tools, spec.chat_template_kwargs) == expected_mode
        original = deepcopy(spec.messages)
        msg = TokenizeMsg(uid=turn, text=spec.messages, tools=spec.template_tools,
                          sampling_params=spec.sampling_params, chat_template_kwargs=spec.chat_template_kwargs,
                          inline_system_policy=spec.inline_system_policy)
        prompts.append(manager.render_prompt(msg))
        ids.append(manager.tokenize([msg])[0].input_ids.tolist())
        assert spec.messages == original
        assert prompts[-1].count("Keep all existing tests.") == 1
        assert "Retained thought 1." in prompts[-1]
        for previous in range(1, turn+1):
            assert prompts[-1].count(f"During tool call {previous}.") == 1
            assert prompts[-1].count(f"Remaining budget: {1000-previous}.") == 1
    common = next((i for i, pair in enumerate(zip(*ids)) if pair[0] != pair[1]), min(map(len, ids)))
    assert common >= len(ids[0]) - 64
    assert common > 2048


@pytest.mark.needs_weights
@pytest.mark.parametrize("tool_count", [0, 20])
def test_native_muse_inline_system_does_not_repeat_environment(tool_count):
    from transformers import AutoTokenizer

    path = os.environ.get("FREETOKEN_MUSE_MODEL")
    if not path:
        pytest.skip("set FREETOKEN_MUSE_MODEL to a local checkpoint")
    manager = TokenizeManager(AutoTokenizer.from_pretrained(path, local_files_only=True))
    tools = [{"type": "function", "function": {"name": f"read_{i}",
              "description": "Inspect the repository source. " * 100,
              "parameters": {"type": "object", "properties": {}}}} for i in range(tool_count)] or None
    messages = [{"role": "system", "content": "Stable instructions."},
                {"role": "user", "content": "Inspect this file.\n" + "value = 1\n" * 256}]
    assert manager.inline_system_mode(tools, {}) == "fold"
    for turn in range(3):
        messages.extend([{"role": "assistant", "content": f"Answer {turn}."},
                         {"role": "user", "content": f"Question {turn}."},
                         {"role": "system", "content": f"Budget reminder {turn}."}])
        original = deepcopy(messages)
        msg = TokenizeMsg(uid=turn, text=messages, tools=tools, sampling_params=SamplingParams(),
                          inline_system_policy="auto")
        prompt = manager.render_prompt(msg)
        assert prompt.count("Reasoning strength:") == 1
        assert prompt.count("# Valid recipients:") == 1
        assert prompt.count("// Function schemas") == bool(tools)
        for previous in range(turn+1):
            assert prompt.count(f"Budget reminder {previous}.") == 1
        ids = manager.tokenize([msg])[0].input_ids.tolist()
        msg.inline_system_policy = "fold"
        assert ids == manager.tokenize([msg])[0].input_ids.tolist()
        assert messages == original

    latest = [m for i, m in enumerate(messages) if m["role"] != "system" or i in (0, len(messages)-1)]
    updated = deepcopy(latest)
    updated[-1]["content"] = "Budget reminder 900."
    tokens = []
    for conversation in (latest, updated):
        msg = TokenizeMsg(uid=0, text=conversation, tools=tools, sampling_params=SamplingParams(),
                          inline_system_policy="auto")
        tokens.append(manager.tokenize([msg])[0].input_ids.tolist())
    common = next((i for i, pair in enumerate(zip(*tokens)) if pair[0] != pair[1]), min(map(len, tokens)))
    assert common >= min(map(len, tokens)) - 16


class FakeTokenizer:
    def __init__(self) -> None:
        self.chat_template_kwargs = None
        self.chat_template_messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_kwargs = kwargs
        self.chat_template_messages = deepcopy(messages)
        return "rendered prompt"

    def encode(self, prompt, return_tensors=None, add_special_tokens=True):
        assert prompt == "rendered prompt"
        assert return_tensors == "pt"
        # The template rendered every special token already; encode must not
        # add another bos on top (the muse-glimmer/llama double-bos bug).
        assert add_special_tokens is False
        return torch.tensor([[1, 2, 3]], dtype=torch.long)


def test_tokenize_manager_passes_chat_template_kwargs():
    tokenizer = FakeTokenizer()
    manager = TokenizeManager(tokenizer)
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "hello"}],
        sampling_params=SamplingParams(),
        chat_template_kwargs={"enable_thinking": True},
    )

    input_ids = manager.tokenize([msg])[0].input_ids

    assert tokenizer.chat_template_kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }
    assert input_ids.tolist() == [1, 2, 3]


def test_tokenize_manager_passes_tools_to_chat_template():
    tokenizer = FakeTokenizer()
    manager = TokenizeManager(tokenizer)
    tools = [
        {
            "name": "get_weather",
            "description": "Return weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "weather?"}],
        sampling_params=SamplingParams(),
        tools=tools,
    )

    input_ids = manager.tokenize([msg])[0].input_ids

    assert tokenizer.chat_template_kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "tools": tools,
    }
    assert input_ids.tolist() == [1, 2, 3]


@pytest.mark.parametrize("initial_system", [[], [{"role": "system", "content": "stable instructions"}]])
def test_generic_template_folds_system_messages_without_mutating_input(initial_system):
    tokenizer = FakeTokenizer()
    manager = TokenizeManager(tokenizer)
    messages = initial_system + [
        {"role": "user", "content": "inspect files"},
        {"role": "system", "content": "first reminder"},
        {"role": "assistant", "content": "found the bug"},
        {"role": "system", "content": "second reminder"},
    ]
    original = deepcopy(messages)
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    original_tools = deepcopy(tools)
    manager.tokenize([TokenizeMsg(uid=1, text=messages, tools=tools, sampling_params=SamplingParams(),
                                 inline_system_policy="auto")])
    assert tokenizer.chat_template_messages == initial_system + [
        {"role": "user", "content": "inspect files\n\nfirst reminder"},
        {"role": "assistant", "content": "found the bug"},
        {"role": "user", "content": "second reminder"},
    ]
    assert tokenizer.chat_template_kwargs["tools"] == original_tools
    assert messages == original
    assert tools == original_tools


class FakeDsv4Tokenizer:
    chat_template = None

    def __init__(self, model_path) -> None:
        self.name_or_path = str(model_path)
        self.prompt = None

    def encode(self, prompt, return_tensors=None, add_special_tokens=True):
        self.prompt = prompt
        assert return_tensors == "pt"
        # dsv4's own encoder path keeps the default special-token behavior.
        assert add_special_tokens is True
        return torch.tensor([[4, 5, 6]], dtype=torch.long)


def test_dsv41_encoder_gets_image_sources_before_and_after_fetch(tmp_path):
    import copy
    from freetoken.mm.media import collect_image_refs
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding_dsv4.py").write_text('''
def encode_messages(messages, thinking_mode, reasoning_effort=None):
    parts = messages[0]["content"]
    assert parts[1] == {"type": "text", "text": "compare"}
    assert parts[0].get("url") == "https://example.com/image.png" or parts[0].get("data") == b"first image bytes"
    assert parts[2].get("data") in ("aW1hZ2U=", b"second image bytes")
    return "two images"
''')
    messages = [{"role": "user", "content": [
        {"type": "image", "freetoken_ref": {"kind": "url", "data": "https://example.com/image.png"}},
        {"type": "text", "text": "compare"},
        {"type": "image", "freetoken_ref": {"kind": "b64", "data": "aW1hZ2U="}},
    ]}]
    manager = TokenizeManager(FakeDsv4Tokenizer(tmp_path))
    msg = TokenizeMsg(uid=1, text=messages, sampling_params=SamplingParams())
    original = copy.deepcopy(messages)
    assert manager.render_prompt(msg) == "two images"
    assert messages == original
    assert len(collect_image_refs(messages)) == 2
    msg.images = [b"first image bytes", b"second image bytes"]
    stripped = copy.deepcopy(messages)
    assert manager.render_prompt(msg) == "two images"
    assert messages == stripped


def test_tokenize_manager_uses_dsv4_encoder_when_chat_template_is_missing(tmp_path):
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding_dsv4.py").write_text(
        """
def encode_messages(messages, thinking_mode, reasoning_effort=None):
    assert thinking_mode == "thinking"
    assert messages[0]["role"] == "system"
    assert messages[0]["tools"][0]["function"]["name"] == "read"
    assert messages[1]["role"] == "user"
    return "dsv4 prompt"
""".lstrip()
    )
    tokenizer = FakeDsv4Tokenizer(tmp_path)
    manager = TokenizeManager(tokenizer)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}},
            },
        }
    ]
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "inspect files"}],
        sampling_params=SamplingParams(),
        tools=tools,
    )

    input_ids = manager.tokenize([msg])[0].input_ids

    assert tokenizer.prompt == "dsv4 prompt"
    assert input_ids.tolist() == [4, 5, 6]


@pytest.mark.parametrize("initial_system", [[], [{"role": "system", "content": "stable instructions"}]])
def test_dsv4_fold_keeps_instructions_and_tools_on_stable_prefix(tmp_path, initial_system):
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding_dsv4.py").write_text(
        "import json\n"
        "def encode_messages(messages, thinking_mode, reasoning_effort=None):\n"
        "    return json.dumps(messages)\n"
    )
    tokenizer = FakeDsv4Tokenizer(tmp_path)
    manager = TokenizeManager(tokenizer)
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    messages = initial_system + [
        {"role": "user", "content": "inspect files"},
        {"role": "system", "content": "first reminder"},
        {"role": "assistant", "content": "found the bug"},
    ]
    original, original_tools = deepcopy(messages), deepcopy(tools)
    manager.tokenize([TokenizeMsg(uid=1, text=messages, tools=tools, sampling_params=SamplingParams(),
                                 inline_system_policy="fold")])
    first = json.loads(tokenizer.prompt)
    extended = messages + [{"role": "system", "content": "next reminder"}]
    original_extended = deepcopy(extended)
    manager.tokenize([TokenizeMsg(uid=2, text=extended, tools=tools, sampling_params=SamplingParams(),
                                 inline_system_policy="fold")])
    second = json.loads(tokenizer.prompt)
    assert [m["role"] for m in first] == ["system", "user", "assistant"]
    assert first[0]["content"] == (initial_system[0]["content"] if initial_system else "")
    assert first[0]["tools"] == original_tools
    assert first[1]["content"] == "inspect files\n\nfirst reminder"
    assert second[:-1] == first
    assert second[-1] == {"role": "user", "content": "next reminder"}
    assert all("tools" not in m for m in second[1:])
    assert messages == original and extended == original_extended and tools == original_tools


@pytest.fixture(scope="module")
def native_dsv4_manager():
    model_path = os.environ.get("FREETOKEN_DSV4_MODEL")
    if not model_path:
        pytest.skip("set FREETOKEN_DSV4_MODEL to a local DSV4 checkpoint")
    from transformers import AutoTokenizer

    manager = TokenizeManager(AutoTokenizer.from_pretrained(model_path, local_files_only=True))
    assert manager._dsv4_encoder is not None, "checkpoint must supply the native DSV4 encoder"
    return manager


def test_dsv4_no_tools_retains_reminders_when_encoder_drops_developer_turns(tmp_path):
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding_dsv4.py").write_text(
        "import json\n"
        "def encode_messages(messages, thinking_mode, reasoning_effort=None):\n"
        "    if thinking_mode == 'thinking':\n"
        "        messages = [m for m in messages if m['role'] != 'developer']\n"
        "    return json.dumps(messages)\n"
    )
    manager = TokenizeManager(FakeDsv4Tokenizer(tmp_path))
    messages = [
        {"role": "user", "content": "First task."},
        {"role": "system", "content": "Keep this earlier instruction."},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Second task."},
    ]
    original = deepcopy(messages)
    msg = TokenizeMsg(uid=1, text=messages, sampling_params=SamplingParams(),
                      chat_template_kwargs={"enable_thinking": True}, inline_system_policy="fold")
    rendered = json.loads(manager.render_prompt(msg))
    assert all(m["role"] in ("system", "user", "assistant") for m in rendered)
    assert sum(m["content"].count("Keep this earlier instruction.") for m in rendered) == 1
    assert messages == original


@pytest.mark.needs_weights
@pytest.mark.parametrize("system", [None, "You are a coding assistant."])
@pytest.mark.parametrize("change", ["update", "append"])
def test_native_dsv4_reminder_update_preserves_token_prefix(native_dsv4_manager, system, change):
    from freetoken.server.anthropic_api import convert_anthropic_to_genspec
    from freetoken.server.anthropic_models import AnthropicMessagesRequest

    data = {
        "model": "dsv4", "max_tokens": 64,
        "tools": [{"name": "read", "input_schema": {"type": "object", "properties": {}}}],
        "messages": [
            {"role": "user", "content": "Inspect this file:\n" + "value = 1\n" * 512},
            {"role": "system", "content": "Keep the changes minimal."},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I should read the source.", "signature": ""},
                {"type": "tool_use", "id": "call_read", "name": "read", "input": {}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_read", "content": "value = 1"}]},
            {"role": "system", "content": "Remaining token budget: 1000"},
        ],
    }
    if system is not None:
        data["system"] = system
    tokens = []
    for budget in (1000, 900):
        if budget == 900 and change == "append":
            data["messages"].extend([
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "The file is simple.", "signature": ""},
                    {"type": "text", "text": "I inspected the file."},
                ]},
                {"role": "system", "content": "Remaining token budget: 900"},
            ])
        else:
            data["messages"][-1]["content"] = f"Remaining token budget: {budget}"
        req = AnthropicMessagesRequest.model_validate(data)
        original = req.model_dump()
        spec = convert_anthropic_to_genspec(req, {})
        msg = TokenizeMsg(uid=budget, text=spec.messages, tools=spec.template_tools,
                          sampling_params=spec.sampling_params, chat_template_kwargs=spec.chat_template_kwargs,
                          inline_system_policy=spec.inline_system_policy)
        prompt = native_dsv4_manager.render_prompt(msg)
        assert prompt.count("Keep the changes minimal.") == 1
        assert prompt.index("Inspect this file:") < prompt.index(f"Remaining token budget: {budget}")
        tokens.append(native_dsv4_manager.tokenize([msg])[0].input_ids.tolist())
        assert req.model_dump() == original
    common = next((i for i, pair in enumerate(zip(*tokens)) if pair[0] != pair[1]), min(map(len, tokens)))
    assert common > 2048
    assert common >= min(map(len, tokens)) - 64


@pytest.mark.needs_weights
@pytest.mark.parametrize("disable_declared_tools", [False, True])
@pytest.mark.parametrize("thinking_enabled", [False, True])
def test_native_dsv4_no_tools_keeps_earlier_reminder(
    native_dsv4_manager, disable_declared_tools, thinking_enabled
):
    from freetoken.server.anthropic_api import convert_anthropic_to_genspec
    from freetoken.server.anthropic_models import AnthropicMessagesRequest

    data = {
        "model": "dsv4", "max_tokens": 64, "system": "Stable instructions.",
        "thinking": {"type": "enabled" if thinking_enabled else "disabled"},
        "messages": [
            {"role": "user", "content": "First task."},
            {"role": "system", "content": "Keep this earlier instruction."},
            {"role": "assistant", "content": "First answer."},
            {"role": "user", "content": "Second task."},
        ],
    }
    if disable_declared_tools:
        data["tools"] = [{"name": "read", "input_schema": {"type": "object", "properties": {}}}]
        data["tool_choice"] = {"type": "none"}
    req = AnthropicMessagesRequest.model_validate(data)
    original = req.model_dump()
    spec = convert_anthropic_to_genspec(req, {})
    assert not spec.template_tools
    msg = TokenizeMsg(uid=1, text=spec.messages, tools=spec.template_tools,
                      sampling_params=spec.sampling_params, chat_template_kwargs=spec.chat_template_kwargs,
                      inline_system_policy=spec.inline_system_policy)
    prompt = native_dsv4_manager.render_prompt(msg)
    assert prompt.count("Keep this earlier instruction.") == 1
    assert "Second task." in prompt
    assert req.model_dump() == original


def test_dsv4_encoder_gets_tool_call_arguments_as_json_string(tmp_path):
    """Regression: render_messages hands the template dict arguments; the dsv4
    encoder contract is a JSON-object STRING -- a dict trips its fallback that
    wraps every replayed call in a parameter literally named "arguments"."""
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding_dsv4.py").write_text(
        """
import json

def encode_messages(messages, thinking_mode, reasoning_effort=None):
    (tc,) = messages[1]["tool_calls"]
    arguments = tc["function"]["arguments"]
    assert isinstance(arguments, str), f"expected str, got {type(arguments)}"
    assert json.loads(arguments) == {"command": "gog calendar time", "n": 2}
    return "dsv4 prompt"
""".lstrip()
    )
    tokenizer = FakeDsv4Tokenizer(tmp_path)
    manager = TokenizeManager(tokenizer)
    messages = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call0",
                    "type": "function",
                    # dict form, as produced by server render_messages
                    "function": {"name": "exec", "arguments": {"command": "gog calendar time", "n": 2}},
                }
            ],
        },
    ]
    msg = TokenizeMsg(uid=1, text=messages, sampling_params=SamplingParams())

    input_ids = manager.tokenize([msg])[0].input_ids

    assert tokenizer.prompt == "dsv4 prompt"
    assert input_ids.tolist() == [4, 5, 6]
    # caller's messages must not be mutated (message copies are shallow)
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {
        "command": "gog calendar time",
        "n": 2,
    }


def test_dsv4_arguments_str_normalization():
    assert _dsv4_arguments_str({"a": 1, "b": "x"}) == '{"a": 1, "b": "x"}'
    assert _dsv4_arguments_str({"t": "héllo 世界"}) == '{"t": "héllo 世界"}'  # ensure_ascii=False
    assert _dsv4_arguments_str('{"a": 1}') == '{"a": 1}'  # object string passes through verbatim
    assert _dsv4_arguments_str(None) == "{}"
    assert _dsv4_arguments_str("") == "{}"
    assert _dsv4_arguments_str("  ") == "{}"
    for bad in ("[1,2]", "5", "true", '"x"', "not json", [1, 2], 5):
        with pytest.raises(ValueError):
            _dsv4_arguments_str(bad)


class Qwen38LikeTokenizer:
    """Fake whose template grades effort like Qwen3.8: validates the vocabulary
    whenever thinking is not explicitly off, distinct preamble per gear."""

    def __init__(self) -> None:
        self.chat_template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_kwargs = kwargs
        if kwargs.get("enable_thinking") is not False:
            effort = kwargs.get("reasoning_effort", "xhigh")
            if effort not in ("xhigh", "medium", "low"):
                raise ValueError(f"Unexpected reasoning effort {effort}")
            return f"prompt effort={effort}"
        return "prompt effort=off"

    def encode(self, prompt, return_tensors=None, add_special_tokens=True):
        return torch.tensor([[7, 8]], dtype=torch.long)


def test_tokenize_quantizes_foreign_effort_onto_the_template_vocabulary():
    """DeepSeek-dialect "high" must reach a Qwen3.8-style template as its
    nearest supported gear, not raw (raw would raise_exception)."""
    tokenizer = Qwen38LikeTokenizer()
    manager = TokenizeManager(tokenizer)
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "hello"}],
        sampling_params=SamplingParams(),
        chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "high"},
    )

    manager.tokenize([msg])

    assert tokenizer.chat_template_kwargs["reasoning_effort"] == "xhigh"
    assert tokenizer.chat_template_kwargs["enable_thinking"] is True
    # the caller's kwargs stay untouched
    assert msg.chat_template_kwargs["reasoning_effort"] == "high"


def test_tokenize_drops_effort_for_templates_that_ignore_it():
    tokenizer = FakeTokenizer()  # renders the same prompt regardless of kwargs
    manager = TokenizeManager(tokenizer)
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "hello"}],
        sampling_params=SamplingParams(),
        chat_template_kwargs={"reasoning_effort": "high"},
    )

    manager.tokenize([msg])

    assert "reasoning_effort" not in tokenizer.chat_template_kwargs


def test_tokenize_drops_far_effort_for_the_dsv4_encoder(tmp_path):
    """An OpenAI-dialect "medium" has no nearby dsv4 gear, so nothing is sent
    and the encoder default ("low") applies -- never a silent escalation to
    the absolute-maximum "high" prompt."""
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding_dsv4.py").write_text(
        """
SEEN = []

def encode_messages(messages, thinking_mode, reasoning_effort=None):
    effort = reasoning_effort or "low"
    assert effort in ("low", "high", "max"), f"Invalid reasoning effort: {effort}"
    SEEN.append(reasoning_effort)
    return f"dsv4 prompt effort={effort}"
""".lstrip()
    )
    tokenizer = FakeDsv4Tokenizer(tmp_path)
    manager = TokenizeManager(tokenizer)
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "hello"}],
        sampling_params=SamplingParams(),
        chat_template_kwargs={"reasoning_effort": "medium"},
    )

    manager.tokenize([msg])

    assert tokenizer.prompt == "dsv4 prompt effort=low"


def test_tokenize_survives_an_unhashable_effort():
    tokenizer = Qwen38LikeTokenizer()
    manager = TokenizeManager(tokenizer)
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "hello"}],
        sampling_params=SamplingParams(),
        chat_template_kwargs={"reasoning_effort": ["high"]},  # legal JSON on the wire
    )

    manager.tokenize([msg])

    assert "reasoning_effort" not in tokenizer.chat_template_kwargs


@pytest.mark.parametrize("budget", [1, 25, 50, 75, 100])
def test_numeric_encoder_effort_survives_prompt_rendering(tmp_path, budget):
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding.py").write_text('''
def encode_messages(messages, thinking_mode, reasoning_effort=None):
    if thinking_mode != "thinking":
        return "chat"
    effort = "high" if reasoning_effort is None else reasoning_effort
    if isinstance(effort, str):
        effort = {"low": 50, "high": 75, "max": 100}[effort]
    if type(effort) is not int or not 1 <= effort <= 100:
        raise ValueError("invalid budget")
    return f"Reasoning Effort: {effort}"
''')
    manager = TokenizeManager(FakeDsv4Tokenizer(tmp_path))
    msg = TokenizeMsg(
        uid=1, text=[{"role": "user", "content": "hello"}],
        sampling_params=SamplingParams(),
        chat_template_kwargs={"enable_thinking": True, "reasoning_effort": budget},
    )
    assert manager.render_prompt(msg) == f"Reasoning Effort: {budget}"
    assert msg.chat_template_kwargs["reasoning_effort"] == budget
