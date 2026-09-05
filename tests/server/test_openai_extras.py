"""OpenAI protocol coverage: n > 1 fan-out, echo and suffix (FIM) on completions, the
sampling extras reaching SamplingParams, request_id, usage details, and the tokenize /
detokenize routes."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from freetoken.message import TokenizeMsg, UserReply
from freetoken.server.api_models import (
    ChatCompletionRequest,
    CompletionRequest,
    DetokenizeRequest,
    TokenizeRequest,
)
from freetoken.server.openai_api import (
    _usage,
    chat_request_to_genspec,
    handle_chat_completion,
    handle_completion,
    handle_detokenize,
    handle_tokenize,
)
from freetoken.server.stats import prometheus_text


def run(coro):
    return asyncio.run(coro)


class FakeState:
    """Distinct uids per generation; every uid gets the same two-ack reply."""

    def __init__(self, text=("hel", "lo"), reasoning_tokens=0, tokenizer=None):
        self.config = SimpleNamespace(
            model_path="/models/unit-model",
            served_model_name="unit-model",
            tool_call_parser="llama3",
            reasoning_parser=None,
            max_seq_len=4096,
        )
        self.text = text
        self.reasoning_tokens = reasoning_tokens
        self._next = 100
        self.sent: list[TokenizeMsg] = []
        self.aborted: list[int] = []
        if tokenizer is not None:
            manager = SimpleNamespace(
                tokenizer=tokenizer,
                render_prompt=lambda msg: "RENDERED:" + json.dumps(msg.text),
            )
            self.frontend_tokenizer = lambda: manager

    def new_user(self) -> int:
        self._next += 1
        return self._next

    async def send_one(self, msg):
        self.sent.append(msg)

    async def abort_user(self, uid):
        self.aborted.append(uid)

    async def wait_for_ack(self, uid: int):
        yield UserReply(
            uid=uid, incremental_output="", finished=False, prompt_tokens_delta=5
        )
        for i, piece in enumerate(self.text):
            last = i == len(self.text) - 1
            yield UserReply(
                uid=uid,
                incremental_output=piece,
                finished=last,
                completion_tokens_delta=1,
                finish_reason="stop" if last else None,
                reasoning_tokens=self.reasoning_tokens if last else 0,
            )


def chat(**kwargs) -> ChatCompletionRequest:
    body = {"model": "unit-model", "messages": [{"role": "user", "content": "hi"}]}
    body.update(kwargs)
    return ChatCompletionRequest(**body)


def parse_sse(chunks: list[bytes]) -> list:
    out = []
    for chunk in chunks:
        for line in chunk.decode().split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            out.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return out


async def _collect(generator):
    return [item async for item in generator]


# ---- n > 1 ------------------------------------------------------------------------------


def test_chat_n_returns_one_choice_per_generation_and_counts_the_prompt_once():
    state = FakeState(reasoning_tokens=2)
    body = run(handle_chat_completion(chat(n=2), None, state, {}))
    assert [c["index"] for c in body["choices"]] == [0, 1]
    assert all(c["message"]["content"] == "hello" for c in body["choices"])
    assert len(state.sent) == 2 and len({m.uid for m in state.sent}) == 2
    usage = body["usage"]
    assert (
        usage["prompt_tokens"] == 5
        and usage["completion_tokens"] == 4
        and usage["total_tokens"] == 9
    )
    assert usage["completion_tokens_details"] == {"reasoning_tokens": 4}
    assert body["system_fingerprint"] is None


def test_chat_n_stream_interleaves_choice_indices_and_ends_once():
    state = FakeState()
    response = run(
        handle_chat_completion(
            chat(n=2, stream=True, stream_options={"include_usage": True}),
            None,
            state,
            {},
        )
    )
    events = parse_sse(run(_collect(response.body_iterator)))
    assert events.count("[DONE]") == 1 and events[-1] == "[DONE]"
    indices = {c["index"] for ev in events[:-1] for c in ev.get("choices", [])}
    assert indices == {0, 1}
    usage_chunks = [ev for ev in events[:-1] if ev.get("usage")]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"]["prompt_tokens"] == 5
    assert usage_chunks[0]["usage"]["completion_tokens"] == 4
    finals = [
        c for ev in events[:-1] for c in ev.get("choices", []) if c.get("finish_reason")
    ]
    assert len(finals) == 2


def test_n_out_of_range_is_400():
    for n in (0, 17):
        body = run(handle_chat_completion(chat(n=n), None, FakeState(), {}))
        assert body.status_code == 400
        assert json.loads(body.body)["error"]["param"] == "n"


# ---- completions: echo, suffix, n ---------------------------------------------------------


def test_completion_echo_prepends_the_prompt():
    state = FakeState()
    req = CompletionRequest(model="unit-model", prompt="Once upon", echo=True)
    body = run(handle_completion(req, None, state, {}))
    assert body["choices"][0]["text"] == "Once uponhello"


def test_completion_echo_streams_the_prompt_first():
    state = FakeState()
    req = CompletionRequest(
        model="unit-model", prompt="Once upon", echo=True, stream=True
    )
    response = run(handle_completion(req, None, state, {}))
    events = parse_sse(run(_collect(response.body_iterator)))
    assert events[0]["choices"][0]["text"] == "Once upon"
    assert events[-1] == "[DONE]"


class _FimTokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, name):
        return {"<|fim_prefix|>": 10, "<|fim_suffix|>": 11, "<|fim_middle|>": 12}.get(
            name, -1
        )


class _NoFimTokenizer:
    unk_token_id = 3

    def convert_tokens_to_ids(self, name):
        return 3


def test_completion_suffix_builds_a_fill_in_the_middle_prompt():
    state = FakeState(tokenizer=_FimTokenizer())
    req = CompletionRequest(model="unit-model", prompt="def f(", suffix="    return 1")
    run(handle_completion(req, None, state, {}))
    assert (
        state.sent[0].text
        == "<|fim_prefix|>def f(<|fim_suffix|>    return 1<|fim_middle|>"
    )


def test_completion_suffix_needs_fim_tokens():
    state = FakeState(tokenizer=_NoFimTokenizer())
    req = CompletionRequest(model="unit-model", prompt="def f(", suffix="x")
    body = run(handle_completion(req, None, state, {}))
    assert (
        body.status_code == 400 and json.loads(body.body)["error"]["param"] == "suffix"
    )


def test_completion_n_orders_choices_prompt_major():
    state = FakeState()
    req = CompletionRequest(model="unit-model", prompt=["a", "b"], n=2)
    body = run(handle_completion(req, None, state, {}))
    assert [c["index"] for c in body["choices"]] == [0, 1, 2, 3]
    assert [m.text for m in state.sent] == ["a", "a", "b", "b"]
    assert body["usage"]["prompt_tokens"] == 10  # two prompts, each counted once
    assert body["usage"]["completion_tokens"] == 8


# ---- sampling extras --------------------------------------------------------------------


def test_sampling_extras_reach_sampling_params():
    spec = chat_request_to_genspec(
        chat(
            min_p=0.1,
            repetition_penalty=1.2,
            presence_penalty=0.5,
            frequency_penalty=0.3,
            logit_bias={"5": 150, "7": -3},
            min_tokens=3,
            stop_token_ids=[9],
            include_stop_str_in_output=True,
            skip_special_tokens=True,
            seed=7,
            user="u1",
        ),
        {},
    )
    sp = spec.sampling_params
    assert sp.min_p == 0.1 and sp.repetition_penalty == 1.2
    assert sp.presence_penalty == 0.5 and sp.frequency_penalty == 0.3
    assert sp.logit_bias == {5: 100.0, 7: -3.0}  # clamped to the OpenAI range
    assert sp.min_tokens == 3 and sp.stop_token_ids == [9]
    assert sp.include_stop_str_in_output is True and sp.skip_special_tokens is True
    assert sp.needs_logits_processing


@pytest.mark.parametrize(
    "field",
    [
        {"min_p": 1.5},
        {"presence_penalty": 3.0},
        {"frequency_penalty": -2.5},
        {"repetition_penalty": 0.0},
        {"logit_bias": {"x": 1.0}},
        {"min_tokens": -1},
        {"stop_token_ids": [-1]},
    ],
)
def test_bad_sampling_extras_are_400(field):
    body = run(handle_chat_completion(chat(**field), None, FakeState(), {}))
    assert body.status_code == 400, field


def test_continue_final_message_needs_an_assistant_tail_and_reaches_the_template():
    spec = chat_request_to_genspec(
        chat(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Sure,"},
            ],
            continue_final_message=True,
        ),
        {},
    )
    assert spec.chat_template_kwargs["continue_final_message"] is True
    with pytest.raises(ValueError):
        chat(continue_final_message=True)


# ---- ids and usage ------------------------------------------------------------------------


def test_request_id_becomes_the_response_id():
    state = FakeState()
    body = run(handle_chat_completion(chat(request_id="abc-1"), None, state, {}))
    assert body["id"] == "chatcmpl-abc-1"
    response = run(
        handle_chat_completion(chat(request_id="abc-2", stream=True), None, state, {})
    )
    events = parse_sse(run(_collect(response.body_iterator)))
    assert events[0]["id"] == "chatcmpl-abc-2"


def test_usage_details_appear_only_when_nonzero():
    assert "completion_tokens_details" not in _usage(10, 5)
    assert _usage(10, 5, reasoning_tokens=3)["completion_tokens_details"] == {
        "reasoning_tokens": 3
    }
    assert _usage(10, 5, cached_tokens=4)["prompt_tokens_details"] == {
        "cached_tokens": 4
    }


# ---- tokenize / detokenize / metrics -----------------------------------------------------


class _Tokenizer:
    def encode(self, text, add_special_tokens=True):
        ids = [ord(c) for c in text]
        return ([1] + ids) if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if not (skip_special_tokens and i == 1))

    def convert_ids_to_tokens(self, ids):
        return [chr(i) for i in ids]


def test_tokenize_prompt_and_messages_and_detokenize():
    state = FakeState(tokenizer=_Tokenizer())
    body = run(handle_tokenize(TokenizeRequest(prompt="ab"), state))
    assert body == {"count": 3, "max_model_len": 4096, "tokens": [1, 97, 98]}
    body = run(
        handle_tokenize(
            TokenizeRequest(
                prompt="ab", add_special_tokens=False, return_token_strs=True
            ),
            state,
        )
    )
    assert body["tokens"] == [97, 98] and body["token_strs"] == ["a", "b"]
    body = run(
        handle_tokenize(
            TokenizeRequest(messages=[{"role": "user", "content": "x"}]), state
        )
    )
    assert body["count"] == len(
        "RENDERED:" + json.dumps([{"role": "user", "content": "x"}])
    )
    bad = run(
        handle_tokenize(
            TokenizeRequest(prompt="a", messages=[{"role": "user", "content": "x"}]),
            state,
        )
    )
    assert bad.status_code == 400
    body = run(
        handle_detokenize(
            DetokenizeRequest(tokens=[1, 104, 105], skip_special_tokens=True), state
        )
    )
    assert body == {"prompt": "hi"}


def test_metrics_exposition_lists_the_stats_document():
    doc = {
        "instance_id": "abc",
        "model": {"id": "unit-model"},
        "uptime_s": 12,
        "kv": {"used_pages": 3, "total_pages": 10, "page_size": 64},
        "mamba": {"used_slots": 1, "total_slots": 4},
        "swa": None,
        "vram_bytes": 123,
        "throughput": {"decode_tps": 99.5, "prefill_tps": 1000.0},
        "requests": {
            "active": 1,
            "completed": 7,
            "p95_ms": 50,
            "ttft_mean_ms": 20,
            "prompt_tokens_total": 100,
            "completion_tokens_total": 40,
        },
    }
    text = prometheus_text(doc)
    assert 'freetoken_info{instance_id="abc",model="unit-model"} 1' in text
    assert "freetoken_requests_active 1" in text
    assert "freetoken_requests_completed_total 7" in text
    assert "freetoken_kv_pages_total 10" in text
    assert "freetoken_mamba_slots_used 1" in text
    assert "freetoken_decode_tokens_per_second 99.5" in text
    assert "# TYPE freetoken_prompt_tokens_total counter" in text
