"""Real SDK model/stream validation over the real HTTP adapters and GenSpec path.

Only the scheduler/tokenizer transport is scripted. In-process ASGI transport
keeps these deterministic and CPU-only; it does not prove TCP chunk timing,
disconnect behavior, tokenizer templates, or a real model's output quality.
"""
import asyncio
import json
from types import SimpleNamespace

import anthropic
import httpx
import openai
import pytest
from fastapi import FastAPI

from freetoken.message import TokenizeMsg, UserReply
from freetoken.server.anthropic_api import register_anthropic_routes
from freetoken.server.openai_api import register_openai_routes
from freetoken.server.responses_api import register_responses_routes


SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}}
TOOL = {"type": "function", "function": {"name": "get_weather", "parameters": SCHEMA}}
CALL = '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'


class ScriptedBackend:
    def __init__(self, text, *, reasoning=False):
        self.config = SimpleNamespace(
            model_path="/models/test", served_model_name="test", max_seq_len=4096,
            tool_call_parser="qwen25", reasoning_parser="qwen3" if reasoning else None,
        )
        self.text = text
        self.sent = []
        self.maintenance_state = "serving"

    def new_user(self):
        return 42

    async def send_one(self, message):
        assert isinstance(message, TokenizeMsg)
        self.sent.append(message)

    async def wait_for_ack(self, uid):
        # One character per backend reply exercises adapter stream aggregation
        # as well as the SDK SSE decoder. Usage deltas remain authoritative.
        for i, char in enumerate(self.text):
            yield UserReply(uid=uid, incremental_output=char,
                            finished=i == len(self.text) - 1,
                            prompt_tokens_delta=5 if i == 0 else 0,
                            completion_tokens_delta=1)

    async def stream_with_cancellation(self, events, request, uid):
        async for event in events:
            yield event


def client(state, vendor="openai"):
    app = FastAPI()
    for register in (register_openai_routes, register_anthropic_routes, register_responses_routes):
        register(app, lambda: state, dict)
    # Recent Anthropic SDKs use httpx2 and reject httpx client instances.
    # Select the transport matching its public default client class; older SDKs
    # in the supported range still use httpx. Neither path bypasses SDK validation.
    http = httpx
    if vendor == "anthropic" and not issubclass(anthropic.DefaultAsyncHttpxClient, httpx.AsyncClient):
        import httpx2
        http = httpx2
    transport = http.AsyncClient(transport=http.ASGITransport(app=app))
    cls = openai.AsyncOpenAI if vendor == "openai" else anthropic.AsyncAnthropic
    return cls(base_url="http://test/v1" if vendor == "openai" else "http://test",
               api_key="test", http_client=transport, max_retries=0,
               _strict_response_validation=True)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool", [False, True])
def test_openai_chat_sdk_parses_reasoning_text_tools_and_usage(stream, tool):
    async def run():
        state = ScriptedBackend("<think>consider</think>" + (CALL if tool else "Hello"), reasoning=True)
        async with client(state) as sdk:
            response = await sdk.chat.completions.create(
                model="test", messages=[{"role": "user", "content": "Hi"}],
                tools=[TOOL], stream=stream, stream_options={"include_usage": True},
            )
            if stream:
                chunks = [chunk async for chunk in response]
                choices = [choice for chunk in chunks for choice in chunk.choices]
                assert "".join(getattr(c.delta, "reasoning_content", "") or "" for c in choices) == "consider"
                assert [c.finish_reason for c in choices if c.finish_reason] == ["tool_calls" if tool else "stop"]
                if tool:
                    calls = [tc for c in choices for tc in c.delta.tool_calls or []]
                    assert [tc.function.name for tc in calls if tc.function.name] == ["get_weather"]
                    assert json.loads("".join(tc.function.arguments or "" for tc in calls)) == {"city": "Paris"}
                    assert all(tc.index == 0 for tc in calls)
                else:
                    assert "".join(c.delta.content or "" for c in choices) == "Hello"
                usage = chunks[-1].usage
            else:
                assert response.choices[0].message.reasoning_content == "consider"
                assert response.choices[0].finish_reason == ("tool_calls" if tool else "stop")
                if tool:
                    call = response.choices[0].message.tool_calls[0]
                    assert call.id and call.function.name == "get_weather"
                    assert json.loads(call.function.arguments) == {"city": "Paris"}
                else:
                    assert response.choices[0].message.content == "Hello"
                usage = response.usage
            assert usage.prompt_tokens == 5
            assert usage.completion_tokens == len(state.text)
            assert usage.total_tokens == 5 + len(state.text)
            assert len(state.sent) == 1
    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool", [False, True])
def test_anthropic_sdk_assembles_typed_message(stream, tool):
    async def run():
        state = ScriptedBackend("<think>consider</think>" + (CALL if tool else "Hello"), reasoning=True)
        async with client(state, "anthropic") as sdk:
            kwargs = dict(model="test", max_tokens=256,
                          messages=[{"role": "user", "content": "Hi"}],
                          tools=[{"name": "get_weather", "input_schema": SCHEMA}])
            if stream:
                async with sdk.messages.stream(**kwargs) as response:
                    events = [event async for event in response]
                    result = await response.get_final_message()
                assert events[0].type == "message_start"
                assert events[-1].type == "message_stop"
            else:
                result = await sdk.messages.create(**kwargs)
            assert result.role == "assistant" and result.model == "test"
            assert result.stop_reason == ("tool_use" if tool else "end_turn")
            assert result.content[0].type == "thinking"
            assert result.content[0].thinking == "consider"
            if tool:
                # The established full-response shape includes an empty text
                # block before tools; preserving that is part of compatibility.
                assert all(b.text == "" for b in result.content if b.type == "text")
                block = next(b for b in result.content if b.type == "tool_use")
                assert block.id
                assert block.name == "get_weather" and block.input == {"city": "Paris"}
            else:
                block = result.content[1]
                assert block.type == "text" and block.text == "Hello"
            assert result.usage.input_tokens == 5
            assert result.usage.output_tokens == len(state.text)
            assert len(state.sent) == 1
    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool", [False, True])
def test_responses_sdk_parses_output_and_events(stream, tool):
    async def run():
        state = ScriptedBackend(CALL if tool else "Hello")
        async with client(state) as sdk:
            response = await sdk.responses.create(
                model="test", input="Hi", stream=stream,
                tools=[{"type": "function", "name": "get_weather", "parameters": SCHEMA}],
            )
            if stream:
                events = [event async for event in response]
                assert events[0].type == "response.created"
                assert events[-1].type == "response.completed"
                assert [e.sequence_number for e in events] == list(range(len(events)))
                result = events[-1].response
                deltas = [e for e in events if e.type == ("response.function_call_arguments.delta" if tool else "response.output_text.delta")]
                text = "".join(e.delta for e in deltas)
                assert json.loads(text) == {"city": "Paris"} if tool else text == "Hello"
            else:
                result = response
            assert result.status == "completed"
            if tool:
                call = next(item for item in result.output if item.type == "function_call")
                assert call.call_id and call.name == "get_weather"
                assert json.loads(call.arguments) == {"city": "Paris"}
            else:
                assert result.output_text == "Hello"
            assert result.usage.input_tokens == 5
            assert result.usage.output_tokens == len(state.text)
            assert result.usage.total_tokens == 5 + len(state.text)
            assert len(state.sent) == 1
    asyncio.run(run())


@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic"])
def test_sdk_classifies_invalid_request_as_400(protocol):
    async def run():
        state = ScriptedBackend("unused")
        vendor = "anthropic" if protocol == "anthropic" else "openai"
        error = anthropic.BadRequestError if vendor == "anthropic" else openai.BadRequestError
        async with client(state, vendor) as sdk:
            with pytest.raises(error) as caught:
                if protocol == "chat":
                    await sdk.chat.completions.create(model="test", messages=[{"role": "user", "content": "Hi"}], n=2)
                elif protocol == "responses":
                    await sdk.responses.create(model="test", input="Hi", background=True)
                else:
                    await sdk.messages.create(model="test", max_tokens=0, messages=[{"role": "user", "content": "Hi"}])
            assert caught.value.status_code == 400
            assert "invalid_request_error" in str(caught.value)
            assert state.sent == []
    asyncio.run(run())
