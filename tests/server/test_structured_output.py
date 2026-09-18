"""Contract/transport tests; deliberately no pretend grammar-backed generation."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from freetoken.core import SamplingParams
from freetoken.message import BaseBackendMsg, BaseTokenizerMsg, ErrorReplyMsg, UserMsg
from freetoken.scheduler import structured_output
from freetoken.scheduler.scheduler import Scheduler
from freetoken.server.api_models import ChatCompletionRequest
from freetoken.server.generation import GenerationError, GenSpec, parse_response_format, submit_generation
from freetoken.server.openai_api import chat_request_to_genspec, register_openai_routes
from freetoken.server.responses_api import (
    ResponsesRequest, convert_responses_to_genspec, register_responses_routes,
)


SCHEMA = {
    "type": "object", "properties": {"answer": {"type": "string"}},
    "required": ["answer"], "additionalProperties": False,
}


def response_format(schema=None):
    return {"type": "json_schema", "json_schema": {
        "name": "answer", "schema": SCHEMA if schema is None else schema, "strict": True,
    }}


class State:
    def __init__(self):
        self.config = SimpleNamespace(
            model_path="/test", served_model_name="test",
            tool_call_parser="llama3", reasoning_parser=None,
        )
        self.admitted = 0
        self.sent = []

    def new_user(self):
        self.admitted += 1
        return self.admitted

    async def send_one(self, msg):
        self.sent.append(msg)


@pytest.mark.parametrize("schema", [
    {"type": "invalid"}, {"required": "answer"}, {"properties": []},
    {"$schema": []}, {"$schema": "https://example.invalid/unknown-dialect"},
])
def test_invalid_schemas_fail_meta_validation(schema):
    with pytest.raises(GenerationError):
        parse_response_format(response_format(schema))


@pytest.mark.parametrize("descriptor", [
    None, {}, {"name": "answer"}, {"name": "bad name", "schema": SCHEMA},
    {"name": "answer", "schema": []}, {"name": "answer", "schema": SCHEMA, "strict": "yes"},
])
def test_invalid_wire_wrappers_fail(descriptor):
    with pytest.raises(GenerationError):
        parse_response_format({"type": "json_schema", "json_schema": descriptor})


def test_schema_is_copied_and_both_wire_contracts_use_genspec():
    fmt = response_format()
    chat = chat_request_to_genspec(ChatCompletionRequest(
        model="test", messages=[{"role": "user", "content": "hi"}], response_format=fmt,
    ), {})
    responses = convert_responses_to_genspec(ResponsesRequest(
        model="test", input="hi", text={"format": {"type": "json_schema", **fmt["json_schema"]}},
    ), {})
    assert chat.structured_output_schema == responses.structured_output_schema == SCHEMA
    chat.structured_output_schema["properties"]["extra"] = {"type": "number"}
    assert "extra" not in SCHEMA["properties"]


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_valid_but_unsupported_schema_returns_400_before_admission_or_sse(path, stream):
    state = State()
    app = FastAPI()
    register_openai_routes(app, lambda: state, lambda: {})
    register_responses_routes(app, lambda: state, lambda: {})
    payload = {"model": "test", "stream": stream}
    if path.endswith("completions"):
        payload |= {"messages": [{"role": "user", "content": "hi"}], "response_format": response_format()}
    else:
        payload |= {"input": "hi", "text": {"format": {"type": "json_schema", **response_format()["json_schema"]}}}
    with TestClient(app) as http:
        response = http.post(path, json=payload)
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "unsupported_response_format"
    assert "xgrammar" in response.json()["error"]["message"]
    assert state.admitted == 0 and state.sent == []


def test_scheduler_capability_boundary_is_explicit():
    structured_output.ensure_structured_output_supported(None)
    with pytest.raises(NotImplementedError, match="xgrammar"):
        structured_output.ensure_structured_output_supported({})


def test_direct_scheduler_ipc_fails_before_engine_access():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.engine = None  # Any context/KV access before rejection would fail the test.
    sent = []
    scheduler.send_result = sent.extend
    scheduler._process_one_msg(UserMsg(
        uid=9, input_ids=torch.tensor([1], dtype=torch.int32),
        sampling_params=SamplingParams(structured_output_schema=SCHEMA),
    ))
    assert len(sent) == 1
    assert isinstance(sent[0], ErrorReplyMsg)
    assert sent[0].uid == 9
    assert sent[0].code == "unsupported_response_format"


def test_schema_transport_survives_both_ipc_hops(monkeypatch):
    # Open only the capability gate to exercise TRANSPORT, not constrained output.
    # Production keeps it closed until matcher state and both sampler masks exist.
    monkeypatch.setattr(structured_output, "ensure_structured_output_supported", lambda schema: None)
    schema = {**SCHEMA, "properties": {
        "__type__": {"type": "string"}, "__raw_dict__": {"type": "object"},
    }}
    params = SamplingParams()
    spec = GenSpec(messages=[{"role": "user", "content": "hi"}],
                   sampling_params=params, structured_output_schema=schema)
    state = State()
    uid = asyncio.run(submit_generation(spec, state))
    tokenized = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(state.sent[0]))
    assert tokenized.sampling_params.structured_output_schema == schema
    backend = UserMsg(uid=uid, input_ids=torch.tensor([1], dtype=torch.int32),
                      sampling_params=tokenized.sampling_params)
    decoded = BaseBackendMsg.decoder(backend.encoder())
    assert decoded.sampling_params.structured_output_schema == schema
    assert params.structured_output_schema is None  # caller's sampling object is not mutated


def test_plain_text_submission_keeps_the_original_sampling_object():
    state = State()
    params = SamplingParams()
    spec = GenSpec(messages=[], sampling_params=params)
    asyncio.run(submit_generation(spec, state))
    assert state.sent[0].sampling_params is params
    assert parse_response_format({"type": "text"}) is None
