import json
from types import SimpleNamespace

import pytest
import torch

import freetoken.grammar.json_schema as json_schema
import freetoken.grammar.matcher as matcher
from freetoken.grammar.matcher import starts_open
from freetoken.message import ErrorReplyMsg, UserMsg
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.scheduler import Scheduler


def test_a_json_schema_request_waits_for_its_token_in_flight():
    """Under overlap a step launches before the last one drains; a constrained request's mask is made from every token
    before it, so it sits out the launch while one of its tokens is in flight, and only it does."""
    constrained, plain = SimpleNamespace(uid=1), SimpleNamespace(uid=2)
    in_flight = (SimpleNamespace(batch=SimpleNamespace(reqs=[constrained, plain])), None)
    scheduler = SimpleNamespace(_grammars={1: object()})
    assert Scheduler._awaiting_inflight(scheduler, in_flight) == [constrained]
    assert Scheduler._awaiting_inflight(scheduler, None) == []
    assert Scheduler._awaiting_inflight(SimpleNamespace(_grammars={}), in_flight) == []


class _Req:
    def __init__(self, uid: int) -> None:
        self.uid = uid


def test_a_held_request_sits_out_the_decode_step():
    held, other = _Req(1), _Req(2)
    manager = SimpleNamespace(running_reqs={held, other})
    assert DecodeManager.schedule_next_batch(manager, hold=[held]).reqs == [other]
    assert DecodeManager.schedule_next_batch(manager, hold=[held, other]) is None


class _Recorder:
    def __init__(self) -> None:
        self.advanced = []

    def advance(self, tokens) -> None:
        self.advanced.append(list(tokens))


class _Decoding:
    """Just what _commit_tokens reads of a request."""

    def __init__(self, uid: int, max_device_len: int) -> None:
        self.uid = uid
        self.input_ids = torch.tensor([1, 2], dtype=torch.int32)
        self.max_device_len = max_device_len
        self.sampling_params = SimpleNamespace(ignore_eos=False, stop_strs=None)
        self.toolcall_anchor_len = None

    def append_host(self, token: torch.Tensor) -> None:
        self.input_ids = torch.cat([self.input_ids, token])


def test_a_json_schema_request_mask_advances_by_exactly_the_tokens_committed():
    """The next mask is made from the reply so far, so the matcher steps over the committed tokens only: not the ones
    past a stop, and not another request's."""
    cut, plain, idle = _Recorder(), _Recorder(), _Recorder()
    scheduler = SimpleNamespace(eos_token_ids={9}, toolcall_anchor_id=None, _match_stop_str=lambda _req: None,
                                _grammars={1: cut, 2: plain, 3: idle})

    msg, n = Scheduler._commit_tokens(scheduler, _Decoding(1, 10), torch.tensor([4, 9, 5, 6]))
    assert (msg.next_tokens, n) == ([4, 9], 2)
    msg, n = Scheduler._commit_tokens(scheduler, _Decoding(2, 10), torch.tensor([7, 8]))
    assert (msg.next_tokens, n) == ([7, 8], 2)

    assert cut.advanced == [[4, 9]]
    assert plain.advanced == [[7, 8]]
    assert idle.advanced == []


def test_the_overlap_launch_leaves_out_a_json_schema_request_with_a_token_in_flight():
    constrained, plain = _Req(1), _Req(2)
    decode = SimpleNamespace(running_reqs={constrained, plain})
    decode.schedule_next_batch = lambda hold: DecodeManager.schedule_next_batch(decode, hold)
    launched = []
    scheduler = SimpleNamespace(
        _grammars={1: object()}, receive_msg=lambda blocking: [], _pending_rebuild=None, _pending_flush=None,
        _check_pending_flush=lambda: None, _process_last_data=lambda data: None, _flush_abort_acks=lambda: None,
        prefill_manager=SimpleNamespace(runnable=False, schedule_next_batch=lambda budget: None),
        decode_manager=decode, prefill_budget=0, stream=SimpleNamespace(wait_stream=lambda stream: None),
        engine=SimpleNamespace(stream=None), _report_prompt_admissions=lambda batch: None,
        _prepare_batch=lambda batch: launched.append(batch.reqs),
    )
    scheduler._awaiting_inflight = lambda data: Scheduler._awaiting_inflight(scheduler, data)
    scheduler._schedule_next_batch = lambda hold=(): Scheduler._schedule_next_batch(scheduler, hold)
    in_flight = (SimpleNamespace(batch=SimpleNamespace(reqs=[constrained, plain])), None)

    assert Scheduler.overlap_loop(scheduler, in_flight) is None
    assert launched == [[plain]]


def test_a_prompt_that_leaves_thinking_open_starts_free():
    open_ids, close_ids = (5,), (6, 7)
    assert starts_open([1, 2, 5], open_ids, close_ids)
    assert starts_open([5, 3, 4], open_ids, close_ids)
    assert not starts_open([5, 6, 7], open_ids, close_ids)
    assert not starts_open([1, 2, 3], open_ids, close_ids)


def test_a_refused_tokenizer_or_schema_is_not_read_again_on_a_resend(monkeypatch):
    """A json_schema request this server refuses (a tokenizer without byte tokens, a schema over the automaton cap) is
    refused from a kept reason on the next one, not by rereading the vocabulary or recompiling on the scheduler's
    thread."""
    reads = []

    def no_byte_tokens(tokenizer):
        reads.append(tokenizer)
        raise ValueError("json_schema needs a token for every byte")

    monkeypatch.setattr(matcher, "token_bytes", no_byte_tokens)
    msg = SimpleNamespace(uid=1, sampling_params=SimpleNamespace(json_schema='{"type":"integer"}'))
    scheduler = SimpleNamespace(_grammar_vocab=None, _grammars={}, tokenizer="tok")
    for _ in range(3):
        assert "every byte" in Scheduler._track_grammar(scheduler, msg)
    assert len(reads) == 1 and scheduler._grammars == {}

    vocab = matcher.Vocab([bytes([b]) for b in range(256)], 256, [])
    compiles = []
    real = json_schema.compile_schema

    def counted(schema):
        compiles.append(schema)
        return real(schema)

    monkeypatch.setattr(matcher, "compile_schema", counted)
    monkeypatch.setattr(json_schema, "MAX_DFA_STATES", 8)
    big = json.dumps({"type": "string", "maxLength": 50})
    for _ in range(3):
        with pytest.raises(json_schema.SchemaError, match="8 automaton states"):
            vocab.grammar(big)
    assert len(compiles) == 1


def test_the_thinking_tags_resolve_to_token_ids_for_a_tag_pair_parser_only():
    from freetoken.server.reasoning_parser import think_tags

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return {"<think>": [7], "</think>": [8, 9]}.get(text, [])

    assert think_tags("qwen3") == ("<think>", "</think>")
    assert think_tags("gpt_oss") is None and think_tags(None) is None
    assert matcher.think_ids(Tokenizer(), think_tags("qwen3")) == ((7,), (8, 9))
    assert matcher.think_ids(Tokenizer(), ("<a>", "</think>")) is None
    assert matcher.think_ids(Tokenizer(), None) is None


class _Planned(Exception):
    """Stops _prepare_batch once the step's sampling arguments are built."""


def _prepare_until_sampling(grammars: dict) -> SimpleNamespace:
    sample_args = SimpleNamespace(grammar=None)
    reqs = [SimpleNamespace(uid=1, can_decode=True, decode_batch_idx=0),
            SimpleNamespace(uid=2, can_decode=True, decode_batch_idx=0)]
    batch = SimpleNamespace(is_decode=True, is_prefill=False, phase="decode", rows_per_req=1, reqs=reqs)

    def stop(_reqs):
        raise _Planned

    scheduler = SimpleNamespace(
        spec=None, _grammars=grammars,
        engine=SimpleNamespace(graph_runner=SimpleNamespace(pad_batch=lambda batch: None),
                               sampler=SimpleNamespace(prepare=lambda batch: sample_args)),
        cache_manager=SimpleNamespace(maybe_free_swa_out_of_window=lambda reqs: None, allocate_paged=stop),
    )
    scheduler._plan_grammar = lambda batch: Scheduler._plan_grammar(scheduler, batch)
    with pytest.raises(_Planned):
        Scheduler._prepare_batch(scheduler, batch)
    return sample_args


def test_a_step_with_a_json_schema_request_samples_with_its_mask():
    constrained = object()
    plan = _prepare_until_sampling({2: constrained}).grammar
    assert plan.matchers == [None, constrained] and plan.width == 1
    assert _prepare_until_sampling({}).grammar is None
    assert _prepare_until_sampling({3: constrained}).grammar is None


def _admit(json_schema, reason):
    sent, admitted, tracked = [], [], []

    def track(msg):
        tracked.append(msg.uid)
        return reason

    scheduler = SimpleNamespace(
        engine=SimpleNamespace(encoder_cache=None, max_seq_len=64), send_result=sent.extend, _track_grammar=track,
        prefill_manager=SimpleNamespace(add_one_req=admitted.append),
    )
    msg = UserMsg(uid=7, input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
                  sampling_params=SimpleNamespace(json_schema=json_schema, max_tokens=8))
    Scheduler._process_one_msg(scheduler, msg)
    return sent, admitted, tracked


def test_a_json_schema_the_server_cannot_serve_is_refused_before_admission():
    sent, admitted, tracked = _admit('{"type":"integer"}', "response_format json_schema: no")
    assert admitted == [] and tracked == [7]
    assert len(sent) == 1 and isinstance(sent[0], ErrorReplyMsg)
    assert (sent[0].uid, sent[0].error) == (7, "response_format json_schema: no")

    sent, admitted, tracked = _admit('{"type":"integer"}', None)
    assert sent == [] and [m.uid for m in admitted] == [7] and tracked == [7]
    sent, admitted, tracked = _admit(None, "unused")
    assert sent == [] and [m.uid for m in admitted] == [7] and tracked == []


def test_a_finished_json_schema_request_drops_its_matcher():
    """Otherwise matchers pile up and every later step builds a mask plan."""
    kept = object()
    scheduler = SimpleNamespace(
        _grammars={1: object(), 2: kept}, spec=None,
        cache_manager=SimpleNamespace(cache_req=lambda req, finished: None),
        table_manager=SimpleNamespace(free=lambda idx: None),
    )
    req = SimpleNamespace(uid=1, table_idx=3)
    Scheduler._free_req_resources(scheduler, req)
    assert scheduler._grammars == {2: kept} and req.table_idx == -1
