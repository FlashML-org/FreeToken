"""Encoder/decoder round-trips for the ZMQ control messages (no GPU).

Every message that crosses api -> tokenizer -> scheduler -> tokenizer -> api must survive the
wire with its fields intact; these pin the ones carrying state a later consumer reads back
(rebuild control, prompt admission, per-reply token deltas and KV usage).
"""

from __future__ import annotations

import torch

from freetoken.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    CacheRebuildBackendMsg,
    CacheRebuildMsg,
    CacheRebuildReply,
    CacheRebuildResultMsg,
    CacheStatsBackendMsg,
    CacheStatsMsg,
    CacheStatsReply,
    CacheStatsResultMsg,
    PromptAdmittedMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from freetoken.core import SamplingParams


def test_cache_rebuild_msg_roundtrip():
    msg = CacheRebuildMsg(request_id="abc", moe_cache_size=8, num_pages=1024, mode="if_idle")
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, CacheRebuildMsg)
    assert (out.request_id, out.moe_cache_size, out.num_pages, out.mode) == ("abc", 8, 1024, "if_idle")


def test_cache_rebuild_backend_msg_roundtrip():
    msg = CacheRebuildBackendMsg(request_id="r1", moe_cache_size=None, num_pages=256, mode="drain")
    out = BaseBackendMsg.decoder(msg.encoder())
    assert isinstance(out, CacheRebuildBackendMsg)
    assert (out.request_id, out.moe_cache_size, out.num_pages, out.mode) == ("r1", None, 256, "drain")


def test_cache_rebuild_result_msg_roundtrip():
    msg = CacheRebuildResultMsg(request_id="r2", status="ok", moe_cache_size=16, num_pages=512)
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, CacheRebuildResultMsg)
    assert (out.request_id, out.status, out.moe_cache_size, out.num_pages, out.error) == (
        "r2", "ok", 16, 512, None,
    )


def test_cache_rebuild_reply_roundtrip():
    msg = CacheRebuildReply(request_id="r3", status="failed", error="boom")
    out = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(msg))
    assert isinstance(out, CacheRebuildReply)
    assert (out.request_id, out.status, out.error) == ("r3", "failed", "boom")


def test_cache_stats_messages_roundtrip():
    """The read-only cache-statistics control path preserves nested counter data on every hop."""

    request = CacheStatsMsg(request_id="stats-request")
    backend = CacheStatsBackendMsg(request_id="stats-request")
    result = CacheStatsResultMsg(
        request_id="stats-request",
        stats={"available": True, "summary": {"miss_rate": 0.25}},
    )
    reply = CacheStatsReply(
        request_id="stats-request",
        stats={"available": True, "summary": {"miss_rate": 0.25}},
    )

    assert isinstance(BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(request)), CacheStatsMsg)
    assert isinstance(BaseBackendMsg.decoder(backend.encoder()), CacheStatsBackendMsg)
    decoded_result = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(result))
    decoded_reply = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(reply))
    assert isinstance(decoded_result, CacheStatsResultMsg)
    assert isinstance(decoded_reply, CacheStatsReply)
    assert decoded_result.stats == reply.stats
    assert decoded_reply.stats == reply.stats


def test_prompt_admitted_msg_roundtrip():
    msg = PromptAdmittedMsg(uid=42, prompt_tokens=1234, cached_tokens=500)
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, PromptAdmittedMsg)
    assert (out.uid, out.prompt_tokens, out.cached_tokens) == (42, 1234, 500)


def test_user_reply_token_deltas_round_trip():
    msg = UserReply(
        uid=7,
        incremental_output="hello",
        finished=False,
        prompt_tokens_delta=11,
        completion_tokens_delta=3,
        cached_tokens=4,
        kv_used_pages=40,
        kv_total_pages=512,
        gpu_mem_bytes=64 * (1 << 30),
    )

    decoded = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(msg))

    assert isinstance(decoded, UserReply)
    assert decoded.uid == 7
    assert decoded.incremental_output == "hello"
    assert decoded.finished is False
    assert decoded.prompt_tokens_delta == 11
    assert decoded.completion_tokens_delta == 3
    assert decoded.cached_tokens == 4
    assert decoded.kv_used_pages == 40
    assert decoded.kv_total_pages == 512
    assert decoded.gpu_mem_bytes == 64 * (1 << 30)


def test_detokenize_msg_carries_kv_usage_round_trip():
    msg = DetokenizeMsg(
        uid=3, next_token=42, finished=True,
        kv_used_pages=10, kv_total_pages=256, gpu_mem_bytes=1 << 30,
        mamba_used_slots=7, mamba_total_slots=64,
        swa_used_tokens=8448, swa_total_tokens=76800,
    )
    decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(decoded, DetokenizeMsg)
    assert (decoded.kv_used_pages, decoded.kv_total_pages, decoded.gpu_mem_bytes) == (10, 256, 1 << 30)
    assert (decoded.mamba_used_slots, decoded.mamba_total_slots) == (7, 64)
    assert (decoded.swa_used_tokens, decoded.swa_total_tokens) == (8448, 76800)


def test_client_dicts_with_the_wire_tag_key_survive_intact():
    """Tool JSON Schemas and chat_template_kwargs are free-form client data. A field literally
    named ``__type__`` (a common discriminator) must not be read back as a serialized class --
    that used to kill the tokenizer worker on an unknown/incompatible name."""
    hostile = [
        {"__type__": "AbortMsg"},                                    # a real class name
        {"__type__": "NoSuchClassAnywhere"},                         # an unknown one
        {"type": "object", "properties": {"__type__": {"type": "string"}}},
        {"__raw_dict__": {"a": 1}},                                  # collides with the escape key
        {"deep": {"__type__": "AbortMsg", "l": [{"__type__": "x"}]}},
    ]
    for payload in hostile:
        msg = TokenizeMsg(
            uid=1, text="hi", sampling_params=SamplingParams(),
            chat_template_kwargs=payload,
            tools=[{"type": "function", "function": {"name": "f", "parameters": payload}}],
        )
        out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
        assert isinstance(out, TokenizeMsg)
        assert out.chat_template_kwargs == payload
        assert out.tools[0]["function"]["parameters"] == payload


def test_backend_wire_preserves_multidimensional_cpu_tensors():
    """Gemma 4 patch data and image positions survive the tokenizer scheduler ZMQ hop."""
    msg = UserMsg(
        uid=9,
        input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        sampling_params=SamplingParams(),
        # ``mm_embeds`` is used by the in-process offline path. Online requests
        # instead move these CPU tensors to the scheduler, where its ROCm-owned
        # model instance runs the vision tower and projector.
        mm_pixel_values=torch.arange(24, dtype=torch.float32).reshape(1, 2, 12),
        mm_image_position_ids=torch.tensor([[[0, 0], [0, 1]]], dtype=torch.int64),
    )
    decoded = BaseBackendMsg.decoder(msg.encoder())
    assert isinstance(decoded, UserMsg)
    assert decoded.mm_pixel_values is not None
    assert decoded.mm_image_position_ids is not None
    assert decoded.mm_pixel_values.shape == (1, 2, 12)
    assert decoded.mm_pixel_values.dtype == torch.float32
    assert decoded.mm_image_position_ids.shape == (1, 2, 2)
    assert decoded.mm_image_position_ids.dtype == torch.int64
    assert torch.equal(decoded.mm_pixel_values, msg.mm_pixel_values)
    assert torch.equal(decoded.mm_image_position_ids, msg.mm_image_position_ids)
