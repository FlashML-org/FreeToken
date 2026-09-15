"""Encoder/decoder round-trips for the ZMQ control messages (no GPU).

Every message that crosses api -> tokenizer -> scheduler -> tokenizer -> api must survive the
wire with its fields intact; these pin the ones carrying state a later consumer reads back
(rebuild control, prompt admission, per-reply token deltas and KV usage).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    MMItem,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    CacheRebuildBackendMsg,
    CacheRebuildMsg,
    CacheRebuildReply,
    CacheRebuildResultMsg,
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


def test_tensor_wire_nd_and_dtype_roundtrip():
    import numpy as np
    import torch
    from freetoken.message.utils import deserialize_type, serialize_type

    cases = [
        torch.arange(6, dtype=torch.int32),
        torch.arange(12, dtype=torch.int64).reshape(3, 4),
        torch.randn(2, 3, 5, dtype=torch.float32),
        torch.randn(4, 1536, dtype=torch.bfloat16),
        torch.randn(3, 4)[:, :2],  # non-contiguous input
        torch.tensor(7, dtype=torch.int32),  # 0-d
    ]
    for t in cases:
        out = deserialize_type({}, serialize_type(t))
        assert out.dtype == t.dtype
        assert out.shape == t.shape
        assert torch.equal(out, t)

    # legacy payloads carry no shape and decode as 1-D
    legacy = {
        "__type__": "Tensor",
        "buffer": np.arange(4, dtype=np.int32).tobytes(),
        "dtype": "torch.int32",
    }
    out = deserialize_type({}, legacy)
    assert out.shape == (4,) and out.dtype == torch.int32

    # 1-D payloads still omit the shape field (old decoders can read them)
    assert "shape" not in serialize_type(torch.arange(3, dtype=torch.int32))


def test_user_msg_with_mm_items_survives_the_wire():
    item = MMItem(
        modality="image",
        hash=0x1234ABCD,
        pad_value=1_000_000 + 0x1234ABCD,
        offsets=[[7, 11]],
        feature=torch.randn(16, 1536, dtype=torch.bfloat16),
        model_specific_data={"grid_thw": [1, 4, 4]},
    )
    msg = UserMsg(
        uid=9,
        input_ids=torch.arange(16, dtype=torch.int32),
        sampling_params=SamplingParams(),
        mm_items=[item],
        mrope_positions=torch.zeros(3, 16, dtype=torch.int32),
        mrope_delta=-3,
    )
    out = BaseBackendMsg.decoder(msg.encoder())
    assert isinstance(out, UserMsg)
    got = out.mm_items[0]
    assert (got.hash, got.pad_value, got.offsets) == (0x1234ABCD, 1_000_000 + 0x1234ABCD, [[7, 11]])
    assert got.num_tokens == 4 and got.grid_thw == [1, 4, 4]  # model_specific_data reads as attributes
    assert got.precomputed_embeddings is None
    assert torch.equal(got.feature, item.feature)
    assert out.mrope_positions.shape == (3, 16) and out.mrope_delta == -3


def test_mm_item_shape_rules():
    t = torch.zeros(2, 4)
    item = MMItem(modality="image", hash=1, pad_value=1, offsets=[[3, 5], [9, 12]], feature=t)
    item.validate()
    assert item.num_tokens == 5
    assert getattr(item, "grid_thw", None) is None  # missing model-specific key -> AttributeError path
    with pytest.raises(ValueError, match="half-open"):
        MMItem(modality="image", hash=1, pad_value=1, offsets=[[5, 5]], feature=t).validate()
    with pytest.raises(ValueError, match="exactly one"):
        MMItem(modality="image", hash=1, pad_value=1, offsets=[[0, 2]]).validate()
    with pytest.raises(ValueError, match="exactly one"):
        MMItem(
            modality="image", hash=1, pad_value=1, offsets=[[0, 2]], feature=t, precomputed_embeddings=t
        ).validate()
