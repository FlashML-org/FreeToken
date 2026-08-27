from __future__ import annotations

import multiprocessing as mp
import base64
import io
from typing import Any, List
from urllib.parse import unquote_to_bytes, urlparse
from urllib.request import Request, urlopen

import torch
from freetoken.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    CacheRebuildBackendMsg,
    CacheRebuildMsg,
    CacheRebuildReply,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    PromptAdmittedMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from freetoken.utils import (
    ZmqPullQueue,
    ZmqPushQueue,
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
)


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


def _prompt_admitted_reply(msg: PromptAdmittedMsg) -> UserReply:
    """Translate the scheduler's admission signal onto the existing frontend usage path."""
    return UserReply(
        uid=msg.uid,
        incremental_output="",
        finished=False,
        prompt_tokens_delta=msg.prompt_tokens,
        cached_tokens=msg.cached_tokens,
    )


def _error_reply(msg: ErrorReplyMsg) -> UserReply:
    return UserReply(
        uid=msg.uid, incremental_output="", finished=True, error=msg.error, error_code=msg.code,
    )


def _put_user_replies(send_frontend: Any, replies: List[UserReply]) -> None:
    if replies:
        send_frontend.put(
            replies[0] if len(replies) == 1 else BatchFrontendMsg(data=replies)
        )


def _send_generation_replies(
    send_frontend: Any,
    admitted: List[UserReply],
    sampled: List[UserReply],
    terminal_errors: List[UserReply],
) -> None:
    """Preserve the accounting barrier within one tokenizer queue drain.

    Scheduler abort acknowledgements are terminal: every already-sampled DetokenizeMsg
    drained alongside them must reach FrontendManager first. Admission messages remain
    first so per-request usage precedes that request's sampled completion.
    """
    _put_user_replies(send_frontend, admitted)
    _put_user_replies(send_frontend, sampled)
    _put_user_replies(send_frontend, terminal_errors)


def _tokenize_requests(
    tokenize_manager: Any,
    multimodal_processor: Any,
    messages: List[TokenizeMsg],
    logger: Any,
) -> tuple[List[TokenizeMsg], List[torch.Tensor], List[dict[str, torch.Tensor] | None], List[UserReply]]:
    """Tokenize independently, returning backend work plus terminal frontend errors.

    Successful tokenization deliberately emits no prompt-token reply: accounting starts
    only when the scheduler later confirms first-prefill admission.
    """
    ok_msgs: List[TokenizeMsg] = []
    ok_tensors: List[torch.Tensor] = []
    ok_multimodal: List[dict[str, torch.Tensor] | None] = []
    errors: List[UserReply] = []
    for msg in messages:
        try:
            tokens, mm = multimodal_processor.encode(msg, tokenize_manager)
        except Exception as exc:  # noqa: BLE001 — isolate, never crash the worker
            logger.warning(f"tokenization failed for request {msg.uid}: {exc!r}")
            errors.append(
                UserReply(
                    uid=msg.uid,
                    incremental_output="",
                    finished=True,
                    error=f"could not encode request: {exc}",
                )
            )
            continue
        # A zero-token prompt would trip the scheduler's input_len > 0 invariant and
        # crash the worker; reject it here as a terminal error instead.
        if tokens.numel() == 0:
            errors.append(
                UserReply(
                    uid=msg.uid,
                    incremental_output="",
                    finished=True,
                    error="prompt must contain at least one token",
                )
            )
            continue
        ok_msgs.append(msg)
        ok_tensors.append(tokens)
        ok_multimodal.append(mm)
    return ok_msgs, ok_tensors, ok_multimodal, errors


_MAX_IMAGE_BYTES = 64 * 1024 * 1024


def _image_source(part: dict[str, Any]) -> str:
    value = part.get("image_url", part.get("image"))
    if isinstance(value, dict):
        value = value.get("url")
    if not isinstance(value, str) or not value:
        raise ValueError("image content part needs a non-empty image URL")
    return value


def _download_image(source: str) -> bytes:
    if source.startswith("data:"):
        header, separator, payload = source.partition(",")
        if not separator:
            raise ValueError("invalid image data URL")
        try:
            data = base64.b64decode(payload, validate=True) if ";base64" in header else unquote_to_bytes(payload)
        except Exception as exc:
            raise ValueError("invalid image data URL") from exc
    elif urlparse(source).scheme in {"http", "https"}:
        request = Request(source, headers={"User-Agent": "FreeToken/vision"})
        with urlopen(request, timeout=30) as response:  # noqa: S310 - explicit http(s) check above
            data = response.read(_MAX_IMAGE_BYTES + 1)
    else:
        raise ValueError("image URL must use data, http, or https")
    if len(data) > _MAX_IMAGE_BYTES:
        raise ValueError("image exceeds the 64 MiB input limit")
    return data


def _message_image_sources(text: str | List[dict[str, Any]]) -> list[str]:
    if not isinstance(text, list):
        return []
    sources: list[str] = []
    for message in text:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"image", "image_url"} or "image" in part or "image_url" in part:
                sources.append(_image_source(part))
    return sources


class _MultimodalProcessor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.processor = None

    def encode(
        self, msg: TokenizeMsg, tokenize_manager: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        sources = _message_image_sources(msg.text)
        if not sources:
            return tokenize_manager.tokenize([msg])[0], None

        from PIL import Image
        from transformers import AutoProcessor

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        prompt = tokenize_manager.render_prompt(msg)
        images = []
        try:
            for source in sources:
                with Image.open(io.BytesIO(_download_image(source))) as image:
                    images.append(image.convert("RGB"))
            encoded = self.processor(text=[prompt], images=images, return_tensors="pt")
        finally:
            for image in images:
                image.close()

        required = {"input_ids", "pixel_values", "image_grid_thw", "mm_token_type_ids"}
        missing = sorted(required.difference(encoded))
        if missing:
            raise ValueError(f"model processor did not return: {', '.join(missing)}")
        return encoded["input_ids"][0].to(dtype=torch.int32), {
            "pixel_values": encoded["pixel_values"].to(dtype=torch.bfloat16),
            "image_grid_thw": encoded["image_grid_thw"].to(dtype=torch.int64),
            "mm_token_type_ids": encoded["mm_token_type_ids"][0].to(dtype=torch.int32),
        }


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)
    multimodal_processor = _MultimodalProcessor(tokenizer_path)
    detokenize_manager = DetokenizeManager(
        tokenizer, load_eos_token_ids(tokenizer_path, tokenizer)
    )

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]
            prompt_admitted_msg = [m for m in pending_msg if isinstance(m, PromptAdmittedMsg)]
            error_reply_msg = [m for m in pending_msg if isinstance(m, ErrorReplyMsg)]
            # Cache-rebuild control messages are pure passthrough (no tokenization):
            # CacheRebuildMsg (api -> scheduler) and CacheRebuildResultMsg (scheduler -> api).
            for m in pending_msg:
                if isinstance(m, CacheRebuildMsg):
                    send_backend.put(
                        CacheRebuildBackendMsg(
                            request_id=m.request_id,
                            moe_cache_size=m.moe_cache_size,
                            num_pages=m.num_pages,
                            num_mamba_slots=m.num_mamba_slots,
                            num_swa_pages=m.num_swa_pages,
                            mode=m.mode,
                        )
                    )
                elif isinstance(m, CacheRebuildResultMsg):
                    send_frontend.put(
                        CacheRebuildReply(
                            request_id=m.request_id,
                            status=m.status,
                            moe_cache_size=m.moe_cache_size,
                            num_pages=m.num_pages,
                            mamba_slots=m.mamba_slots,
                            num_swa_pages=m.num_swa_pages,
                            error=m.error,
                        )
                    )
            n_control = sum(
                isinstance(
                    m,
                    (CacheRebuildMsg, CacheRebuildResultMsg, ErrorReplyMsg, PromptAdmittedMsg),
                )
                for m in pending_msg
            )
            assert (
                len(detokenize_msg) + len(tokenize_msg) + len(abort_msg) + n_control
                == len(pending_msg)
            )
            sampled_replies: List[UserReply] = []
            if len(detokenize_msg) > 0:
                replies = detokenize_manager.detokenize(detokenize_msg)
                sampled_replies = [
                    UserReply(
                        uid=msg.uid,
                        incremental_output=reply,
                        finished=msg.finished,
                        finish_reason=msg.finish_reason,
                        matched_stop=msg.matched_stop,
                        completion_tokens_delta=1,
                        kv_used_pages=msg.kv_used_pages,
                        kv_total_pages=msg.kv_total_pages,
                        mamba_used_slots=msg.mamba_used_slots,
                        mamba_total_slots=msg.mamba_total_slots,
                        swa_used_tokens=msg.swa_used_tokens,
                        swa_total_tokens=msg.swa_total_tokens,
                        gpu_mem_bytes=msg.gpu_mem_bytes,
                    )
                    for msg, reply in zip(detokenize_msg, replies, strict=True)
                ]

            # An error reply and a client abort are both terminal for their uid, and neither
            # produces the finished DetokenizeMsg that would release the decode state.
            for msg in error_reply_msg:
                detokenize_manager.discard(msg.uid)
            for msg in abort_msg:
                detokenize_manager.discard(msg.uid)

            _send_generation_replies(
                send_frontend,
                [_prompt_admitted_reply(msg) for msg in prompt_admitted_msg],
                sampled_replies,
                [_error_reply(msg) for msg in error_reply_msg],
            )

            if len(tokenize_msg) > 0:
                # Tokenize per-message so a single un-renderable request (e.g. a chat template
                # that rejects the message layout) becomes a terminal error reply for THAT uid
                # instead of an uncaught exception that kills the worker and bricks the server.
                ok_msgs, ok_tensors, ok_multimodal, errors = _tokenize_requests(
                    tokenize_manager, multimodal_processor, tokenize_msg, logger
                )
                if errors:
                    send_frontend.put(
                        errors[0] if len(errors) == 1 else BatchFrontendMsg(data=errors)
                    )
                if ok_msgs:
                    backend = []
                    for msg, tokens, mm in zip(ok_msgs, ok_tensors, ok_multimodal, strict=True):
                        backend.append(
                            UserMsg(
                                uid=msg.uid,
                                input_ids=tokens,
                                sampling_params=msg.sampling_params,
                                mm_pixel_values=(mm or {}).get("pixel_values"),
                                mm_image_grid_thw=(mm or {}).get("image_grid_thw"),
                                mm_token_type_ids=(mm or {}).get("mm_token_type_ids"),
                            )
                        )
                    send_backend.put(backend[0] if len(backend) == 1 else BatchBackendMsg(data=backend))
            if len(abort_msg) > 0:
                batch_output = BatchBackendMsg(
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        pass
