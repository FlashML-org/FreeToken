from __future__ import annotations

import multiprocessing as mp
from typing import Any, List

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


def _backend_msg(msg, t) -> UserMsg:
    mm_inputs = getattr(t, "mm_inputs", None)
    cache_ids = mm_inputs.pop("cache_ids", None) if mm_inputs else None
    return UserMsg(
        uid=msg.uid,
        input_ids=t,
        sampling_params=msg.sampling_params,
        mm_inputs=mm_inputs,
        cache_ids=cache_ids,
    )


def _tokenize_requests(
    tokenize_manager: Any,
    messages: List[TokenizeMsg],
    logger: Any,
) -> tuple[List[TokenizeMsg], List[torch.Tensor], List[UserReply]]:
    """Tokenize independently, returning backend work plus terminal frontend errors. The
    returned tensors carry the request's vision-processor outputs as ``.mm_inputs`` (None for
    text-only requests).

    Successful tokenization deliberately emits no prompt-token reply: accounting starts
    only when the scheduler later confirms first-prefill admission.
    """
    ok_msgs: List[TokenizeMsg] = []
    ok_tensors: List[torch.Tensor] = []
    errors: List[UserReply] = []
    for msg in messages:
        try:
            tokens, mm_inputs = tokenize_manager.tokenize_one(msg)
            tokens.mm_inputs = mm_inputs
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
    return ok_msgs, ok_tensors, errors


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

    tokenize_manager = TokenizeManager(tokenizer, model_path=tokenizer_path)
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
                ok_msgs, ok_tensors, errors = _tokenize_requests(
                    tokenize_manager, tokenize_msg, logger
                )
                if errors:
                    send_frontend.put(
                        errors[0] if len(errors) == 1 else BatchFrontendMsg(data=errors)
                    )
                if ok_msgs:
                    backend = [
                        _backend_msg(msg, t) for msg, t in zip(ok_msgs, ok_tensors, strict=True)
                    ]
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
