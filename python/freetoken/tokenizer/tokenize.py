from __future__ import annotations

import importlib.util
import json
import os
import threading
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, List

import torch
from freetoken.message import TokenizeMsg, UserMsg
from freetoken.utils import init_logger
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from freetoken.mm.processor import MMProcessor

from .effort import (
    EffortProfile,
    ThinkingProfile,
    probe_effort_profile,
    probe_thinking_profile,
    quantize_effort,
)

logger = init_logger(__name__)


def resolve_thinking_mode(chat_template_kwargs: dict[str, Any] | None, tools: Any | None) -> str:
    """Resolve the thinking mode (``"thinking"`` or ``"chat"``) for a chat request.

    The single source of truth for this decision: the encode side
    (``_apply_dsv4_chat_encoder`` below) uses it to pick the prompt the model
    sees, and the frontend parse side (``server/openai_api.py``) imports it to
    decide whether the model's output begins inside a reasoning block. Keeping
    one implementation prevents the two sides from disagreeing. Thinking is on
    when tools are offered (dsv4 only emits well-formed tool calls in thinking
    mode) or when the caller requests it via ``chat_template_kwargs``.
    """
    ctk = chat_template_kwargs or {}
    mode = str(ctk.get("thinking_mode") or "chat")
    if tools or ctk.get("enable_thinking") or ctk.get("thinking"):
        mode = "thinking"
    if mode not in ("chat", "thinking"):
        mode = "chat"
    return mode


_EFFORT_PROBE_MESSAGES = [{"role": "user", "content": "ping"}]


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, mm_processor: MMProcessor | None = None) -> None:
        self.tokenizer = tokenizer
        self.mm_processor = mm_processor  # None: the model takes no images
        self._dsv4_encoder = _load_dsv4_encoder_if_needed(tokenizer)
        self._dsv41 = bool(getattr(self._dsv4_encoder, "IS_DSV41", False))
        self._vision_args = _dsv41_vision_args(_load_dsv41_config(tokenizer)) if self._dsv41 else None
        self._effort_profile: EffortProfile | None = None
        self._thinking_profile: ThinkingProfile | None = None
        self._effort_lock = threading.Lock()
        self._logged_effort_maps: set[tuple[Any, str | None]] = set()

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[UserMsg]:
        results: List[UserMsg] = []
        # TODO: batch tokenization
        for msg in msgs:
            if self._dsv41 and isinstance(msg.text, list):
                from freetoken.models.deepseek_v41.image_processor import prepare_vl_inputs

                prompt, payload = _apply_dsv4_chat_encoder(
                    self._dsv4_encoder, _dsv41_image_sources(msg.text, msg.images), msg.tools,
                    self._sanitize_effort(msg.chat_template_kwargs or {}), return_media=True,
                )
                args = self.mm_processor.args if self.mm_processor is not None else self._vision_args
                ids, _types, images = prepare_vl_inputs(prompt, payload["images"], self.tokenizer, args)
                input_ids = torch.tensor(ids, dtype=torch.int32)
                if self.mm_processor is not None:
                    mm = self.mm_processor.from_media(input_ids, images or [])
                    results.append(UserMsg(
                        uid=msg.uid, input_ids=mm.input_ids, sampling_params=msg.sampling_params,
                        mm_items=mm.mm_items, mrope_positions=mm.mrope_positions, mrope_delta=mm.mrope_delta,
                    ))
                else:
                    msg.media = [vars(item) for item in images] if images else None
                    results.append(UserMsg(
                        uid=msg.uid, input_ids=input_ids, sampling_params=msg.sampling_params, media=msg.media,
                    ))
                continue
            prompt = self.render_prompt(msg)
            # A jinja chat template owns every special token (HF's apply_chat_template
            # tokenizes with add_special_tokens=False for the same reason): tokenizers
            # that auto-add bos (muse-glimmer's, llama's) would otherwise double it --
            # the template already rendered one. Raw-string prompts and the dsv4
            # encoder path keep the default.
            templated = isinstance(msg.text, list) and self._dsv4_encoder is None
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(
                    prompt, return_tensors="pt", add_special_tokens=not templated
                )
            )
            input_ids = input_ids.view(-1).to(torch.int32)
            if msg.images:
                if self.mm_processor is None:
                    raise ValueError("image input is not supported for this model")
                mm = self.mm_processor.apply(input_ids, msg.images)
                results.append(
                    UserMsg(
                        uid=msg.uid,
                        input_ids=mm.input_ids,
                        sampling_params=msg.sampling_params,
                        mm_items=mm.mm_items,
                        mrope_positions=mm.mrope_positions,
                        mrope_delta=mm.mrope_delta,
                    )
                )
            else:
                results.append(
                    UserMsg(uid=msg.uid, input_ids=input_ids, sampling_params=msg.sampling_params)
                )
        return results

    def render_prompt(self, msg: TokenizeMsg) -> str:
        """The template/encoder half of ``tokenize``, exposed so the frontend can
        validate a request before committing an SSE stream. Sanitizes
        ``reasoning_effort`` first: every render path (worker, frontend
        validation, count_tokens) must quantize identically."""
        if not isinstance(msg.text, list):
            return msg.text
        messages = _dsv41_image_sources(msg.text, msg.images) if self._dsv41 else msg.text
        return self._render(
            messages, msg.tools, self._sanitize_effort(msg.chat_template_kwargs or {})
        )

    def _render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any],
    ) -> str:
        """Raw render, no effort sanitation — the probe needs unsupported values
        to actually reach the template so rejection is observable."""
        if not self._dsv41 and self.mm_processor is None and _contains_images(messages):
            raise ValueError("this model does not support image content")
        if self._dsv4_encoder is not None:
            return _apply_dsv4_chat_encoder(
                self._dsv4_encoder, messages, tools, chat_template_kwargs
            )
        # Broadcast the effort in every spelling the ecosystem's templates read
        # (muse-glimmer grades ``reasoning_strength``; Jinja ignores undeclared
        # variables) -- the same rule the thinking toggles use. An explicit
        # caller-provided spelling wins over the broadcast.
        if "reasoning_effort" in chat_template_kwargs:
            chat_template_kwargs = dict(chat_template_kwargs)
            chat_template_kwargs.setdefault(
                "reasoning_strength", chat_template_kwargs["reasoning_effort"]
            )
        if tools is not None:
            chat_template_kwargs = {**chat_template_kwargs, "tools": tools}
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        assert isinstance(prompt, str)
        return prompt

    def effort_profile(self) -> EffortProfile:
        """The checkpoint's effort vocabulary, probed on first use and cached
        for the process lifetime."""
        with self._effort_lock:
            if self._effort_profile is None:
                self._effort_profile = probe_effort_profile(self._probe_render)
                logger.info(
                    "reasoning-effort profile: supported=%s default=%s",
                    sorted(self._effort_profile.supported) or "(none)",
                    self._effort_profile.default,
                )
            return self._effort_profile

    def thinking_profile(self) -> ThinkingProfile:
        """The checkpoint's thinking controls (toggle behavior + effort
        vocabulary), probed on first use and cached for the process lifetime.
        Feeds the /v1/cache/status gear derivation."""
        efforts = self.effort_profile()
        with self._effort_lock:
            if self._thinking_profile is None:
                self._thinking_profile = probe_thinking_profile(self._probe_render, efforts)
            return self._thinking_profile

    def _probe_render(
        self, kwargs: dict[str, Any], tools: list[dict[str, Any]] | None
    ) -> str:
        return self._render(_EFFORT_PROBE_MESSAGES, tools, kwargs)

    def _sanitize_effort(self, chat_template_kwargs: dict[str, Any]) -> dict[str, Any]:
        if "reasoning_effort" not in chat_template_kwargs:
            return chat_template_kwargs
        raw = chat_template_kwargs.get("reasoning_effort")
        if self._dsv41 and type(raw) is int:
            if not 1 <= raw <= 100:
                raise ValueError("DeepSeek-V4.1 reasoning_effort must be between 1 and 100")
            return chat_template_kwargs
        mapped = quantize_effort(raw, self.effort_profile())
        if mapped == raw:
            return chat_template_kwargs
        # raw is client-controlled and may be unhashable (a JSON list/dict).
        key = (raw if isinstance(raw, str) else repr(raw), mapped)
        if key not in self._logged_effort_maps:
            self._logged_effort_maps.add(key)
            logger.info(
                "reasoning_effort %r is not supported by this checkpoint; using %s",
                raw,
                mapped if mapped is not None else "the template default",
            )
        sanitized = dict(chat_template_kwargs)
        if mapped is None:
            del sanitized["reasoning_effort"]
        else:
            sanitized["reasoning_effort"] = mapped
        return sanitized


def _contains_images(value) -> bool:
    if isinstance(value, dict):
        return value.get("type") in ("image", "image_url", "input_image") or any(_contains_images(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_images(v) for v in value)
    return False


def _dsv41_image_sources(messages: list[dict], images: list[bytes] | None) -> list[dict]:
    """Bind image bytes before the native encoder reorders tool-result messages."""
    image_index = 0

    def blocks(content):
        nonlocal image_index
        if not isinstance(content, list):
            return content
        result = []
        for block in content:
            if not isinstance(block, dict):
                result.append(block)
                continue
            part = dict(block)
            if part.get("type") in ("image", "image_url"):
                if images is not None:
                    if image_index >= len(images):
                        raise ValueError("image parts and supplied image bytes do not match")
                    part = {"type": "image", "data": images[image_index]}
                    image_index += 1
                elif isinstance(part.get("freetoken_ref"), dict):
                    ref = part["freetoken_ref"]
                    key = "data" if ref.get("kind") == "b64" else "url"
                    part = {"type": "image", key: ref.get("data")}
            elif part.get("type") == "tool_result":
                part["content"] = blocks(part.get("content"))
            result.append(part)
        return result

    rendered = []
    for message in messages:
        item = dict(message)
        for key in ("content", "content_blocks"):
            if key in item:
                item[key] = blocks(item[key])
        rendered.append(item)
    if images is not None and image_index != len(images):
        raise ValueError("image parts and supplied image bytes do not match")
    return rendered


def _load_dsv41_config(tokenizer) -> dict | None:
    model_path = str(getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "_name_or_path", ""))
    config_path = os.path.join(model_path, "config.json")
    data = None
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as handle:
            data = json.load(handle)
    elif "deepseek" in model_path.lower() and any(v in model_path.lower() for v in ("v4.1", "v41")):
        from freetoken.utils import cached_load_hf_config

        data = cached_load_hf_config(model_path).to_dict()
    if data and (data.get("model_type") == "deepseek_v41" or "DeepseekV41ForCausalLM" in data.get("architectures", [])):
        return data
    return None


def _dsv41_vision_args(data: dict):
    vision = data.get("vision_config") or {}
    return SimpleNamespace(
        image_token_id=data.get("image_token_id", 129264),
        vision_enabled=bool(vision.get("num_hidden_layers", 0)),
        vision_patch_size=vision.get("patch_size", 14),
        vision_downsample_ratio=vision.get("downsample_ratio", 3),
        vision_max_wh_ratio=vision.get("max_wh_ratio"),
        vision_min_pixels=vision.get("min_pixels", 295936),
        vision_max_n_token=vision.get("max_image_tokens", 1024),
    )


def _load_dsv4_encoder_if_needed(tokenizer: PreTrainedTokenizerBase) -> ModuleType | None:
    if _load_dsv41_config(tokenizer) is not None:
        from freetoken.models.deepseek_v41 import encoding

        return encoding
    if getattr(tokenizer, "chat_template", None):
        return None
    model_path = getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "_name_or_path", "")
    if not model_path:
        return None
    encoder_path = os.path.join(str(model_path), "encoding", "encoding_dsv4.py")
    if not os.path.isfile(encoder_path):
        return None
    spec = importlib.util.spec_from_file_location("encoding_dsv4", encoder_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "encode_messages"):
        return None
    return module


def _apply_dsv4_chat_encoder(
    encoder: ModuleType,
    messages: list[dict],
    tools: list[dict] | None,
    chat_template_kwargs: dict,
    *,
    return_media: bool = False,
):
    rendered_messages = [dict(message) for message in messages]
    for message in rendered_messages:
        if message.get("tool_calls"):
            message["tool_calls"] = _dsv4_tool_calls(message["tool_calls"])
    if tools:
        _attach_tools_to_dsv4_messages(rendered_messages, tools)

    # No effort filtering here: the caller sanitized already, and the probe
    # needs raw values to reach the encoder's own validation.
    extra = {"return_multi_modal_data": True} if return_media else {}
    return encoder.encode_messages(
        rendered_messages,
        thinking_mode=resolve_thinking_mode(chat_template_kwargs, tools),
        reasoning_effort=chat_template_kwargs.get("reasoning_effort"),
        **extra,
    )


def _dsv4_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """The dsv4 encoder's contract is ``function.arguments`` = JSON-object STRING
    (it json.loads then iterates .items()); a dict (what ``render_messages``
    produces for Jinja templates) trips its bare-except fallback, which wraps the
    whole payload in a bogus parameter literally named ``arguments``. Re-serialize
    here. Copies each tool-call dict: the outer message copy is shallow, so these
    are shared with the caller."""
    rendered = []
    for tc in tool_calls:
        tc = dict(tc)
        fn = dict(tc.get("function") or {})
        fn["arguments"] = _dsv4_arguments_str(fn.get("arguments"))
        tc["function"] = fn
        rendered.append(tc)
    return rendered


def _dsv4_arguments_str(arguments: Any) -> str:
    """Missing/empty means no arguments (vLLM parity); anything else that is not
    a JSON object is rejected -- ValueError becomes a per-request "could not
    encode request" error, never a worker crash -- matching sglang's
    validate-then-400. A non-object would otherwise raise uncaught in the
    encoder's .items() or be wrapped as garbage."""
    if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
        return "{}"
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    shown = f"{arguments!r:.200}"
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"tool call function.arguments must be valid JSON, got {shown}"
            ) from err
        if isinstance(parsed, dict):
            return arguments
    raise ValueError(f"tool call function.arguments must be a JSON object, got {shown}")


def _attach_tools_to_dsv4_messages(messages: list[dict], tools: list[dict]) -> None:
    for message in messages:
        if message.get("role") == "system":
            message["tools"] = tools
            return
    messages.insert(0, {"role": "system", "content": "", "tools": tools})
