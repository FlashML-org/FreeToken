# Adapted from LightLLM [https://github.com/ModelTC/lightllm/blob/main/lightllm/server/function_call_parser.py]
# Copyright 2025 ModelTC Team
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import json
import os
import logging
import re
from abc import ABC, abstractmethod
from json import JSONDecodeError, JSONDecoder
from json.decoder import WHITESPACE
from typing import Any, Dict, List, Optional, Tuple, Type

import partial_json_parser
from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow
from pydantic import BaseModel

from .api_models import Function, Tool
from .reasoning_parser import (
    ATEM_CLOSING_TOKENS,
    ATEM_HEADER_SPAN,
    ATEM_INLINE_HEADER_RE,
    ATEM_MESSAGE,
    ATEM_RECIPIENT_RE,
    ATEM_START,
    atem_hold_len,
    atem_marker_inside,
)

try:
    import orjson
except ModuleNotFoundError:  # FreeToken does not require orjson.
    class _OrjsonCompat:
        @staticmethod
        def loads(value: str) -> Any:
            return json.loads(value)

    orjson = _OrjsonCompat()

logger = logging.getLogger(__name__)
_TRUE_ENV_VALUES = {"ON", "TRUE", "1", "YES"}
_LIGHTLLM_ENABLE_TOOL_NAME_CHECK = os.getenv("LIGHTLLM_ENABLE_TOOL_NAME_CHECK")
if _LIGHTLLM_ENABLE_TOOL_NAME_CHECK is not None:
    FORWARD_UNKNOWN_TOOLS = _LIGHTLLM_ENABLE_TOOL_NAME_CHECK.upper() not in _TRUE_ENV_VALUES
else:
    FORWARD_UNKNOWN_TOOLS = (
        os.getenv("FREETOKEN_FORWARD_UNKNOWN_TOOLS", os.getenv("SGLANG_FORWARD_UNKNOWN_TOOLS", "True")).upper()
        in _TRUE_ENV_VALUES
    )


def _should_forward_unknown_tool(name: Any) -> bool:
    return FORWARD_UNKNOWN_TOOLS or (isinstance(name, str) and ":" in name)


TOOLS_TAG_LIST = [
    "<|plugin|>",
    "<|tool_call>",
    "<|tool_call_begin|>",
    "<|channel|>",
    "<function=",
    "<tool_call>",
    "<minimax:tool_call>",
    "]<]minimax[>[<tool_call>",
    "<|python_tag|>",
    "[TOOL_CALLS]",
    "<｜DSML｜function_calls>",
    "<｜DSML｜tool_calls>",
    "<｜DSML｜invoke",
    "<atem:function_calls>",
]


class ToolCallItem(BaseModel):
    """Simple encapsulation of the parsed ToolCall result for easier usage in streaming contexts."""

    tool_index: int
    name: Optional[str] = None
    parameters: str  # JSON string


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(self, normal_text: str = "", calls: Optional[List[ToolCallItem]] = None):
        self.normal_text = normal_text
        self.calls = calls or []


def _first_existing_pos(text: str, tokens: List[str]) -> int:
    positions = [text.find(token) for token in tokens if token in text]
    return min(positions) if positions else -1


def _find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def _partial_json_loads(input_str: str, flags: Allow) -> Tuple[Any, int]:
    """
    Parse incomplete or partial JSON strings commonly encountered during streaming.

    Args:
        input_str (str): The potentially incomplete JSON string to parse.
        flags (Allow): Bitwise flags controlling what types of partial data are allowed.
            Common flags include:
            - Allow.STR: Allow partial strings (e.g., '"hello wo' -> 'hello wo')
            - Allow.OBJ: Allow partial objects (e.g., '{"key":' -> {'key': None})
            - Allow.ARR: Allow partial arrays (e.g., '[1, 2,' -> [1, 2])
            - Allow.ALL: Allow all types of partial data

    Returns:
        Tuple[Any, int]: A tuple containing:
            - parsed_object: The Python object parsed from the JSON
            - consumed_length: Number of characters consumed from input_str
    """
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except (JSONDecodeError, IndexError) as e:
        msg = getattr(e, "msg", str(e))
        if "Extra data" in msg or "pop from empty list" in msg:
            start = WHITESPACE.match(input_str, 0).end()
            obj, end = JSONDecoder().raw_decode(input_str, start)
            return obj, end
        raise


def _is_complete_json(input_str: str) -> bool:
    try:
        orjson.loads(input_str)
        return True
    except JSONDecodeError:
        return False


def _parse_loose_json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if value.lower() == "null":
            return None
        return value


def _parse_first_json_value(text: str) -> Any:
    decoder = JSONDecoder()
    for idx, char in enumerate(text.strip()):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text.strip(), idx)
            return value
        except JSONDecodeError:
            return None
    return None


def _split_top_level(text: str, delimiter: str) -> List[str]:
    parts: List[str] = []
    start = 0
    depth = 0
    in_gemma_string = False
    i = 0
    while i < len(text):
        if text.startswith('<|"|>', i):
            in_gemma_string = not in_gemma_string
            i += len('<|"|>')
            continue
        if not in_gemma_string:
            if text[i] in "[{":
                depth += 1
            elif text[i] in "]}":
                depth -= 1
            elif text[i] == delimiter and depth == 0:
                parts.append(text[start:i])
                start = i + 1
        i += 1
    parts.append(text[start:])
    return parts


def _parse_gemma_value(value: str) -> Any:
    value = value.strip()
    if value.startswith('<|"|>') and value.endswith('<|"|>'):
        return value[len('<|"|>') : -len('<|"|>')]
    if value == "true":
        return True
    if value == "false":
        return False
    if value in ("null", "none", "None"):
        return None
    if value.startswith("{") and value.endswith("}"):
        return _parse_gemma_call_args(value[1:-1])
    if value.startswith("[") and value.endswith("]"):
        return [_parse_gemma_value(item) for item in _split_top_level(value[1:-1], ",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_gemma_call_args(text: str) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for item in _split_top_level(text, ","):
        if not item.strip():
            continue
        key, sep, value = item.partition(":")
        if not sep:
            continue
        args[key.strip()] = _parse_gemma_value(value.strip())
    return args


class BaseFormatDetector(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    # Detectors whose parse_streaming_increment is not incremental-safe (e.g. re-emit
    # already-released text) set this False; the serving layer then falls back to
    # buffering the whole generation and calling parse_non_stream at the end.
    supports_streaming = True

    # Whether emitted argument fragments always concatenate to a prefix of the
    # call's FINAL arguments JSON. Adapters whose clients concatenate fragments
    # (Anthropic input_json_delta, OpenAI tool_calls deltas) only stream fragments
    # when this holds; otherwise they send the full arguments once at close.
    args_fragments_prefix_stable = True

    # The marker that UNIQUELY opens a tool-call block in this wire format, or None when
    # no such marker exists (gpt-oss: <|channel|> opens every channel header, tool call or
    # not; DSV32/DSV4: the DSML opener is a multi-piece composite). Consumed by the
    # scheduler's special-token checkpoint, which only uses it when the tokenizer encodes
    # it as a single token. Often equals bot_token, but declared separately because
    # bot_token is a parse trigger, not a uniqueness claim.
    toolcall_opener: str | None = None

    def __init__(self):
        # Streaming state management
        # Buffer for accumulating incomplete patterns that arrive across multiple streaming chunks
        self._buffer = ""
        # Stores complete tool call info (name and arguments) for each tool being parsed.
        # Used by serving layer for completion handling when streaming ends.
        # Format: [{"name": str, "arguments": dict}, ...]
        self.prev_tool_call_arr: List[Dict] = []
        # Index of currently streaming tool call. Starts at -1 (no active tool),
        # increments as each tool completes. Tracks which tool's arguments are streaming.
        self.current_tool_id: int = -1
        # Flag for whether current tool's name has been sent to client.
        # Tool names sent first with empty parameters, then arguments stream incrementally.
        self.current_tool_name_sent: bool = False
        # Tracks raw JSON string content streamed to client for each tool's arguments.
        # Critical for serving layer to calculate remaining content when streaming ends.
        # Each index corresponds to a tool_id. Example: ['{"location": "San Francisco"', '{"temp": 72']
        self.streamed_args_for_tool: List[str] = []

        # Token configuration (override in subclasses)
        self.bot_token = ""
        self.eot_token = ""
        self.tool_call_separator = ", "

    def _get_tool_indices(self, tools: List[Tool]) -> Dict[str, int]:
        """
        Get a mapping of tool names to their indices in the tools list.

        This utility method creates a dictionary mapping function names to their
        indices in the tools list, which is commonly needed for tool validation
        and ToolCallItem creation.

        Args:
            tools: List of available tools

        Returns:
            Dictionary mapping tool names to their indices
        """
        return {tool.function.name: i for i, tool in enumerate(tools) if tool.function.name}

    def parse_base_json(self, action: Any, tools: List[Tool]) -> List[ToolCallItem]:
        tool_indices = self._get_tool_indices(tools)
        if not isinstance(action, list):
            action = [action]

        results = []
        for act in action:
            name = act.get("name")
            if not (name and name in tool_indices):
                logger.warning(f"Model attempted to call undefined function: {name}")
                if not _should_forward_unknown_tool(name):
                    continue

            results.append(
                ToolCallItem(
                    tool_index=tool_indices.get(name, -1),
                    name=name,
                    parameters=json.dumps(
                        act.get("parameters") or act.get("arguments", {}),
                        ensure_ascii=False,
                    ),
                )
            )

        return results

    @abstractmethod
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parses the text in one go. Returns success=True if the format matches, otherwise False.
        Note that leftover_text here represents "content that this parser will not consume further".
        """
        action = orjson.loads(text)
        return StreamingParseResult(calls=self.parse_base_json(action, tools))

    def _ends_with_partial_token(self, buffer: str, bot_token: str) -> int:
        """
        Check if buffer ends with a partial bot_token.
        Return the length of the partial bot_token.

        For some format, the bot_token is not a token in model's vocabulary, such as
        `[TOOL_CALLS] [` in Mistral.
        """
        for i in range(1, min(len(buffer) + 1, len(bot_token))):
            if bot_token.startswith(buffer[-i:]):
                return i
        return 0

    def _get_param_config(self, func_name: str, tools: List[Tool]) -> Dict:
        """Extract the parameter properties (JSON schema) for one tool."""
        for tool in tools:
            if tool.function.name == func_name and tool.function.parameters:
                params = tool.function.parameters
                if isinstance(params, dict) and "properties" in params:
                    return params["properties"]
                elif isinstance(params, dict):
                    return params
        return {}

    def _convert_param_value(self, value: str, param_name: str, param_config: Dict, func_name: str) -> Any:
        """Convert parameter value based on schema type. Safe alternative to eval()."""
        if value.lower() == "null":
            return None

        if param_name not in param_config:
            return value

        prop = param_config.get(param_name, {})
        param_type = str(prop.get("type", "string")).strip().lower() if isinstance(prop, dict) else "string"

        if param_type in ("string", "str", "enum"):
            return value
        elif param_type.startswith("int") or param_type == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        elif param_type in ("number", "float", "double"):
            try:
                fv = float(value)
                return int(fv) if fv == int(fv) else fv
            except (ValueError, TypeError):
                return value
        elif param_type in ("boolean", "bool"):
            return value.lower() == "true"
        elif param_type in ("object", "array"):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError, ValueError):
                try:
                    return ast.literal_eval(value)
                except (ValueError, SyntaxError, TypeError):
                    return value
        return value

    def _schema_param_type(self, param_name: str, param_config: Dict, missing: str = "string") -> str:
        """Normalized schema type for a parameter; ``missing`` when undeclared."""
        if param_name not in param_config:
            return missing
        prop = param_config.get(param_name, {})
        if not isinstance(prop, dict):
            return "string"
        return str(prop.get("type", "string")).strip().lower()

    @staticmethod
    def _json_escape_chunk(text: str) -> str:
        """JSON-string-escape a fragment of a value (escaping is per-character, so
        chunk boundaries are safe)."""
        return json.dumps(text, ensure_ascii=False)[1:-1]

    def block_close_tokens(self) -> tuple:
        """Tokens that end a tool block — used to locate text AFTER the last call
        in one-shot parsing (detect_and_parse only keeps text before the first)."""
        return (self.eot_token,) if self.eot_token else ()

    def finish_streaming(self) -> str:
        """End-of-stream drain: return residual buffered text that should be surfaced
        as normal content. Suppressed when the buffer holds an unfinished tool call
        (raw tool markup must not leak into content), or when it is only markup
        debris (closing tag / bare separator) left behind by completed calls."""
        residual, self._buffer = self._buffer, ""
        if not residual:
            return ""
        if self.has_tool_call(residual) or self.current_tool_name_sent:
            return ""
        if self.eot_token and self.eot_token in residual:
            residual = residual.replace(self.eot_token, "")
        if self.prev_tool_call_arr and residual.strip() in ("", self.tool_call_separator.strip()):
            return ""
        return residual


    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.

        This base implementation works best with formats where:
        1. bot_token is followed immediately by JSON (e.g., bot_token + JSON_array)
        2. JSON can be parsed incrementally using partial_json_loads
        3. Multiple tool calls are separated by "; " or ", "

        Examples of incompatible formats (need custom implementation, may reuse some logic from this class):
        - Each tool call is wrapped in a separate block: See Qwen25Detector
        - Multiple separate blocks: [TOOL_CALLS] [...] \n [TOOL_CALLS] [...]
        - Tool call is Pythonic style

        For incompatible formats, detectors should override this method with custom logic.
        """
        # Append new text to buffer
        self._buffer += new_text
        current_text = self._buffer

        # The current_text has tool_call if it is the start of a new tool call sequence
        # or it is the start of a new tool call after a tool call separator, when there is a previous tool call
        if not (
            self.has_tool_call(current_text)
            or (self.current_tool_id > 0 and current_text.startswith(self.tool_call_separator))
        ):
            # Only clear buffer if we're sure no tool call is starting
            if not self._ends_with_partial_token(self._buffer, self.bot_token):
                normal_text = self._buffer
                self._buffer = ""
                if self.eot_token in normal_text:
                    normal_text = normal_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=normal_text)
            else:
                # Might be partial bot_token, keep buffering
                return StreamingParseResult()

        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            try:
                tool_call_pos = current_text.find(self.bot_token)
                if tool_call_pos > 0:
                    # Normal text precedes the tool tag in this buffer (text and tag
                    # landed in one chunk): release it before parsing the call so it
                    # isn't silently dropped when the buffer is later trimmed.
                    prefix = current_text[:tool_call_pos]
                    self._buffer = current_text[tool_call_pos:]
                    if self.eot_token and self.eot_token in prefix:
                        prefix = prefix.replace(self.eot_token, "")
                    return StreamingParseResult(normal_text=prefix)
                if tool_call_pos != -1:
                    start_idx = tool_call_pos + len(self.bot_token)
                elif self.current_tool_id > 0 and current_text.startswith(self.tool_call_separator):
                    start_idx = len(self.tool_call_separator)
                else:
                    start_idx = 0

                if start_idx >= len(current_text):
                    return StreamingParseResult()

                try:
                    obj, end_idx = _partial_json_loads(current_text[start_idx:], flags)
                except MalformedJSON:
                    if tool_call_pos == -1 and not self._ends_with_partial_token(
                        self._buffer, self.bot_token
                    ):
                        after = current_text[start_idx:].lstrip()
                        if after and after[0] not in "{[":
                            # Reached via the separator heuristic after a completed
                            # call, but what follows can never become another call's
                            # JSON: trailing normal text — release it instead of
                            # holding forever. A bare separator / whitespace tail
                            # stays held (the next call's JSON may still arrive).
                            normal_text = self._buffer
                            self._buffer = ""
                            if self.eot_token and self.eot_token in normal_text:
                                normal_text = normal_text.replace(self.eot_token, "")
                            return StreamingParseResult(normal_text=normal_text)
                    return StreamingParseResult()

                is_current_complete = _is_complete_json(current_text[start_idx : start_idx + end_idx])

                # Validate tool name if present
                if (
                    "name" in obj
                    and obj["name"] not in self._tool_indices
                    and not _should_forward_unknown_tool(obj["name"])
                ):
                    # Invalid tool name - reset state
                    self._buffer = ""
                    self.current_tool_id = -1
                    self.current_tool_name_sent = False
                    if self.streamed_args_for_tool:
                        self.streamed_args_for_tool.pop()
                    return StreamingParseResult()

                # Handle parameters/arguments consistency
                # NOTE: we assume here that the obj is always partial of a single tool call
                if "parameters" in obj:
                    assert "arguments" not in obj, "model generated both parameters and arguments"
                    obj["arguments"] = obj["parameters"]

                current_tool_call = obj

            except MalformedJSON:
                return StreamingParseResult()

            if not current_tool_call:
                return StreamingParseResult()

            # Case 1: Handle tool name streaming
            # This happens when we encounter a tool but haven't sent its name yet
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")

                if function_name and (
                    function_name in self._tool_indices or _should_forward_unknown_tool(function_name)
                ):
                    # If this is a new tool (current_tool_id was -1), initialize it
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.streamed_args_for_tool.append("")
                    # If this is a subsequent tool, ensure streamed_args_for_tool is large enough
                    elif self.current_tool_id >= len(self.streamed_args_for_tool):
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                    # Send the tool name with empty parameters
                    res = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        ],
                    )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            # Case 2: Handle streaming arguments
            # This happens when we've already sent the tool name and now need to stream arguments incrementally
            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()

                # NOTE: `is not None`, not truthiness — an empty-arguments call
                # ({}) must still take the completion path below so its buffer is
                # consumed and the call closes exactly once (vLLM's hermes parser
                # gates on JSON completeness for the same reason).
                if cur_arguments is not None:
                    # Calculate how much of the arguments we've already streamed
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_arguments = None
                    if self.current_tool_id < len(self.prev_tool_call_arr):
                        prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get("arguments")

                    argument_diff = None

                    # If the current tool's JSON is complete, send all remaining arguments
                    if is_current_complete:
                        argument_diff = cur_args_json[sent:]
                        completing_tool_id = self.current_tool_id  # Save the ID of the tool that's completing

                        # Only remove the processed portion, keep unprocessed content;
                        # also consume the block's closing tag so it can't jam the
                        # tool_call_separator heuristic on the next increment.
                        self._buffer = current_text[start_idx + end_idx :]
                        if self.eot_token:
                            after = self._buffer.lstrip()
                            if after.startswith(self.eot_token):
                                self._buffer = after[len(self.eot_token):]

                        if self.current_tool_id < len(self.prev_tool_call_arr):
                            self.prev_tool_call_arr[self.current_tool_id].clear()
                        self.current_tool_name_sent = False
                        self.streamed_args_for_tool[self.current_tool_id] = ""
                        self.current_tool_id += 1

                    # If the tool is still being parsed, send incremental changes
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                        if cur_args_json != prev_args_json:
                            prefix = _find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    # Send the argument diff if there's something new
                    if argument_diff is not None:
                        # Use the correct tool_index: completing_tool_id for completed tools,
                        # current_tool_id for ongoing
                        tool_index_to_use = completing_tool_id if is_current_complete else self.current_tool_id
                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=tool_index_to_use,
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        if not is_current_complete:
                            self.streamed_args_for_tool[self.current_tool_id] += argument_diff

            # Update prev_tool_call_arr with current state
            if self.current_tool_id >= 0:
                # Ensure prev_tool_call_arr is large enough
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call

            return res

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult()



class InvokeParamStreamMixin:
    """Value-level streaming for invoke/parameter block formats (qwen3_coder,
    minimax — the same shape vLLM's ParserEngine covers): text outside blocks
    streams live; parameter VALUES whose schema type is string stream char-by-char
    as JSON-escaped, prefix-stable fragments; typed values buffer until the
    parameter closes so their JSON form is schema-correct.

    Subclasses define the grammar via class attributes:
      _ps_outer_open/_ps_outer_close  wrapper block tokens (may equal invoke tokens)
      _ps_invoke_open_prefix/_ps_invoke_open_re (group 1 = name)/_ps_invoke_close
      _ps_param_open_prefix/_ps_param_open_re (group 1 = key)/_ps_param_close
      _ps_trim       chars trimmed around values
      _ps_trim_single  True: at most ONE leading/trailing trim char (qwen3_coder)
      _ps_missing_type schema type assumed for undeclared params ("string" streams,
                       "loose" buffers and loose-parses — per-family legacy typing)
    """

    _ps_outer_open: str = ""
    _ps_outer_close: str = ""
    _ps_trim: str = "\n"
    _ps_trim_single: bool = False
    _ps_missing_type: str = "string"

    def _ps_reset(self) -> None:
        self._ps_mode = "idle"
        self._ps_key = ""
        self._ps_lead = "{"
        self._ps_emitted_any = False
        self._ps_lead_trimmed = False
        self._ps_param_config: Dict = {}

    def _ps_convert_value(self, key: str, raw: str) -> Any:
        if key in self._ps_param_config or self._ps_missing_type != "loose":
            return self._convert_param_value(raw, key, self._ps_param_config, "")
        return _parse_loose_json_value(raw)

    def _ps_canonical_name(self, name: str) -> str:
        """Hook: normalize an invoke's function name before validation (identity by
        default; muse_glimmer collapses template-doubled ``name.name`` recipients)."""
        return name

    def _ps_trim_leading(self, text: str) -> str:
        if self._ps_trim_single:
            return text[1:] if text[:1] and text[:1] in self._ps_trim else text
        return text.lstrip(self._ps_trim)

    def _ps_trim_trailing(self, text: str) -> str:
        if self._ps_trim_single:
            return text[:-1] if text[-1:] and text[-1:] in self._ps_trim else text
        return text.rstrip(self._ps_trim)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        self._buffer += new_text
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)
        if not hasattr(self, "_ps_mode"):
            self._ps_reset()

        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []

        def _emit(fragment: str) -> None:
            if fragment:
                self.streamed_args_for_tool[self.current_tool_id] += fragment
                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, name=None, parameters=fragment)
                )

        def _update_prev() -> None:
            ledger = self.streamed_args_for_tool[self.current_tool_id]
            for probe in (ledger, ledger + "}"):
                try:
                    parsed = json.loads(probe)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(parsed, dict):
                    self.prev_tool_call_arr[self.current_tool_id]["arguments"] = parsed
                return

        while True:
            buf = self._buffer
            if not buf:
                break
            mode = self._ps_mode

            if mode == "idle":
                if calls:
                    break  # text after a call defers to the next step (wire order)
                pos = buf.find(self._ps_outer_open)
                if pos == -1:
                    hold = self._ends_with_partial_token(buf, self._ps_outer_open)
                    release = buf[: len(buf) - hold] if hold else buf
                    if release:
                        normal_parts.append(release)
                        self._buffer = buf[len(release):]
                    break
                if pos > 0:
                    normal_parts.append(buf[:pos])
                    self._buffer = buf[pos:]
                    continue
                self._buffer = buf[len(self._ps_outer_open):]
                self._ps_mode = "block"
                continue

            if mode == "block":
                inv = buf.find(self._ps_invoke_open_prefix)
                close = buf.find(self._ps_outer_close) if self._ps_outer_close else -1
                if close != -1 and (inv == -1 or close < inv):
                    self._buffer = buf[close + len(self._ps_outer_close):]
                    self._ps_mode = "idle"
                    continue
                if inv != -1:
                    m = self._ps_invoke_open_re.search(buf, inv)
                    if m is None:
                        break  # invoke tag still streaming
                    func_name = self._ps_canonical_name(m.group(1).strip())
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
                    self._buffer = buf[m.end():]
                    if func_name in self._tool_indices or _should_forward_unknown_tool(func_name):
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id, name=func_name, parameters=""
                            )
                        )
                        self.prev_tool_call_arr[self.current_tool_id] = {
                            "name": func_name,
                            "arguments": {},
                        }
                        self._args_started = False
                        self._ps_param_config = self._get_param_config(func_name, tools)
                        self._ps_mode = "invoke"
                    else:
                        logger.warning(f"Model attempted to call undefined function: {func_name}")
                        self._ps_mode = "invoke_skip"
                    continue
                hold = max(
                    self._ends_with_partial_token(buf, self._ps_invoke_open_prefix),
                    self._ends_with_partial_token(buf, self._ps_outer_close)
                    if self._ps_outer_close
                    else 0,
                )
                if len(buf) - hold > 0:
                    self._buffer = buf[len(buf) - hold:]  # inter-invoke whitespace
                break

            if mode in ("invoke", "invoke_skip"):
                p = buf.find(self._ps_param_open_prefix)
                e = buf.find(self._ps_invoke_close)
                if e != -1 and (p == -1 or e < p):
                    if mode == "invoke":
                        _emit("}" if self._args_started else "{}")
                        _update_prev()
                        self.current_tool_id += 1
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")
                    self._buffer = buf[e + len(self._ps_invoke_close):]
                    self._ps_mode = "block"
                    continue
                if p != -1:
                    m = self._ps_param_open_re.search(buf, p)
                    if m is None:
                        break  # parameter tag still streaming
                    self._buffer = buf[m.end():]
                    if mode == "invoke_skip":
                        self._ps_mode = "pskip"
                        continue
                    self._ps_key = m.group(1).strip()
                    lead = "{" if not self._args_started else ","
                    self._args_started = True
                    ptype = self._schema_param_type(
                        self._ps_key, self._ps_param_config, self._ps_missing_type
                    )
                    if ptype in ("string", "str", "enum"):
                        _emit(lead + json.dumps(self._ps_key, ensure_ascii=False) + ':"')
                        self._ps_emitted_any = False
                        self._ps_lead_trimmed = False
                        self._ps_mode = "pstr"
                    else:
                        self._ps_lead = lead
                        self._ps_mode = "pbuf"
                    continue
                hold = max(
                    self._ends_with_partial_token(buf, self._ps_param_open_prefix),
                    self._ends_with_partial_token(buf, self._ps_invoke_close),
                )
                if len(buf) - hold > 0:
                    self._buffer = buf[len(buf) - hold:]  # whitespace between parameters
                break

            if mode == "pstr":
                if not self._ps_lead_trimmed:
                    trimmed = self._ps_trim_leading(buf)
                    if trimmed != buf:
                        self._buffer = trimmed
                        if self._ps_trim_single or trimmed:
                            self._ps_lead_trimmed = bool(trimmed) or self._ps_trim_single
                        continue
                    if buf:
                        self._ps_lead_trimmed = True
                end = buf.find(self._ps_param_close)
                if end == -1:
                    hold = self._ends_with_partial_token(buf, self._ps_param_close)
                    safe = buf[: len(buf) - hold] if hold else buf
                    keep = len(safe) - len(safe.rstrip(self._ps_trim))
                    emit_now = safe[: len(safe) - keep]
                    if emit_now:
                        _emit(self._json_escape_chunk(emit_now))
                        self._ps_emitted_any = True
                        self._buffer = buf[len(emit_now):]
                    break
                tail = self._ps_trim_trailing(buf[:end])
                _emit(self._json_escape_chunk(tail) + '"')
                _update_prev()
                self._buffer = buf[end + len(self._ps_param_close):]
                self._ps_mode = "invoke"
                continue

            if mode in ("pbuf", "pskip"):
                end = buf.find(self._ps_param_close)
                if end == -1:
                    break  # hold the whole value until the parameter closes
                if mode == "pbuf":
                    raw = self._ps_trim_trailing(self._ps_trim_leading(buf[:end]))
                    converted = self._ps_convert_value(self._ps_key, raw)
                    _emit(
                        self._ps_lead
                        + json.dumps(self._ps_key, ensure_ascii=False)
                        + ":"
                        + json.dumps(converted, ensure_ascii=False)
                    )
                    _update_prev()
                self._buffer = buf[end + len(self._ps_param_close):]
                self._ps_mode = "invoke" if mode == "pbuf" else "invoke_skip"
                continue

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def finish_streaming(self) -> str:
        residual, self._buffer = self._buffer, ""
        mode = getattr(self, "_ps_mode", "idle")
        self._ps_reset()
        if mode != "idle" or (self._ps_outer_open and self._ps_outer_open in residual):
            return ""
        if self.prev_tool_call_arr and residual.strip() == "":
            return ""
        return residual


class Qwen25Detector(BaseFormatDetector):
    """
    Detector for Qwen 2.5 and Qwen 3 model function call format.

    Format Structure:
    ```
    <tool_call>\n{"name":"func1", "arguments":{...}}\n
    </tool_call>\n<tool_call>\n{"name":"func2", "arguments":{...}}\n</tool_call>
    ```

    Key Components:
    - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
    - Function Call Object: JSON object with "name" and "arguments" fields

    Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?chat_template=default
    """
    toolcall_opener = "<tool_call>"
    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.tool_call_separator = "\n"
        self._normal_text_buffer = ""  # Buffer for handling partial end tokens

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Qwen 2.5 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Find all <tool_call>\n...\n</tool_call> blocks
        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match_result in match_result_list:
            try:
                parsed_call = json.loads(match_result.strip())
                calls.extend(self.parse_base_json(parsed_call, tools))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON part: {match_result}, JSON parse error: {str(e)}")
                continue
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Streaming incremental parsing for Qwen 2.5 tool calls.
        Uses base class implementation with buffering to handle partial end tokens.
        """
        result = super().parse_streaming_increment(new_text, tools)

        # Handle partial end tokens that are streamed character by character
        if result.normal_text:
            self._normal_text_buffer += result.normal_text

            # Check if buffer contains complete end token (without leading newline)
            end_token_without_newline = self.eot_token  # "</tool_call>"
            if end_token_without_newline in self._normal_text_buffer:
                cleaned_text = self._normal_text_buffer.replace(end_token_without_newline, "")
                self._normal_text_buffer = ""
                result.normal_text = cleaned_text
            else:
                # Check if buffer might contain partial end token at the end
                partial_match_len = self._ends_with_partial_token(self._normal_text_buffer, end_token_without_newline)

                if partial_match_len:
                    # Keep potential partial match in buffer, return the rest
                    result.normal_text = self._normal_text_buffer[:-partial_match_len]
                    self._normal_text_buffer = self._normal_text_buffer[-partial_match_len:]
                else:
                    # No partial match, return all buffered text
                    result.normal_text = self._normal_text_buffer
                    self._normal_text_buffer = ""

        return result

    def finish_streaming(self) -> str:
        residual = super().finish_streaming()
        held, self._normal_text_buffer = self._normal_text_buffer, ""
        return held + residual


class MistralDetector(BaseFormatDetector):
    """
    Detector for Mistral model function call format.

    The Mistral format uses a simple bracket-delimited structure with JSON arrays
    containing function call objects.

    Format Structure:
    ```
    [TOOL_CALLS] [{"name": "function_name", "arguments": {json_args}}, ...]
    ```

    Reference: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3?chat_template=default
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "[TOOL_CALLS] ["
        self.eot_token = "]"
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)
        self.tool_call_separator = ", "

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Mistral format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Extract the JSON array part from [TOOL_CALLS] [...]
        # Use bracket counting to properly handle nested brackets in JSON content
        json_array_str = self._extract_json_array(text)
        if not json_array_str:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        try:
            function_call_arr = json.loads(json_array_str)
            # Handle both single object and array of objects
            if not isinstance(function_call_arr, list):
                function_call_arr = [function_call_arr]
            calls = self.parse_base_json(function_call_arr, tools)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON part: {json_array_str}, JSON parse error: {str(e)}")

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _extract_json_array(self, text: str) -> str:
        """
        Extract the JSON array part using bracket counting to handle nested brackets.

        :param text: The complete text containing [TOOL_CALLS] [...]
        :return: The JSON array string or None if not found
        """
        start_idx = text.find(self.bot_token)
        if start_idx == -1:
            return None

        # Start from the opening bracket after [TOOL_CALLS]
        json_start = start_idx + len(self.bot_token) - 1  # -1 to include the opening bracket
        bracket_count = 0
        in_string = False
        escape_next = False

        for i in range(json_start, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        return text[json_start : i + 1]

        return None


class Llama32Detector(BaseFormatDetector):
    """
    Detector for Llama 3.2 models with json tool call format.

    Format Structure:
    ```
    <python_tag>{"name":"xxx", "arguments":{...}}
    ```
    """
    toolcall_opener = "<|python_tag|>"
    def __init__(self):
        super().__init__()
        self.bot_token = "<|python_tag|>"
        # NOTE: technically Llama3.2 doesn't support well with parallel tool calls
        # They need specific prompt engineering to support parallel tool calls
        # Here we use ';' as the separator, which might have compatibility issues
        # if users define to use a different separator in their prompt
        self.tool_call_separator = ";"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Llama 3.2 format tool call."""
        # depending on the prompt format the Llama model may or may not
        # prefix the output with the <|python_tag|> token
        return "<|python_tag|>" in text or text.startswith("{")

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse function calls from text, handling multiple JSON objects."""
        if "<|python_tag|>" not in text and not text.startswith("{"):
            return StreamingParseResult(normal_text=text, calls=[])

        if "<|python_tag|>" in text:
            normal_text, action_text = text.split("<|python_tag|>", maxsplit=1)
        else:
            normal_text, action_text = "", text

        decoder = json.JSONDecoder()
        idx = 0
        safe_idx = idx  # the index of the last valid JSON object
        all_actions = []
        action_text_len = len(action_text)
        while idx < action_text_len:
            try:
                obj, end = decoder.raw_decode(action_text[idx:])
                all_actions.append(obj)
                idx += end + len(self.tool_call_separator)
                safe_idx = idx
            except json.JSONDecodeError as e:
                # Find where next `{"name"` appears and try again
                logger.warning(f"Failed to parse JSON part: {action_text[idx:]}, JSON parse error: {str(e)}")
                next_obj_start = action_text.find('{"name":', idx + 1)
                if next_obj_start == -1:
                    break
                idx = next_obj_start
                continue

        # Only process if we found valid JSON objects
        calls = self.parse_base_json(all_actions, tools) if all_actions else []
        # Use safe_idx to avoid idx containing the last part of an invalid JSON object
        trailing_text = action_text[safe_idx:].strip() if safe_idx < action_text_len else ""
        return StreamingParseResult(normal_text=normal_text + trailing_text, calls=calls)





class Glm47Detector(BaseFormatDetector):
    """
    Detector for GLM-4.7/GLM-4.7-Flash model function call format.

    The GLM-4.7 format uses an XML-style envelope with arg_key/arg_value pairs
    instead of JSON arguments.

    Format Structure:
    ```
    <tool_call>function_name
    <arg_key>param1</arg_key>
    <arg_value>value1</arg_value>
    <arg_key>param2</arg_key>
    <arg_value>value2</arg_value>
    </tool_call>
    ```

    Example:
    ```
    <tool_call>tool_brave_web_search_post
    <arg_key>query</arg_key>
    <arg_value>test search</arg_value>
    <arg_key>count</arg_key>
    <arg_value>5</arg_value>
    </tool_call>
    ```

    Key Components:
    - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
    - Function Name: Appears on the first line after `<tool_call>`
    - Arguments: Pairs of `<arg_key>name</arg_key>` and `<arg_value>value</arg_value>`

    Reference: https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/glm4_moe_tool_parser.py
    """
    toolcall_opener = "<tool_call>"
    _preserve_string_whitespace = False

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.tool_call_separator = "\n"

        # Regex patterns for parsing GLM-4.7 tool calls
        # Match complete tool call blocks
        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        # Extract function name and arguments from a tool call block
        # Function name can be followed by newline OR directly by <arg_key>
        # Pattern: <tool_call>function_name(\n|<arg_key>)...
        self.func_detail_regex = re.compile(
            r"<tool_call>([^<\n]+?)(?:\n|(?=<arg_key>)|(?=</tool_call>))(.*?)</tool_call>", re.DOTALL
        )
        # Extract arg_key/arg_value pairs
        self.func_arg_regex = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)

        self._last_arguments = ""
        self._normal_text_buffer = ""

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a GLM-4.7 format tool call."""
        return self.bot_token in text

    def _argument_is_string(self, key: str, param_config: Dict) -> bool:
        """Whether an XML argument uses the incremental string path.

        Poolside V1 follows JSON Schema literally: only an explicit
        ``type: string`` preserves the raw value. GLM keeps its legacy loose
        aliases and defaults for backward compatibility.
        """
        if self._preserve_string_whitespace:
            schema = param_config.get(key, {})
            return isinstance(schema, dict) and schema.get("type") == "string"
        return self._schema_param_type(key, param_config, "loose") in (
            "string",
            "str",
            "enum",
        )

    def _convert_xml_argument(
        self, value: str, key: str, param_config: Dict, func_name: str
    ):
        raw = value.strip()
        if self._preserve_string_whitespace:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return raw
        return self._convert_param_value(raw, key, param_config, func_name)

    def _parse_xml_arguments(self, arg_text: str, param_config: Dict | None = None, func_name: str = "") -> dict:
        """
        Parse XML-style arguments into a dictionary.

        Args:
            arg_text: The text containing <arg_key>/<arg_value> pairs

        Returns:
            Dictionary of argument name to value
        """
        if not arg_text:
            return {}

        args = {}
        matches = self.func_arg_regex.findall(arg_text)
        for key, value in matches:
            key = key.strip()
            if param_config and key in param_config:
                if self._preserve_string_whitespace and self._argument_is_string(
                    key, param_config
                ):
                    args[key] = value
                    continue
                # Schema-first: the declared type wins (a string-typed "5" stays "5").
                args[key] = self._convert_xml_argument(value, key, param_config, func_name)
                continue
            # Undeclared parameter: legacy loose typing.
            value = value.strip()
            try:
                parsed_value = json.loads(value)
                args[key] = parsed_value
            except (json.JSONDecodeError, ValueError):
                args[key] = value
        return args

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: StreamingParseResult with normal_text and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx] if idx != -1 else text
        if idx != -1 and not self._preserve_string_whitespace:
            normal_text = normal_text.strip()

        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls = []

        # Find all <tool_call>...</tool_call> blocks
        match_result_list = self.func_call_regex.findall(text)

        for match_result in match_result_list:
            try:
                # Extract function name and arguments
                func_detail = self.func_detail_regex.search(match_result)
                if not func_detail:
                    logger.warning(f"Failed to parse GLM-4.7 tool call: {match_result}")
                    continue

                func_name = func_detail.group(1).strip()
                arg_text = func_detail.group(2) if func_detail.group(2) else ""

                # Validate function name
                if func_name not in tool_indices and not _should_forward_unknown_tool(func_name):
                    logger.warning(f"Model attempted to call undefined function: {func_name}")
                    continue

                # Parse XML arguments to JSON (schema-first typing, loose fallback)
                func_args = self._parse_xml_arguments(
                    arg_text, self._get_param_config(func_name, tools), func_name
                )

                calls.append(
                    ToolCallItem(
                        tool_index=tool_indices.get(func_name, len(calls)),
                        name=func_name,
                        parameters=json.dumps(func_args, ensure_ascii=False),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to parse GLM-4.7 tool call: {match_result}, error: {str(e)}")
                continue

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    _G_KEY_OPEN = "<arg_key>"
    _G_KEY_CLOSE = "</arg_key>"
    _G_VAL_OPEN = "<arg_value>"
    _G_VAL_CLOSE = "</arg_value>"

    def _g_reset(self) -> None:
        self._g_mode = "idle"  # idle|name|invoke|invoke_skip|key|key_skip|preval|preval_skip|pstr|pbuf|pskip
        self._g_key = ""
        self._g_lead = "{"
        self._g_config: Dict = {}
        self._g_lead_trimmed = False

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Value-level streaming for GLM-4.7: text outside blocks streams live;
        string-typed <arg_value> content streams char-by-char as JSON-escaped
        prefix-stable fragments; typed values buffer until the value closes."""
        self._buffer += new_text
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)
        if not hasattr(self, "_g_mode"):
            self._g_reset()

        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []

        def _emit(fragment: str) -> None:
            if fragment:
                self.streamed_args_for_tool[self.current_tool_id] += fragment
                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, name=None, parameters=fragment)
                )

        def _update_prev() -> None:
            ledger = self.streamed_args_for_tool[self.current_tool_id]
            for probe in (ledger, ledger + "}"):
                try:
                    parsed = json.loads(probe)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(parsed, dict):
                    self.prev_tool_call_arr[self.current_tool_id]["arguments"] = parsed
                return

        while True:
            buf = self._buffer
            if not buf:
                break
            mode = self._g_mode

            if mode == "idle":
                if calls:
                    break  # text after a call defers to the next step (wire order)
                pos = buf.find(self.bot_token)
                if pos == -1:
                    hold = self._ends_with_partial_token(buf, self.bot_token)
                    release = buf[: len(buf) - hold] if hold else buf
                    if release:
                        normal_parts.append(release)
                        self._buffer = buf[len(release):]
                    break
                if pos > 0:
                    normal_parts.append(buf[:pos])
                    self._buffer = buf[pos:]
                    continue
                self._buffer = buf[len(self.bot_token):]
                self._g_mode = "name"
                continue

            if mode == "name":
                # The function name runs until a newline, the first <arg_key>, or
                # the closing tag (whichever comes first).
                ends = [p for p in (buf.find("\n"), buf.find(self._G_KEY_OPEN), buf.find(self.eot_token)) if p != -1]
                if not ends:
                    break  # name still streaming (names are short — hold)
                cut = min(ends)
                func_name = buf[:cut].strip()
                self._buffer = buf[cut + 1:] if buf[cut] == "\n" else buf[cut:]
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                if func_name and (
                    func_name in self._tool_indices or _should_forward_unknown_tool(func_name)
                ):
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, name=func_name, parameters="")
                    )
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                    self._args_started = False
                    self._g_config = self._get_param_config(func_name, tools)
                    self._g_mode = "invoke"
                else:
                    logger.warning(f"Model attempted to call undefined function: {func_name}")
                    self._g_mode = "invoke_skip"
                continue

            if mode in ("invoke", "invoke_skip"):
                k = buf.find(self._G_KEY_OPEN)
                e = buf.find(self.eot_token)
                if e != -1 and (k == -1 or e < k):
                    if mode == "invoke":
                        _emit("}" if self._args_started else "{}")
                        _update_prev()
                        self.current_tool_id += 1
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")
                    self._buffer = buf[e + len(self.eot_token):]
                    self._g_mode = "idle"
                    continue
                if k != -1:
                    self._buffer = buf[k + len(self._G_KEY_OPEN):]
                    self._g_mode = "key" if mode == "invoke" else "key_skip"
                    continue
                hold = max(
                    self._ends_with_partial_token(buf, self._G_KEY_OPEN),
                    self._ends_with_partial_token(buf, self.eot_token),
                )
                if len(buf) - hold > 0:
                    self._buffer = buf[len(buf) - hold:]  # whitespace between elements
                break

            if mode in ("key", "key_skip"):
                end = buf.find(self._G_KEY_CLOSE)
                if end == -1:
                    break  # keys are short — hold until complete
                self._g_key = buf[:end].strip()
                self._buffer = buf[end + len(self._G_KEY_CLOSE):]
                self._g_mode = "preval" if mode == "key" else "preval_skip"
                continue

            if mode in ("preval", "preval_skip"):
                v = buf.find(self._G_VAL_OPEN)
                if v == -1:
                    hold = self._ends_with_partial_token(buf, self._G_VAL_OPEN)
                    if len(buf) - hold > 0:
                        self._buffer = buf[len(buf) - hold:]  # whitespace between key and value
                    break
                self._buffer = buf[v + len(self._G_VAL_OPEN):]
                if mode == "preval_skip":
                    self._g_mode = "pskip"
                    continue
                lead = "{" if not self._args_started else ","
                self._args_started = True
                if self._argument_is_string(self._g_key, self._g_config):
                    _emit(lead + json.dumps(self._g_key, ensure_ascii=False) + ':"')
                    self._g_lead_trimmed = False
                    self._g_mode = "pstr"
                else:
                    self._g_lead = lead
                    self._g_mode = "pbuf"
                continue

            if mode == "pstr":
                if not self._preserve_string_whitespace and not self._g_lead_trimmed:
                    trimmed = buf.lstrip()
                    if trimmed != buf:
                        self._buffer = trimmed
                        continue
                    self._g_lead_trimmed = True
                end = buf.find(self._G_VAL_CLOSE)
                if end == -1:
                    hold = self._ends_with_partial_token(buf, self._G_VAL_CLOSE)
                    safe = buf[: len(buf) - hold] if hold else buf
                    keep = len(safe) - len(safe.rstrip())
                    emit_now = safe[: len(safe) - keep]
                    if emit_now:
                        _emit(self._json_escape_chunk(emit_now))
                        self._buffer = buf[len(emit_now):]
                    break
                tail = (
                    buf[:end]
                    if self._preserve_string_whitespace
                    else buf[:end].rstrip()
                )
                _emit(self._json_escape_chunk(tail) + '"')
                _update_prev()
                self._buffer = buf[end + len(self._G_VAL_CLOSE):]
                self._g_mode = "invoke"
                continue

            if mode in ("pbuf", "pskip"):
                end = buf.find(self._G_VAL_CLOSE)
                if end == -1:
                    break  # hold the whole value until it closes
                if mode == "pbuf":
                    raw = buf[:end].strip()
                    if self._g_key in self._g_config:
                        converted = self._convert_xml_argument(
                            raw, self._g_key, self._g_config, ""
                        )
                    else:
                        try:
                            converted = json.loads(raw)
                        except (json.JSONDecodeError, ValueError):
                            converted = raw
                    _emit(
                        self._g_lead
                        + json.dumps(self._g_key, ensure_ascii=False)
                        + ":"
                        + json.dumps(converted, ensure_ascii=False)
                    )
                    _update_prev()
                self._buffer = buf[end + len(self._G_VAL_CLOSE):]
                self._g_mode = "invoke" if mode == "pbuf" else "invoke_skip"
                continue

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def finish_streaming(self) -> str:
        residual, self._buffer = self._buffer, ""
        mode = getattr(self, "_g_mode", "idle")
        self._g_reset()
        if mode != "idle" or self.bot_token in residual:
            return ""
        if self.eot_token in residual:
            residual = residual.replace(self.eot_token, "")
        if self.prev_tool_call_arr and residual.strip() == "":
            return ""
        return residual


class PoolsideV1Detector(Glm47Detector):
    """Poolside V1 tool protocol used by Laguna.

    Its envelope matches GLM-4.7, but string arguments are raw text. Preserve
    leading/trailing whitespace so source code and patches survive parsing.
    """

    _preserve_string_whitespace = True


class DeepSeekV32Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V3.2 model function call format using DSML
    (DeepSeek Markup Language).

    Format Structure:
    ```
    <｜DSML｜function_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">Hangzhou</｜DSML｜parameter>
    <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Key Components:
    - Function Calls Block: `<｜DSML｜function_calls>` ... `</｜DSML｜function_calls>`
    - Individual Invocation: `<｜DSML｜invoke name="func">` ... `</｜DSML｜invoke>`
    - Parameters: `<｜DSML｜parameter name="key" string="true|false">value</｜DSML｜parameter>`
      - string="true": value is plain text (will be JSON-escaped)
      - string="false": value is JSON (numbers, booleans, arrays, objects)
    - Supports multiple parallel tool calls

    Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3.2
    """

    def __init__(self):
        super().__init__()
        self.dsml_token = "｜DSML｜"
        self.bot_token = f"<{self.dsml_token}function_calls>"
        self.eot_token = f"</{self.dsml_token}function_calls>"
        self.alt_bot_token = f"<{self.dsml_token}tool_calls>"
        self.alt_eot_token = f"</{self.dsml_token}tool_calls>"
        self.invoke_start_prefix = f"<{self.dsml_token}invoke"
        self.invoke_end_token = f"</{self.dsml_token}invoke>"
        self.param_end_token = f"</{self.dsml_token}parameter>"

        # Regex for complete invoke extraction
        _de = re.escape(self.dsml_token)
        self.invoke_regex = re.compile(
            rf'<{_de}invoke\s+name="([^"]+)"\s*>(.*?)</{_de}invoke>',
            re.DOTALL,
        )
        # Regex for parameter extraction
        self.param_regex = re.compile(
            rf'<{_de}parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>(.*?)</{_de}parameter>',
            re.DOTALL,
        )
        # Regex for partial invoke (name known, body still streaming)
        self.partial_invoke_regex = re.compile(
            rf'<{_de}invoke\s+name="([^"]+)"\s*>(.*)',
            re.DOTALL,
        )
        # Streaming state machine tag regexes (anchored matches over the buffer).
        self.invoke_open_regex = re.compile(rf'<{_de}invoke\s+name="([^"]+)"\s*>')
        self.param_open_regex = re.compile(
            rf'<{_de}parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>'
        )

        self._last_arguments = ""
        self._accumulated_params: List[tuple] = []
        self._in_function_calls = False  # Track if we're inside a function_calls block
        # Streaming state machine (vLLM deepseekv32-parser style):
        # idle | block | invoke | invoke_skip | pstr | pbuf | pskip
        self._ds_mode = "idle"
        self._args_started = False
        self._param_name = ""
        self._param_lead = "{"

    def block_close_tokens(self) -> tuple:
        return (self.eot_token, self.alt_eot_token)

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text or self.alt_bot_token in text

    def _param_fragment(self, index: int, name: str, is_str: str, value: str) -> str:
        """Prefix-stable arguments fragment for one closed DSML parameter: the
        object opener (or separator) plus ``"name": value``, serialized exactly as
        json.dumps of the full dict would, so concatenated fragments + the closing
        brace equal the final arguments JSON byte-for-byte."""
        if is_str == "true":
            parsed: Any = value
        else:
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                parsed = value
        lead = "{" if index == 0 else ", "
        return (
            lead
            + json.dumps(name, ensure_ascii=False)
            + ": "
            + json.dumps(parsed, ensure_ascii=False)
        )

    def _dsml_params_to_json(self, params: List[tuple]) -> str:
        """Convert DSML parameter tuples (name, is_str, value) to a JSON arguments string."""
        args = {}
        for name, is_str, value in params:
            if is_str == "true":
                args[name] = value
            else:
                try:
                    args[name] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    args[name] = value
        return json.dumps(args, ensure_ascii=False)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing for DSML format tool calls."""
        idx = _first_existing_pos(text, [self.bot_token, self.alt_bot_token])
        normal_text = text[:idx].strip() if idx != -1 else text
        if idx == -1:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls = []

        invoke_matches = self.invoke_regex.findall(text)
        for func_name, invoke_body in invoke_matches:
            if func_name not in tool_indices and not _should_forward_unknown_tool(func_name):
                logger.warning(f"Model attempted to call undefined function: {func_name}")
                continue

            param_matches = self.param_regex.findall(invoke_body)
            args_json = self._dsml_params_to_json(param_matches)

            calls.append(
                ToolCallItem(
                    tool_index=tool_indices.get(func_name, len(calls)),
                    name=func_name,
                    parameters=args_json,
                )
            )

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Streaming incremental parsing for DSML tool calls, modeled on vLLM's
        deepseekv32 parser: text outside blocks streams live; ``string="true"``
        parameter VALUES stream char-by-char as JSON-escaped, prefix-stable
        argument fragments (``{"key":"`` at parameter open, escaped value chars,
        ``"`` at close, ``}`` at invoke close); non-string values buffer until the
        parameter closes because their JSON form needs the complete text."""
        self._buffer += new_text
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        param_open_token = f"<{self.dsml_token}parameter"

        def _emit_args(fragment: str) -> None:
            if fragment:
                self.streamed_args_for_tool[self.current_tool_id] += fragment
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id, name=None, parameters=fragment
                    )
                )

        def _update_prev_args() -> None:
            ledger = self.streamed_args_for_tool[self.current_tool_id]
            for probe in (ledger, ledger + "}"):
                try:
                    parsed = json.loads(probe)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(parsed, dict):
                    self.prev_tool_call_arr[self.current_tool_id]["arguments"] = parsed
                return

        while True:
            buf = self._buffer
            if not buf:
                break

            if self._ds_mode == "idle":
                if calls:
                    break  # text after a call defers to the next step (wire order)
                pos = _first_existing_pos(buf, [self.bot_token, self.alt_bot_token])
                if pos == -1:
                    hold = max(
                        self._ends_with_partial_token(buf, self.bot_token),
                        self._ends_with_partial_token(buf, self.alt_bot_token),
                    )
                    release = buf[: len(buf) - hold] if hold else buf
                    if release:
                        for e_token in (self.eot_token, self.alt_eot_token, self.invoke_end_token):
                            if e_token in release:
                                release = release.replace(e_token, "")
                        if release:
                            normal_parts.append(release)
                        self._buffer = buf[len(buf) - hold:] if hold else ""
                    break
                if pos > 0:
                    normal_parts.append(buf[:pos])
                    self._buffer = buf[pos:]
                    continue
                opener = self.bot_token if buf.startswith(self.bot_token) else self.alt_bot_token
                self._buffer = buf[len(opener):]
                self._in_function_calls = True
                self._ds_mode = "block"
                continue

            if self._ds_mode == "block":
                inv = buf.find(self.invoke_start_prefix)
                close = _first_existing_pos(buf, [self.eot_token, self.alt_eot_token])
                if close != -1 and (inv == -1 or close < inv):
                    matched = (
                        self.eot_token
                        if buf[close : close + len(self.eot_token)] == self.eot_token
                        else self.alt_eot_token
                    )
                    self._buffer = buf[close + len(matched):]
                    self._in_function_calls = False
                    self._ds_mode = "idle"
                    continue
                if inv != -1:
                    m = self.invoke_open_regex.search(buf, inv)
                    if m is None:
                        break  # invoke tag still streaming
                    func_name = m.group(1)
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
                    self._buffer = buf[m.end():]
                    if func_name in self._tool_indices or _should_forward_unknown_tool(func_name):
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id, name=func_name, parameters=""
                            )
                        )
                        self.prev_tool_call_arr[self.current_tool_id] = {
                            "name": func_name,
                            "arguments": {},
                        }
                        self._args_started = False
                        self._ds_mode = "invoke"
                    else:
                        logger.warning(f"Model attempted to call undefined function: {func_name}")
                        self._ds_mode = "invoke_skip"
                    continue
                hold = max(
                    self._ends_with_partial_token(buf, self.invoke_start_prefix),
                    self._ends_with_partial_token(buf, self.eot_token),
                    self._ends_with_partial_token(buf, self.alt_eot_token),
                )
                if len(buf) - hold > 0:
                    self._buffer = buf[len(buf) - hold:]  # inter-invoke whitespace
                break

            if self._ds_mode in ("invoke", "invoke_skip"):
                p = buf.find(param_open_token)
                e = buf.find(self.invoke_end_token)
                if e != -1 and (p == -1 or e < p):
                    if self._ds_mode == "invoke":
                        _emit_args("}" if self._args_started else "{}")
                        _update_prev_args()
                        self.current_tool_id += 1
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")
                    self._buffer = buf[e + len(self.invoke_end_token):]
                    self._ds_mode = "block"
                    continue
                if p != -1:
                    m = self.param_open_regex.search(buf, p)
                    if m is None:
                        break  # parameter tag still streaming
                    self._buffer = buf[m.end():]
                    if self._ds_mode == "invoke_skip":
                        self._ds_mode = "pskip"
                        continue
                    self._param_name = m.group(1)
                    lead = "{" if not self._args_started else ","
                    self._args_started = True
                    if m.group(2) == "true":
                        _emit_args(lead + json.dumps(self._param_name, ensure_ascii=False) + ':"')
                        self._ds_mode = "pstr"
                    else:
                        self._param_lead = lead
                        self._ds_mode = "pbuf"
                    continue
                hold = max(
                    self._ends_with_partial_token(buf, param_open_token),
                    self._ends_with_partial_token(buf, self.invoke_end_token),
                )
                if len(buf) - hold > 0:
                    self._buffer = buf[len(buf) - hold:]  # whitespace between parameters
                break

            if self._ds_mode == "pstr":
                end = buf.find(self.param_end_token)
                if end == -1:
                    hold = self._ends_with_partial_token(buf, self.param_end_token)
                    emit_len = len(buf) - hold
                    if emit_len > 0:
                        _emit_args(json.dumps(buf[:emit_len], ensure_ascii=False)[1:-1])
                        self._buffer = buf[emit_len:]
                    break
                _emit_args(json.dumps(buf[:end], ensure_ascii=False)[1:-1] + '"')
                self._buffer = buf[end + len(self.param_end_token):]
                _update_prev_args()
                self._ds_mode = "invoke"
                continue

            if self._ds_mode in ("pbuf", "pskip"):
                end = buf.find(self.param_end_token)
                if end == -1:
                    break  # hold the whole value until the parameter closes
                if self._ds_mode == "pbuf":
                    value = buf[:end]
                    try:
                        parsed: Any = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        parsed = value
                    _emit_args(
                        self._param_lead
                        + json.dumps(self._param_name, ensure_ascii=False)
                        + ":"
                        + json.dumps(parsed, ensure_ascii=False)
                    )
                    _update_prev_args()
                self._buffer = buf[end + len(self.param_end_token):]
                self._ds_mode = "invoke" if self._ds_mode == "pbuf" else "invoke_skip"
                continue

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def finish_streaming(self) -> str:
        residual, self._buffer = self._buffer, ""
        mode, self._ds_mode = self._ds_mode, "idle"
        self._in_function_calls = False
        if mode != "idle" or self.has_tool_call(residual):
            return ""
        for e_token in (self.eot_token, self.alt_eot_token, self.invoke_end_token):
            if e_token in residual:
                residual = residual.replace(e_token, "")
        if self.prev_tool_call_arr and residual.strip() == "":
            return ""
        return residual


class Qwen3CoderDetector(InvokeParamStreamMixin, BaseFormatDetector):
    toolcall_opener = "<tool_call>"
    _ps_trim = "\n"
    _ps_trim_single = True
    _ps_missing_type = "string"

    """
    Detector for Qwen3-Coder XML-style function call format.

    Format Structure:
    ```
    <tool_call>
    <function=function_name>
    <parameter=param1>
    value1
    </parameter>
    <parameter=param2>
    value2
    </parameter>
    </function>
    </tool_call>
    ```

    Key differences from Qwen25Detector (JSON-based):
    - Parameters are XML key-value pairs, not JSON objects
    - Function name is embedded in the <function=> tag attribute
    - Values need schema-aware type conversion (string by default)

    Reference: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Coder-480B-A35B.html
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.tool_call_separator = "\n"

        # Regex patterns
        self.tool_call_block_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.function_regex = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
        self.parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.DOTALL
        )
        self._normal_text_buffer = ""

        # InvokeParamStreamMixin grammar
        self._ps_outer_open = "<tool_call>"
        self._ps_outer_close = "</tool_call>"
        self._ps_invoke_open_prefix = "<function="
        self._ps_invoke_open_re = re.compile(r"<function=([^>]*)>")
        self._ps_invoke_close = "</function>"
        self._ps_param_open_prefix = "<parameter="
        self._ps_param_open_re = re.compile(r"<parameter=([^>]*)>")
        self._ps_param_close = "</parameter>"
        self._ps_reset()

    def has_tool_call(self, text: str) -> bool:
        return "<function=" in text or self.bot_token in text


    def _parse_function_call(self, function_str: str, tools: List[Tool]) -> Optional[ToolCallItem]:
        """Parse a single <function=name>...</function> block into a ToolCallItem."""
        try:
            end_index = function_str.index(">")
        except ValueError:
            return None

        func_name = function_str[:end_index].strip()
        tool_indices = self._get_tool_indices(tools)
        if func_name not in tool_indices and not _should_forward_unknown_tool(func_name):
            logger.warning(f"Model attempted to call undefined function: {func_name}")
            return None

        parameters_text = function_str[end_index + 1 :]
        param_config = self._get_param_config(func_name, tools)
        param_dict = {}

        for match in self.parameter_regex.findall(parameters_text):
            try:
                idx = match.index(">")
            except ValueError:
                continue
            param_name = match[:idx].strip()
            param_value = match[idx + 1 :]
            # Strip leading/trailing newlines from value
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(param_value, param_name, param_config, func_name)

        return ToolCallItem(
            tool_index=tool_indices.get(func_name, 0),
            name=func_name,
            parameters=json.dumps(param_dict, ensure_ascii=False),
        )

    def _build_partial_arguments_json(self, func_name: str, partial_body: str, tools: List[Tool]) -> Optional[str]:
        """Build the current argument JSON from a partial XML tool-call body."""
        param_matches = self.parameter_regex.findall(partial_body)
        if not param_matches:
            return None

        param_config = self._get_param_config(func_name, tools)
        param_dict = {}
        has_visible_value = False

        for match in param_matches:
            try:
                idx = match.index(">")
            except ValueError:
                continue

            param_name = match[:idx].strip()
            param_value = match[idx + 1 :]
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            if param_value.strip():
                has_visible_value = True
            elif (
                f"<parameter={param_name}>" in partial_body
                and f"<parameter={param_name}>{param_value}</parameter>" in partial_body
            ):
                # Closed empty-string parameter. We can safely emit it.
                has_visible_value = True
            else:
                # Parameter tag is present but its value has not started streaming yet.
                continue

            param_dict[param_name] = self._convert_param_value(param_value, param_name, param_config, func_name)

        if not param_dict and not has_visible_value:
            return None

        return json.dumps(param_dict, ensure_ascii=False)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if "<function=" not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Extract function blocks from tool_call blocks (or raw text as fallback)
        tool_call_blocks = self.tool_call_block_regex.findall(text)
        if not tool_call_blocks:
            tool_call_blocks = [text]

        calls = []
        for block in tool_call_blocks:
            func_matches = self.function_regex.findall(block)
            for match in func_matches:
                func_str = match[0] if match[0] else match[1]
                item = self._parse_function_call(func_str, tools)
                if item:
                    item.tool_index = len(calls)
                    calls.append(item)

        return StreamingParseResult(normal_text=normal_text, calls=calls)


class Gemma4Detector(BaseFormatDetector):
    """FreeToken serving adapter for Gemma4's compact tool-call format."""
    toolcall_opener = "<|tool_call>"
    def __init__(self):
        super().__init__()
        self.bot_token = "<|tool_call>"
        self.eot_token = "<tool_call|>"
        self.call_regex = re.compile(
            r"<\|tool_call>\s*call:([A-Za-z_][\w.:-]*)\{(.*?)\}<tool_call\|>",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if idx == -1:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        for func_name, arg_text in self.call_regex.findall(text):
            if func_name not in tool_indices and not _should_forward_unknown_tool(func_name):
                logger.warning(f"Model attempted to call undefined function: {func_name}")
                continue
            calls.append(
                ToolCallItem(
                    tool_index=len(calls),
                    name=func_name,
                    parameters=json.dumps(_parse_gemma_call_args(arg_text), ensure_ascii=False),
                )
            )
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    _G4_QUOTE = '<|"|>'

    def _g4_reset(self) -> None:
        self._g4_mode = "idle"  # idle|header|key|dispatch|pstr|vbuf|await_eot|swallow
        self._g4_key = ""
        self._g4_scanned = 0
        self._g4_depth = 0
        self._g4_in_str = False

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Value-level streaming for Gemma4's compact call syntax: text outside
        blocks streams live; ``<|"|>``-quoted string values stream char-by-char as
        JSON-escaped prefix-stable fragments; unquoted values (numbers, booleans,
        nested objects/arrays) buffer to their top-level ``,``/``}`` terminator and
        are typed exactly like the non-streaming parser."""
        self._buffer += new_text
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)
        if not hasattr(self, "_g4_mode"):
            self._g4_reset()

        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        Q = self._G4_QUOTE

        def _emit(fragment: str) -> None:
            if fragment:
                self.streamed_args_for_tool[self.current_tool_id] += fragment
                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, name=None, parameters=fragment)
                )

        def _update_prev() -> None:
            ledger = self.streamed_args_for_tool[self.current_tool_id]
            for probe in (ledger, ledger + "}"):
                try:
                    parsed = json.loads(probe)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(parsed, dict):
                    self.prev_tool_call_arr[self.current_tool_id]["arguments"] = parsed
                return

        while True:
            buf = self._buffer
            if not buf:
                break
            mode = self._g4_mode

            if mode == "idle":
                if calls:
                    break  # text after a call defers to the next step (wire order)
                pos = buf.find(self.bot_token)
                if pos == -1:
                    hold = self._ends_with_partial_token(buf, self.bot_token)
                    release = buf[: len(buf) - hold] if hold else buf
                    if release:
                        normal_parts.append(release)
                        self._buffer = buf[len(release):]
                    break
                if pos > 0:
                    normal_parts.append(buf[:pos])
                    self._buffer = buf[pos:]
                    continue
                self._buffer = buf[len(self.bot_token):]
                self._g4_mode = "header"
                continue

            if mode == "header":
                brace = buf.find("{")
                if brace == -1:
                    break  # header still streaming (short — hold)
                header = buf[:brace].strip()
                self._buffer = buf[brace + 1:]
                func_name = header[len("call:"):].strip() if header.startswith("call:") else ""
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                if func_name and (
                    func_name in self._tool_indices or _should_forward_unknown_tool(func_name)
                ):
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, name=func_name, parameters="")
                    )
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                    self._args_started = False
                    self._g4_mode = "key"
                else:
                    logger.warning(f"Model attempted to call undefined function: {func_name!r}")
                    self._g4_mode = "swallow"
                continue

            if mode == "key":
                stripped = buf.lstrip()
                if stripped != buf:
                    self._buffer = stripped
                    continue
                if buf.startswith("}"):
                    _emit("}" if self._args_started else "{}")
                    _update_prev()
                    self.current_tool_id += 1
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
                    self._buffer = buf[1:]
                    self._g4_mode = "await_eot"
                    continue
                colon = buf.find(":")
                if colon == -1:
                    break  # key still streaming (short — hold)
                self._g4_key = buf[:colon].strip()
                self._buffer = buf[colon + 1:]
                self._g4_mode = "dispatch"
                continue

            if mode == "dispatch":
                stripped = buf.lstrip()
                if stripped != buf:
                    self._buffer = stripped
                    continue
                lead = "{" if not self._args_started else ","
                if buf.startswith(Q):
                    self._args_started = True
                    _emit(lead + json.dumps(self._g4_key, ensure_ascii=False) + ':"')
                    self._buffer = buf[len(Q):]
                    self._g4_mode = "pstr"
                    continue
                if self._ends_with_partial_token(buf, Q) == len(buf):
                    break  # could still become the opening quote marker
                self._args_started = True
                self._g4_lead = lead
                self._g4_scanned = 0
                self._g4_depth = 0
                self._g4_in_str = False
                self._g4_mode = "vbuf"
                continue

            if mode == "pstr":
                end = buf.find(Q)
                if end == -1:
                    hold = self._ends_with_partial_token(buf, Q)
                    emit_now = buf[: len(buf) - hold] if hold else buf
                    if emit_now:
                        _emit(self._json_escape_chunk(emit_now))
                        self._buffer = buf[len(emit_now):]
                    break
                _emit(self._json_escape_chunk(buf[:end]) + '"')
                _update_prev()
                self._buffer = buf[end + len(Q):]
                self._g4_mode = "key_sep"
                continue

            if mode == "key_sep":
                stripped = buf.lstrip()
                if stripped != buf:
                    self._buffer = stripped
                    continue
                if buf.startswith(","):
                    self._buffer = buf[1:]
                    self._g4_mode = "key"
                    continue
                if buf.startswith("}"):
                    self._g4_mode = "key"  # key state handles the close
                    continue
                break  # separator still streaming

            if mode == "vbuf":
                # Scan for the top-level , or } terminator, quote/depth aware.
                hold = self._ends_with_partial_token(buf, Q)
                limit = len(buf) - hold
                i = self._g4_scanned
                term = -1
                while i < limit:
                    if buf.startswith(Q, i):
                        self._g4_in_str = not self._g4_in_str
                        i += len(Q)
                        continue
                    ch = buf[i]
                    if not self._g4_in_str:
                        if ch in "{[":
                            self._g4_depth += 1
                        elif ch in "]}" and self._g4_depth > 0:
                            self._g4_depth -= 1
                        elif self._g4_depth == 0 and ch in ",}":
                            term = i
                            break
                    i += 1
                if term == -1:
                    self._g4_scanned = i
                    break  # value still streaming — keep buffering
                raw = buf[:term].strip()
                converted = _parse_gemma_value(raw)
                _emit(
                    self._g4_lead
                    + json.dumps(self._g4_key, ensure_ascii=False)
                    + ":"
                    + json.dumps(converted, ensure_ascii=False)
                )
                _update_prev()
                self._g4_scanned = 0
                if buf[term] == ",":
                    self._buffer = buf[term + 1:]
                    self._g4_mode = "key"
                else:
                    self._buffer = buf[term:]
                    self._g4_mode = "key"  # key state emits the close on '}'
                continue

            if mode == "await_eot":
                pos = buf.find(self.eot_token)
                if pos == -1:
                    hold = self._ends_with_partial_token(buf, self.eot_token)
                    if len(buf) - hold > 0 and buf[: len(buf) - hold].strip() == "":
                        self._buffer = buf[len(buf) - hold:]
                        break
                    if hold:
                        break
                    # No closing marker and non-whitespace content: treat as done.
                    self._g4_mode = "idle"
                    continue
                self._buffer = buf[pos + len(self.eot_token):]
                self._g4_mode = "idle"
                continue

            if mode == "swallow":
                pos = buf.find(self.eot_token)
                if pos == -1:
                    hold = self._ends_with_partial_token(buf, self.eot_token)
                    self._buffer = buf[len(buf) - hold:] if hold else ""
                    break
                self._buffer = buf[pos + len(self.eot_token):]
                self._g4_mode = "idle"
                continue

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def finish_streaming(self) -> str:
        residual, self._buffer = self._buffer, ""
        mode = getattr(self, "_g4_mode", "idle")
        self._g4_reset()
        if mode != "idle" or self.bot_token in residual:
            return ""
        if self.eot_token in residual:
            residual = residual.replace(self.eot_token, "")
        if self.prev_tool_call_arr and residual.strip() == "":
            return ""
        return residual


class MiniMaxDetector(InvokeParamStreamMixin, BaseFormatDetector):
    toolcall_opener = "<minimax:tool_call>"
    _ps_trim = "\n"
    _ps_trim_single = False
    _ps_missing_type = "loose"

    """FreeToken serving adapter for MiniMax-M2 XML tool-call blocks.

    LightLLM routes MiniMax mostly through reasoning parser/template handling; FreeToken's
    OpenAI-compatible server consumes the final ``<minimax:tool_call>`` block here.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<minimax:tool_call>"
        self.eot_token = "</minimax:tool_call>"
        self.invoke_regex = re.compile(r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', re.DOTALL)
        self.param_regex = re.compile(r'<parameter\s+name="([^"]+)"\s*>(.*?)</parameter>', re.DOTALL)

        # InvokeParamStreamMixin grammar
        self._ps_outer_open = "<minimax:tool_call>"
        self._ps_outer_close = "</minimax:tool_call>"
        self._ps_invoke_open_prefix = "<invoke"
        self._ps_invoke_open_re = re.compile(r'<invoke\s+name="([^"]+)"\s*>')
        self._ps_invoke_close = "</invoke>"
        self._ps_param_open_prefix = "<parameter"
        self._ps_param_open_re = re.compile(r'<parameter\s+name="([^"]+)"\s*>')
        self._ps_param_close = "</parameter>"
        self._ps_reset()

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if idx == -1:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        for func_name, invoke_body in self.invoke_regex.findall(text):
            if func_name not in tool_indices and not _should_forward_unknown_tool(func_name):
                logger.warning(f"Model attempted to call undefined function: {func_name}")
                continue
            args = {}
            config = self._get_param_config(func_name, tools)
            for name, value in self.param_regex.findall(invoke_body):
                key = name.strip()
                raw = value.strip("\n")
                if key in config:
                    args[key] = self._convert_param_value(raw, key, config, func_name)
                else:
                    args[key] = _parse_loose_json_value(raw)
            calls.append(
                ToolCallItem(
                    tool_index=len(calls),
                    name=func_name,
                    parameters=json.dumps(args, ensure_ascii=False),
                )
            )
        return StreamingParseResult(normal_text=normal_text, calls=calls)



class MiniMaxM3Detector(BaseFormatDetector):
    """FreeToken serving adapter for MiniMax-M3 namespace-delimited XML tool calls.

    M3 is NOT M2 with renamed tags (vLLM's Rust ``MinimaxM3ToolParser`` grammar is
    the reference): every structural tag is prefixed with the ``]<]minimax[>[``
    namespace marker, one wrapper block holds multiple ``<invoke>`` tags, and the
    arguments are parameter-NAME XML elements rendered recursively -- objects nest
    tags, arrays repeat ``<item>``::

        ]<]minimax[>[<tool_call>
        ]<]minimax[>[<invoke name="create_order">
        ]<]minimax[>[<user_id>42]<]minimax[>[</user_id>
        ]<]minimax[>[<shipping>
        ]<]minimax[>[<city>Singapore]<]minimax[>[</city>
        ]<]minimax[>[</shipping>
        ]<]minimax[>[</invoke>
        ]<]minimax[>[</tool_call>

    Leaf values are typed from the tool schema at every nesting level (loose JSON
    when undeclared). Element scanning is lenient -- malformed tags and stray text
    never void the parameters that did parse, and a truncated trailing element
    salvages every complete sibling (which is what makes ``recover_truncated_call``
    work for calls cut off by max_tokens).

    Streaming: text outside the wrapper streams live (holding back a partial
    marker suffix); inside the wrapper each completed ``</invoke>`` emits its call
    in one piece, wire order preserved. The buffer keeps the wrapper opener while
    inside a block so ``finish_streaming`` suppresses truncated-markup residue and
    ``recover_truncated_call`` can re-parse it.
    """

    NS = "]<]minimax[>["

    def __init__(self):
        super().__init__()
        self.bot_token = self.NS + "<tool_call>"
        self.eot_token = self.NS + "</tool_call>"
        # The template renders double quotes but models emit single quotes too.
        self.invoke_open_re = re.compile(
            re.escape(self.NS) + r'<invoke\s+name=["\']([^"\']+)["\']\s*>'
        )
        self._invoke_close = self.NS + "</invoke>"
        # While True, self._buffer starts with bot_token (the retained opener).
        self._m3_in_block = False

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def _partial_marker_suffix(self, buffer: str) -> int:
        """Length of the LONGEST buffer suffix that is a proper prefix of the
        wrapper opener. The base ``_ends_with_partial_token`` returns the shortest
        match, which under-holds for this marker: ``]`` recurs inside
        ``]<]minimax[>[...``, so a buffer ending ``]<]`` would hold only 1 char and
        leak ``]<`` as content, breaking the marker forever."""
        for i in range(min(len(buffer), len(self.bot_token) - 1), 0, -1):
            if self.bot_token.startswith(buffer[-i:]):
                return i
        return 0

    # ---- lenient recursive parameter-XML scanning ------------------------------------
    _TAG_BAD_CHARS = frozenset(' "<>/')

    def _scan_elements(self, text: str) -> tuple:
        """Scan ``text`` for ``NS<k>...NS</k>`` elements; returns ``(items, stray)``
        where ``stray`` is the inter-element text (surfaced as ``"$text"`` for
        mixed content). Lenient: malformed tags and dangling closers are stepped
        over without dropping later siblings, an unterminated trailing element
        salvages everything before it, and same-name nesting is depth-matched.

        Known leniency trade-off: a leaf value quoting a well-formed element pair
        parses as structure, and one quoting its own closer ends the element
        early -- the strict alternative voided every parameter over one stray
        character. (Quoted invoke/wrapper markers are handled structurally by
        ``_scan_invoke_interior``.)
        """
        items: List[tuple] = []
        stray_parts: List[str] = []
        pos, n = 0, len(text)
        while pos < n:
            nxt = text.find(self.NS + "<", pos)
            if nxt == -1:
                stray_parts.append(text[pos:])
                break
            stray_parts.append(text[pos:nxt])
            pos = nxt
            if text.startswith(self.NS + "</", pos):
                # A close tag we did not open: model noise -- skip the marker and
                # keep scanning (aborting dropped every later sibling).
                pos += len(self.NS) + 2
                continue
            tag_start = pos + len(self.NS) + 1
            gt = text.find(">", tag_start)
            if gt == -1:
                break  # truncated open tag
            name = text[tag_start:gt]
            if not name or any(c in self._TAG_BAD_CHARS for c in name):
                pos = gt + 1  # malformed tag (e.g. an <invoke ...> echo): step over
                continue
            open_tag = self.NS + "<" + name + ">"
            close_tag = self.NS + "</" + name + ">"
            depth, search, inner_end = 1, gt + 1, -1
            while depth:
                cpos = text.find(close_tag, search)
                if cpos == -1:
                    # unterminated element: salvage what parsed
                    return items, "".join(stray_parts)
                opos = text.find(open_tag, search)
                if opos != -1 and opos < cpos:
                    depth += 1
                    search = opos + len(open_tag)
                else:
                    depth -= 1
                    inner_end = cpos
                    search = cpos + len(close_tag)
            items.append((name, text[gt + 1 : inner_end]))
            pos = search
        return items, "".join(stray_parts)

    def _typed_leaf(self, raw: str, prop: Any) -> Any:
        """Leaf dequoting. The template renders leaf values verbatim, so string
        leaves round-trip exactly (stripping would corrupt multi-line arguments).
        Declared non-string types tolerate surrounding whitespace; undeclared
        leaves fall back to loose JSON but keep the verbatim text when the parse
        yields a string anyway."""
        if prop is not None:
            if isinstance(prop, dict) and "type" not in prop:
                subs = prop.get("anyOf") or prop.get("oneOf")
                if isinstance(subs, list) and subs:
                    return self._union_leaf(raw, subs)
            ptype = (
                str(prop.get("type", "string")).strip().lower()
                if isinstance(prop, dict)
                else "string"
            )
            if ptype in ("string", "str", "enum"):
                return raw
            if ptype in ("number", "float", "double"):
                # Integer literals stay int; "5.0" stays 5.0.
                s = raw.strip()
                try:
                    return int(s)
                except ValueError:
                    try:
                        return float(s)
                    except ValueError:
                        return raw
            if ptype in ("object", "array") and raw.strip() == "":
                # Empty container-typed element -> typed empty container.
                return {} if ptype == "object" else []
            return self._convert_param_value(raw.strip(), "_", {"_": prop}, "")
        if raw.strip() == "":
            # An empty element is the empty STRING; loose-JSON would make it {}.
            return ""
        value = _parse_loose_json_value(raw.strip())
        return raw if isinstance(value, str) else value

    def _union_leaf(self, raw: str, subs: list) -> Any:
        """``anyOf``/``oneOf``: try each member's strict coercion in declared
        order; verbatim text when only the string member (or nothing) matches."""
        s = raw.strip()
        for sub in subs:
            t = str(sub.get("type", "")).strip().lower() if isinstance(sub, dict) else ""
            if t in ("integer", "int"):
                try:
                    return int(s)
                except ValueError:
                    continue
            if t in ("number", "float", "double"):
                try:
                    return int(s)
                except ValueError:
                    try:
                        return float(s)
                    except ValueError:
                        continue
            if t in ("boolean", "bool") and s.lower() in ("true", "false"):
                return s.lower() == "true"
            if t == "null" and s.lower() == "null":
                return None
            if t in ("object", "array"):
                try:
                    v = json.loads(s)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(v, dict if t == "object" else list):
                    return v
        return raw

    def _nested_value(self, raw: str, schema: Any = None) -> Any:
        items, stray = self._scan_elements(raw)
        if not items:
            return self._typed_leaf(raw, schema)
        return self._structure(items, schema, stray=stray)

    def _structure(self, items: List[tuple], schema: Any = None, stray: str = "") -> Any:
        """Composite value from scanned elements, threading the JSON schema down
        (``properties`` / ``items`` / ``additionalProperties``). Only ``<item>``
        children render as a bare array -- the template's array convention;
        repeated siblings under any other tag stay an object with an array-valued
        key. Mixed text+children keeps the text under ``"$text"``."""
        props = schema.get("properties") if isinstance(schema, dict) else None
        item_schema = schema.get("items") if isinstance(schema, dict) else None

        def _sub_schema(key: str) -> Any:
            sub = props.get(key) if isinstance(props, dict) else None
            if sub is None and isinstance(schema, dict):
                ap = schema.get("additionalProperties")
                if isinstance(ap, dict):
                    sub = ap
            return sub

        names = {k for k, _ in items}
        if names == {"item"}:
            return [self._nested_value(raw, item_schema) for _, raw in items]
        counts: Dict[str, int] = {}
        for k, _ in items:
            counts[k] = counts.get(k, 0) + 1
        out: Dict[str, Any] = {}
        for key, raw in items:
            sub = _sub_schema(key)
            # A repeated key's elements are the array's members: unwrap its array
            # schema to the item schema. A singleton key keeps the array schema
            # (its <item> children unwrap inside).
            if (
                counts[key] > 1
                and isinstance(sub, dict)
                and str(sub.get("type", "")).lower() == "array"
            ):
                sub = sub.get("items")
            value = self._nested_value(raw, sub)
            if key in out:
                prev = out[key]
                out[key] = (prev if isinstance(prev, list) else [prev]) + [value]
            else:
                out[key] = value
        if stray.strip():
            out["$text"] = stray.strip()
        return out

    def _args_from_items(
        self, func_name: str, items: List[tuple], tools: List[Tool]
    ) -> Dict:
        config = self._get_param_config(func_name, tools)
        args: Dict[str, Any] = {}
        for key, raw in items:
            prop = config.get(key) if isinstance(config, dict) and key in config else None
            nested, stray = self._scan_elements(raw)
            if nested:
                value: Any = self._structure(nested, prop, stray=stray)
            else:
                value = self._typed_leaf(raw, prop)
            if key in args:
                prev = args[key]
                args[key] = (prev if isinstance(prev, list) else [prev]) + [value]
            else:
                args[key] = value
        return args

    def _scan_invoke_interior(self, text: str, pos: int) -> tuple:
        """Collect parameter elements from ``pos`` (just past an invoke opener)
        until a top-level ``NS</invoke>`` / ``NS</tool_call>`` / end of text.
        Close markers are only interpreted between elements, so a value quoting
        wire syntax can neither end the invoke early nor truncate the block.
        Returns ``(items, end_pos, closer)`` with ``closer`` in
        {"invoke", "eot", None}; ``end_pos`` points at the closer marker
        (``len(text)`` when the text ran out -- truncation salvage)."""
        items: List[tuple] = []
        n = len(text)
        while pos < n:
            nxt = text.find(self.NS + "<", pos)
            if nxt == -1:
                return items, n, None
            if text.startswith(self._invoke_close, nxt):
                return items, nxt, "invoke"
            if text.startswith(self.eot_token, nxt):
                return items, nxt, "eot"
            pos = nxt
            if text.startswith(self.NS + "</", pos):
                # Dangling closer we did not open: model noise -- skip it and
                # keep collecting (the references keep well-formed siblings).
                pos += len(self.NS) + 2
                continue
            tag_start = pos + len(self.NS) + 1
            gt = text.find(">", tag_start)
            if gt == -1:
                return items, n, None  # truncated open tag
            name = text[tag_start:gt]
            if not name or any(c in self._TAG_BAD_CHARS for c in name):
                pos = gt + 1  # malformed tag (e.g. a quoted <invoke ...> echo)
                continue
            open_tag = self.NS + "<" + name + ">"
            close_tag = self.NS + "</" + name + ">"
            depth, search, inner_end = 1, gt + 1, -1
            while depth:
                cpos = text.find(close_tag, search)
                if cpos == -1:
                    return items, n, None  # unterminated element: salvage
                opos = text.find(open_tag, search)
                if opos != -1 and opos < cpos:
                    depth += 1
                    search = opos + len(open_tag)
                else:
                    depth -= 1
                    inner_end = cpos
                    search = cpos + len(close_tag)
            items.append((name, text[gt + 1 : inner_end]))
            pos = search
        return items, n, None

    def _parse_block(
        self, text: str, pos: int, tools: List[Tool], calls: List[ToolCallItem]
    ) -> int:
        """Parse a wrapper interior starting at ``pos`` (just past ``bot_token``),
        appending calls; returns the position just past the block's
        ``NS</tool_call>``, or ``len(text)`` when the wrapper never closes
        (end-of-generation truncation -- a truncated trailing invoke still
        salvages its complete parameters, the recover_truncated_call path)."""
        tool_indices = self._get_tool_indices(tools)
        n = len(text)
        while pos < n:
            nxt = text.find(self.NS + "<", pos)
            if nxt == -1:
                return n
            if text.startswith(self.eot_token, nxt):
                return nxt + len(self.eot_token)
            m = self.invoke_open_re.match(text, nxt)
            if m is None:
                pos = nxt + len(self.NS) + 1  # stray marker at block level
                continue
            items, end, closer = self._scan_invoke_interior(text, m.end())
            func_name = m.group(1)
            if func_name in tool_indices or _should_forward_unknown_tool(func_name):
                args = self._args_from_items(func_name, items, tools)
                calls.append(
                    ToolCallItem(
                        tool_index=len(calls),
                        name=func_name,
                        parameters=json.dumps(args, ensure_ascii=False),
                    )
                )
            else:
                logger.warning(f"Model attempted to call undefined function: {func_name}")
            if closer == "invoke":
                pos = end + len(self._invoke_close)
            elif closer == "eot":
                return end + len(self.eot_token)
            else:
                return n
        return n

    # ---- one-shot -------------------------------------------------------------------
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse EVERY wrapper block (text between and after blocks stays content).
        A block without ``</tool_call>`` (end-of-generation truncation) still
        yields its parseable calls -- recover_truncated_call appends the closer
        and re-enters here."""
        calls: List[ToolCallItem] = []
        normal_parts: List[str] = []
        pos = 0
        while True:
            b = text.find(self.bot_token, pos)
            if b == -1:
                normal_parts.append(text[pos:])
                break
            normal_parts.append(text[pos:b])
            pos = self._parse_block(text, b + len(self.bot_token), tools, calls)
        return StreamingParseResult(
            normal_text="".join(normal_parts).strip(), calls=calls
        )

    # ---- streaming ------------------------------------------------------------------
    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        self._buffer += new_text
        calls: List[ToolCallItem] = []
        normal_parts: List[str] = []
        tool_indices = self._get_tool_indices(tools)

        while True:
            if not self._m3_in_block:
                pos = self._buffer.find(self.bot_token)
                if pos == -1:
                    hold = self._partial_marker_suffix(self._buffer)
                    release = self._buffer[: len(self._buffer) - hold]
                    if release and calls:
                        break  # wire order: text after a call defers to the next step
                    self._buffer = self._buffer[len(release) :]
                    if release:
                        normal_parts.append(release)
                    break
                if pos > 0:
                    if calls:
                        break  # wire order (text precedes a SECOND block)
                    normal_parts.append(self._buffer[:pos])
                    self._buffer = self._buffer[pos:]
                self._m3_in_block = True
                continue

            # In-block: the buffer keeps the opener (finish_streaming and
            # recover_truncated_call key off has_tool_call(buffer)). Markers are
            # interpreted structurally, same as detect_and_parse: the first
            # top-level invoke opener or wrapper closer wins.
            body = self._buffer[len(self.bot_token) :]
            action = None
            scan = 0
            while True:
                nxt = body.find(self.NS + "<", scan)
                if nxt == -1:
                    break
                if body.startswith(self.eot_token, nxt):
                    action = ("eot", nxt, None)
                    break
                m = self.invoke_open_re.match(body, nxt)
                if m is not None:
                    action = ("invoke", nxt, m)
                    break
                scan = nxt + len(self.NS) + 1  # stray marker at block level
            if action is None:
                break  # inside the block, nothing complete yet -> hold
            if action[0] == "eot":
                # Wrapper closed: drop the block, loop back to the idle scan so the
                # residue gets partial-marker holds and second-block detection
                # instead of leaking raw.
                self._buffer = body[action[1] + len(self.eot_token) :]
                self._m3_in_block = False
                continue
            m = action[2]
            items, end, closer = self._scan_invoke_interior(body, m.end())
            if closer != "invoke":
                # Invoke still streaming (or eot-truncated: left to
                # finish_streaming / recover_truncated_call) -> hold.
                break
            func_name = m.group(1)
            self._buffer = self.bot_token + body[end + len(self._invoke_close) :]
            if func_name not in tool_indices and not _should_forward_unknown_tool(
                func_name
            ):
                logger.warning(
                    f"Model attempted to call undefined function: {func_name}"
                )
                continue
            args = self._args_from_items(func_name, items, tools)
            args_json = json.dumps(args, ensure_ascii=False)
            self.current_tool_id += 1
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": func_name,
                "arguments": args,
            }
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id, name=func_name, parameters=""
                )
            )
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id, name=None, parameters=args_json
                )
            )
            self.streamed_args_for_tool[self.current_tool_id] = args_json
            continue

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)


class GptOssDetector(BaseFormatDetector):
    """FreeToken serving adapter for Harmony ``to=functions.*`` tool calls.

    LightLLM preserves these Harmony blocks in its reasoning parser. FreeToken's current
    OpenAI-compatible path does not run a separate reasoning parser before tool parsing, so
    this detector extracts the function call directly from the preserved Harmony text.

    Streaming is a small channel state machine (vLLM's harmony rail is token-id
    driven via StreamableParser; this is the text-marker translation since only
    detokenized text is available here, with the same semantics): text outside
    harmony blocks streams live; a ``commentary ... to=functions.NAME`` header
    opens a call whose body bytes (``<|constrain|>json`` — the raw arguments
    JSON) stream as prefix-stable fragments until a closing boundary; blocks of
    other channels are swallowed (they only reach this detector when no reasoning
    parser runs upstream, matching detect_and_parse which drops them too).
    """

    _BLOCK_OPENERS = ("<|start|>", "<|channel|>")
    _CLOSING_TOKENS = ("<|end|>", "<|return|>", "<|call|>")
    _name_regex = re.compile(r"to=functions\.([^\s<]+)")

    def __init__(self):
        super().__init__()
        self.bot_token = "<|channel|>"
        self.eot_token = "<|end|>"
        self.call_regex = re.compile(
            r"<\|channel\|>commentary\s+to=functions\.([^\s<]+)"
            r".*?<\|message\|>(.*?)(?=<\|end\|>|<\|start\|>|$)",
            re.DOTALL,
        )
        self._mode = "text"  # "text" | "tool_body" | "skip_body"
        self._args_acc = ""

    def has_tool_call(self, text: str) -> bool:
        return "to=functions." in text and "<|message|>" in text

    def block_close_tokens(self) -> tuple:
        return self._CLOSING_TOKENS

    @staticmethod
    def _partial_suffix(text: str, tokens) -> int:
        best = 0
        for tok in tokens:
            for k in range(min(len(tok) - 1, len(text)), best, -1):
                if text.endswith(tok[:k]):
                    best = k
                    break
        return best

    @staticmethod
    def _earliest(text: str, tokens):
        best_pos, best_tok = -1, None
        for tok in tokens:
            pos = text.find(tok)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos, best_tok = pos, tok
        return best_pos, best_tok

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = _first_existing_pos(text, ["<|start|>", "<|channel|>"])
        normal_text = text[:idx].strip() if idx != -1 else text
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=normal_text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        for func_name, payload in self.call_regex.findall(text):
            if func_name not in tool_indices and not _should_forward_unknown_tool(func_name):
                logger.warning(f"Model attempted to call undefined function: {func_name}")
                continue
            args = _parse_first_json_value(payload)
            if args is None:
                continue
            calls.append(
                ToolCallItem(
                    tool_index=len(calls),
                    name=func_name,
                    parameters=json.dumps(args, ensure_ascii=False),
                )
            )
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        self._buffer += new_text
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)
        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        while True:
            buf = self._buffer
            if not buf:
                break
            if self._mode == "text":
                if calls:
                    break  # text after a call defers to the next step (wire order)
                pos, _tok = self._earliest(buf, self._BLOCK_OPENERS)
                if pos == -1:
                    hold = self._partial_suffix(buf, self._BLOCK_OPENERS)
                    release = buf[: len(buf) - hold] if hold else buf
                    if release:
                        normal_parts.append(release)
                        self._buffer = buf[len(release):]
                    break
                if pos > 0:
                    normal_parts.append(buf[:pos])
                    self._buffer = buf[pos:]
                    continue
                msg = buf.find("<|message|>")
                if msg == -1:
                    break  # header still streaming
                header = buf[:msg]
                self._buffer = buf[msg + len("<|message|>"):]
                m = self._name_regex.search(header) if "commentary" in header else None
                func_name = m.group(1) if m else None
                if func_name and (
                    func_name in self._tool_indices or _should_forward_unknown_tool(func_name)
                ):
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id, name=func_name, parameters=""
                        )
                    )
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                    self._args_acc = ""
                    self._mode = "tool_body"
                else:
                    self._mode = "skip_body"
                continue
            # Body modes: stream (tool) or swallow (other channels) until a boundary.
            boundaries = self._CLOSING_TOKENS + self._BLOCK_OPENERS
            pos, tok = self._earliest(buf, boundaries)
            if pos == -1:
                hold = self._partial_suffix(buf, boundaries)
                emit_len = len(buf) - hold
                if emit_len > 0:
                    piece = buf[:emit_len]
                    self._buffer = buf[emit_len:]
                    if self._mode == "tool_body":
                        self._args_acc += piece
                        self.streamed_args_for_tool[self.current_tool_id] += piece
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id, name=None, parameters=piece
                            )
                        )
                break
            piece = buf[:pos]
            if tok in self._CLOSING_TOKENS:
                self._buffer = buf[pos + len(tok):]
            else:
                self._buffer = buf[pos:]  # an opener belongs to the NEXT block
            if self._mode == "tool_body":
                if piece:
                    self._args_acc += piece
                    self.streamed_args_for_tool[self.current_tool_id] += piece
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id, name=None, parameters=piece
                        )
                    )
                parsed_args = _parse_first_json_value(self._args_acc)
                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = (
                    parsed_args if isinstance(parsed_args, dict) else {}
                )
                self.streamed_args_for_tool[self.current_tool_id] = ""
                self.current_tool_id += 1
                self._args_acc = ""
            self._mode = "text"
        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def finish_streaming(self) -> str:
        residual, self._buffer = self._buffer, ""
        mode, self._mode = self._mode, "text"
        if mode != "text" or any(op in residual for op in self._BLOCK_OPENERS):
            return ""
        return residual



class MuseGlimmerDetector(InvokeParamStreamMixin, BaseFormatDetector):
    """Detector for Muse Glimmer's ATEM tool-call protocol.

    Wire format (inside a ``<|start|>assistant to=<tool><|message|> ... <|eot|>``
    channel; the ``MuseGlimmerReasoningParser`` upstream preserves such blocks
    verbatim and unwraps everything else):

    ```
    <atem:function_calls>
    <atem:invoke name="tool.fn">
    <atem:parameter name="param">value</atem:parameter>
    </atem:invoke>
    </atem:function_calls>
    ```

    The invoke/parameter body is the InvokeParamStreamMixin grammar; per the
    template's own contract values are parsed with regexes (not strict XML) and
    string values are NOT whitespace-trimmed (``_ps_trim = ""``). A channel layer
    around the mixin swallows the ``<|start|>...<|message|>`` headers and the
    ``<|eot|>/<|eom|>`` terminators, streams ``to=user`` bodies as text, and drops
    ``to=self`` bodies (they only reach this detector when no reasoning parser
    runs upstream). Channels also open with a mid-stream headerless
    ``to=X<|message|>`` switch (the model leaves a channel without ``<|eom|>``).

    ``header_open`` initializes the detector from the prompt it continues:
    True when the detector receives the raw turn bytes directly (no muse
    reasoning parser stacked above), whose templated prompt ends with
    ``<|start|>assistant`` -- a synthetic ``<|start|>`` is seeded so the bare
    first header goes through the ordinary full-header machinery. False (the
    default) downstream of ``MuseGlimmerReasoningParser``, which delivers tool
    slices with their full headers and everything else already classified.

    ATEM markup is EXECUTED only inside a tool-recipient channel (vLLM's rule).
    A block quoted in a ``to=user`` body -- or the system prompt's own ATEM
    example echoed back -- renders as plain text instead of becoming a real
    call.
    """

    # <atem:function_calls> is not a unique tool opener under the channel-scoped
    # execution rule (a to=user body may quote it), so no checkpoint anchor.
    toolcall_opener = None
    _ps_trim = ""  # "spaces for string values are not stripped" (chat-template contract)
    _ps_missing_type = "string"
    accepts_header_open = True  # FunctionCallParser passes the turn-start state

    def __init__(self, header_open: bool = False):
        super().__init__()
        self.bot_token = "<atem:function_calls>"
        self.eot_token = "</atem:function_calls>"
        self.tool_call_separator = "\n"

        # InvokeParamStreamMixin grammar
        self._ps_outer_open = "<atem:function_calls>"
        self._ps_outer_close = "</atem:function_calls>"
        self._ps_invoke_open_prefix = "<atem:invoke"
        self._ps_invoke_open_re = re.compile(r'<atem:invoke name="([^"]*)">')
        self._ps_invoke_close = "</atem:invoke>"
        self._ps_param_open_prefix = "<atem:parameter"
        self._ps_param_open_re = re.compile(r'<atem:parameter name="([^"]*)">')
        self._ps_param_close = "</atem:parameter>"
        self._ps_reset()

        # Channel layer state: "text" (to=user / outside channels) | "skip"
        # (non-user, non-tool channel body) | "tool" (tool channel body,
        # delegated to the mixin).
        self._ch_mode = "text"
        self._header_open = header_open
        # The seed is synthetic: if it is ever ruled a non-header, the marker
        # text itself is not delivered -- the model never emitted it.
        self._buffer = ATEM_START if header_open else ""
        self._synthetic_open = header_open
        # A channel boundary fired mid-invoke: until the NEXT boundary, incoming
        # text is that broken channel's residue (raw ATEM markup), not content.
        self._truncated_channel = False
        # Dropped tool-channel prose, accumulated for ONE warning per channel
        # (a per-fragment warning logs once per generated character).
        self._dropped_prose: List[str] = []

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def block_close_tokens(self) -> tuple:
        return (self.eot_token,) + ATEM_CLOSING_TOKENS

    def _ps_canonical_name(self, name: str) -> str:
        """The template renders a bare-name tool's recipient namespace as
        ``name.*``, so the model emits ``get_weather.get_weather``: collapse the
        doubled form to the registered head (never when the doubled form itself
        is a registered tool)."""
        head, dot, tail = name.partition(".")
        indices = getattr(self, "_tool_indices", {})
        if dot and head == tail and head in indices and name not in indices:
            return head
        return name

    # ------------------------------------------------------------------
    # streaming
    # ------------------------------------------------------------------
    def _enter_recipient(self, recipient: str) -> None:
        if recipient == "self":
            self._ch_mode = "skip"
        elif recipient == "user":
            self._ch_mode = "text"
        else:
            self._ch_mode = "tool"

    @staticmethod
    def _channel_boundary(buf: str):
        """Earliest channel boundary in ``buf``: ``(pos, kind, payload)`` with kind
        "closer" (payload = token), "start", or "inline" (payload = the headerless
        ``to=X<|message|>`` match); ``(-1, None, None)`` when none is present."""
        best = (len(buf) + 1, None, None)
        for tok in ATEM_CLOSING_TOKENS:
            pos = buf.find(tok)
            if pos != -1 and pos < best[0]:
                best = (pos, "closer", tok)
        pos = buf.find(ATEM_START)
        if pos != -1 and pos < best[0]:
            best = (pos, "start", None)
        m = ATEM_INLINE_HEADER_RE.search(buf)
        if m is not None and m.start() < best[0]:
            best = (m.start(), "inline", m)
        if best[1] is None:
            return -1, None, None
        return best

    def _finalize_truncated_invoke(self, reason: str = "tool channel closed") -> List[ToolCallItem]:
        """The channel (or the whole stream) ended while the ATEM machinery was
        mid-flight. If an invoke already streamed its Start, close its argument
        JSON so the streamed fragments stay valid (never emit broken arguments),
        ledger the completed parameters, and advance the ordinal so the next
        call cannot merge into this one. Anything less than an open invoke is
        markup debris: warn and reset."""
        self._warn_dropped_prose()
        mode = getattr(self, "_ps_mode", "idle")
        if mode == "idle":
            return []
        self._truncated_channel = True  # what follows is the broken channel's residue
        calls: List[ToolCallItem] = []
        if mode in ("invoke", "pstr", "pbuf"):
            ledger = self.streamed_args_for_tool[self.current_tool_id]
            if mode == "pstr":
                frag = '"}'  # the ledger ends inside an open string value
            elif ledger:
                frag = "}"  # ends after a complete "key": value pair
            else:
                frag = "{}"  # invoke opened, no parameter emitted yet
            self.streamed_args_for_tool[self.current_tool_id] += frag
            calls.append(
                ToolCallItem(tool_index=self.current_tool_id, name=None, parameters=frag)
            )
            try:
                parsed = json.loads(self.streamed_args_for_tool[self.current_tool_id])
            except (json.JSONDecodeError, ValueError):
                parsed = {}
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = (
                parsed if isinstance(parsed, dict) else {}
            )
            self.streamed_args_for_tool[self.current_tool_id] = ""
            self.current_tool_id += 1
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")
            logger.warning("muse_glimmer: %s mid-invoke; arguments truncated", reason)
        else:  # block / invoke_skip / pskip: no open call, just partial markup
            logger.warning("muse_glimmer: %s mid-block; partial tool markup dropped", reason)
        self._ps_reset()
        return calls

    def finalize_streaming(self) -> List[ToolCallItem]:
        """End-of-stream hook (FunctionCallParser.finalize_stream): the generation
        ran out (max_tokens) while an invoke was still streaming -- close it the
        same way a channel boundary would, so the client's concatenated argument
        fragments end as valid JSON instead of an unterminated string."""
        return self._finalize_truncated_invoke("generation ended")

    # The truncated channel's trailing-markup shapes: what a broken-off invoke
    # leaves behind when the model closes the tags it opened.
    _CLOSING_MARKUP = ("</atem:parameter>", "</atem:invoke>", "</atem:function_calls>")

    @classmethod
    def _channel_residue_end(cls, buf: str) -> tuple[int, bool]:
        """Length of the leading run of ATEM closing markup (closing tags plus
        whitespace) in ``buf``, and whether the scan DECIDED. False means the
        buffer ends inside the run or a partial tag: hold for more input.

        Known cost of shape-based dropping: a reply that literally BEGINS with a
        complete closing tag ("</atem:parameter> is the closing tag") loses that
        leading tag. Partial-tag-shaped prose ("</a", "</atem") is held and
        released intact, so no ordinary word can be eaten -- only a verbatim
        full tag right after a truncated invoke, which is indistinguishable from
        the residue this filter exists to drop."""
        i = 0
        while i < len(buf):
            if buf[i] in " \t\r\n":
                i += 1
                continue
            matched = False
            for tag in cls._CLOSING_MARKUP:
                if buf.startswith(tag, i):
                    i += len(tag)
                    matched = True
                    break
            if matched:
                continue
            rest = buf[i:]
            if any(tag.startswith(rest) for tag in cls._CLOSING_MARKUP):
                return i, False  # a partial closing tag at the end: hold
            return i, True  # first non-markup character: the residue ended
        return i, False  # consumed the whole buffer; the run may continue

    def _drop_channel_prose(self, text: str) -> None:
        """Text inside a tool channel that is not ATEM markup is discarded (only
        tool-recipient markup is executed; the template puts nothing else there).
        Accumulated and logged ONCE per channel by ``_warn_dropped_prose`` --
        token-by-token serving would otherwise warn per generated character."""
        if text and text.strip():
            self._dropped_prose.append(text)

    def _warn_dropped_prose(self) -> None:
        if not self._dropped_prose:
            return
        dropped = "".join(self._dropped_prose)
        self._dropped_prose = []
        logger.warning(
            "muse_glimmer: dropped %d chars of non-markup text inside a tool channel: %.120r",
            len(dropped),
            dropped,
        )

    def _run_mixin(self, atem_part: str, remainder: str, tools: List[Tool]) -> StreamingParseResult:
        """Feed ``atem_part`` through the invoke/parameter machinery; whatever the
        mixin leaves unconsumed stays buffered ahead of ``remainder``."""
        self._buffer = atem_part
        result = InvokeParamStreamMixin.parse_streaming_increment(self, "", tools)
        self._buffer += remainder
        return result

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        self._buffer += new_text
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)
        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        while True:
            buf = self._buffer
            if not buf:
                break

            if self._ch_mode == "text":
                if calls:
                    break  # text after a call defers to the next step (wire order)
                if self._truncated_channel:
                    # After a mid-invoke channel break, drop the broken channel's
                    # trailing markup BY SHAPE (closing tags + whitespace) and
                    # clear at the first non-markup character -- a real reply
                    # following the break must flow, not be swallowed until some
                    # later boundary that the production pipeline never delivers.
                    consumed, decided = self._channel_residue_end(buf)
                    if consumed:
                        self._buffer = buf[consumed:]
                    if not decided:
                        break
                    self._truncated_channel = False
                    self._warn_dropped_prose()
                    continue
                pos, kind, payload = self._channel_boundary(buf)
                if pos == -1:
                    hold = atem_hold_len(buf)
                    release = buf[: len(buf) - hold] if hold else buf
                    if release:
                        normal_parts.append(release)
                        self._buffer = buf[len(release):]
                    break
                if pos > 0:
                    normal_parts.append(buf[:pos])
                    self._buffer = buf[pos:]
                    continue
                if kind == "closer":
                    self._buffer = buf[len(payload):]  # stray terminator: drop
                    continue
                if kind == "inline":
                    self._enter_recipient(payload.group(1))
                    self._buffer = buf[payload.end():]
                    continue
                # kind == "start": parse the channel header. While the header-open
                # seed is unconsumed it is the leftmost marker, so each consuming
                # branch owns the synthetic flag: a synthetic marker ruled a
                # non-header is dropped, never delivered (the model never emitted it).
                synthetic = self._synthetic_open
                msg = buf.find(ATEM_MESSAGE)
                if atem_marker_inside(
                    buf, len(ATEM_START), msg if msg != -1 else len(buf)
                ):
                    # A control token inside the candidate: headers never contain
                    # markers, so this <|start|> is literal text (a synthetic
                    # seed is simply dropped) -- mirrors the reasoning parser.
                    if not synthetic:
                        normal_parts.append(ATEM_START)
                    self._synthetic_open = False
                    self._buffer = buf[len(ATEM_START):]
                    continue
                if msg == -1:
                    # +len(<|message|>) slack: a protocol-legal header whose marker
                    # is still mid-arrival must not be cut at the nominal span.
                    if len(buf) > len(ATEM_START) + ATEM_HEADER_SPAN + len(ATEM_MESSAGE):
                        # Past a plausible header span the marker cannot open a
                        # header anymore: release it as literal text and move on
                        # (mirrors the reasoning parser's bound).
                        if not synthetic:
                            normal_parts.append(ATEM_START)
                        self._synthetic_open = False
                        self._buffer = buf[len(ATEM_START):]
                        continue
                    break  # header still streaming
                if msg - len(ATEM_START) > ATEM_HEADER_SPAN:
                    # The found <|message|> is too far away to belong to THIS
                    # marker (a stray literal <|start|> followed by junk, then the
                    # NEXT segment's real header): the marker is literal text.
                    if not synthetic:
                        normal_parts.append(ATEM_START)
                    self._synthetic_open = False
                    self._buffer = buf[len(ATEM_START):]
                    continue
                m = ATEM_RECIPIENT_RE.search(buf[len(ATEM_START):msg])
                self._enter_recipient(m.group(1) if m else "user")
                self._synthetic_open = False
                self._buffer = buf[msg + len(ATEM_MESSAGE):]
                continue

            pos, kind, payload = self._channel_boundary(buf)

            if self._ch_mode == "skip":
                if pos == -1:
                    hold = atem_hold_len(buf)
                    self._buffer = buf[len(buf) - hold:] if hold else ""
                    break
                # Drop the skipped body; transition exactly like text mode would.
                self._buffer = buf[pos:]
                self._ch_mode = "text"
                continue

            # self._ch_mode == "tool": the body up to the channel boundary belongs
            # to the ATEM machinery; text the mixin releases inside the channel is
            # not executed (only tool-recipient markup is) and is dropped.
            if pos == -1:
                hold = atem_hold_len(buf)
                atem_part = buf[: len(buf) - hold] if hold else buf
                if not atem_part:
                    break
                result = self._run_mixin(atem_part, buf[len(atem_part):], tools)
                self._drop_channel_prose(result.normal_text)
                calls.extend(result.calls)
                if self._buffer == buf:
                    break  # no progress: the mixin is holding a partial marker
                continue
            result = self._run_mixin(buf[:pos], buf[pos:], tools)
            self._drop_channel_prose(result.normal_text)
            calls.extend(result.calls)
            if self._buffer != buf:
                # The mixin advanced (it stops after each call's close, wire-order):
                # more blocks may still precede the boundary -- a second invoke block
                # in the same undrained buffer must parse, not vanish as debris.
                continue
            # Inert: nothing before the boundary the machinery can still consume.
            # Finalize a mid-flight invoke (its Start already streamed) instead of
            # letting the next channel's call merge into it, then transition; any
            # residue ahead of the boundary is markup debris and drops with it.
            calls.extend(self._finalize_truncated_invoke())
            if kind == "closer":
                self._buffer = buf[pos + len(payload):]
                self._ch_mode = "text"
                # NOTE: the truncation mark (when finalize set it) survives the
                # closer on purpose: on the production pipeline the broken
                # channel's markup residue arrives AFTER the reasoning parser's
                # synthetic terminator. Text mode clears the mark by SHAPE -- at
                # the first non-markup character -- so a real reply flows
                # immediately while trailing closing tags are dropped.
            elif kind == "inline":
                self._enter_recipient(payload.group(1))
                self._buffer = buf[payload.end():]
            else:  # abutting <|start|>: reprocess it in text mode
                self._buffer = buf[pos:]
                self._ch_mode = "text"
            continue
        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def finish_streaming(self) -> str:
        self._warn_dropped_prose()
        residual, self._buffer = self._buffer, ATEM_START if self._header_open else ""
        ch_mode, self._ch_mode = self._ch_mode, "text"
        synthetic, self._synthetic_open = self._synthetic_open, self._header_open
        truncated, self._truncated_channel = self._truncated_channel, False
        ps_mode = getattr(self, "_ps_mode", "idle")
        self._ps_reset()
        if ch_mode != "text" or ps_mode != "idle" or truncated:
            return ""
        if synthetic:
            residual = residual[len(ATEM_START):]  # the seed: never model output
        # At end of stream a <|start|> that never received its <|message|> is NOT
        # a header: deliver the text, drop only the marker(s). Any capped discard
        # here diverged from the layer above -- its span runs on raw bytes while
        # this layer sees closer-stripped text, so the two edges can never agree;
        # removing the drop window entirely makes the agreement trivial.
        residual = residual.replace(ATEM_START, "")
        for tok in ATEM_CLOSING_TOKENS:
            residual = residual.replace(tok, "")
        if self.prev_tool_call_arr and residual.strip() == "":
            return ""
        return residual

    # ------------------------------------------------------------------
    # one-shot
    # ------------------------------------------------------------------
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-shot parse by replaying the streaming machinery on a fresh clone,
        so both paths share one definition of channel classification (a raw
        ``to=self`` body never leaks into content), execution scoping and value
        semantics. Streamed fragments are reassembled into complete calls; a
        call truncated by end-of-input closes from the parse ledger, exactly as
        the serving layer would."""
        clone = type(self)(header_open=self._header_open)
        raw_calls: List[ToolCallItem] = []
        normal_parts: List[str] = []
        result = clone.parse_streaming_increment(text, tools)
        normal_parts.append(result.normal_text)
        raw_calls.extend(result.calls)
        while True:  # drain wire-order deferrals (one call boundary per step)
            result = clone.parse_streaming_increment("", tools)
            if not result.normal_text and not result.calls:
                break
            normal_parts.append(result.normal_text)
            raw_calls.extend(result.calls)
        raw_calls.extend(clone._finalize_truncated_invoke("input ended"))  # mid-invoke
        normal_parts.append(clone.finish_streaming())

        order: List[int] = []
        by_index: Dict[int, Dict[str, str]] = {}
        for item in raw_calls:
            if item.name is not None and item.tool_index not in by_index:
                by_index[item.tool_index] = {"name": item.name, "params": ""}
                order.append(item.tool_index)
            if item.parameters and item.tool_index in by_index:
                by_index[item.tool_index]["params"] += item.parameters
        calls = [
            ToolCallItem(
                tool_index=ordinal,
                name=by_index[idx]["name"],
                parameters=by_index[idx]["params"] or "{}",
            )
            for ordinal, idx in enumerate(order)
        ]
        return StreamingParseResult(normal_text="".join(normal_parts).strip(), calls=calls)


class FunctionCallParser:
    """
    Parser for function/tool calls in model outputs.

    This class handles both streaming and non-streaming parsing of function calls using a detector.
    In streaming scenarios, each time new_text is received, it calls detector.parse_streaming_increment
    and returns the resulting normal_text and calls to the upper layer (or SSE).
    """

    ToolCallParserEnum: Dict[str, Type[BaseFormatDetector]] = {
        "deepseekv32": DeepSeekV32Detector,
        "gemma4": Gemma4Detector,
        "gpt-oss": GptOssDetector,
        "gpt_oss": GptOssDetector,
        "glm47": Glm47Detector,
        "llama3": Llama32Detector,
        "minimax": MiniMaxDetector,
        "minimax_m3": MiniMaxM3Detector,
        "mistral": MistralDetector,
        "muse_glimmer": MuseGlimmerDetector,
        "poolside_v1": PoolsideV1Detector,
        "qwen": Qwen25Detector,
        "qwen25": Qwen25Detector,
        "qwen3_coder": Qwen3CoderDetector,
    }

    def __init__(self, tools: List[Tool], tool_call_parser: str, turn_starts_open: bool = False):
        detector: Type[BaseFormatDetector] = None
        detector_class = self.ToolCallParserEnum.get(tool_call_parser)
        if detector_class is None:
            raise ValueError(f"Unsupported tool_call_parser: {tool_call_parser}")
        if getattr(detector_class, "accepts_header_open", False):
            # Turn-start parse state read from the prompt: the templated chat
            # prompt ends inside a channel header (``<|start|>assistant``), so a
            # detector that receives the raw turn bytes starts header-open.
            detector = detector_class(header_open=turn_starts_open)
        else:
            detector = detector_class()

        self.detector = detector
        self.tools = _coerce_tools(tools)

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a tool call in the format supported by this parser.
        This delegates to the detector's implementation.

        Args:
            text: The text to check for tool calls

        Returns:
            True if the text contains a tool call, False otherwise
        """
        if not self.tools:
            return False
        return self.detector.has_tool_call(text)

    def parse_non_stream(self, full_text: str) -> StreamingParseResult:
        """
        One-time parsing of the full text to extract tool calls.

        Args:
            full_text: The complete text to parse

        Returns:
            StreamingParseResult with normal_text and parsed calls.
        """
        if not self.tools:
            return StreamingParseResult(normal_text=full_text, calls=[])
        parsed_result = self.detector.detect_and_parse(full_text, self.tools)
        tool_call_list = parsed_result.calls
        if tool_call_list:
            # Keep text after the LAST tool block: the streaming path emits it, so
            # one-shot parsing must agree instead of silently dropping it.
            tail_start = -1
            for tok in self.detector.block_close_tokens():
                pos = full_text.rfind(tok)
                if pos != -1:
                    tail_start = max(tail_start, pos + len(tok))
            if tail_start != -1:
                tail = full_text[tail_start:]
                normal = parsed_result.normal_text or ""
                # Only plain text: an unterminated final block would make the
                # "last closer" precede it and leak markup into content.
                if tail.strip() and tail.strip() not in normal and not self.detector.has_tool_call(tail):
                    parsed_result.normal_text = normal + tail
            return parsed_result
        else:
            return StreamingParseResult(normal_text=full_text, calls=[])

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, list[ToolCallItem]]:
        """
        Streaming incremental parsing of chunks of text as they arrive.

        Args:
            chunk_text: The new chunk of text to parse

        Returns:
            A tuple containing:
            - The normal text that should be displayed to the user
            - A list of tool calls parsed from the chunk
        """
        normal_parts: list[str] = []
        final_calls: list[ToolCallItem] = []
        for kind, payload in self.parse_stream_events(chunk_text):
            if kind == "text":
                normal_parts.append(payload)
            else:
                final_calls.extend(payload)
        return "".join(normal_parts), final_calls

    def parse_stream_events(self, chunk_text: str) -> list[tuple[str, Any]]:
        """Ordered streaming parse: ``[("text", str) | ("calls", [ToolCallItem]), ...]``
        segments in generation order. Detectors return text-before-calls within one
        increment; segments across drain iterations keep the true interleaving
        (pre-text, call, trailing text) that a flat (text, calls) pair loses."""
        if not self.tools:
            return [("text", chunk_text)] if chunk_text else []
        segments: list[tuple[str, Any]] = []

        # Drain loop: an increment may return after consuming only part of its buffer
        # (e.g. one complete invoke of several, leaving the rest and the closing tag
        # behind). Re-feed "" until the detector stops making progress.
        text = chunk_text
        for _ in range(16):
            sp_result = self.detector.parse_streaming_increment(text, self.tools)
            if sp_result.normal_text:
                segments.append(("text", sp_result.normal_text))
            if sp_result.calls:
                segments.append(("calls", sp_result.calls))
            if text == "" and not sp_result.calls and not sp_result.normal_text:
                break
            text = ""
        return segments

    def recover_truncated_call(self) -> List[ToolCallItem]:
        """Best-effort parse of a tool call cut off by end-of-generation while its
        tag block was still open: append the closing tag and re-parse. Consumes the
        detector buffer on success; returns [] when nothing recoverable."""
        det = self.detector
        buf = det._buffer
        if not buf or not det.eot_token or not det.has_tool_call(buf):
            return []
        try:
            parsed = det.detect_and_parse(buf + det.eot_token, self.tools)
        except Exception:  # noqa: BLE001 — recovery is best-effort by definition
            return []
        if parsed.calls:
            det._buffer = ""
        return parsed.calls

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        normal_text, calls = self.parse_stream_chunk(new_text)
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def supports_streaming(self) -> bool:
        """Whether the underlying detector can parse incrementally; when False the
        serving layer must buffer the full generation and use parse_non_stream."""
        return bool(self.detector.supports_streaming)

    def args_fragments_prefix_stable(self) -> bool:
        """Whether streamed argument fragments always concatenate to a prefix of the
        final arguments JSON (safe for clients that concatenate fragments)."""
        return bool(self.detector.args_fragments_prefix_stable)

    def finish_stream(self) -> str:
        """End-of-stream drain: residual buffered text that is plain content (empty
        when the buffer holds unfinished tool-call markup)."""
        return self.detector.finish_streaming()

    def unstreamed_arguments(self, tool_ordinal: int) -> str | None:
        """Best-effort complete arguments JSON for a call whose argument stream was cut
        short (generation truncated mid-call), from the detector's partial-parse state."""
        arr = self.detector.prev_tool_call_arr
        if 0 <= tool_ordinal < len(arr):
            args = arr[tool_ordinal].get("arguments")
            # `is not None`, not truthiness: {} is a legitimate ledger entry (an
            # empty-argument call) and must serialize instead of falling through.
            if args is not None:
                return json.dumps(args, ensure_ascii=False)
        return None

    def finalize_stream(self) -> List[ToolCallItem]:
        """End-of-stream finalize: argument fragments that CLOSE a call cut off
        mid-arguments, from detectors that can (muse_glimmer closes the streamed
        JSON so the fragments the client concatenated stay valid). The serving
        layer routes these before it closes the open call; [] for detectors
        without the hook."""
        finalize = getattr(self.detector, "finalize_streaming", None)
        return finalize() if finalize is not None else []


SUPPORTED_TOOL_CALL_PARSERS = list(FunctionCallParser.ToolCallParserEnum.keys())


def toolcall_opener_for(tool_call_parser: str) -> str | None:
    """The configured parser's unique tool-call opening marker, or None when the format has
    no single unique opener (see ``BaseFormatDetector.toolcall_opener``)."""
    detector = FunctionCallParser.ToolCallParserEnum.get(tool_call_parser)
    return detector.toolcall_opener if detector is not None else None


def _coerce_tools(tools: List[Any] | None) -> List[Tool]:
    return [_coerce_tool(tool) for tool in tools or []]


def _coerce_tool(tool: Any) -> Tool:
    if isinstance(tool, Tool):
        return tool
    if isinstance(tool, dict):
        return Tool.model_validate(tool)
    function = getattr(tool, "function", None)
    if function is None:
        raise TypeError(f"Unsupported tool schema: {tool!r}")
    return Tool(
        type=getattr(tool, "type", "function"),
        function=Function(
            name=getattr(function, "name", None),
            description=getattr(function, "description", None),
            parameters=getattr(function, "parameters", None),
        ),
    )
