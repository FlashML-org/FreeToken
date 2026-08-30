"""Reasoning-content parsing for OpenAI-compatible responses.

Separates a model completion into ``reasoning_content`` (the ``<think>…</think>``
block) and ``content`` (everything else), running *before* the tool-call parser
just like SGLang's ``serving_chat`` does. Adapted from SGLang's
``python/sglang/srt/parser/reasoning_parser.py`` and trimmed to what FreeToken's
text-only edge models need.

DeepSeek-V4 / V3.2 inject the opening ``<think>`` at the generation prompt, so
the model output *starts inside* the reasoning block and only ever emits the
closing ``</think>`` (see ``tokenizer/tokenize.py``). When
thinking is active the caller must construct the parser with
``force_reasoning=True`` so the leading text is attributed to reasoning. The
model also sometimes skips ``</think>`` and jumps straight to the ``｜DSML｜``
tool block; ``tool_start_token`` ends reasoning at the first DSML marker and
preserves the block for the tool-call parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

# DeepSeek-V4 / V3.2 protocol special-token strings. The detokenizer decodes with
# skip_special_tokens=False (so the DSML tool markers survive for the tool parser),
# which means BOS/EOS and the think tags can leak into the text. The reasoning and
# tool parsers strip <think>/</think> and the DSML block themselves; this set is the
# residual cleanup applied to the final content / reasoning fields.
BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
DSML_TOKEN = "｜DSML｜"

DSV4_SPECIAL_TOKENS: List[str] = [
    BOS_TOKEN,
    EOS_TOKEN,
    THINK_START_TOKEN,
    THINK_END_TOKEN,
]


def strip_special_tokens(text: str, tokens: List[str]) -> str:
    """Remove special-token strings that leaked through the decode.

    Empty entries in ``tokens`` are ignored so they cannot corrupt the text.
    """
    for token in tokens:
        if token:
            text = text.replace(token, "")
    return text


HARMONY_CHANNEL = "<|channel|>"
HARMONY_MESSAGE = "<|message|>"
# Tokens that bound a Harmony channel segment's body.
HARMONY_BOUNDARY_TOKENS = (
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|start|>",
    "<|channel|>",
)
# Every Harmony control token, used to hold back a partial marker mid-stream.
HARMONY_ALL_TOKENS = (HARMONY_CHANNEL, HARMONY_MESSAGE) + HARMONY_BOUNDARY_TOKENS


def _longest_harmony_partial_suffix(text: str) -> int:
    """Length of the longest suffix of ``text`` that is a *proper* prefix of any
    Harmony control token. Lets a marker split across stream chunks reassemble on
    the next chunk instead of leaking into the emitted body."""
    best = 0
    for tok in HARMONY_ALL_TOKENS:
        for k in range(min(len(tok) - 1, len(text)), best, -1):
            if text.endswith(tok[:k]):
                best = k
                break
    return best


@dataclass
class ReasoningParseResult:
    """Result of (incremental or one-shot) reasoning extraction."""

    reasoning_text: str = ""
    normal_text: str = ""


class BaseReasoningParser:
    """One-shot and streaming reasoning extraction.

    Args:
        think_start_token: Opening reasoning marker (e.g. ``<think>``).
        think_end_token: Closing reasoning marker (e.g. ``</think>``).
        force_reasoning: If True, the completion is assumed to *start* inside a
            reasoning block even when no opening marker is emitted (the dsv4
            implicit-open convention).
        stream_reasoning: If True, reasoning is streamed incrementally; if False,
            it is buffered until the closing marker.
        tool_start_token: Optional marker that ends reasoning without a closing
            ``think_end_token`` (the block is preserved in ``normal_text``).
    """

    # Max bytes a suspected tool block is held while waiting for a possible later
    # </think> that would reclaim it as reasoning. Short quoted markers stay
    # recoverable; anything longer commits to "real tool call" and streams live.
    TOOL_HOLD_MAX = 512

    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
        tool_start_token: Optional[str] = None,
    ) -> None:
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self.tool_start_token = tool_start_token
        self.force_reasoning = force_reasoning
        self.stream_reasoning = stream_reasoning

        self._in_reasoning = force_reasoning
        self._buffer = ""
        self._stripped_think_start = False

    def detect_and_parse(self, text: str) -> ReasoningParseResult:
        """One-shot parse of a complete completion."""
        in_reasoning = self._in_reasoning or self.think_start_token in text
        if not in_reasoning:
            return ReasoningParseResult(normal_text=text)

        processed_text = text.replace(self.think_start_token, "").strip()

        if self.think_end_token not in processed_text:
            # No closing </think>. If a tool block starts, reasoning ends there
            # and the block stays in normal_text for the tool-call parser.
            if (
                self.tool_start_token is not None
                and self.tool_start_token in processed_text
            ):
                tool_idx = processed_text.find(self.tool_start_token)
                return ReasoningParseResult(
                    reasoning_text=processed_text[:tool_idx].strip(),
                    normal_text=processed_text[tool_idx:],
                )
            # Otherwise reasoning was truncated before the end marker.
            return ReasoningParseResult(reasoning_text=processed_text)

        reasoning_text, normal_text = processed_text.split(self.think_end_token, 1)
        return ReasoningParseResult(
            reasoning_text=reasoning_text, normal_text=normal_text.strip()
        )

    def parse_streaming_increment(self, new_text: str) -> ReasoningParseResult:
        """Incremental parse.

        Chunks are detokenizer deltas (roughly token-aligned). A marker's leading
        ``<`` may arrive glued to preceding text in one token (e.g. ``.<`` before
        ``｜DSML｜tool_calls>``), so we hold back any trailing *partial* of a
        tracked token (see ``_split_trailing_partial``) and reassemble it on the
        next chunk rather than streaming it out and losing the marker boundary.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Strip an explicit opening <think> if a complete one is present.
        if not self._stripped_think_start and self.think_start_token in current_text:
            current_text = current_text.replace(self.think_start_token, "", 1)
            self._stripped_think_start = True
            self._in_reasoning = True

        # A complete </think> ends reasoning and WINS over any tool marker -- this
        # is what keeps a ｜DSML｜ literal quoted inside reasoning as reasoning.
        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)
            self._buffer = ""
            self._in_reasoning = False
            return ReasoningParseResult(
                reasoning_text=current_text[:end_idx].rstrip(),
                normal_text=current_text[end_idx + len(self.think_end_token) :],
            )

        # A complete tool marker while still reasoning => the model skipped
        # </think>. HOLD from the marker (stream the reasoning before it) and keep
        # buffering; a later </think> reclaims it as reasoning, else flush() emits
        # the held block as content for the tool-call parser. The hold is BOUNDED:
        # past TOOL_HOLD_MAX the block is almost certainly a real tool call (not a
        # quoted marker inside reasoning), so commit "reasoning ended here" and
        # release it live — otherwise a long tool call streams nothing until
        # end-of-generation and trips client idle timeouts (codex: 300s).
        if self._in_reasoning and self.tool_start_token and self.tool_start_token in current_text:
            tool_idx = current_text.find(self.tool_start_token)
            self._buffer = current_text[tool_idx:]
            if len(self._buffer) > self.TOOL_HOLD_MAX:
                held = self._buffer
                self._buffer = ""
                self._in_reasoning = False
                return ReasoningParseResult(
                    reasoning_text=current_text[:tool_idx], normal_text=held
                )
            return ReasoningParseResult(reasoning_text=current_text[:tool_idx])

        # No complete marker. Hold back any trailing partial of a tracked token so
        # a marker split across chunks -- e.g. its leading '<' glued to preceding
        # text as one token (".<", "><") -- is reassembled on the next chunk.
        safe, held = self._split_trailing_partial(current_text)
        self._buffer = held
        if self._in_reasoning and not self.stream_reasoning:
            self._buffer = current_text  # accumulate until the closing marker
            return ReasoningParseResult()
        if self._in_reasoning:
            return ReasoningParseResult(reasoning_text=safe)
        return ReasoningParseResult(normal_text=safe)

    def _split_trailing_partial(self, text: str) -> tuple[str, str]:
        """Split off the longest suffix of ``text`` that is a *proper* prefix of a
        tracked token, returning ``(safe, held)``. Lets a marker whose leading
        characters are glued to preceding text reassemble on the next chunk
        instead of being streamed out and breaking marker detection."""
        tokens = [self.think_end_token]
        if not self._stripped_think_start:
            tokens.append(self.think_start_token)
        if self.tool_start_token:
            tokens.append(self.tool_start_token)
        best = 0
        for tok in tokens:
            for k in range(min(len(tok) - 1, len(text)), best, -1):
                if text.endswith(tok[:k]):
                    best = k
                    break
        if best == 0:
            return text, ""
        return text[:-best], text[-best:]

    def flush(self) -> ReasoningParseResult:
        """Drain any residue left in the buffer at end-of-stream.

        A held tool block (the model skipped </think> and ran straight into the
        ｜DSML｜ block) is attributed to content so the tool-call parser sees it;
        any other residue is attributed by the current reasoning state (a
        truncated </think> prefix stays reasoning; a trailing '<' of normal
        content stays content). Without this, text stuck in the prefix buffer or
        a held block would be silently dropped from the streamed response.
        """
        buf = self._buffer
        self._buffer = ""
        if not buf:
            return ReasoningParseResult()
        if self.tool_start_token and buf.lstrip().startswith(self.tool_start_token):
            self._in_reasoning = False
            return ReasoningParseResult(normal_text=buf)
        if self._in_reasoning:
            return ReasoningParseResult(reasoning_text=buf)
        return ReasoningParseResult(normal_text=buf)


class DeepSeekV32ReasoningParser(BaseReasoningParser):
    """Reasoning parser for DeepSeek-V4 and DeepSeek-V3.2 (same ``<think>`` +
    ``｜DSML｜`` protocol). ``force_reasoning`` defaults to True because these
    checkpoints start their output inside the reasoning block."""

    def __init__(
        self, force_reasoning: bool = True, stream_reasoning: bool = True
    ) -> None:
        super().__init__(
            think_start_token=THINK_START_TOKEN,
            think_end_token=THINK_END_TOKEN,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            # First DSML structural marker (tool_calls / function_calls / invoke)
            # ends reasoning when </think> was skipped.
            tool_start_token=f"<{DSML_TOKEN}",
        )


class GptOssHarmonyReasoningParser(BaseReasoningParser):
    """Reasoning parser for gpt-oss Harmony output.

    Routes the ``analysis`` channel to ``reasoning_text`` and the ``final``
    channel (unwrapped) to ``normal_text``. A ``commentary ... to=functions.*``
    tool-call block is preserved verbatim (markers included) in ``normal_text``
    so the downstream ``GptOssDetector`` tool parser can still extract it. All
    three methods are overridden; the base ``<think>`` machinery is unused.
    """

    def __init__(self, force_reasoning: bool = False, stream_reasoning: bool = True) -> None:
        super().__init__(
            think_start_token="",
            think_end_token="",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )
        self._buffer = ""
        self._emitted_reasoning = 0
        self._emitted_content = 0

    def detect_and_parse(self, text: str) -> ReasoningParseResult:
        if HARMONY_CHANNEL not in text:
            return ReasoningParseResult(normal_text=text)
        reasoning, content = self._scan(text, hold_partial=False)
        return ReasoningParseResult(
            reasoning_text=reasoning.strip(), normal_text=content.strip()
        )

    def parse_streaming_increment(self, new_text: str) -> ReasoningParseResult:
        self._buffer += new_text
        reasoning, content = self._scan(self._buffer, hold_partial=True)
        result = ReasoningParseResult(
            reasoning_text=reasoning[self._emitted_reasoning :],
            normal_text=content[self._emitted_content :],
        )
        self._emitted_reasoning = len(reasoning)
        self._emitted_content = len(content)
        return result

    def flush(self) -> ReasoningParseResult:
        reasoning, content = self._scan(self._buffer, hold_partial=False)
        result = ReasoningParseResult(
            reasoning_text=reasoning[self._emitted_reasoning :],
            normal_text=content[self._emitted_content :],
        )
        self._emitted_reasoning = len(reasoning)
        self._emitted_content = len(content)
        return result

    def _segment_end(self, text: str, start: int) -> tuple[int, str | None]:
        """Return ``(end_index, matched_token)`` for a segment body starting at
        ``start``: the earliest boundary token and its string, or
        ``(len(text), None)`` when none is present yet (the still-streaming last
        segment)."""
        best_pos: int | None = None
        best_tok: str | None = None
        for tok in HARMONY_BOUNDARY_TOKENS:
            pos = text.find(tok, start)
            if pos != -1 and (best_pos is None or pos < best_pos):
                best_pos = pos
                best_tok = tok
        if best_pos is None:
            return len(text), None
        return best_pos, best_tok

    # Boundary tokens that CLOSE a segment (the token itself belongs to the
    # block and must be included in the verbatim slice for tool blocks).
    _CLOSING_BOUNDARY_TOKENS = frozenset(("<|end|>", "<|return|>", "<|call|>"))

    def _scan(self, text: str, *, hold_partial: bool) -> tuple[str, str]:
        reasoning: list[str] = []
        content: list[str] = []
        i = 0
        while True:
            ch = text.find(HARMONY_CHANNEL, i)
            if ch == -1:
                break
            msg = text.find(HARMONY_MESSAGE, ch + len(HARMONY_CHANNEL))
            if msg == -1:
                break  # header still streaming
            header = text[ch + len(HARMONY_CHANNEL) : msg]
            body_start = msg + len(HARMONY_MESSAGE)
            end, matched_token = self._segment_end(text, body_start)
            terminated = matched_token is not None
            parts = header.split()
            channel = parts[0] if parts else ""
            is_tool = channel == "commentary" and "to=functions" in header
            if is_tool:
                # Include a CLOSING terminator (<|end|>, <|return|>, <|call|>)
                # in the verbatim slice so the downstream tool parser sees the
                # complete block.  Abutting tokens (<|channel|>, <|start|>) are
                # NOT included — they open the *next* segment.  Loop advancement
                # always uses `end` (the boundary START) so a <|channel|> is
                # re-found as the next segment's opener.
                if matched_token in self._CLOSING_BOUNDARY_TOKENS:
                    slice_end = end + len(matched_token)
                else:
                    slice_end = end
                content.append(text[ch:slice_end])  # verbatim, markers included
            else:
                body = text[body_start:end]
                if not terminated and hold_partial:
                    held = _longest_harmony_partial_suffix(body)
                    if held:
                        body = body[:-held]
                if channel == "analysis":
                    reasoning.append(body)
                else:  # final, plain commentary, or unknown -> content
                    content.append(body)
            if not terminated:
                break
            i = end
        return "".join(reasoning), "".join(content)


class ThinkReasoningParser(BaseReasoningParser):
    """Generic ``<think>``/``</think>`` reasoning parser for models that wrap their
    chain-of-thought in think tags (Qwen3/3.5, GLM-4.x, MiniMax-M2). No DSML tool
    marker — reasoning ends at ``</think>``.

    Assumes the model closes ``</think>`` before any tool call. These families do
    so in well-formed thinking mode; unlike dsv4 there is no ``tool_start_token``
    fallback, so a (malformed) turn that skips ``</think>`` and runs straight into
    a tool call would fold that block into reasoning. Add a family-appropriate
    ``tool_start_token`` (as ``DeepSeekV32ReasoningParser`` does) if that surfaces.
    """

    def __init__(self, force_reasoning: bool = False, stream_reasoning: bool = True) -> None:
        super().__init__(
            think_start_token=THINK_START_TOKEN,
            think_end_token=THINK_END_TOKEN,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )


class MiniMaxM3ReasoningParser(BaseReasoningParser):
    """Reasoning parser for MiniMax-M3's ``<mm:think>...</mm:think>`` protocol.

    M3's template has three thinking modes: "enabled" pre-opens ``<mm:think>`` at
    the generation prompt (the model emits only the closing tag -- the caller
    passes ``force_reasoning=True``), "adaptive" leaves the model to open the tag
    itself, and "disabled" pre-closes it (no reasoning). The tool block's
    namespaced opener ends reasoning when a (malformed) turn skips
    ``</mm:think>`` and runs straight into a tool call (dsv4 precedent).

    Adaptive quirk (matches vLLM / llama.cpp): a NON-thinking adaptive turn
    starts with a bare ``</mm:think>`` written by the model itself ("thinking
    off for this turn"). Without special handling that literal marker would
    stream into content on the DEFAULT gear's most common path, so a single
    leading closer (whitespace-tolerant head: the detokenizer may open with a
    newline before the marker; anything else before a closer keeps it visible)
    is stripped in both one-shot and streaming modes -- never when
    ``force_reasoning`` is set, where the closer genuinely terminates the
    template-opened think block.
    """

    THINK_START = "<mm:think>"
    THINK_END = "</mm:think>"

    def __init__(self, force_reasoning: bool = False, stream_reasoning: bool = True) -> None:
        super().__init__(
            think_start_token=self.THINK_START,
            think_end_token=self.THINK_END,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="]<]minimax[>[<tool_call>",
        )
        # Streaming state for the leading-bare-closer check: hold the head of the
        # stream while it is still a prefix of ``</mm:think>``.
        self._leading_closer_pending = not force_reasoning
        self._head_buffer = ""

    def detect_and_parse(self, text: str) -> ReasoningParseResult:
        """One-shot parse, positional and verbatim, matching the streaming path:
        reasoning is anchored at the first opener the model wrote, prose before
        it stays content, later marker occurrences are data, and nothing is
        whitespace-trimmed. (The base one-shot relabels prose before a
        mid-content opener and strips both sides -- wrong for this grammar.)"""
        end = self.think_end_token
        if self.force_reasoning:
            # enabled gear: the template pre-opened the think block.
            reasoning, sep, rest = text.partition(end)
            if sep:
                return ReasoningParseResult(reasoning_text=reasoning, normal_text=rest)
            # No closer: a malformed turn running straight into a tool block
            # ends reasoning there (dsv4 precedent); else truncated reasoning.
            if self.tool_start_token is not None and self.tool_start_token in text:
                t = text.find(self.tool_start_token)
                return ReasoningParseResult(
                    reasoning_text=text[:t], normal_text=text[t:]
                )
            return ReasoningParseResult(reasoning_text=text)
        # adaptive: a leading bare closer is "thinking off for this turn"
        # (whitespace-tolerant head, same as streaming).
        head = text.lstrip()
        if head.startswith(end):
            return ReasoningParseResult(normal_text=head[len(end) :])
        start = text.find(self.think_start_token)
        if start == -1:
            return ReasoningParseResult(normal_text=text)
        before = text[:start]
        body = text[start + len(self.think_start_token) :]
        reasoning, sep, rest = body.partition(end)
        if sep:
            return ReasoningParseResult(
                reasoning_text=reasoning, normal_text=before + rest
            )
        if self.tool_start_token is not None and self.tool_start_token in body:
            t = body.find(self.tool_start_token)
            return ReasoningParseResult(
                reasoning_text=body[:t], normal_text=before + body[t:]
            )
        return ReasoningParseResult(reasoning_text=body)

    def parse_streaming_increment(self, new_text: str) -> ReasoningParseResult:
        if self._leading_closer_pending:
            self._head_buffer += new_text
            # Whitespace-tolerant head: the detokenizer may open with a newline
            # before the model's bare closer.
            head = self._head_buffer.lstrip()
            if head.startswith(self.think_end_token):
                # Bare leading closer: strip it once, everything after is content.
                self._leading_closer_pending = False
                self._head_buffer = ""
                self._in_reasoning = False
                rest = head[len(self.think_end_token) :]
                return (
                    super().parse_streaming_increment(rest)
                    if rest
                    else ReasoningParseResult()
                )
            if not head or self.think_end_token.startswith(head):
                return ReasoningParseResult()  # still a (whitespace+) prefix: hold
            # Diverged: not a leading closer -- replay the held head as normal
            # (whitespace included; it was real output).
            replay, self._head_buffer = self._head_buffer, ""
            self._leading_closer_pending = False
            return super().parse_streaming_increment(replay)
        return super().parse_streaming_increment(new_text)

    def flush(self) -> ReasoningParseResult:
        if self._leading_closer_pending and self._head_buffer:
            # Stream ended while the head was still a closer prefix (e.g. "</mm:t"):
            # it was never a bare closer, replay it before the base flush.
            head, self._head_buffer = self._head_buffer, ""
            self._leading_closer_pending = False
            first = super().parse_streaming_increment(head)
            rest = super().flush()
            return ReasoningParseResult(
                reasoning_text=first.reasoning_text + rest.reasoning_text,
                normal_text=first.normal_text + rest.normal_text,
            )
        return super().flush()


class GemmaThoughtReasoningParser(BaseReasoningParser):
    """Reasoning parser for Gemma-4's thought channel: the model emits its thought,
    then a closing ``<channel|>`` marker, then the visible answer. The opening
    ``<|channel>thought\\n`` marker is injected by the template (implicit-think),
    so ``force_reasoning`` is set by the caller's thinking mode."""

    def __init__(self, force_reasoning: bool = False, stream_reasoning: bool = True) -> None:
        super().__init__(
            think_start_token="<|channel>thought\n",
            think_end_token="<channel|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )


# Muse Glimmer ATEM protocol control tokens. Generation starts right after the
# template's ``<|start|>assistant``, so the FIRST segment arrives as a bare header
# continuation (`` to=self<|message|>...``) -- the parser is constructed
# header-open and seeds a synthetic ``<|start|>`` so those bytes go through the
# ordinary full-header machinery instead of a duplicate guessing path. Later
# segments open with a full ``<|start|>assistant to=X<|message|>`` -- or, when
# the model leaves a channel without emitting ``<|eom|>`` first, with a bare
# ``to=X<|message|>`` (a complete match of that shape is a segment boundary;
# vLLM applies the same rule). ``<|eom|>`` separates segments within one turn;
# ``<|eot|>`` / ``<|end_of_text|>`` are the stops.
ATEM_START = "<|start|>"
ATEM_MESSAGE = "<|message|>"
ATEM_CLOSING_TOKENS = ("<|eot|>", "<|eom|>", "<|end_of_text|>")
ATEM_ALL_TOKENS = (ATEM_START, ATEM_MESSAGE) + ATEM_CLOSING_TOKENS
# Recipient names cap at 64 chars in EVERY header shape (inline switches and the
# bare stream-start header use the same bound below), so streaming and one-shot
# agree on which recipients are valid and >64 degrades identically everywhere.
ATEM_RECIPIENT_RE = re.compile(r"to=([^\s<]{1,64})")
# A complete headerless channel switch: ``to=<name>`` abutting ``<|message|>``.
# The name length is capped so the streaming hold-back below can bound how much
# tail it withholds while a potential switch is still arriving.
ATEM_INLINE_HEADER_RE = re.compile(r"to=([^\s<]{1,64})<\|message\|>")
# Lookback window for the streaming hold-back's inline-switch scan: a headerless
# ``to=<name><|message|>`` switch can occupy at most this many trailing bytes.
_ATEM_HOLD_WINDOW = 96
# Longest span after "<|start|>" a real header can occupy before its <|message|>
# arrives ("assistant to=<name>" + slack). Past this, the marker cannot open a
# header anymore and is released as literal content instead of held forever.
ATEM_HEADER_SPAN = 128


def atem_marker_inside(text: str, start: int, end: int) -> bool:
    """True when a complete ATEM control token lies fully inside ``text[start:end]``.
    A channel header can never contain one, so a header candidate with a marker
    before its ``<|message|>`` is not a header."""
    for tok in ATEM_ALL_TOKENS:
        if tok == ATEM_MESSAGE:
            continue  # the terminator being sought, not a header byte
        if text.find(tok, start, end) != -1:
            return True
    return False


def _longest_atem_partial_suffix(text: str) -> int:
    """Length of the longest suffix of ``text`` that is a *proper* prefix of any ATEM
    control token (the Harmony hold-back, for Muse Glimmer's marker set)."""
    best = 0
    for tok in ATEM_ALL_TOKENS:
        for k in range(min(len(tok) - 1, len(text)), best, -1):
            if text.endswith(tok[:k]):
                best = k
                break
    return best


def atem_hold_len(text: str) -> int:
    """Streaming hold-back: the longest suffix of ``text`` that could still grow into
    an ATEM marker OR a headerless ``to=<name><|message|>`` channel switch. Withheld
    text is re-examined on the next chunk, so an emitted body never has to shrink
    when the boundary completes."""
    best = _longest_atem_partial_suffix(text)
    window = text[-_ATEM_HOLD_WINDOW:]
    i = window.rfind("to=")
    if i != -1:
        tail = window[i:]
        if re.fullmatch(r"to=[^\s<]{0,64}", tail):
            best = max(best, len(tail))  # name (or "to=") still streaming
        else:
            m = re.fullmatch(r"to=([^\s<]{1,64})(.+)", tail, re.DOTALL)
            if m and ATEM_MESSAGE.startswith(m.group(2)):
                best = max(best, len(tail))  # partial <|message|> after the name
    for prefix in ("t", "to"):  # proper prefixes of "to=" (one-chunk delay at most)
        if len(prefix) > best and text.endswith(prefix):
            best = len(prefix)
    return best


class MuseGlimmerReasoningParser(BaseReasoningParser):
    """Reasoning parser for Muse Glimmer's ATEM channel output.

    Routes ``to=self`` segments to ``reasoning_text`` and ``to=user`` (or
    recipient-less) segments, unwrapped, to ``normal_text``. A tool-recipient
    segment (``to=<tool>``, carrying an ``<atem:function_calls>`` block) is
    preserved verbatim (header included) in ``normal_text`` for the downstream
    ``MuseGlimmerDetector`` -- when the channel ends without its own terminator
    (an abutting ``<|start|>`` header or a headerless switch), a synthetic
    ``<|eom|>`` is appended so the detector always receives a delimited block.
    Segments open with ``<|start|>...`` headers or with a headerless
    ``to=X<|message|>`` switch (the model leaves a channel without ``<|eom|>``;
    vLLM's rule). The parser is initialized from the prompt it continues:
    every request that reaches it is a templated chat generation whose prompt
    ends with ``<|start|>assistant``, so ``header_open=True`` (the default)
    seeds a synthetic ``<|start|>`` and the turn's first bytes go through the
    ordinary full-header machinery -- a real ``<|message|>`` closes the header
    (a junk recipient yields the empty channel the model asked for), and a
    header that never gets its ``<|message|>`` falls to the unfinished-header
    end-of-stream rule below. This replaces a prefix-guessing "start" mode that
    kept misclassifying content shaped like a header (the same mechanism
    ``force_reasoning`` is for the ``<think>`` families, read from the prompt
    instead of re-derived). Text with no ATEM markers at all passes through as
    content.

    Streaming consumes the buffer as it emits (O(n) over the stream; the naive
    re-scan of the full buffer per chunk was quadratic and stalled the serving
    loop on this family's long default CoT). At end-of-stream, held text is
    delivered as content rather than dropped -- a whole reply that merely looks
    like a bare-header prefix, or prose stranded after a literal closer, must
    not vanish. All three methods are overridden; the base ``<think>`` machinery
    is unused.
    """

    def __init__(
        self,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
        header_open: bool = True,
    ) -> None:
        super().__init__(
            think_start_token="",
            think_end_token="",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )
        self._header_open = header_open
        # header_open: the prompt ended inside a channel header (the template's
        # ``<|start|>assistant``); seed the marker so the turn's first bytes are
        # parsed by the same machinery as every later header. The seed is
        # synthetic -- if it is ever ruled a non-header (released past the span
        # bound, or unfinished at end of stream), the marker text itself is NOT
        # delivered: the model never emitted it.
        self._buffer = ATEM_START if header_open else ""
        self._synthetic_open = header_open
        # "body" (streaming a segment body) | "seek" (between segments, looking
        # for the next header).
        self._mode = "seek"
        self._recipient = "user"

    def detect_and_parse(self, text: str) -> ReasoningParseResult:
        # Closers count as markers too: a literal stray <|eot|> must take the same
        # replay path as streaming (which strips it), not pass through verbatim.
        if not any(tok in text for tok in ATEM_ALL_TOKENS):
            return ReasoningParseResult(normal_text=text)
        clone = type(self)(header_open=self._header_open)
        first = clone.parse_streaming_increment(text)
        rest = clone.flush()
        return ReasoningParseResult(
            reasoning_text=(first.reasoning_text + rest.reasoning_text).strip(),
            normal_text=(first.normal_text + rest.normal_text).strip(),
        )

    def parse_streaming_increment(self, new_text: str) -> ReasoningParseResult:
        self._buffer += new_text
        return self._drain(final=False)

    def flush(self) -> ReasoningParseResult:
        return self._drain(final=True)

    @staticmethod
    def _segment_body_end(buf: str):
        """Earliest segment-body boundary in ``buf``: ``(end, kind, payload)`` with
        kind "closer" (payload = the token), "start" (an abutting ``<|start|>``),
        "inline" (payload = the headerless ``to=X<|message|>`` match), or
        ``(len(buf), None, None)`` while the body is still streaming."""
        best = (len(buf), None, None)
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
        return best

    def _drain(self, *, final: bool) -> ReasoningParseResult:
        out_reasoning: list[str] = []
        out_content: list[str] = []

        def emit(recipient: str, piece: str) -> None:
            if not piece:
                return
            if recipient == "self":
                out_reasoning.append(piece)
            else:  # user bodies unwrapped, tool slices verbatim
                out_content.append(piece)

        def begin_body(recipient: str, header_text: str) -> None:
            self._recipient = recipient
            self._mode = "body"
            if recipient not in ("self", "user"):
                emit(recipient, header_text)  # tool slices keep their header

        def emit_seek(piece: str) -> None:
            # Inter-segment text is content (deliver, don't drop); redundant closer
            # tokens in it are protocol debris and are stripped -- the same filter
            # the detector applies on its text path.
            for tok in ATEM_CLOSING_TOKENS:
                piece = piece.replace(tok, "")
            if piece:
                out_content.append(piece)

        while True:
            buf = self._buffer
            if not buf:
                break

            if self._mode == "seek":
                s = buf.find(ATEM_START)
                m = ATEM_INLINE_HEADER_RE.search(buf)
                if m is not None and (s == -1 or m.start() < s):
                    emit_seek(buf[: m.start()])
                    self._buffer = buf[m.end():]
                    begin_body(m.group(1), buf[m.start(): m.end()])
                    continue
                if s != -1:
                    # While the header-open seed is unconsumed it is the leftmost
                    # marker, so every consuming branch below owns clearing the
                    # synthetic flag; a synthetic marker ruled a non-header is
                    # dropped, never delivered (the model did not emit it).
                    synthetic = self._synthetic_open
                    msg = buf.find(ATEM_MESSAGE, s + len(ATEM_START))
                    if atem_marker_inside(
                        buf, s + len(ATEM_START), msg if msg != -1 else len(buf)
                    ):
                        # A control token inside the candidate: headers never
                        # contain markers, so this <|start|> is literal content
                        # (and a synthetic seed is simply dropped). Decides
                        # immediately -- no waiting for the span bound or EOS.
                        emit_seek(buf[:s])
                        if not synthetic:
                            out_content.append(ATEM_START)
                        self._synthetic_open = False
                        self._buffer = buf[s + len(ATEM_START):]
                        continue
                    if msg != -1 and msg - s - len(ATEM_START) > ATEM_HEADER_SPAN:
                        # The found <|message|> is too far away to belong to THIS
                        # marker (a stray literal <|start|>, junk, then the NEXT
                        # segment's real header): the marker is literal content --
                        # the span bound applies whether or not a <|message|> is
                        # already in the buffer, or one-shot and streaming diverge.
                        emit_seek(buf[:s])
                        if not synthetic:
                            out_content.append(ATEM_START)
                        self._synthetic_open = False
                        self._buffer = buf[s + len(ATEM_START):]
                        continue
                    if msg != -1:
                        emit_seek(buf[:s])
                        header = buf[s + len(ATEM_START): msg]
                        rm = ATEM_RECIPIENT_RE.search(header)
                        body_start = msg + len(ATEM_MESSAGE)
                        self._buffer = buf[body_start:]
                        self._synthetic_open = False
                        begin_body(rm.group(1) if rm else "user", buf[s:body_start])
                        continue
                    # A complete <|start|> whose <|message|> hasn't arrived: text
                    # before it is content NOW (never re-dropped), the candidate is
                    # held -- bounded, with slack for a <|message|> mid-arrival (a
                    # protocol-legal long-name header must not be cut at the nominal
                    # span). Past the bound it cannot open a header anymore: release
                    # the marker as literal content and resume scanning behind it,
                    # so a degenerate wire neither stalls nor eats the turn.
                    emit_seek(buf[:s])
                    self._buffer = buf[s:]
                    if len(self._buffer) - len(ATEM_START) > ATEM_HEADER_SPAN + len(ATEM_MESSAGE):
                        if not synthetic:
                            out_content.append(ATEM_START)  # verbatim: deliver, don't drop
                        self._synthetic_open = False
                        self._buffer = self._buffer[len(ATEM_START):]
                        continue
                    if final:
                        # At end of stream a candidate that never received its
                        # <|message|> is NOT a header: deliver the tail, drop only
                        # the marker. (A capped discard here kept a drop window
                        # whose edge moved between layers; removing it makes the
                        # detector's agreement trivial -- both deliver.)
                        emit_seek(self._buffer[len(ATEM_START):])
                        self._synthetic_open = False
                        self._buffer = ""
                    break
                # No marker in sight: stream eagerly as content, holding only the
                # tail that could still grow into one. The buffer is CONSUMED here
                # -- the previous hold-until-flush was quadratic (re-scanned from
                # byte 0 per chunk) and stalled the stream on long headerless
                # continuations.
                hold = 0 if final else atem_hold_len(buf)
                piece = buf[: len(buf) - hold] if hold else buf
                emit_seek(piece)
                self._buffer = buf[len(piece):]
                break

            # self._mode == "body"
            recipient = self._recipient
            end, kind, payload = self._segment_body_end(buf)
            if kind is None:
                hold = 0 if final else atem_hold_len(buf)
                piece = buf[: len(buf) - hold] if hold else buf
                emit(recipient, piece)
                self._buffer = buf[len(piece):]
                break
            body = buf[:end]
            if recipient not in ("self", "user"):
                # A closing terminator belongs to the preserved block; a channel
                # left without one (abutting header / headerless switch) gets a
                # synthetic <|eom|> so the detector still sees a delimited block.
                body += payload if kind == "closer" else "<|eom|>"
            emit(recipient, body)
            if kind == "closer":
                self._buffer = buf[end + len(payload):]
                self._mode = "seek"
            elif kind == "start":
                self._buffer = buf[end:]
                self._mode = "seek"
            else:  # headerless switch: the next segment is fully known
                self._buffer = buf[payload.end():]
                begin_body(payload.group(1), buf[end: payload.end()])
        return ReasoningParseResult(
            reasoning_text="".join(out_reasoning), normal_text="".join(out_content)
        )


class ReasoningParser:
    """Wraps a reasoning detector for streaming and non-streaming use."""

    ReasoningParserEnum: Dict[str, Type[BaseReasoningParser]] = {
        "deepseekv32": DeepSeekV32ReasoningParser,
        "gpt_oss": GptOssHarmonyReasoningParser,
        "qwen3": ThinkReasoningParser,
        "glm": ThinkReasoningParser,
        "laguna": ThinkReasoningParser,
        "minimax": ThinkReasoningParser,
        "minimax_m3": MiniMaxM3ReasoningParser,
        "muse_glimmer": MuseGlimmerReasoningParser,
        "gemma4": GemmaThoughtReasoningParser,
    }

    def __init__(
        self,
        reasoning_parser: str,
        force_reasoning: bool = True,
        stream_reasoning: bool = True,
    ) -> None:
        parser_class = self.ReasoningParserEnum.get(reasoning_parser)
        if parser_class is None:
            raise ValueError(f"Unsupported reasoning_parser: {reasoning_parser}")
        self.detector = parser_class(
            force_reasoning=force_reasoning, stream_reasoning=stream_reasoning
        )

    def parse_non_stream(self, full_text: str) -> Tuple[str, str]:
        """Return ``(reasoning_text, normal_text)`` for a complete completion."""
        result = self.detector.detect_and_parse(full_text)
        return result.reasoning_text, result.normal_text

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, str]:
        """Return ``(reasoning_delta, normal_delta)`` for one streaming chunk."""
        result = self.detector.parse_streaming_increment(chunk_text)
        return result.reasoning_text, result.normal_text

    def flush(self) -> Tuple[str, str]:
        """Drain buffered residue at end-of-stream as ``(reasoning_delta, normal_delta)``."""
        result = self.detector.flush()
        return result.reasoning_text, result.normal_text


def build_reasoning_parser(config, force_reasoning: bool) -> Optional[ReasoningParser]:
    """Construct the configured reasoning parser, or ``None`` when the server has
    no reasoning parser set. Shared by every protocol adapter's generation path."""
    name = getattr(config, "reasoning_parser", None)
    if not name:
        return None
    if name == "minimax":
        # MiniMax-M2's template always starts generation inside an implicit
        # <think> block and the model only emits the closing </think> marker.
        force_reasoning = True
    return ReasoningParser(name, force_reasoning=force_reasoning)


SUPPORTED_REASONING_PARSERS = list(ReasoningParser.ReasoningParserEnum.keys())
