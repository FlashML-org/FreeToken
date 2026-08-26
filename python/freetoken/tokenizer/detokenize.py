from dataclasses import dataclass
from typing import Dict, FrozenSet, List

from freetoken.message import DetokenizeMsg
from transformers import PreTrainedTokenizerBase

# Borrowed from sglang


def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        return text[: text.rfind(" ") + 1]


def _stop_prefix_holdback(text: str, stop_strs: list[str]) -> int:
    """Length of the longest trailing suffix of ``text`` that is a proper prefix of some
    stop string. Those chars are withheld until a later token resolves whether the stop
    completes, so a partial stop is never streamed and then needs retracting."""
    hold = 0
    for stop in stop_strs:
        for i in range(min(len(stop) - 1, len(text)), 0, -1):
            if text.endswith(stop[:i]):
                hold = max(hold, i)
                break
    return hold


@dataclass
class DecodeStatus:
    decoded_ids: List[int]
    decoded_str: str
    read_offset: int  # length of read ids
    surr_offset: int  # length of surr ids
    sent_offset: int  # length of sent out string


class DetokenizeManager:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, eos_token_ids: FrozenSet[int] | None = None
    ) -> None:
        # uid -> DecodeStatus
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_ids = (
            eos_token_ids if eos_token_ids is not None else frozenset({tokenizer.eos_token_id})
        )

    def discard(self, uid: int) -> None:
        """Drop a uid's decode state without a finished DetokenizeMsg. An aborted or errored
        request never sends one (its terminal reply is an ErrorReplyMsg), so without this its
        accumulated ids and text stay in ``decode_map`` for the life of the worker."""
        self.decode_map.pop(uid, None)

    def _append_token(self, s: DecodeStatus, token: int, *, skip: bool = False) -> str:
        """Advance one request by one token, preserving tokenizer boundary context."""
        if not skip:
            s.decoded_ids.append(token)
        read_str, surr_str = self.tokenizer.batch_decode(
            [
                s.decoded_ids[s.surr_offset :],
                s.decoded_ids[s.surr_offset : s.read_offset],
            ]
        )
        new_text = read_str[len(surr_str) :]
        if len(new_text) > 0 and not new_text.endswith("�"):
            output_str = s.decoded_str + new_text
            s.decoded_str = output_str
            s.surr_offset = s.read_offset
            s.read_offset = len(s.decoded_ids)
        else:
            new_text = find_printable_text(new_text)
            output_str = s.decoded_str + new_text
        return output_str

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        """Detokenize each request's newly emitted token block in order.

        A speculative step can deliver multiple accepted tokens for one uid. The
        original two-pass batch implementation appended every token before updating
        the uid's offsets, so each decoded prefix was appended cumulatively. vLLM's
        speculative output processor advances tokens sequentially; doing so here is
        both exact and safe when multiple messages for one uid are coalesced.
        """
        incremental_strs: List[str] = []
        for msg in msgs:
            if msg.uid not in self.decode_map:
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.uid]
            tokens = msg.token_ids if msg.token_ids is not None else [msg.next_token]
            output_str = s.decoded_str
            for pos, token in enumerate(tokens):
                final_eos = (
                    msg.finished
                    and pos == len(tokens) - 1
                    and token in self.eos_token_ids
                )
                output_str = self._append_token(s, token, skip=final_eos)

            prev_sent = s.sent_offset
            if msg.finished:
                # Generation is over: flush everything, trimming at the matched stop string.
                if msg.matched_stop:
                    cut = output_str.find(msg.matched_stop)
                    emit_end = cut if cut >= 0 else len(output_str)
                else:
                    emit_end = len(output_str)
            elif msg.stop_strs:
                # Hold back a trailing suffix that could still grow into a stop string.
                emit_end = len(output_str) - _stop_prefix_holdback(output_str, msg.stop_strs)
            else:
                emit_end = len(output_str)
            incremental_output = output_str[prev_sent:emit_end] if emit_end > prev_sent else ""
            s.sent_offset = max(prev_sent, emit_end)
            incremental_strs.append(incremental_output)
            if msg.finished:
                del self.decode_map[msg.uid]

        return incremental_strs
