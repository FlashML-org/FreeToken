"""Per-request detokenizer options: include_stop_str_in_output, skip_special_tokens, and
the reasoning token count reported on the finished reply."""

from __future__ import annotations

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager

EOS, SPECIAL, THINK_END = 0, 99, 7


class _Tok:
    """Token ids decode to letters; 99 is a special marker, 7 the reasoning end tag."""

    eos_token_id = EOS
    unk_token_id = -1

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for t in ids:
            if t == SPECIAL:
                if not skip_special_tokens:
                    out.append("<s>")
            elif t == THINK_END:
                out.append("</think>")
            else:
                out.append(chr(ord("a") + t))
        return "".join(out)

    def batch_decode(self, ids_list, skip_special_tokens=False):
        return [self.decode(ids, skip_special_tokens) for ids in ids_list]


def _run(manager, uid, tokens, *, finished_kwargs=None, **msg_kwargs):
    text = ""
    meta = 0
    for i, t in enumerate(tokens):
        last = i == len(tokens) - 1
        kwargs = dict(msg_kwargs)
        if last and finished_kwargs:
            kwargs.update(finished_kwargs)
        strs, counts = manager.detokenize_with_meta(
            [DetokenizeMsg(uid=uid, next_token=t, finished=last, **kwargs)]
        )
        text += strs[0]
        meta = counts[0]
    return text, meta


def test_stop_string_is_trimmed_by_default_and_kept_on_request():
    manager = DetokenizeManager(_Tok(), frozenset({EOS}))
    trimmed, _ = _run(
        manager, 1, [1, 2, 3], stop_strs=["cd"], finished_kwargs={"matched_stop": "cd"}
    )
    assert trimmed == "b"
    kept, _ = _run(
        manager,
        2,
        [1, 2, 3],
        stop_strs=["cd"],
        keep_stop_str=True,
        finished_kwargs={"matched_stop": "cd"},
    )
    assert kept == "bcd"


def test_skip_special_tokens_is_per_request():
    manager = DetokenizeManager(_Tok(), frozenset({EOS}))
    shown, _ = _run(manager, 1, [1, SPECIAL, 2, EOS])
    hidden, _ = _run(manager, 2, [1, SPECIAL, 2, EOS], skip_special_tokens=True)
    assert shown == "b<s>c"
    assert hidden == "bc"


def test_reasoning_tokens_count_up_to_the_end_tag():
    manager = DetokenizeManager(_Tok(), frozenset({EOS}), think_end_id=THINK_END)
    text, reasoning = _run(manager, 1, [1, 2, THINK_END, 3, 4, EOS])
    assert text == "bc</think>de"
    assert reasoning == 3  # b, c and the tag itself
    _, none = _run(manager, 2, [1, 2, EOS])
    assert none == 0
