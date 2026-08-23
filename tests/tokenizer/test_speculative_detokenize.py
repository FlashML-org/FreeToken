from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager


class _PieceTokenizer:
    eos_token_id = 0

    _pieces = {
        0: "<eos>",
        1: "pres",
        2: "idencia",
        3: " de",
        4: " López",
        5: " Obrador",
        6: ".",
    }

    def batch_decode(self, rows):
        return ["".join(self._pieces[token] for token in row) for row in rows]


def _manager() -> DetokenizeManager:
    return DetokenizeManager(_PieceTokenizer(), frozenset({0}))


def test_speculative_block_emits_each_token_once():
    manager = _manager()

    chunks = manager.detokenize(
        [
            DetokenizeMsg(
                uid=7,
                next_token=5,
                token_ids=[1, 2, 3, 4, 5],
                finished=False,
            )
        ]
    )

    assert chunks == ["presidencia de López Obrador"]


def test_coalesced_messages_for_one_request_do_not_repeat_prefixes():
    manager = _manager()

    chunks = manager.detokenize(
        [
            DetokenizeMsg(uid=9, next_token=2, token_ids=[1, 2], finished=False),
            DetokenizeMsg(uid=9, next_token=5, token_ids=[3, 4, 5], finished=False),
            DetokenizeMsg(uid=9, next_token=0, finished=True),
        ]
    )

    assert chunks == ["presidencia", " de López Obrador", ""]
    assert 9 not in manager.decode_map


def test_finished_block_counts_text_before_eos_once():
    manager = _manager()

    chunks = manager.detokenize(
        [
            DetokenizeMsg(
                uid=11,
                next_token=0,
                token_ids=[1, 2, 6, 0],
                finished=True,
                finish_reason="stop",
            )
        ]
    )

    assert chunks == ["presidencia."]
    assert 11 not in manager.decode_map
