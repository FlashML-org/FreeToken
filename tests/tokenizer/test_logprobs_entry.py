from freetoken.tokenizer.detokenize import build_logprobs_entry


class _FakeTokenizer:
    def __init__(self, token_table: dict[int, str]) -> None:
        self.token_table = token_table

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.token_table[token_id] for token_id in token_ids)


def test_build_logprobs_entry_matches_contract() -> None:
    tokenizer = _FakeTokenizer({1: "hel", 2: "lo", 3: "é"})
    entry = build_logprobs_entry(
        tokenizer,
        token_id=3,
        chosen_logprob=-0.123,
        top_ids=[3, 2, 1],
        top_logprobs=[-0.1, -0.9, -1.5],
    )

    assert entry["token_id"] == 3
    assert entry["token"] == "é"
    assert entry["bytes"] == list("é".encode("utf-8"))
    assert entry["logprob"] == -0.123

    top = entry["top"]
    assert len(top) == 3
    assert top[0] == {
        "token_id": 3,
        "token": "é",
        "bytes": list("é".encode("utf-8")),
        "logprob": -0.1,
    }
    assert top[1] == {
        "token_id": 2,
        "token": "lo",
        "bytes": list("lo".encode("utf-8")),
        "logprob": -0.9,
    }
    assert top[2] == {
        "token_id": 1,
        "token": "hel",
        "bytes": list("hel".encode("utf-8")),
        "logprob": -1.5,
    }


def test_build_logprobs_entry_empty_top() -> None:
    tokenizer = _FakeTokenizer({7: "x", 8: "y"})
    entry = build_logprobs_entry(
        tokenizer,
        token_id=7,
        chosen_logprob=-2.0,
        top_ids=[],
        top_logprobs=[],
    )

    assert entry["top"] == []
