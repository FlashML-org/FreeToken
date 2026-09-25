"""Regression coverage for GGUF tokenizer control-token registration."""

from freetoken.models.gguf.tokenizer import _register_embedded_special_tokens


class _FakeTokenizer:
    """Minimal tokenizer recorder that keeps the helper test independent of model files."""

    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token = "<pad>"

    def __init__(self) -> None:
        self.calls: list[dict[str, list[str]]] = []

    def add_special_tokens(self, values: dict[str, list[str]]) -> int:
        """Record the exact registration request made by the GGUF helper."""
        self.calls.append(values)
        return len(values["additional_special_tokens"])


def test_register_embedded_control_and_user_defined_tokens() -> None:
    """Qwen's thinking marker stays atomic after a GGUF tokenizer conversion."""
    tokenizer = _FakeTokenizer()

    _register_embedded_special_tokens(
        tokenizer,
        ["ordinary", "<bos>", "<|im_start|>", "<think>", "<eos>"],
        [1, 3, 3, 4, 3],
    )

    assert tokenizer.calls == [
        {"additional_special_tokens": ["<|im_start|>", "<think>"]}
    ]


def test_register_embedded_special_tokens_ignores_invalid_metadata() -> None:
    """Malformed optional type metadata cannot block otherwise valid GGUF loading."""
    tokenizer = _FakeTokenizer()

    _register_embedded_special_tokens(tokenizer, ["<think>"], None)
    _register_embedded_special_tokens(tokenizer, ["<think>"], [4, 4])

    assert tokenizer.calls == []
