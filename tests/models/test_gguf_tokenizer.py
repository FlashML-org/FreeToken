from __future__ import annotations


def _tiny_fast_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    # With no merge rule the base BPE spells <think> as three tokens even though
    # the complete string has an assigned vocabulary id.  GGUF USER_DEFINED
    # registration must make the complete string win atomically.
    backend = Tokenizer(
        BPE(
            vocab={"<": 0, "think": 1, ">": 2, "<think>": 3, "<ctrl>": 4},
            merges=[],
            unk_token=None,
        )
    )
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def test_registers_user_defined_tokens_atomically_without_hiding_them():
    from freetoken.models.gguf.tokenizer import _register_gguf_added_tokens

    tokenizer = _tiny_fast_tokenizer()
    assert tokenizer.encode("<think>", add_special_tokens=False) != [3]

    _register_gguf_added_tokens(
        tokenizer,
        ["<", "think", ">", "<think>", "<ctrl>"],
        [1, 1, 1, 4, 3],
    )

    assert tokenizer.encode("<think>", add_special_tokens=False) == [3]
    assert tokenizer.decode([3], skip_special_tokens=True) == "<think>"
    assert tokenizer.encode("<ctrl>", add_special_tokens=False) == [4]
    assert tokenizer.decode([4], skip_special_tokens=True) == ""


def test_ignores_missing_or_malformed_token_type_table():
    from freetoken.models.gguf.tokenizer import _register_gguf_added_tokens

    tokenizer = _tiny_fast_tokenizer()
    before = tokenizer.encode("<think>", add_special_tokens=False)
    _register_gguf_added_tokens(tokenizer, ["<think>"], None)
    _register_gguf_added_tokens(tokenizer, ["<think>"], [4, 4])
    assert tokenizer.encode("<think>", add_special_tokens=False) == before
