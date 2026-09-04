"""CPU-only regression tests for GGUF control-token registration.

The GGUF->fast-tokenizer converter leaves CONTROL/USER_DEFINED vocab entries
(e.g. Qwen's think tags, tool-call tags) unregistered: unless they happen to be
reachable through BPE merges they silently split into plain pieces, feeding the
model garbage ids ("thought then empty" symptom). ``_register_control_tokens``
must re-register every such string against its EXISTING id -- vocab size and id
assignments may not change, and already-atomic tokens must be left alone.

Runs on any machine: no GPU, no model download -- builds synthetic BPE
tokenizers whose vocab mirrors the broken shape (control string in vocab,
unreachable because no merge path leads to it).
"""

from __future__ import annotations

from tokenizers import Tokenizer, models, pre_tokenizers

from freetoken.models.gguf.tokenizer import _register_control_tokens
from transformers import PreTrainedTokenizerFast


def _tok(vocab: dict[str, int], merges: list[tuple[str, str]] = []):
    backend = Tokenizer(models.BPE(vocab=vocab, merges=list(merges)))
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def test_unmergeable_control_token_registered_at_existing_id():
    # Mirrors the real Qwen GGUF: '<think>' is IN the vocab (id 11) but no merge
    # path produces it, so bare conversion emits per-character pieces.
    vocab = {
        "<": 1, "t": 2, "h": 3, "i": 4, "n": 5, "k": 6, ">": 7,
        "a": 8, "b": 9, "<think>": 11,
    }
    tok = _tok(vocab)
    before = tok.encode("<think>", add_special_tokens=False)
    assert before != [11], "sanity: must start out broken (split), like the real bug"

    names_by_id = sorted(vocab, key=vocab.get)
    types = [1] * len(names_by_id)
    types[names_by_id.index("<think>")] = 4  # USER_DEFINED
    _register_control_tokens(tok, names_by_id, types)

    assert tok.encode("<think>", add_special_tokens=False) == [11]
    assert tok.vocab_size == len(vocab), "vocab must not grow"
    assert tok.convert_tokens_to_ids("<think>") == 11, "id must be preserved"


def test_merge_reachable_control_token_left_alone():
    # A CONTROL entry that is ALREADY atomic (reachable via merges) must not be
    # re-registered: the filter is encode(name) != [own id].
    vocab = {"a": 1, "b": 2, "ab": 3}
    tok = _tok(vocab, merges=[("a", "b")])
    assert tok.encode("ab", add_special_tokens=False) == [3]

    names_by_id = ["a", "b", "ab"]
    types = [1, 1, 3]
    _register_control_tokens(tok, names_by_id, types)
    assert tok.encode("ab", add_special_tokens=False) == [3]
    assert tok.vocab_size == 3


def test_normal_tokens_never_registered():
    # NORMAL entries that split must stay split: only CONTROL/UNKNOWN/USER_DEFINED
    # types are eligible (BYTE/UNUSED excluded too).
    vocab = {"a": 1, "b": 2, "c": 3}
    tok = _tok(vocab)
    names_by_id = ["a", "b", "c"]
    _register_control_tokens(tok, names_by_id, [1, 5, 6])
    assert tok.encode("ab", add_special_tokens=False) == [1, 2]


def test_empty_types_is_noop():
    vocab = {"a": 1}
    tok = _tok(vocab)
    _register_control_tokens(tok, ["a"], [])
    assert tok.vocab_size == 1
