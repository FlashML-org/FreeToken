"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF model config from the GGUF, which needs the PyPI ``gguf`` package and an architecture transformers knows (``qwen4exp`` is not one).
So we run only its tokenizer half: ``get_gguf_tokenizer`` and ``convert_gguf_tokenizer``.
"""

from __future__ import annotations

import json

from .reader import load_gguf_metadata

# end-of-generation token names from llama.cpp (llama_vocab::impl::load in src/llama-vocab.cpp): a GGUF stores one eos id and no stop list
_EOG_NAMES = frozenset((
    "<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>", "<end_of_turn>",
    "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>", "<end_of_utterance>",
    "<eos>", "<turn|>", "<|tool_response>", "<\uff5cend\u2581of\u2581sentence\uff5c>", "[e~[", "<|ifm|im_end|>",
))

# BPE pre-tokenizers that llama.cpp adds a bos for when the file has no add_bos_token key (llama_vocab::impl::load)
_BOS_PRE = frozenset((
    "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0", "lfm2", "jina-v5-nano", "tekken", "chameleon",
))

# Llama 3 and GLM pre-tokenizers, whose HF tokenizer and llama.cpp take a word that is one vocabulary token as is instead of merging it
_IGNORE_MERGES_PRE = frozenset(("llama3", "llama-v3", "llama-bpe", "glm4", "glm5"))


def _add_bos(meta: dict) -> bool:
    if "tokenizer.ggml.add_bos_token" in meta:
        return bool(meta["tokenizer.ggml.add_bos_token"])
    model = meta.get("tokenizer.ggml.model")
    return model == "llama" or (model == "gpt2" and meta.get("tokenizer.ggml.pre") in _BOS_PRE)


def _fix_added_tokens(fast, meta: dict):
    """Register the control and user-defined tokens that transformers skipped at their file ids, and drop added tokens past the file's vocabulary."""
    from tokenizers import Tokenizer
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    tokens = meta["tokenizer.ggml.tokens"]
    spelling = bytes_to_unicode()
    alphabet = set(spelling.values())
    # transformers' byte-level vocabulary spells a token with characters outside the byte alphabet in bytes and then skips it (DeepSeek's full-width-bar tokens, MiniMax's tokens with spaces)
    skipped = [
        (i, text, kind == 3)
        for i, (text, kind) in enumerate(zip(tokens, meta.get("tokenizer.ggml.token_type") or ()))
        if kind in (3, 4) and meta.get("tokenizer.ggml.model") == "gpt2" and not set(text) <= alphabet and fast.token_to_id(text) != i
    ]
    # its GPT2Tokenizer also adds a default <|endoftext|> when the vocabulary lacks one
    if not skipped and fast.get_vocab_size(with_added_tokens=True) <= len(tokens):
        return fast
    # a piece the merges use or build stays a plain BPE token, since renaming it breaks the merges (unsloth's MiniMax-M3 marks U+FFFD as control)
    pieces = {p for m in meta.get("tokenizer.ggml.merges") or () for p in (*m.split(" "), m.replace(" ", ""))}
    state = json.loads(fast.to_str())
    vocab = state["model"]["vocab"]
    added = [t for t in state["added_tokens"] if t["id"] < len(tokens)]
    for i, text, special in skipped:
        spelled = "".join(spelling[b] for b in text.encode("utf-8"))
        if spelled in pieces or vocab.get(spelled) != i:
            continue
        # renamed so the added token keeps the file id instead of taking a new one
        vocab[text] = vocab.pop(spelled)
        added.append({"id": i, "content": text, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": special})
    state["added_tokens"] = sorted(added, key=lambda t: t["id"])
    return Tokenizer.from_str(json.dumps(state))


def _known_specials(special: dict, fast) -> dict:
    """Drop a bos/eos/unk/pad the vocabulary lacks, which PreTrainedTokenizerFast would add past it (unsloth's MiniMax-M3 unk is U+FFFD stored decoded)."""
    vocab = fast.get_vocab(with_added_tokens=True)
    return {k: v for k, v in special.items() if k == "chat_template" or v is None or v in vocab}


def load_gguf_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.gguf import convert_gguf_tokenizer, get_gguf_tokenizer

    arch, tok_dict, special = get_gguf_tokenizer(model_path)
    fast, _ = convert_gguf_tokenizer(arch, tok_dict)
    meta = load_gguf_metadata(model_path)
    fast = _fix_added_tokens(fast, meta)
    pre = meta.get("tokenizer.ggml.pre") if meta.get("tokenizer.ggml.model") == "gpt2" else None
    if pre in ("glm4", "glm5"):
        # llama.cpp spells the same GLM pre-tokenizer "glm4" and "glm5" (llama-vocab.cpp maps both
        # to CHATGLM4 with ignore_merges); transformers maps no split for either
        from tokenizers import Regex, pre_tokenizers
        from transformers.integrations.gguf.gguf_tokenizer_mapping import GGUF_PRE_TOKENIZER_SPLITS

        fast.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(GGUF_PRE_TOKENIZER_SPLITS["llama3"]), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    if pre in _IGNORE_MERGES_PRE:
        fast.model.ignore_merges = True
    # special: bos/eos/unk/pad as the file names them (None when absent) and the chat template
    return PreTrainedTokenizerFast(
        tokenizer_object=fast,
        **_known_specials(special, fast),
        add_bos_token=_add_bos(meta),
        add_eos_token=bool(meta.get("tokenizer.ggml.add_eos_token")),
    )


def gguf_eos_token_ids(model_path: str) -> set[int]:
    """Stop ids for GGUF generation, picked the way llama.cpp picks its end-of-generation tokens."""
    meta = load_gguf_metadata(model_path)
    tokens = meta["tokenizer.ggml.tokens"]
    ids = {i for i, t in enumerate(tokens) if t in _EOG_NAMES}
    ids |= {int(meta[f"tokenizer.ggml.{k}_token_id"]) for k in ("eos", "eot", "eom") if f"tokenizer.ggml.{k}_token_id" in meta}
    names = {tokens[i]: i for i in ids if i < len(tokens)}
    # llama.cpp's exceptions: <|end|> only ends a message when <|call|> and <|return|> end the turn (gpt-oss), and gemma4's </s> is a normal token
    if "<|end|>" in names and names.keys() & {"<|call|>", "<|calls|>"} and names.keys() & {"<|return|>", "<|flush|>"}:
        ids.discard(names["<|end|>"])
    if "</s>" in names and names.keys() & {"<|tool_response>", "<|plamo:eos|>"}:
        ids.discard(names["</s>"])
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
