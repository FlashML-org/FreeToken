"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF model config from the GGUF, which needs the PyPI ``gguf`` package and an architecture transformers knows (``qwen4exp`` is not one).
So we run only its tokenizer half: ``get_gguf_tokenizer`` and ``convert_gguf_tokenizer``.
"""

from __future__ import annotations

from .reader import load_gguf_metadata

# end-of-generation token names from llama.cpp (llama_vocab::impl::load in src/llama-vocab.cpp): a GGUF stores one eos id and no stop list
_EOG_NAMES = frozenset((
    "<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>", "<end_of_turn>",
    "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>", "<end_of_utterance>",
    "<eos>", "<turn|>", "<|tool_response>", "<\uff5cend\u2581of\u2581sentence\uff5c>", "[e~[", "<|ifm|im_end|>",
))


def load_gguf_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.gguf import convert_gguf_tokenizer, get_gguf_tokenizer

    arch, tok_dict, special = get_gguf_tokenizer(model_path)
    fast, _ = convert_gguf_tokenizer(arch, tok_dict)
    meta = load_gguf_metadata(model_path)
    # special: bos/eos/unk/pad as the file names them (None when absent) and the chat template
    return PreTrainedTokenizerFast(
        tokenizer_object=fast,
        **special,
        add_bos_token=bool(meta.get("tokenizer.ggml.add_bos_token")),
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
