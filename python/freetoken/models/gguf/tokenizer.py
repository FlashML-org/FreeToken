"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF
config, which the gemma4 strict dataclass rejects (per-layer ``num_key_value_heads``
array). So we call the GGUF->fast tokenizer converter directly on the
``tokenizer.ggml.*`` metadata, bypassing config entirely.
"""

from __future__ import annotations

from typing import Any

from .reader import gguf_architecture, load_gguf_metadata

# GGUF architecture -> transformers GGUF tokenizer-converter key. transformers keys its
# converters by *its own* model_type, not by the GGUF arch string, so an arch it has never
# heard of raises KeyError inside convert_gguf_tokenizer. qwen35moe is a GPT2-style BPE
# with merges (tokenizer.ggml.model == "gpt2", pre == "qwen35"), which the qwen2 converter
# handles; there is no qwen3.5-specific converter and it would be the same BPE anyway.
# qwen3moe is also a GPT2-style BPE (tokenizer.ggml.model == "gpt2", pre == "qwen3").
_TOKENIZER_ARCH = {
    "gemma4": "gemma4_text",
    "qwen35moe": "qwen2",
    "qwen35": "qwen2",
    "qwen3moe": "qwen2",
    # tokenizer.ggml.model is gpt2 (BPE), pre joyai-llm, 129280 entries.
    "deepseek4": "llama",
}

# Per-arch chat/stop tokens, in preference order: the first one present in the vocab
# becomes eos (so chat generation halts on the turn end rather than the formal document
# eos), and every one present is a stop id. Keyed by GGUF arch because these names are
# vocab-specific -- gemma4 ends a turn with <turn|>, Qwen with <|im_end|>. An arch absent
# here falls back to tokenizer.ggml.eos_token_id alone.
_STOP_TOKENS: dict[str, tuple[str, ...]] = {
    "gemma4": ("<turn|>", "<eos>"),
    "qwen35moe": ("<|im_end|>", "<|endoftext|>"),
    # Dense sibling: same vocab and same chat markers as the MoE variant.
    "qwen35": ("<|im_end|>", "<|endoftext|>"),
    "qwen3moe": ("<|im_end|>", "<|endoftext|>"),
    # Read from the vocab: eos id 1 is the document end, <|EOT|> (128805) ends a
    # chat turn. <｜User｜> is deliberately not a stop -- the template emits it
    # before the model speaks, not after.
    "deepseek4": ("<|EOT|>", "<｜end▁of▁sentence｜>"),
}


def load_gguf_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.ggml import convert_gguf_tokenizer

    meta = load_gguf_metadata(model_path)
    arch = gguf_architecture(model_path)
    conv_arch = _TOKENIZER_ARCH.get(arch, arch)
    tok_dict: dict[str, Any] = {
        k[len("tokenizer.ggml.") :]: v
        for k, v in meta.items()
        if k.startswith("tokenizer.ggml.")
    }
    fast, _extra = convert_gguf_tokenizer(conv_arch, tok_dict)

    tokens = tok_dict["tokens"]

    def tok_for(id_key: str, default: str | None) -> str | None:
        """The token named by ``tokenizer.ggml.<id_key>``, else ``default`` if it is in the
        vocab. A default that is not in this vocab is dropped: handing
        PreTrainedTokenizerFast an unknown special token would append it to the vocab and
        shift nothing but confuse decoding (Qwen has no <unk> at all)."""
        tid = meta.get(f"tokenizer.ggml.{id_key}")
        if tid is not None and int(tid) < len(tokens):
            return tokens[int(tid)]
        return default if default is not None and default in tokens else None

    # Prefer the chat turn end as eos so chat generation halts there; the formal document
    # eos stays a stop id (see gguf_eos_token_ids).
    turn_end = next((t for t in _STOP_TOKENS.get(arch, ()) if t in tokens), None)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast,
        bos_token=tok_for("bos_token_id", "<bos>"),
        eos_token=turn_end or tok_for("eos_token_id", "<eos>"),
        unk_token=tok_for("unknown_token_id", "<unk>"),
        pad_token=tok_for("padding_token_id", "<pad>"),
    )
    chat_template = meta.get("tokenizer.chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def gguf_eos_token_ids(model_path: str, tokenizer) -> set[int]:
    """Stop ids for GGUF generation: the formal <eos> plus the chat turn end <turn|>."""
    meta = load_gguf_metadata(model_path)
    tokens = meta["tokenizer.ggml.tokens"]
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    eid = meta.get("tokenizer.ggml.eos_token_id")
    if eid is not None:
        ids.add(int(eid))
    # Look the stop tokens up in the vocab directly (convert_tokens_to_ids would map an
    # absent name to <unk>, wrongly adding it as a stop id). Names are per-arch: gemma4's
    # <eos>/<turn|> do not exist in a Qwen vocab and vice versa.
    for name in _STOP_TOKENS.get(gguf_architecture(model_path), ()):
        try:
            ids.add(tokens.index(name))
        except ValueError:
            pass
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
