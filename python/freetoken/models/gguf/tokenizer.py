"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF
config, which the gemma4 strict dataclass rejects (per-layer ``num_key_value_heads``
array). So we call the GGUF->fast tokenizer converter directly on the
``tokenizer.ggml.*`` metadata, bypassing config entirely.
"""

from __future__ import annotations

from typing import Any

from .reader import gguf_architecture, load_gguf_metadata

# GGUF architecture -> transformers GGUF tokenizer-converter key.
_TOKENIZER_ARCH = {"gemma4": "gemma4_text", "qwen35moe": "qwen3_moe"}


def _register_embedded_special_tokens(
    tokenizer: Any, tokens: list[Any], token_types: Any
) -> None:
    """Restore GGUF CONTROL and USER_DEFINED token matching on a fast tokenizer.

    The GGUF converter supplies the vocabulary ids, but some transformer releases
    do not install USER_DEFINED entries as fast-tokenizer special tokens.  This
    helper deliberately registers only GGML token classes 3 (CONTROL) and 4
    (USER_DEFINED), excluding the four roles already configured on the tokenizer.
    ``add_special_tokens`` retains a pre-existing vocabulary id when the spelling
    is already present, so model weights and prompt ids remain aligned.
    """
    if not isinstance(token_types, (list, tuple)) or len(token_types) != len(tokens):
        return
    configured_specials = {
        tokenizer.bos_token,
        tokenizer.eos_token,
        tokenizer.unk_token,
        tokenizer.pad_token,
    }
    embedded_specials = [
        str(token)
        for token, token_type in zip(tokens, token_types)
        if token_type in (3, 4) and str(token) not in configured_specials
    ]
    if embedded_specials:
        tokenizer.add_special_tokens({"additional_special_tokens": embedded_specials})


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

    def tok_for(id_key: str, default: str) -> str:
        tid = meta.get(f"tokenizer.ggml.{id_key}")
        return tokens[int(tid)] if tid is not None and int(tid) < len(tokens) else default

    # gemma4 chat turns end with <turn|>; prefer it as eos so chat generation halts
    # (the formal <eos> is also a stop id, see gguf_eos_token_ids).
    turn_end = "<turn|>" if "<turn|>" in tokens else None
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast,
        bos_token=tok_for("bos_token_id", "<bos>"),
        eos_token=turn_end or tok_for("eos_token_id", "<eos>"),
        unk_token=tok_for("unknown_token_id", "<unk>"),
        pad_token=tok_for("padding_token_id", "<pad>"),
    )

    # ``convert_gguf_tokenizer`` preserves every vocabulary entry but does not
    # consistently restore GGUF's USER_DEFINED token class as an atomic special
    # token.  Qwen3.6 declares ``<think>`` and ``</think>`` in that class.  If
    # they are not registered here, a caller-rendered ``<think>`` prompt is
    # split into three ordinary pieces (``<th``, ``ink``, ``>``), so FreeToken
    # runs a different token sequence from llama.cpp despite receiving exactly
    # the same UTF-8 request body.  GGUF token types 3 and 4 are CONTROL and
    # USER_DEFINED respectively.  Registering both groups retains their
    # existing vocabulary ids while making their matching semantics explicit.
    _register_embedded_special_tokens(tokenizer, tokens, tok_dict.get("token_type"))
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
    # absent name to <unk>, wrongly adding it as a stop id).
    for name in ("<eos>", "<turn|>"):
        try:
            ids.add(tokens.index(name))
        except ValueError:
            pass
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
