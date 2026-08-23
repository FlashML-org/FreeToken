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
# laguna ships a plain gpt2-style BPE (tokenizer.ggml.model = "gpt2"); transformers
# has no "laguna" converter, so route it to the gpt2 one.
_TOKENIZER_ARCH = {"gemma4": "gemma4_text", "laguna": "gpt2"}


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
    tokens = tok_dict["tokens"]
    # Some converters (gpt2) read .bos_token/.eos_token off the skeleton, which the
    # GGUF metadata only carries as ids -- materialize the token strings.
    for name in ("bos", "eos"):
        tid = tok_dict.get(f"{name}_token_id")
        if f"{name}_token" not in tok_dict and tid is not None and int(tid) < len(tokens):
            tok_dict[f"{name}_token"] = tokens[int(tid)]
    fast, _extra = convert_gguf_tokenizer(conv_arch, tok_dict)

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
    chat_template = meta.get("tokenizer.chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def gguf_eos_token_ids(model_path: str, tokenizer) -> set[int]:
    """Stop ids for GGUF generation: the formal <eos>, the chat turn end <turn|>, the
    GGUF-declared eot, and gemma4's tool-response opener <|tool_response> (the model
    emits it right after closing a tool call, so it is a stop id upstream too --
    generation_config.json ships eos_token_id [1, 106, 50])."""
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
    eot = meta.get("tokenizer.ggml.eot_token_id")
    if eot is not None:
        ids.add(int(eot))
    for name in ("<eos>", "<turn|>", "<|tool_response>"):
        try:
            ids.add(tokens.index(name))
        except ValueError:
            pass
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
