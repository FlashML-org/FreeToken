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
_TOKENIZER_ARCH = {"gemma4": "gemma4_text"}

# LFM2's byte-level BPE splits on a GPT-4-style pattern, which the converter's generic
# GPT-2 path does not produce (digits group in 1-3s, punctuation runs stay whole).
# llama.cpp names it ``tokenizer.ggml.pre = "lfm2"``; the pattern below is the one the
# released LFM2/LFM2.5 ``tokenizer.json`` carries.
_LFM2_PRE = "lfm2"
_LFM2_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# Markers llama.cpp writes as CONTROL although the released tokenizer ships them
# non-special: they delimit text a client parses, so they must survive
# decode(skip_special_tokens=True).
_PLAIN_MARKERS_BY_PRE = {
    _LFM2_PRE: ("<|tool_call_start|>", "<|tool_call_end|>"),
}


def _register_gguf_special_tokens(
    tokenizer, tokens: list, token_types, plain_markers: tuple[str, ...] = ()
) -> None:
    """Encoder-isolate the tokens the GGUF marks CONTROL (special) and USER_DEFINED.

    llama.cpp splits them out before BPE, so without this the chat markers
    (``<|im_start|>``, ``<think>``, ...) would BPE-split into pieces. The generic GPT-2
    converter path used by families without their own converter (LFM2) registers none.
    """
    from tokenizers import AddedToken

    types = list(token_types or [])
    specials, plain = [], []
    for i, kind in enumerate(types):
        if i >= len(tokens):
            break
        if kind not in (3, 4):  # gguf CONTROL / USER_DEFINED
            continue
        token = tokens[i]
        special = kind == 3 and token not in plain_markers
        (specials if special else plain).append(
            AddedToken(token, normalized=False, special=special)
        )
    if specials:
        tokenizer.add_special_tokens(specials)
    if plain:
        tokenizer.add_tokens(plain)


def _apply_pre_tokenizer_override(tokenizer, pre: str | None) -> None:
    """Give ``pre`` its trained pre-tokenizer + decoder (the converter has no entry for it)."""
    if pre != _LFM2_PRE:
        return
    from tokenizers import Regex, decoders, pre_tokenizers

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(_LFM2_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)


def load_gguf_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.ggml import convert_gguf_tokenizer

    meta = load_gguf_metadata(model_path)
    arch = gguf_architecture(model_path)
    tok_dict: dict[str, Any] = {
        k[len("tokenizer.ggml.") :]: v
        for k, v in meta.items()
        if k.startswith("tokenizer.ggml.")
    }
    # Families whose converter key differs from the GGUF architecture name (gemma4) are
    # listed; everything else uses the file's own ``tokenizer.ggml.model`` (the BPE variant
    # llama.cpp wrote, e.g. "gpt2" for LFM2's byte-level BPE).
    conv_arch = _TOKENIZER_ARCH.get(arch) or str(tok_dict.get("model") or arch)
    fast, _extra = convert_gguf_tokenizer(conv_arch, tok_dict)
    _register_gguf_special_tokens(
        fast,
        tok_dict["tokens"],
        tok_dict.get("token_type"),
        _PLAIN_MARKERS_BY_PRE.get(tok_dict.get("pre"), ()),
    )
    _apply_pre_tokenizer_override(fast, tok_dict.get("pre"))

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
