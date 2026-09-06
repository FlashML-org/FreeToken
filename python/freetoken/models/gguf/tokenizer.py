"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF
config, which the gemma4 strict dataclass rejects (per-layer ``num_key_value_heads``
array). So we call the GGUF->fast tokenizer converter directly on the
``tokenizer.ggml.*`` metadata, bypassing config entirely.
"""

from __future__ import annotations

from typing import Any

from freetoken.utils import init_logger

from .reader import gguf_architecture, load_gguf_metadata

logger = init_logger(__name__)

# GGUF architecture -> transformers GGUF tokenizer-converter key.
_TOKENIZER_ARCH = {
    "gemma4": "gemma4_text",
    "qwen35moe": "qwen3_moe",
}


_GGML_SPECIAL_TYPES = (2, 3, 4)


def _register_control_tokens(tokenizer, tokens: list[str], types: list[int]) -> None:
    """Register unmerged control/user-defined tokens without changing their IDs."""
    missing = [
        name
        for i, (name, ty) in enumerate(zip(tokens, types))
        if int(ty) in _GGML_SPECIAL_TYPES
        and tokenizer.encode(name, add_special_tokens=False) != [i]
    ]
    if missing:
        tokenizer.backend_tokenizer.add_tokens(missing)
        logger.info("registered %d unmerged GGUF control tokens", len(missing))


def _resolve_chat_template(meta: dict[str, Any], model_path: str) -> str | None:
    """Resolve sidecar, optional Hub mirror, then embedded GGUF chat template."""
    import os

    sidecar = os.path.join(os.path.dirname(model_path), "chat_template.jinja")
    if os.path.isfile(sidecar):
        with open(sidecar, encoding="utf-8") as fh:
            return fh.read()
    repo = os.environ.get("FT_CHAT_TEMPLATE_REPO")
    if repo:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=repo, filename="chat_template.jinja")
            with open(path, encoding="utf-8") as fh:
                return fh.read()
        except Exception as exc:  # offline/bad mirror must not break embedded fallback
            logger.warning("FT_CHAT_TEMPLATE_REPO=%s fetch failed: %s", repo, exc)
    embedded = meta.get("tokenizer.chat_template")
    return embedded if isinstance(embedded, str) and embedded.strip() else None


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
    tokens = tok_dict.get("tokens")
    if not isinstance(tokens, (list, tuple)) or not tokens:
        raise ValueError(f"{model_path}: GGUF tokenizer.ggml.tokens is missing or empty")
    fast, _extra = convert_gguf_tokenizer(conv_arch, tok_dict)

    def tok_for(id_key: str, default: str) -> str:
        tid = meta.get(f"tokenizer.ggml.{id_key}")
        return tokens[int(tid)] if tid is not None and 0 <= int(tid) < len(tokens) else default

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
    types = meta.get("tokenizer.ggml.token_type") or []
    if types:
        _register_control_tokens(tokenizer, list(tokens), list(types))
    chat_template = _resolve_chat_template(meta, str(model_path))
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def gguf_eos_token_ids(model_path: str, tokenizer) -> set[int]:
    """Stop ids for GGUF generation: the formal <eos> plus the chat turn end <turn|>."""
    meta = load_gguf_metadata(model_path)
    tokens = meta.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, (list, tuple)) or not tokens:
        raise ValueError(f"{model_path}: GGUF tokenizer.ggml.tokens is missing or empty")
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
