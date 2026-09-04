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


# ggml token_type enum values that mark non-mergeable control strings.
# 1=NORMAL 2=UNKNOWN 3=CONTROL 4=USER_DEFINED 5=UNUSED 6=BYTE
_GGML_SPECIAL_TYPES = (2, 3, 4)


def _register_control_tokens(tokenizer, tokens: list[str], types: list[int]) -> None:
    """Force every CONTROL/USER_DEFINED/UNKNOWN vocab entry to tokenize atomically.

    The GGUF->fast-tokenizer converter relies on BPE merges for special strings:
    ``<|im_start|>`` happens to be a merge-table entry and survives, but ``<think>``
    is NOT (it never appears in merged training text), so it silently splits into
    plain pieces ('<th', 'ink', '>'). A model fed those garbage ids answers with
    gibberish and EOS -- observed as "thought then returned empty" on Qwen3.6 GGUFs.
    Registering each string as an added token makes the AddedVocabulary extract it
    before BPE; because the string already exists in the base vocab, the existing id
    is reused and the vocab never grows (transformers' own ``add_tokens`` wrapper
    no-ops for in-vocab strings, so this goes through the backend directly).
    """
    missing = [
        name
        for i, (name, ty) in enumerate(zip(tokens, types))
        if int(ty) in _GGML_SPECIAL_TYPES
        # Skip BYTE-fallback and unused; skip anything already atomic at its id.
        and tokenizer.encode(name, add_special_tokens=False) != [i]
    ]
    if missing:
        tokenizer.backend_tokenizer.add_tokens(missing)
        logger.info(
            "registered %d unmerged control tokens for atomic encoding (e.g. %s)",
            len(missing),
            ", ".join(repr(t) for t in missing[:6]),
        )


def _resolve_chat_template(meta: dict[str, Any], model_path: str) -> str | None:
    """Chat template for GGUF checkpoints: explicit mirrors win, then metadata.

    Priority: a ``chat_template.jinja`` dropped NEXT TO the .gguf file, then
    ``FT_CHAT_TEMPLATE_REPO`` (<repo-id> on the HF Hub, fetched via huggingface_hub),
    then the template embedded in the GGUF's ``tokenizer.chat_template`` metadata.
    GGUF packagers (llama.cpp/unsloth) sometimes ship modified variants of the
    official template — placing the official file beside the checkpoint overrides
    it without repacking. The embedded one is the last resort, never wrong-by-default.
    """
    import os

    sidecar = os.path.join(os.path.dirname(model_path), "chat_template.jinja")
    if os.path.isfile(sidecar):
        logger.info("using chat template sidecar %s", sidecar)
        with open(sidecar, encoding="utf-8") as fh:
            return fh.read()
    repo = os.environ.get("FT_CHAT_TEMPLATE_REPO")
    if repo:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=repo, filename="chat_template.jinja")
            with open(path, encoding="utf-8") as fh:
                return fh.read()
        except Exception as exc:  # noqa: BLE001 — offline/bad repo is not fatal
            logger.warning("FT_CHAT_TEMPLATE_REPO=%s fetch failed: %s", repo, exc)
    embedded = meta.get("tokenizer.chat_template")
    if isinstance(embedded, str) and embedded.strip():
        return embedded
    return None


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
    # GGUFs are not required to carry per-token types; absent means all-normal.
    types = meta.get("tokenizer.ggml.token_type") or []
    if types:
        _register_control_tokens(tokenizer, tokens, types)
    chat_template = _resolve_chat_template(meta, str(model_path))
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
