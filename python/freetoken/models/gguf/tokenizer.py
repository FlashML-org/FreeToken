"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF
config, which the gemma4 strict dataclass rejects (per-layer ``num_key_value_heads``
array). So we call the GGUF->fast tokenizer converter directly on the
``tokenizer.ggml.*`` metadata, bypassing config entirely.
"""

from __future__ import annotations

from typing import Any

from .reader import gguf_architecture, gguf_tensor_names, load_gguf_metadata

# GGUF architecture -> transformers GGUF tokenizer-converter key.
_TOKENIZER_ARCH = {"gemma4": "gemma4_text"}


def load_gguf_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.ggml import convert_gguf_tokenizer
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.decoders import Metaspace
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers import AddedToken

    meta = load_gguf_metadata(model_path)
    arch = gguf_architecture(model_path)
    conv_arch = _TOKENIZER_ARCH.get(arch, arch)
    tok_dict: dict[str, Any] = {
        k[len("tokenizer.ggml.") :]: v
        for k, v in meta.items()
        if k.startswith("tokenizer.ggml.")
    }
    tokens = tok_dict["tokens"]

    # This repaired GGUF contains the 248044 base BPE entries but omits Qwen's
    # tokenizer.json added_tokens. Their IDs are part of the model vocabulary and
    # the chat template depends on them as atomic tokens.
    qwen_added = (
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>",
        "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>",
        "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
        "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
        "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
        "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
        "<think>", "</think>", "<|audio_start|>", "<|audio_end|>", "<tts_pad>",
        "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>", "<|audio_pad|>",
    )
    base_vocab_size = len(tokens)
    added_vocab = {
        token: base_vocab_size + index
        for index, token in enumerate(qwen_added)
        if token not in tokens
    }
    if added_vocab:
        tokens = tokens + [token for token, _ in sorted(added_vocab.items(), key=lambda item: item[1])]

    def tok_for(id_key: str, default: str) -> str:
        tid = meta.get(f"tokenizer.ggml.{id_key}")
        return tokens[int(tid)] if tid is not None and int(tid) < len(tokens) else default

    unknown_token = tok_for("unknown_token_id", tokens[0] if tokens else "<unk>")
    # The hf_nvfp4 marker describes the weight layout, not the tokenizer family. Gemma4
    # checkpoints can carry the same marker but use their own SentencePiece-compatible
    # converter; only Qwen GPT2/BPE vocabularies belong in this custom ByteLevel path.
    hf_nvfp4 = arch.startswith("qwen") and tok_dict.get("model") == "gpt2"
    if hf_nvfp4:
        # The embedded Qwen vocabulary is GPT-2 byte-level BPE. Treating it like a
        # SentencePiece vocabulary turns spaces into the fallback unknown token.
        vocab = {token: index for index, token in enumerate(tok_dict["tokens"])}
        merges = [tuple(merge.split(" ", 1)) for merge in tok_dict["merges"]]
        fast = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                unk_token=unknown_token,
                fuse_unk=True,
                byte_fallback=True,
            )
        )
        fast.pre_tokenizer = ByteLevel(add_prefix_space=False)
        fast.decoder = ByteLevelDecoder()
        # Some repaired GGUFs already carry the Qwen control strings in the base token
        # array, while others omit them and require reconstruction above. Mark both cases
        # as special: leaving an existing <|im_end|> as ordinary BPE text can decode it as
        # the visible fragment ``|im_end|>`` instead of terminating the assistant turn.
        fast.add_special_tokens([AddedToken(token, special=True) for token in qwen_added])
    else:
        fast, _extra = convert_gguf_tokenizer(conv_arch, tok_dict)
        # Gemma's vocabulary uses SentencePiece's U+2581 word marker. Some versions of
        # the GGUF converter build the tokenizer model but leave a byte/BPE-style decoder
        # attached, exposing ``▁`` and replacement characters in generated text. Restore
        # the SentencePiece-compatible decoder explicitly for non-GPT2 vocabularies.
        if tok_dict.get("model") != "gpt2":
            fast.decoder = Metaspace()

    # gemma4 chat turns end with <turn|>; prefer it as eos so chat generation halts
    # (the formal <eos> is also a stop id, see gguf_eos_token_ids).
    turn_end = "<turn|>" if "<turn|>" in tokens else None
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast,
        bos_token=tok_for("bos_token_id", "<bos>"),
        eos_token=turn_end or tok_for("eos_token_id", "<eos>"),
        unk_token=unknown_token,
        pad_token=tok_for("padding_token_id", "<pad>"),
    )
    if not hf_nvfp4 and tok_dict.get("model") != "gpt2":
        # Reassign after the HF wrapper is built as well. The GGUF converter and some
        # transformers versions replace the backend decoder during wrapping, which leaves
        # SentencePiece's ``▁`` marker and malformed byte sequences visible in Gemma text.
        tokenizer.backend_tokenizer.decoder = Metaspace(
            replacement="▁", prepend_scheme="always"
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
    # Look the stop tokens up in the completed vocab, which includes reconstructed Qwen
    # added tokens. Avoid convert_tokens_to_ids for absent names because it maps them to unk.
    for name in ("<eos>", "<turn|>", "<|im_end|>"):
        token_id = tokenizer.get_vocab().get(name)
        if token_id is not None:
            ids.add(int(token_id))
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
