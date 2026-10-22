"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF model config from the GGUF, which needs the PyPI ``gguf`` package and an architecture transformers knows (``qwen4exp`` is not one).
So we run only its tokenizer half: ``get_gguf_tokenizer`` and ``convert_gguf_tokenizer``.
"""

from __future__ import annotations

import json
from typing import Any

from .reader import gguf_architecture, gguf_tensor_names, load_gguf_metadata

# end-of-generation token names from llama.cpp (llama_vocab::impl::load in src/llama-vocab.cpp): a GGUF stores one eos id and no stop list
_EOG_NAMES = frozenset((
    "<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>", "<end_of_turn>",
    "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>", "<end_of_utterance>",
    "<eos>", "<turn|>", "<|tool_response>", "<\uff5cend\u2581of\u2581sentence\uff5c>", "[e~[", "<|ifm|im_end|>",
))

_QWEN_ADDED_TOKENS = (
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>",
    "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>",
    "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
    "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
    "<think>", "</think>", "<|audio_start|>", "<|audio_end|>", "<tts_pad>",
    "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>", "<|audio_pad|>",
)

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
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.decoders import Metaspace
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers import AddedToken

    arch, converter_tok_dict, _ = get_gguf_tokenizer(model_path)
    meta = load_gguf_metadata(model_path)
    arch = gguf_architecture(model_path)
    tok_dict: dict[str, Any] = {
        k[len("tokenizer.ggml.") :]: v
        for k, v in meta.items()
        if k.startswith("tokenizer.ggml.")
    }
    tokens = tok_dict["tokens"]

    # This repaired GGUF contains the 248044 base BPE entries but omits Qwen's
    # tokenizer.json added_tokens. Their IDs are part of the model vocabulary and
    # the chat template depends on them as atomic tokens.
    base_vocab_size = len(tokens)
    added_vocab = {
        token: base_vocab_size + index
        for index, token in enumerate(_QWEN_ADDED_TOKENS)
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
        fast.add_special_tokens([AddedToken(token, special=True) for token in _QWEN_ADDED_TOKENS])
    else:
        fast, _extra = convert_gguf_tokenizer(arch, converter_tok_dict)
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


def gguf_eos_token_ids(model_path: str) -> set[int]:
    """Stop ids for GGUF generation, picked the way llama.cpp picks its end-of-generation tokens."""
    meta = load_gguf_metadata(model_path)
    tokens = meta["tokenizer.ggml.tokens"]
    ids = {index for index, token in enumerate(tokens) if token in _EOG_NAMES}
    ids.update(
        int(meta[f"tokenizer.ggml.{key}_token_id"])
        for key in ("eos", "eot", "eom")
        if f"tokenizer.ggml.{key}_token_id" in meta
    )
    names = {tokens[index]: index for index in ids if index < len(tokens)}
    if "<|end|>" in names and names.keys() & {"<|call|>", "<|calls|>"} and names.keys() & {"<|return|>", "<|flush|>"}:
        ids.discard(names["<|end|>"])
    if "</s>" in names and names.keys() & {"<|tool_response>", "<|plamo:eos|>"}:
        ids.discard(names["</s>"])

    if gguf_architecture(model_path).startswith("qwen") and meta.get("tokenizer.ggml.model") == "gpt2":
        present = set(tokens)
        next_id = len(tokens)
        for token in _QWEN_ADDED_TOKENS:
            if token not in present:
                if token in _EOG_NAMES:
                    ids.add(next_id)
                next_id += 1
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
