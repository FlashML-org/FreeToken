"""The GGUF tokenizer: pre-tokenizer spellings llama.cpp writes and how FreeToken maps them."""

from __future__ import annotations

import struct

# GGUF v3 value types (the vendored gguf-py GGUFValueType numbering)
_T_UINT32, _T_FLOAT32, _T_STRING, _T_ARRAY = 4, 6, 8, 9

# a byte-level BPE with no merges: "Hello" is one vocabulary token, and without
# ignore_merges the encoder falls back to its four characters
_TOKENS = ["d", "o", "n", "'", "t", "don't", "Hello", "H", "e", "l", "!"]


def _pack_val(v):
    if isinstance(v, str):
        b = v.encode()
        return struct.pack("<Q", len(b)) + b
    if isinstance(v, int):
        return struct.pack("<I", v)
    if isinstance(v, float):
        return struct.pack("<f", v)
    if isinstance(v, list):
        elem = {int: _T_UINT32, float: _T_FLOAT32, str: _T_STRING}[type(v[0])]
        return struct.pack("<I", elem) + struct.pack("<Q", len(v)) + b"".join(_pack_val(x) for x in v)
    raise TypeError(v)


def _tokenizer_gguf(path, pre: str) -> str:
    """A header-only GGUF carrying a gpt2 tokenizer with the given ``tokenizer.ggml.pre``."""
    kvs = {
        "general.architecture": (_T_STRING, "glm5-next"),
        "tokenizer.ggml.model": (_T_STRING, "gpt2"),
        "tokenizer.ggml.pre": (_T_STRING, pre),
        "tokenizer.ggml.tokens": (_T_ARRAY, list(_TOKENS)),
        "tokenizer.ggml.token_type": (_T_ARRAY, [1] * len(_TOKENS)),
        "tokenizer.ggml.scores": (_T_ARRAY, [0.0] * len(_TOKENS)),
    }
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs)))
        for k, (t, v) in kvs.items():
            f.write(struct.pack("<Q", len(k.encode())) + k.encode() + struct.pack("<I", t) + _pack_val(v))
    return str(path)


def test_glm5_pre_routes_like_glm4(tmp_path):
    """llama.cpp maps its "glm4" and "glm5" spellings to the same pre-tokenizer (CHATGLM4
    with ignore_merges); a glm5 file must encode exactly like the glm4 one."""
    from freetoken.models.gguf.tokenizer import load_gguf_tokenizer

    glm4 = load_gguf_tokenizer(_tokenizer_gguf(tmp_path / "tok4.gguf", "glm4"))
    glm5 = load_gguf_tokenizer(_tokenizer_gguf(tmp_path / "tok5.gguf", "glm5"))
    for text in ("Hello!", "don't", "don't Hello!"):
        assert glm4.encode(text) == glm5.encode(text), text
    # the treatment bites: the whole-word token wins over the character fallback
    assert glm5.encode("Hello!") == [_TOKENS.index("Hello"), _TOKENS.index("!")]
