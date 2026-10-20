"""Compare the tokenizer FreeToken builds from a GGUF with the model's own HF tokenizer.

    python scripts/gguf_check/tok_compare.py <gguf> <HF model dir or tokenizer.json> [--combos 20000]

Encodes, without special tokens, a set of edge strings, every normal vocabulary token, and random joins of 2-4 vocabulary tokens, and counts the texts whose ids differ.
The HF tokenizer is the reference: a stub works, since only the GGUF metadata is read.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

EDGE = [
    "Hello world",
    "  leading spaces",
    "trailing spaces   ",
    "tabs\t\tand\nnewlines\n\n\n",
    "I'm you're it's we'll they'd DON'T",
    "1234567890 3.14159 1,000,000",
    "def f(x):\n    return x ** 2  # square\n",
    "数学题：求 x 的值。日本語のテキスト。한국어 문장.",
    "emoji 😀👍🏽 and combining é ä",
    "Ünïcödé àccents ﬁ ligature",
    "<|im_start|> <s> </s> [INST] literal special-looking text",
    "$$\\frac{a}{b} + \\sqrt{x}$$",
    "URL https://example.com/a?b=c&d=e",
]
_NORMAL = 1


def vocab_words(meta: dict) -> list[str]:
    """The text of every normal vocabulary token, for byte-level BPE and SentencePiece vocabularies."""
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    tokens, types = meta["tokenizer.ggml.tokens"], meta.get("tokenizer.ggml.token_type") or [_NORMAL] * len(meta["tokenizer.ggml.tokens"])
    model = meta.get("tokenizer.ggml.model")
    byte_of = {c: b for b, c in bytes_to_unicode().items()}
    words = []
    for token, kind in zip(tokens, types):
        if kind != _NORMAL:
            continue
        if model == "gpt2":
            try:
                words.append(bytes(byte_of[c] for c in token).decode())
            except (KeyError, UnicodeDecodeError):
                pass
        elif model == "llama":
            words.append(token.replace("\u2581", " "))
        else:
            words.append(token)
    return words


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf")
    ap.add_argument("hf", help="HF model dir or its tokenizer.json")
    ap.add_argument("--combos", type=int, default=20000)
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    from freetoken.models.gguf.reader import load_gguf_metadata
    from freetoken.models.gguf.tokenizer import load_gguf_tokenizer

    hf = Tokenizer.from_file(os.path.join(args.hf, "tokenizer.json") if os.path.isdir(args.hf) else args.hf)
    ours = load_gguf_tokenizer(args.gguf).backend_tokenizer
    meta = load_gguf_metadata(args.gguf)
    words = vocab_words(meta)
    rng = random.Random(0)
    combos = [rng.choice(["", " ", "\n"]).join(rng.choice(words) for _ in range(rng.randint(2, 4))) for _ in range(args.combos)]
    out = {"gguf": args.gguf, "pre": meta.get("tokenizer.ggml.pre")}
    differ = 0
    for name, texts in (("edge", EDGE), ("vocab", words), ("combos", combos)):
        ref = [e.ids for e in hf.encode_batch(texts, add_special_tokens=False)]
        got = [e.ids for e in ours.encode_batch(texts, add_special_tokens=False)]
        bad = [(t, r, g) for t, r, g in zip(texts, ref, got) if r != g]
        differ += len(bad)
        out[name] = {"texts": len(texts), "differ": len(bad), "examples": [{"text": t, "hf": r[:16], "gguf": g[:16]} for t, r, g in bad[: args.examples]]}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
