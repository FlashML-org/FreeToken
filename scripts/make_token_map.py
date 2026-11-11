"""Write a --speculative-token-map file: the token ids the MTP draft head keeps.

    python scripts/make_token_map.py MODEL [--strategy low-id|topk] [--text FILE ...] [--size 106496] [-o token_map.bin]

MODEL is the target checkpoint, an HF directory or a GGUF file. Only its tokenizer is read, so every model that shares the tokenizer can use the same file.
The file holds all special tokens plus the strategy's first ids, --size ids in all, as little-endian int32.

Strategies:
- low-id: lowest ids first. BPE tokenizers mostly number tokens in merge order, so earlier tokens are usually the more common ones. Some tokenizers do not follow this (Gemma 4 puts single characters last); use topk for them.
- topk: the tokens that occur most often in the --text files, tokenized with MODEL's tokenizer. Tokens the text never uses follow in id order.

With --text, both strategies also print how much of that text the map covers.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from freetoken.utils.hf import load_tokenizer

# each strategy ranks every token id from its counts in --text, most wanted first
STRATEGIES = {
    "low-id": lambda counts: np.arange(counts.size),
    "topk": lambda counts: np.lexsort((np.arange(counts.size), -counts)),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="target checkpoint: an HF directory or a GGUF file")
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="low-id", help="how to rank the token ids (default: low-id)")
    p.add_argument("--text", nargs="+", default=[], metavar="FILE", help="UTF-8 text files to count tokens in; required by topk")
    p.add_argument("--size", type=int, default=106496, help="ids in the file, special tokens included (default: 106496, a multiple of 128 so the draft head needs no padding rows)")
    p.add_argument("-o", "--out", default="token_map.bin", help="output file (default: token_map.bin)")
    args = p.parse_args(argv)
    if args.strategy == "topk" and not args.text:
        p.error("--strategy topk needs --text")

    tokenizer = load_tokenizer(args.model)
    vocab = len(tokenizer)
    special = sorted(set(tokenizer.get_added_vocab().values()) | set(tokenizer.all_special_ids))
    if not len(special) <= args.size <= vocab:
        p.error(f"--size must be between the {len(special)} special tokens and the vocabulary size {vocab}")

    counts = np.zeros(vocab, dtype=np.int64)
    for path in args.text:
        with open(path, encoding="utf-8") as f:
            # small chunks stay under the tokenizer's max length, which it warns about
            while lines := f.readlines(1 << 16):
                counts += np.bincount(tokenizer.encode("".join(lines), add_special_tokens=False), minlength=vocab)

    taken = set(special)
    ranked = (int(i) for i in STRATEGIES[args.strategy](counts) if i not in taken)
    ids = np.array(special + [next(ranked) for _ in range(args.size - len(special))], dtype="<i4")
    ids.tofile(args.out)
    print(f"{args.out}: {ids.size} of {vocab} token ids ({100 * ids.size / vocab:.1f}%), strategy {args.strategy}")
    if counts.any():
        print(f"covers {100 * counts[ids].sum() / counts.sum():.2f}% of the {counts.sum()} tokens in --text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
