"""Token masks of a schema DFA, the per-request matcher the scheduler advances, and a step's plan the sampler applies.

A token is allowed in a DFA state when its bytes walk to a live state; an end-of-sequence token only where the state
accepts. Special and added tokens are never allowed: the detokenizer may drop them, so their bytes are not the text.
A request whose prompt leaves a thinking span open is unconstrained until the span's close tag is committed.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .json_schema import SchemaDFA, SchemaError, compile_schema

# a matcher key: the DFA state, or one of these
FREE = -1  # inside an open thinking span: no mask
DEAD = 0  # a token no DFA state takes was committed or drafted: no mask (the row is never used)

_GRAMMARS = 8  # compiled schemas kept per vocabulary
_REFUSED = 64  # refused schemas remembered per vocabulary
_MASKS = 1024  # packed state masks kept per schema (about 31 KB each at a 248k vocabulary)
# a template that opens thinking itself does so within the prompt's last tokens
OPEN_TAIL = 8


def _byte_level_decoder() -> dict:
    """GPT-2's printable stand-in character -> the byte it stands for."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _decoder_types(tokenizer) -> set:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return set()
    decoder = json.loads(backend.to_str()).get("decoder") or {}
    found, stack = set(), [decoder]
    while stack:
        d = stack.pop()
        if isinstance(d, dict):
            found.add(d.get("type"))
            stack.extend(d.get("decoders") or [])
    return found


def token_bytes(tokenizer) -> List[bytes | None]:
    """Each token id's bytes as the detokenizer writes them, None for a token never allowed (special, added, or not
    representable); raises ValueError for a tokenizer whose token bytes cannot be read, or that lacks a one-byte token
    of a byte JSON text can hold (a mask could then allow a token no token can follow)."""
    out = _token_bytes(tokenizer)
    singles = {t[0] for t in out if t is not None and len(t) == 1}
    missing = [b for b in range(256) if b not in singles and b not in _NEVER_IN_UTF8]
    if missing:
        raise ValueError(f"json_schema needs a token for every byte; this tokenizer has none for {len(missing)}, e.g. {missing[0]:#04x}")
    return out


# bytes well-formed UTF-8 never holds
_NEVER_IN_UTF8 = frozenset([0xC0, 0xC1, *range(0xF5, 0x100)])


def _token_bytes(tokenizer) -> List[bytes | None]:
    kinds = _decoder_types(tokenizer)
    vocab = tokenizer.get_vocab()
    size = max(vocab.values()) + 1 if vocab else 0
    out: List[bytes | None] = [None] * size
    added = set(getattr(tokenizer, "added_tokens_decoder", {}) or {}) | set(getattr(tokenizer, "all_special_ids", []) or [])
    if "ByteLevel" in kinds:
        table = _byte_level_decoder()
        for text, i in vocab.items():
            if i in added:
                continue
            try:
                out[i] = bytes(table[c] for c in text)
            except KeyError:
                pass
        return out
    if "Metaspace" in kinds or "ByteFallback" in kinds:
        for text, i in vocab.items():
            if i in added:
                continue
            if len(text) == 6 and text.startswith("<0x") and text.endswith(">"):
                out[i] = bytes([int(text[3:5], 16)])
            else:
                out[i] = text.replace("▁", " ").encode("utf-8")
        return out
    raise ValueError(f"json_schema needs a byte-level or SentencePiece tokenizer; this one decodes with {sorted(k for k in kinds if k) or 'nothing known'}")


def think_ids(tokenizer, tags: Tuple[str, str] | None) -> Tuple[Tuple[int, ...], Tuple[int, ...]] | None:
    """The token ids of the thinking (open tag, close tag), None without a tag pair or when either has no tokens."""
    if tags is None:
        return None
    ids = tuple(tuple(int(t) for t in tokenizer.encode(tag, add_special_tokens=False)) for tag in tags)
    return (ids[0], ids[1]) if ids[0] and ids[1] else None


def _ends_with(tokens: Sequence[int], tail: Tuple[int, ...]) -> bool:
    n = len(tail)
    return len(tokens) >= n and tuple(tokens)[-n:] == tail


def starts_open(prompt_tail: Sequence[int], open_ids: Tuple[int, ...], close_ids: Tuple[int, ...]) -> bool:
    """Whether the prompt leaves thinking open: its last tokens hold the open tag with no close tag after it."""
    tail = list(prompt_tail)[-OPEN_TAIL:]
    for end in range(len(tail), 0, -1):
        if _ends_with(tail[:end], close_ids):
            return False
        if _ends_with(tail[:end], open_ids):
            return True
    return False


class Vocab:
    """The model's token bytes as padded arrays, grouped by first byte, and the schemas compiled against them."""

    def __init__(self, tokens: Sequence[bytes | None], width: int, eos_ids: Iterable[int]) -> None:
        self.width = width
        self.eos = np.array(sorted(i for i in eos_ids if 0 <= i < width), dtype=np.int64)
        ids = np.array([i for i, t in enumerate(tokens[:width]) if t], dtype=np.int64)
        lengths = np.array([len(tokens[i]) for i in ids], dtype=np.int64)
        longest = int(lengths.max()) if len(ids) else 1
        data = np.zeros((len(ids), longest), dtype=np.uint8)
        for row, i in enumerate(ids):
            data[row, : lengths[row]] = np.frombuffer(tokens[i], dtype=np.uint8)
        order = np.argsort(data[:, 0], kind="stable")
        self.ids, self.lengths, self.data = ids[order], lengths[order], data[order]
        self.first = np.searchsorted(self.data[:, 0], np.arange(257))
        self.bytes = list(tokens)
        self._grammars: OrderedDict[str, Grammar] = OrderedDict()
        # schema text -> why it was refused: a resent refused schema is not compiled again
        self._refused: OrderedDict[str, str] = OrderedDict()

    def grammar(self, schema_text: str) -> Grammar:
        """The schema's grammar, compiled on first use; raises SchemaError for a refused schema, from a cache on a
        resend."""
        if schema_text in self._refused:
            self._refused.move_to_end(schema_text)
            raise SchemaError(self._refused[schema_text])
        g = self._grammars.pop(schema_text, None)
        if g is None:
            try:
                g = Grammar(compile_schema(json.loads(schema_text)), self)
            except SchemaError as exc:
                self._refused[schema_text] = str(exc)
                while len(self._refused) > _REFUSED:
                    self._refused.popitem(last=False)
                raise
        self._grammars[schema_text] = g
        while len(self._grammars) > _GRAMMARS:
            self._grammars.popitem(last=False)
        return g


class Grammar:
    """One schema's DFA over one vocabulary, with each visited state's packed token mask."""

    def __init__(self, dfa: SchemaDFA, vocab: Vocab) -> None:
        self.dfa = dfa
        self.vocab = vocab
        self._masks: OrderedDict = OrderedDict()

    def step(self, state: int, token: int) -> int:
        data = self.vocab.bytes[token] if 0 <= token < len(self.vocab.bytes) else None
        return DEAD if not data else self.dfa.walk(state, data)

    def allowed(self, state: int) -> np.ndarray:
        """[width] bool: the tokens allowed in ``state``."""
        packed = self._masks.pop(state, None)
        if packed is None:
            packed = np.packbits(self._compute(state))
        self._masks[state] = packed
        while len(self._masks) > _MASKS:
            self._masks.popitem(last=False)
        return np.unpackbits(packed, count=self.vocab.width).view(bool)

    def _compute(self, state: int) -> np.ndarray:
        v, moves, cls = self.vocab, self.dfa.moves, self.dfa.byte_class
        mask = np.zeros(v.width, dtype=bool)
        live = np.flatnonzero(moves[state, cls])
        rows = np.concatenate([np.arange(v.first[b], v.first[b + 1]) for b in live]) if len(live) else np.zeros(0, np.int64)
        st = moves[state, cls[v.data[rows, 0]]]
        p = 1
        while len(rows):
            keep = st != DEAD
            rows, st = rows[keep], st[keep]
            done = v.lengths[rows] <= p
            mask[v.ids[rows[done]]] = True
            rows, st = rows[~done], st[~done]
            if not len(rows):
                break
            st = moves[st, cls[v.data[rows, p]]]
            p += 1
        if self.dfa.accepting[state]:
            mask[v.eos] = True
        if not mask.any():
            # no token goes on from here: end the reply rather than sample from nothing
            mask[v.eos] = True
        return mask


class Matcher:
    """One request's place in its grammar, advanced on the host by the tokens it commits."""

    def __init__(self, grammar: Grammar, close: Tuple[int, ...] = (), thinking: bool = False) -> None:
        self.grammar = grammar
        self.close = close
        self.key = FREE if thinking and close else 1
        self.tail: Tuple[int, ...] = ()

    def _step(self, key: int, tail: Tuple[int, ...], token: int) -> Tuple[int, Tuple[int, ...]]:
        if key == FREE:
            tail = (tail + (token,))[-len(self.close):]
            return (1, ()) if tail == self.close else (FREE, tail)
        if key == DEAD:
            return DEAD, ()
        return self.grammar.step(key, token), ()

    def advance(self, tokens: Iterable[int]) -> None:
        key, tail = self.key, self.tail
        for t in tokens:
            key, tail = self._step(key, tail, int(t))
        self.key, self.tail = key, tail

    def keys(self, drafts: Sequence[int]) -> List[int]:
        """The key of each verify row: row j reads the drafts before it as committed."""
        key, tail = self.key, self.tail
        out = [key]
        for t in drafts:
            key, tail = self._step(key, tail, int(t))
            out.append(key)
        return out


class GrammarPlan:
    """A step's constrained requests: ``matchers[i]`` for request i (None when it is free), ``width`` rows each."""

    def __init__(self, matchers: List[Matcher | None], width: int) -> None:
        self.matchers = matchers
        self.width = width

    def apply(self, logits: torch.Tensor, drafts: torch.Tensor | None = None) -> torch.Tensor:
        """``logits`` with every disallowed token of a constrained row at -inf, out of place; the same tensor when no
        row is constrained. A verify step passes its drafts [bs, k], which are read to the host here."""
        reqs = [i for i, m in enumerate(self.matchers) if m is not None]
        if drafts is not None and reqs:
            host = drafts[reqs].tolist()
        rows, masks = [], []
        for n, i in enumerate(reqs):
            m = self.matchers[i]
            keys = m.keys(host[n]) if drafts is not None else [m.key]
            for j, key in enumerate(keys):
                if key > DEAD:
                    rows.append(i * self.width + j)
                    masks.append(m.grammar.allowed(key))
        if not rows:
            return logits
        allowed = torch.from_numpy(np.stack(masks)).to(logits.device)
        width = logits.shape[1]
        if allowed.shape[1] != width:
            pad = allowed.new_zeros(len(rows), width)
            n = min(width, allowed.shape[1])
            pad[:, :n] = allowed[:, :n]
            allowed = pad
        index = torch.tensor(rows, dtype=torch.int64).to(logits.device)
        out = logits.clone()
        out[index] = logits[index].masked_fill(~allowed, float("-inf"))
        return out
