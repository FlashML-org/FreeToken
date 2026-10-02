"""Engram conditional memory (reference ``inference/engram.py`` + ``Engram``): n-gram hash lookups
added into the residual streams before layers 1 and 14.

Hashing (host side, ``EngramHash``): token ids are first mapped through a compressed vocabulary
(tokens that normalize alike collapse together), then each position is hashed with the
``max_ngram_size - 1`` tokens before it -- a rolling XOR of ``id * multiplier`` over the lookback,
one bucket range (a distinct prime modulus) per (n-gram order, head). Lookback stops at the sequence
start (pad id) and never crosses an image span. Addresses depend
only on the token ids, so the rows can be fetched before the forward starts.

Lookup (``EngramTable`` protocol): the 384M-row fp8 tables live on disk; a backend gathers the
``n_hash_cols`` rows of every token into a device tensor ``[T, n_hash_cols * head_dim]`` in bf16.

Gate (``EngramLayer``): ``wkv`` (fp8 linear) turns the gathered rows into one key per residual
stream plus a shared value; the gate is a normalized dot product of stream and key, signed-sqrt
then sigmoid; ``stream += gate * value``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from freetoken.layers import BaseOP, LinearReplicated

from .args import DeepseekV41Args


# ----- hashing ---------------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 3.3e24 (the bases below are a proven set), matching
    ``sympy.isprime`` on every 64-bit input."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    """The smallest prime above ``start`` not handed out yet (reference ``find_next_prime``)."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def bucket_primes(args: DeepseekV41Args) -> list[list[list[int]]]:
    """``[layer][ngram order - 2][head]`` bucket moduli: distinct primes drawn in order from
    ``engram_vocab_size - 1`` upward, never reused across layers / orders / heads."""
    primes: list[list[list[int]]] = []
    seen: set[int] = set()
    for _ in args.engram_layer_ids:
        per_ngram = []
        for _ in range(args.engram_max_ngram_size - 1):
            sizes, current = [], args.engram_vocab_size - 1
            for _ in range(args.engram_n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    return primes


def hash_multipliers(layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int) -> torch.Tensor:
    """``[n_layers, max_ngram_size]`` odd int64 multipliers from a per-layer RNG (reference
    ``compute_hash_multipliers``), bounded so ``id * multiplier`` cannot overflow int64."""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        values = gen.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Token id -> compressed id: tokens that normalize alike (NFKC, accents stripped, lowercased,
    whitespace collapsed) share one id. Reference ``build_compressed_token_map``; the size feeds the
    multipliers, so it is asserted against ``engram_compressed_vocab_size``."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


class EngramHash:
    """Maps token runs to table rows: ``row_ids(tokens) -> [T, n_layers, n_hash_cols]`` int64.

    ``token_map`` is the compressed vocabulary (``build_compressed_token_map``); ``pad_id`` is the
    compressed id of ``engram_pad_id``. Runs on the host (numpy / CPU torch) since the addresses
    are needed before the forward launches.
    """

    def __init__(self, args: DeepseekV41Args, token_map: list[int] | torch.Tensor, compressed_vocab_size: int):
        if compressed_vocab_size != args.engram_compressed_vocab_size:
            raise ValueError(
                f"compressed vocab has {compressed_vocab_size} ids, config says {args.engram_compressed_vocab_size}; "
                "the hash multipliers derive from it, so the tokenizer does not match the checkpoint"
            )
        self.args = args
        self.max_ngram = args.engram_max_ngram_size
        self.token_map = torch.as_tensor(token_map, dtype=torch.int64)
        self.pad_id = int(self.token_map[args.engram_pad_id])
        primes = bucket_primes(args)
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
        self.primes = torch.tensor(primes, dtype=torch.int64)  # [L, ngram-1, heads]
        self.offsets = torch.tensor(np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat]), dtype=torch.int64)  # [L, cols]
        self.multipliers = hash_multipliers(args.engram_layer_ids, self.max_ngram, compressed_vocab_size)  # [L, ngram]
        self.rows_per_layer = [int(o[-1] + s[-1]) for o, s in zip(self.offsets.tolist(), [[p for pn in layer for p in pn] for layer in primes])]

    @property
    def n_cols(self) -> int:
        return (self.max_ngram - 1) * self.args.engram_n_heads

    def row_ids(self, tokens: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """Rows for ``tokens [T]`` (raw ids, one contiguous run). ``context`` holds the
        ``max_ngram_size - 1`` raw ids immediately before the run (fewer -> the run starts a
        sequence and the missing lookback is the pad id). Returns ``[T, n_layers, n_hash_cols]``."""
        tokens = tokens.to(torch.int64).cpu()
        ctx_len = self.max_ngram - 1
        if context is None:
            context = tokens.new_empty(0)
        context = context.to(torch.int64).cpu()[-ctx_len:] if ctx_len else tokens.new_empty(0)
        raw = torch.cat([context, tokens])
        # Content-pad ids identify image spans; n-grams stop at them, including across chunks.
        dead = (raw < 0) | (raw >= self.token_map.numel())
        compressed = self.token_map[raw.clamp(0, self.token_map.numel() - 1)]
        n_ctx = context.numel()
        T = tokens.numel()
        positions = torch.arange(n_ctx, n_ctx + T)
        cols = []
        blocked = torch.zeros(T, dtype=torch.bool)
        for shift in range(self.max_ngram):
            src = positions - shift
            blocked |= (src < 0) | dead[src.clamp_min(0)]
            cols.append(torch.where(blocked, torch.full_like(src, self.pad_id), compressed[src.clamp_min(0)]))
        stack = torch.stack(cols, dim=-1)  # [T, max_ngram]; column s is the token s places back
        products = stack.unsqueeze(1) * self.multipliers  # [T, L, max_ngram]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])  # [T, L, heads]
        return torch.cat(hashes, dim=-1) + self.offsets  # [T, L, cols]


# ----- lookup ------------------------------------------------------------------------------------


class EngramTable(Protocol):
    """Row store of one Engram layer: ``lookup`` returns the dequantized rows of the current
    forward's tokens as ``[T, n_hash_cols * head_dim]`` bf16 (staged by the host before launch)."""

    def lookup(self, num_tokens: int) -> torch.Tensor: ...


class ZeroEngramTable:
    """Stand-in when no table is attached (dummy weights / unit tests): every lookup reads zeros."""

    def __init__(self, width: int, device: torch.device) -> None:
        self.width = width
        self.device = device

    def lookup(self, num_tokens: int) -> torch.Tensor:
        return torch.zeros(num_tokens, self.width, dtype=torch.bfloat16, device=self.device)


# ----- the layer -----------------------------------------------------------------------------------


class EngramLayer(BaseOP):
    """``streams [T, hc, dim] += gate * value`` with key / value from the gathered n-gram rows."""

    def __init__(self, args: DeepseekV41Args, layer_id: int, *, quant_config=None, prefix: str = ""):
        self.layer_id = layer_id
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.eps = args.norm_eps
        self.clamp_value = 1e-6
        self.width = args.engram_hash_cols * args.engram_head_dim
        self.wkv = LinearReplicated(self.width, args.dim * (args.hc_mult + 1), has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wkv")
        self.q_weight = torch.empty(args.hc_mult, args.dim, dtype=torch.float32)
        self.k_weight = torch.empty(args.hc_mult, args.dim, dtype=torch.float32)
        self._table: EngramTable | None = None

    def attach_table(self, table: EngramTable) -> None:
        self._table = table

    def forward(self, streams: torch.Tensor, rows: torch.Tensor | None = None, *, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``streams [T, hc, dim]``; ``rows`` overrides the table lookup (tests / reference parity)."""
        T = streams.shape[0]
        if rows is None:
            assert self._table is not None, "Engram table was never attached"
            rows = self._table.lookup(T)
        kv = self.wkv.forward(rows.to(streams.dtype))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().view(T, self.hc_mult, self.dim)
        weight = self.q_weight * self.k_weight  # only ever used as a product
        h = streams.float()
        # normalized per (token, stream) over dim, not jointly over the streams
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        out = (h + gate.unsqueeze(-1) * value.float().unsqueeze(1)).to(streams.dtype)
        return out if image_mask is None else torch.where(image_mask[:, None, None], streams, out)


__all__ = [
    "EngramHash",
    "EngramLayer",
    "EngramTable",
    "ZeroEngramTable",
    "build_compressed_token_map",
    "bucket_primes",
    "hash_multipliers",
]
