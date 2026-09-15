# SPDX-License-Identifier: Apache-2.0
"""Fused n-gram hash -> PLE table row ids.

The eager form of this hash (``NGramEmbedding._window`` + ``_shift_ignore_eos`` + the per-ngram
XOR/multiply/remainder/offset loop) is 39 tiny CUDA kernels per PLE layer per step -- ~400 us of
launch wall for a few us of GPU work. This is the same arithmetic as one kernel, one program per
token; ``NGramEmbedding.row_ids_reference`` keeps the torch-op form as the oracle and CPU path.

The key observation that collapses the ``cummax``-over-the-whole-window boundary scan into a
``ngram_size-1`` step walk: ``_shift_ignore_eos`` marks shift ``s`` valid at position ``p`` iff
``p-s >= 0`` and no boundary token sits anywhere in ``[p-s, p-1]``. Only shifts ``< ngram_size``
are ever used, so the scan never needs to look further back than that, and the predicate is
built incrementally as the walk goes.

Window addressing avoids materializing the ``[B, ctx+max_len]`` packed window entirely. Token
``t`` of the forward belongs to request ``req[t]`` at intra-request offset ``local[t]``, so the
token ``s`` places to its left is ``input_ids[t - s]`` when ``local[t] >= s`` and
``ngram_context[req[t], ctx_len + local[t] - s]`` otherwise -- out of range on the left is the
boundary token, exactly as the eos-filled packed window was.

Capture-safe: fixed shapes, every input on device, no host reads.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ple_row_ids_kernel(
    ids_ptr,  # [T] int64 -- this forward's tokens, ragged, in request order
    ctx_ptr,  # [B, CTX_LEN] int64 -- the tokens immediately before each request's first
    req_ptr,  # [T] int32 -- request index of each token
    local_ptr,  # [T] int32 -- intra-request offset of each token
    mult_ptr,  # [NGRAM] int64
    vocab_ptr,  # [NUM_HEADS] int64
    off_ptr,  # [NUM_HEADS] int64
    out_ptr,  # [T, NUM_HEADS] int64
    EOS: tl.constexpr,
    CTX_LEN: tl.constexpr,
    NGRAM: tl.constexpr,
    HEADS_PER: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    req = tl.load(req_ptr + token).to(tl.int64)
    local = tl.load(local_ptr + token).to(tl.int64)

    head = tl.arange(0, BLOCK_H)
    head_ok = head < NUM_HEADS

    # shift 0 is always the token itself; the eos-crossing rule never masks it
    mixed = tl.load(ids_ptr + token).to(tl.int64) * tl.load(mult_ptr).to(tl.int64)
    acc = tl.zeros([BLOCK_H], dtype=tl.int64)

    valid = 1
    for shift in tl.static_range(1, NGRAM):
        column = CTX_LEN + local - shift  # column in the (virtual) packed window
        from_ids = column >= CTX_LEN
        from_ctx = (column >= 0) & (column < CTX_LEN)
        token_ids = tl.load(ids_ptr + (token - shift), mask=from_ids, other=0)
        token_ctx = tl.load(ctx_ptr + req * CTX_LEN + column, mask=from_ctx, other=EOS)
        raw = tl.where(from_ids, token_ids, token_ctx).to(tl.int64)
        # the window may not cross a boundary token, and a boundary token is itself the wall
        valid = valid * tl.where((column >= 0) & (raw != EOS), 1, 0)
        mixed = mixed ^ (
            tl.where(valid == 1, raw, EOS) * tl.load(mult_ptr + shift).to(tl.int64)
        )
        # after ``shift`` taps the (shift+1)-gram mix is complete; it owns one head block
        ngram = shift + 1
        block = (head >= (ngram - 2) * HEADS_PER) & (head < (ngram - 1) * HEADS_PER)
        acc = tl.where(block, mixed, acc)

    vocab = tl.load(vocab_ptr + head, mask=head_ok, other=1).to(tl.int64)
    offset = tl.load(off_ptr + head, mask=head_ok, other=0).to(tl.int64)
    # torch.remainder is floored, triton's % is truncated; the divisor is always positive
    rem = acc % vocab
    rem = tl.where(rem < 0, rem + vocab, rem)
    tl.store(out_ptr + token * NUM_HEADS + head, rem + offset, mask=head_ok)


def ple_row_ids(
    input_ids: torch.Tensor,
    ngram_context: torch.Tensor,
    req_index: torch.Tensor,
    local_index: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    *,
    eos_token_id: int,
    heads_per_ngram: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``[T, num_heads]`` int64 global table rows for this forward's tokens.

    ``input_ids`` [T] and ``ngram_context`` [B, ngram_size-1] are int64 device tensors;
    ``req_index`` / ``local_index`` are [T] int32 device tensors naming each token's request and
    its offset within that request. The three hash constant tensors are int64 and on the same
    device. ``out``, when given, is the destination (a CUDA graph replays into a fixed buffer).
    """
    tokens = input_ids.numel()
    ngram_size = int(multipliers.numel())
    num_heads = int(vocab_sizes.numel())
    ctx_len = int(ngram_context.shape[-1])
    # Checked as raises, not asserts: the kernel addresses the context row and the head
    # blocks by these, and ``python -O`` must not turn a geometry mismatch into an OOB read.
    if ctx_len != ngram_size - 1:
        raise ValueError(
            f"PLE hash: ngram_context has {ctx_len} context ids but ngram_size {ngram_size} "
            f"needs {ngram_size - 1}"
        )
    if num_heads != heads_per_ngram * (ngram_size - 1):
        raise ValueError(
            f"PLE hash: {num_heads} heads is not heads_per_ngram {heads_per_ngram} x "
            f"{ngram_size - 1} n-gram orders"
        )
    if out is None:
        out = torch.empty((tokens, num_heads), dtype=torch.int64, device=input_ids.device)
    if tokens == 0:
        return out
    _ple_row_ids_kernel[(tokens,)](
        input_ids,
        ngram_context,
        req_index,
        local_index,
        multipliers,
        vocab_sizes,
        offsets,
        out,
        EOS=int(eos_token_id),
        CTX_LEN=ctx_len,
        NGRAM=ngram_size,
        HEADS_PER=int(heads_per_ngram),
        NUM_HEADS=num_heads,
        BLOCK_H=triton.next_power_of_2(num_heads),
        num_warps=1,
    )
    return out


__all__ = ["ple_row_ids"]
