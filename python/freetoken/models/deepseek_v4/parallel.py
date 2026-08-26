"""Tensor-parallel helpers for DeepSeek-V4-Flash.

What DSV4 shards, and why:

* **MLA query/output heads.** ``wq_b`` is column-parallel over ``n_heads``,
  ``wo_a`` is sharded over the ``o_groups`` output groups, and ``wo_b`` is
  row-parallel with one all-reduce at the end of the block. ``o_groups`` is the
  binding constraint: a group owns ``n_heads // o_groups`` heads, so a rank must
  own whole groups.
* **MoE intermediate dimension.** The routed FP4 expert banks and the shared
  SwiGLU expert both split ``moe_inter_dim``. This is the memory win that makes
  the model fit: the host expert banks divide by the TP size, so 4 ranks hold
  the same 143 GB in total that 1 rank holds today.

What stays REPLICATED, and why:

* **The latent KV path** (``wkv``, ``kv_norm``, the paged pools). MLA keeps one
  latent KV per token that every head reads, so there is nothing to split; the
  KV pool cost per rank is unchanged.
* **The compressors and the Lightning Indexer.** They select which blocks the
  attention reads. Every rank must select the SAME blocks. Replicating them
  guarantees that without a collective on the selection path.
* **The router (``Gate``) and the hyper-connection mixers.** Small, and the
  routing decision must agree across ranks.
"""

from __future__ import annotations

import torch

from freetoken.distributed import try_get_tp_info


def tp_info() -> tuple[int, int]:
    """``(rank, size)`` for this process; ``(0, 1)`` before the engine sets it."""
    info = try_get_tp_info()
    return (0, 1) if info is None else (info.rank, info.size)


def tp_size() -> int:
    return tp_info()[1]


def tp_rank() -> int:
    return tp_info()[0]


def div_tp(total: int, what: str, *, multiple_of: int = 1) -> int:
    """Per-rank size of ``total``. Fails loudly when the split is not exact.

    ``multiple_of`` guards the block-quantized weights: a 128x128 FP8 scale block
    and a 32-element FP4 scale block cannot be cut in half.
    """
    size = tp_size()
    if total % size != 0:
        raise ValueError(
            f"DeepSeek-V4 tensor parallelism: {what}={total} is not divisible by "
            f"--tensor-parallel-size {size}"
        )
    local = total // size
    if local % multiple_of != 0:
        raise ValueError(
            f"DeepSeek-V4 tensor parallelism: {what}={total} split over {size} ranks "
            f"gives {local}, which is not a multiple of {multiple_of} (quantization "
            f"block size)"
        )
    return local


def shard(t: torch.Tensor, dim: int) -> torch.Tensor:
    """This rank's slice of ``t`` along ``dim``, in its OWN storage.

    The copy is the point. ``narrow`` returns a view, and on dim 0 that view is already
    contiguous, so ``.contiguous()`` hands it straight back -- keeping the whole parent
    tensor alive behind a 1/N-sized view. Every rank then pays for the full weight it
    just sharded: measured at 5.1 GiB per GPU on DeepSeek-V4-Flash at TP=4, silently
    charged to the model and taken out of the KV and expert-cache budget.
    """
    rank, size = tp_info()
    if size == 1:
        return t
    total = t.shape[dim]
    if total % size != 0:
        raise ValueError(
            f"cannot shard a tensor of shape {tuple(t.shape)} on dim {dim} over {size} ranks"
        )
    step = total // size
    return t.narrow(dim, rank * step, step).clone(memory_format=torch.contiguous_format)


def validate_tp(args) -> None:
    """Check every DSV4 split up front, so a bad ``--tensor-parallel-size`` fails
    at config time with one clear message instead of at the first bad reshape."""
    size = tp_size()
    if size == 1:
        return
    div_tp(args.o_groups, "o_groups")
    div_tp(args.n_heads, "n_heads")
    if args.n_heads % args.o_groups != 0:
        raise ValueError(
            f"DeepSeek-V4 expects n_heads ({args.n_heads}) to be a multiple of "
            f"o_groups ({args.o_groups})"
        )
    # wq_b output rows and the FP8 128x128 scale grid.
    div_tp(args.n_heads * args.head_dim, "n_heads*head_dim", multiple_of=128)
    # wo_b input columns.
    div_tp(args.o_groups * args.o_lora_rank, "o_groups*o_lora_rank", multiple_of=128)
    # Shared expert (FP8 128-blocks) and routed FP4 experts (32-element scale blocks).
    div_tp(args.moe_inter_dim, "moe_inter_dim", multiple_of=128)
    # Vocabulary-parallel embedding + head.
    div_tp(args.vocab_size, "vocab_size")


__all__ = ["div_tp", "shard", "tp_info", "tp_rank", "tp_size", "validate_tp"]
