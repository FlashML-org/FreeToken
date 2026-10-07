"""UVA row gather for an input-embedding table kept in pinned host memory (--embed-device cpu).

The table stays in page-locked host RAM and the GPU dereferences it in place over PCIe -- at
its host VA on Linux/UVA, at the mapped device address on WDDM (``kernel/pinned.device_ptr``).
The token ids never leave the device, so the lookup is CUDA-graph safe and works with the
overlap scheduler's device-side decode ids. A decode step reads a few rows (one hidden_size
row per token), so the PCIe cost is microseconds; a prefill chunk reads chunk x row bytes.

Grid: (tokens, column blocks). Ids outside ``[row_start, row_start + num_rows)`` (another TP
rank's vocab shard, or padding) store zeros.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_D = 1024


@triton.jit
def _host_embedding_gather_kernel(
    table_ptr,
    ids_ptr,
    out_ptr,
    row_start,
    num_rows,
    EMB_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = col < EMB_DIM
    idx = tl.load(ids_ptr + row).to(tl.int64) - row_start
    in_range = (idx >= 0) & (idx < num_rows)
    idx = tl.where(in_range, idx, 0)
    # the table is a host allocation: rebuild the typed pointer from the raw address
    base = table_ptr.to(tl.int64).to(tl.pointer_type(out_ptr.dtype.element_ty))
    values = tl.load(base + idx * EMB_DIM + col, mask=mask & in_range, other=0.0)
    tl.store(out_ptr + row.to(tl.int64) * EMB_DIM + col, values, mask=mask)


def host_embedding_gather(
    table_ptr: int,
    num_rows: int,
    embed_dim: int,
    ids: torch.Tensor,
    out: torch.Tensor,
    row_start: int = 0,
) -> torch.Tensor:
    """Gather the rows ``ids - row_start`` of the host table at ``table_ptr`` into ``out``.

    ``ids`` is a flat device int tensor; ``out`` is ``[ids.numel(), embed_dim]`` on the same
    device, in the table's dtype. ``table_ptr`` is the address the GPU must dereference
    (``kernel/pinned.device_ptr``), not necessarily the host ``data_ptr``.
    """
    n = ids.numel()
    assert out.shape == (n, embed_dim) and out.is_contiguous(), out.shape
    if n:
        block = min(_BLOCK_D, triton.next_power_of_2(embed_dim))
        _host_embedding_gather_kernel[(n, triton.cdiv(embed_dim, block))](
            table_ptr,
            ids,
            out,
            row_start,
            num_rows,
            EMB_DIM=embed_dim,
            BLOCK_D=block,
            num_warps=4,
        )
    return out


__all__ = ["host_embedding_gather"]
