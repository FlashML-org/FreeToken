"""Re-export of the shared quant-aware dense-linear factories (models/quant_linear.py).

The dispatch used to live here; muse_glimmer needing the same NVFP4/bf16 pair promoted
it to a model-agnostic home. Kept as a module so existing imports stay valid.
"""

from freetoken.models.quant_linear import (
    make_col_merged,
    make_col_merged_quant,
    make_replicated,
    make_replicated_quant,
    make_row_parallel,
    make_row_parallel_quant,
)

__all__ = [
    "make_col_merged_quant",
    "make_replicated_quant",
    "make_row_parallel_quant",
    "make_replicated",
    "make_col_merged",
    "make_row_parallel",
]
