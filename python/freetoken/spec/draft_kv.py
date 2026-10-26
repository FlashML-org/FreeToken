"""A draft model's own paged KV, addressed through the target's page table."""

from __future__ import annotations

import torch

from freetoken.kvcache.mha_pool import MHAKVCache


class DraftPagedKV(MHAKVCache):
    """KV of a draft's attention layers on the target's slot ids: same page size, same pages plus the dummy page padded rows write to."""

    def __init__(self, num_kv_heads: int, head_dim: int, num_layers: int, num_pages: int, page_size: int,
                 dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(num_kv_heads=num_kv_heads, num_layers=num_layers, head_dim=head_dim,
                         num_pages=num_pages + 1, page_size=page_size, dtype=dtype, device=device)
