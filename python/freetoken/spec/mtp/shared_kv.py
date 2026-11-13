"""The target's paged KV as a read-only pool for a draft that stores none of its own.

llama.cpp gives the gemma4 assistant its draft KV from the target's cache without allocating any
(``llama_model::create_memory``'s KV-share callback): SWA draft layers alias the target's last SWA
slab, the full draft layer aliases the target's last full layer, and both slabs stay distinct. On
an E-series target each alias resolves through the layer's KV-sharing source, the layer the
shared layers' rows originate from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.kvcache.base import BaseKVCachePool

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class ReadOnlyTargetKVView(BaseKVCachePool):
    """Routes each draft layer to the target slab it shares; every write through it is a bug."""

    def __init__(self, target: BaseKVCachePool, draft_config: ModelConfig, target_config: ModelConfig) -> None:
        self._target = target
        swa_group = target_config.swa_attention_group()
        draft_swa = draft_config.swa_attention_group()
        swa_ids = tuple(swa_group.layer_ids) if swa_group is not None else ()
        full_ids = tuple(i for i in range(target_config.num_layers) if not target_config.is_swa_layer(i))
        if draft_swa is not None:
            if not swa_ids:
                raise ValueError(
                    f"the draft has sliding-window layers but target {target_config.architectures[0]} has no "
                    "sliding-window attention group for them to read"
                )
            # the draft reads the target's slab, so a wider window would address sentinel slots the target already freed
            if draft_swa.sliding_window > swa_group.sliding_window:
                raise RuntimeError(
                    f"the draft's sliding window ({draft_swa.sliding_window}) exceeds the target's "
                    f"({swa_group.sliding_window}); the draft would read slots the target has freed"
                )
        if not full_ids and any(not draft_config.is_swa_layer(i) for i in range(draft_config.num_layers)):
            raise ValueError(
                f"the draft has full-attention layers but target {target_config.architectures[0]} has no "
                "full-attention layer for them to read"
            )
        # llama.cpp's share callback: draft swa -> the target's last SWA layer, draft full -> its last full layer
        last_swa = max(swa_ids) if swa_ids else None
        last_full = max(full_ids) if full_ids else None
        e_args = getattr(target_config, "gemma4_e_args", None)
        self._route: list[int] = []
        for layer_id in range(draft_config.num_layers):
            layer = last_swa if draft_config.is_swa_layer(layer_id) else last_full
            # the hop is a no-op while shared target layers copy their rows into their own slab
            if e_args is not None and e_args.is_kv_shared(layer):
                sources = [i for i in target_config.attention_group_for_layer(layer).layer_ids if i < e_args.first_kv_shared_layer]
                if not sources:
                    raise RuntimeError(
                        f"target layer {layer} is KV-shared but its attention group has no source below the "
                        f"sharing boundary {e_args.first_kv_shared_layer} to route draft layer {layer_id}"
                    )
                layer = max(sources)
            self._route.append(layer)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._target.k_cache(self._route[index])

    def v_cache(self, index: int) -> torch.Tensor:
        return self._target.v_cache(self._route[index])

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int) -> None:
        raise RuntimeError(f"the read-only draft view stores nothing, but layer {layer_id} tried to write KV")

    @property
    def swa_paged(self) -> bool:
        return getattr(self._target, "swa_paged", False)

    def translate_loc_from_full_to_swa(self, out_loc: torch.Tensor) -> torch.Tensor:
        return self._target.translate_loc_from_full_to_swa(out_loc)

    @property
    def device(self) -> torch.device:
        return self._target.device

    @property
    def dtype(self) -> torch.dtype:
        return self._target.dtype

    @property
    def num_layers(self) -> int:
        return self._target.num_layers

    # a read window into a live pool: it is never priced, allocated or rebuilt
    @classmethod
    def kv_cost(cls, config, **kwargs) -> tuple[int, int, int, int]:
        raise NotImplementedError("ReadOnlyTargetKVView is not an allocatable pool")

    def rebuild_from_config(self, config, num_pages: int, *, num_swa_pages: int | None = None) -> None:
        raise NotImplementedError("ReadOnlyTargetKVView is rebuilt by the proposer on every attach")

    def unit_bytes(self) -> tuple[int, int]:
        raise NotImplementedError("ReadOnlyTargetKVView owns no buffers")
