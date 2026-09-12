"""Global-to-local routed-expert ownership for tensor-parallel expert sharding.

This module deliberately knows nothing about cache slots or expert weights.  A global router
emits IDs in ``[0, global_num_experts)``; an owner maps those IDs to a local bank row, while the
LRU cache later maps local rows to cache slots.  Keeping the three namespaces separate avoids
feeding a remote global ID or a slot ID into a bank/GEMM kernel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


def same_device(left: torch.device, right: torch.device) -> bool:
    """Compare devices without spuriously failing on an unspecified index.

    ``torch.device("cuda") != torch.device("cuda:0")`` even though tensors allocated on
    either resolve to the same physical device. Cache adapters are often constructed with
    the bare ``"cuda"`` string while route tensors carry ``cuda:0``; treat an unspecified
    index as a match, but never let ``cuda:0`` and ``cuda:1`` compare equal.
    """
    if left.type != right.type:
        return False
    if left.index is None or right.index is None:
        return True
    return left.index == right.index


@dataclass(frozen=True)
class OwnedRoute:
    """A router decision masked for one expert owner.

    ``local_ids`` is always safe to use as a local-bank row index.  Remote entries use row zero
    as a harmless placeholder and have zero ``weights``; callers must not infer ownership from
    the placeholder.  The original router weights are never renormalized per rank.
    """

    local_ids: torch.Tensor
    weights: torch.Tensor
    owned_mask: torch.Tensor

    @property
    def owned_count(self) -> int:
        return int(self.owned_mask.sum().item())

    @property
    def remote_count(self) -> int:
        return int((~self.owned_mask).sum().item())


@dataclass(frozen=True)
class ExpertOwnership:
    """Contiguous expert ownership for one rank in an EP group.

    The first implementation intentionally requires an even partition.  An interleaved or
    history-weighted owner map can be added later, but it must preserve the same explicit
    global/local/slot boundary and be tested independently.
    """

    global_num_experts: int
    world_size: int
    rank: int

    def __post_init__(self) -> None:
        if self.global_num_experts <= 0:
            raise ValueError("global_num_experts must be positive")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} is outside [0, {self.world_size})")
        if self.global_num_experts % self.world_size:
            raise ValueError(
                f"global_num_experts={self.global_num_experts} is not divisible by "
                f"world_size={self.world_size}"
            )

    @property
    def local_num_experts(self) -> int:
        return self.global_num_experts // self.world_size

    @property
    def global_start(self) -> int:
        return self.rank * self.local_num_experts

    @property
    def global_end(self) -> int:
        return self.global_start + self.local_num_experts

    def owns(self, expert_id: int) -> bool:
        return self.global_start <= expert_id < self.global_end

    def owner(self, expert_id: int) -> int:
        if not 0 <= expert_id < self.global_num_experts:
            raise ValueError(
                f"global expert id {expert_id} is outside [0, {self.global_num_experts})"
            )
        return expert_id // self.local_num_experts

    def global_to_local(self, expert_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(local_ids, owned_mask)`` without changing route weights.

        ``local_ids`` is ``-1`` for remote entries.  Consumers must use ``owned_mask`` before
        indexing a local bank; no invalid sentinel may reach a GPU kernel.  The returned mask
        has the same shape as ``expert_ids`` and works on CPU or CUDA tensors.
        """
        ids = expert_ids.to(dtype=torch.int64)
        owned = (ids >= self.global_start) & (ids < self.global_end)
        local = torch.where(owned, ids - self.global_start, torch.full_like(ids, -1))
        return local.to(dtype=expert_ids.dtype), owned

    def local_to_global(self, local_ids: torch.Tensor) -> torch.Tensor:
        """Convert valid local bank rows back to global router IDs."""
        ids = local_ids.to(dtype=torch.int64)
        if torch.any((ids < 0) | (ids >= self.local_num_experts)):
            raise ValueError("local expert IDs must be in [0, local_num_experts)")
        return (ids + self.global_start).to(dtype=local_ids.dtype)

    def validate_global_ids(self, expert_ids: torch.Tensor) -> None:
        """Fail fast on malformed router IDs before owner masking/remapping."""
        ids = expert_ids.to(dtype=torch.int64)
        if torch.any((ids < 0) | (ids >= self.global_num_experts)):
            raise ValueError(
                f"router expert IDs must be in [0, {self.global_num_experts})"
            )

    def partition_route(
        self, weights: torch.Tensor, expert_ids: torch.Tensor
    ) -> OwnedRoute:
        """Mask a global top-k route for this owner without changing its probabilities.

        The returned route is an adapter contract, not a cache implementation: ``local_ids``
        address the owner's source-bank rows, while a future cache layer must create its own
        local-row-to-slot mapping.  This method deliberately preserves duplicate route entries
        and the global top-k weights.  In particular, a rank with zero local experts receives a
        valid all-zero route rather than an invalid ``-1`` index.
        """
        if weights.shape != expert_ids.shape:
            raise ValueError(
                f"route weights and expert IDs must have the same shape, got "
                f"{tuple(weights.shape)} and {tuple(expert_ids.shape)}"
            )
        if expert_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError(f"router expert IDs must be an integer tensor, got {expert_ids.dtype}")
        self.validate_global_ids(expert_ids)
        local_ids, owned = self.global_to_local(expert_ids)
        safe_local_ids = torch.where(owned, local_ids, torch.zeros_like(local_ids))
        local_weights = torch.where(owned, weights, torch.zeros_like(weights))
        return OwnedRoute(
            local_ids=safe_local_ids,
            weights=local_weights,
            owned_mask=owned,
        )


@dataclass(frozen=True)
class OwnerCacheGeometry:
    """Explicit geometry contract for a future owner-local expert cache.

    The current :class:`~freetoken.moe.offload_cache.OffloadMoeCache` uses one global expert
    namespace for route IDs, source-bank rows, and cache IDs.  An EP owner cache must not reuse
    that scalar for ``local_num_experts``: global router IDs first become local bank rows, and
    only then become cache slots.  This contract describes those dimensions without enabling the
    runtime path prematurely.

    ``cache_size`` is the number of local expert slots on this rank.  It is deliberately checked
    against the local expert count, not the global count.  Prefill overlap borrows two complete
    local layers from the unified pool, hence its separate ``2 * local_num_experts`` minimum.
    """

    global_num_experts: int
    world_size: int
    rank: int
    num_layers: int
    cache_size: int
    prefill_overlap: bool = False

    def __post_init__(self) -> None:
        owner = ExpertOwnership(self.global_num_experts, self.world_size, self.rank)
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.cache_size < owner.local_num_experts:
            raise ValueError(
                f"cache_size={self.cache_size} is smaller than local_num_experts="
                f"{owner.local_num_experts}"
            )
        if self.prefill_overlap and self.cache_size < 2 * owner.local_num_experts:
            raise ValueError(
                "prefill_overlap requires cache_size >= 2 * local_num_experts "
                f"({2 * owner.local_num_experts}), got {self.cache_size}"
            )

    @property
    def ownership(self) -> ExpertOwnership:
        return ExpertOwnership(self.global_num_experts, self.world_size, self.rank)

    @property
    def local_num_experts(self) -> int:
        return self.ownership.local_num_experts

    @property
    def global_start(self) -> int:
        return self.ownership.global_start

    @property
    def global_end(self) -> int:
        return self.ownership.global_end

    def validate_cache_binding(
        self,
        *,
        num_layers: int,
        num_experts: int,
        cache_size: int,
        prefill_overlap: bool,
    ) -> None:
        """Validate a cache constructor's global geometry before any allocation.

        ``num_experts`` is intentionally compared with the *global* dimension.  A caller that
        passes ``local_num_experts`` to the legacy cache will fail here instead of silently
        corrupting layer-flat IDs or allocating a bank with the wrong row count.
        """
        expected = {
            "num_layers": (num_layers, self.num_layers),
            "num_experts": (num_experts, self.global_num_experts),
            "cache_size": (cache_size, self.cache_size),
        }
        for name, (actual, wanted) in expected.items():
            if actual != wanted:
                raise ValueError(
                    f"owner cache geometry mismatch for {name}: got {actual}, expected {wanted}"
                )
        if bool(prefill_overlap) != self.prefill_overlap:
            raise ValueError(
                "owner cache geometry mismatch for prefill_overlap: "
                f"got {bool(prefill_overlap)}, expected {self.prefill_overlap}"
            )

    def global_to_local(self, expert_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map global route IDs to local rows and return the ownership mask."""
        return self.ownership.global_to_local(expert_ids)

    def partition_route(
        self, weights: torch.Tensor, expert_ids: torch.Tensor
    ) -> OwnedRoute:
        """Return this rank's safe local-row route without local renormalization."""
        return self.ownership.partition_route(weights, expert_ids)

    def local_to_flat_id(self, layer_id: int, local_ids: torch.Tensor) -> torch.Tensor:
        """Map a local bank row to a layer-local cache ID namespace.

        This ID is only a contract for an owner-aware cache implementation.  It must not be
        passed to today's global-ID cache kernels, whose stride is the global expert count.
        """
        if not 0 <= layer_id < self.num_layers:
            raise ValueError(f"layer_id {layer_id} is outside [0, {self.num_layers})")
        ids = local_ids.to(dtype=torch.int64)
        if torch.any((ids < 0) | (ids >= self.local_num_experts)):
            raise ValueError(
                f"local expert IDs must be in [0, {self.local_num_experts})"
            )
        return (layer_id * self.local_num_experts + ids).to(dtype=local_ids.dtype)

    def flat_id_to_local(self, flat_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a local flat cache ID into ``(layer_id, local_expert_id)``.

        The inverse is intentionally defined only for the owner-local namespace.  It is a
        future cache adapter contract and must not be used to decode today's global-stride
        ``OffloadMoeCache.id_of_slot`` values.
        """
        ids = flat_ids.to(dtype=torch.int64)
        total = self.num_layers * self.local_num_experts
        if torch.any((ids < 0) | (ids >= total)):
            raise ValueError(f"owner flat IDs must be in [0, {total})")
        layers = torch.div(ids, self.local_num_experts, rounding_mode="floor")
        local = torch.remainder(ids, self.local_num_experts)
        return layers.to(dtype=flat_ids.dtype), local.to(dtype=flat_ids.dtype)

    def global_to_local_flat(
        self, layer_id: int, expert_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map global route IDs to safe owner-local flat IDs plus an ownership mask.

        Remote entries use the layer's local row zero as a safe placeholder before flattening;
        callers must apply the returned mask/weights and must never infer ownership from the
        placeholder.  The resulting IDs use ``layer * local_num_experts + local_row`` and are
        not valid inputs to the legacy global-stride cache kernels.
        """
        local, owned = self.global_to_local(expert_ids)
        safe_local = torch.where(owned, local, torch.zeros_like(local))
        return self.local_to_flat_id(layer_id, safe_local), owned

    def validate_source_banks(
        self, sources: Mapping[str, Sequence[torch.Tensor]]
    ) -> None:
        """Check that every bank has one local-row tensor per layer.

        This accepts arbitrary bank names and trailing dimensions so it can validate BF16,
        NVFP4, and future layouts without coupling the ownership contract to a quantizer.
        """
        if not sources:
            raise ValueError("owner source banks must not be empty")
        for name, per_layer in sources.items():
            if len(per_layer) != self.num_layers:
                raise ValueError(
                    f"bank {name!r} has {len(per_layer)} layers, expected {self.num_layers}"
                )
            for layer_id, bank in enumerate(per_layer):
                if bank.ndim == 0 or bank.shape[0] != self.local_num_experts:
                    got = tuple(bank.shape) if hasattr(bank, "shape") else type(bank).__name__
                    raise ValueError(
                        f"bank {name!r} layer {layer_id} has row shape {got}; "
                        f"expected first dimension {self.local_num_experts}"
                    )

    def validate_slot_maps(
        self, slot_for_id: torch.Tensor, id_of_slot: torch.Tensor
    ) -> None:
        """Check the expected local-row-to-slot and reverse-map shapes."""
        expected_forward = (self.num_layers, self.local_num_experts)
        if tuple(slot_for_id.shape) != expected_forward:
            raise ValueError(
                f"owner slot_for_id shape {tuple(slot_for_id.shape)} does not match "
                f"{expected_forward}"
            )
        if tuple(id_of_slot.shape) != (self.cache_size,):
            raise ValueError(
                f"owner id_of_slot shape {tuple(id_of_slot.shape)} does not match "
                f"({self.cache_size},)"
            )
        integer_types = (torch.int8, torch.int16, torch.int32, torch.int64)
        if slot_for_id.dtype not in integer_types or id_of_slot.dtype not in integer_types:
            raise TypeError("owner slot maps must use integer tensors")
        for name, values, upper in (
            ("slot_for_id", slot_for_id, self.cache_size),
            ("id_of_slot", id_of_slot, self.num_layers * self.local_num_experts),
        ):
            values = values.to(dtype=torch.int64)
            if torch.any((values < -1) | (values >= upper)):
                raise ValueError(
                    f"owner {name} entries must be -1 or in [0, {upper})"
                )


@dataclass(frozen=True)
class OwnerCacheUpdate:
    """Result of one owner-local cache admission in the reference adapter.

    ``slot_ids`` is safe for an owner-local GEMM: remote route entries use slot zero and have
    zero ``weights``. ``local_flat_ids`` is the local namespace used by the future cache
    bookkeeping, not the global-stride ID understood by today's legacy cache.
    """

    slot_ids: torch.Tensor
    local_ids: torch.Tensor
    local_flat_ids: torch.Tensor
    weights: torch.Tensor
    owned_mask: torch.Tensor
    missing_local_ids: torch.Tensor
    evicted_flat_ids: torch.Tensor


class OwnerCacheAdapter:
    """Small deterministic owner-local cache reference implementation.

    This adapter deliberately runs ordinary Python/Torch bookkeeping instead of flashlib or
    Triton kernels. It is the executable P3 contract for the three namespaces and is intended
    for tests and later GPU-kernel bring-up; it is not wired into the serving path yet.
    """

    def __init__(
        self, geometry: OwnerCacheGeometry, device: torch.device | str = "cpu"
    ) -> None:
        self.geometry = geometry
        self.device = torch.device(device)
        self.slot_for_id = torch.full(
            (geometry.num_layers, geometry.local_num_experts),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.id_of_slot = torch.full(
            (geometry.cache_size,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.usage = torch.zeros(
            (geometry.cache_size,), dtype=torch.int64, device=self.device
        )
        self.step = 0

    @property
    def cache_size(self) -> int:
        return self.geometry.cache_size

    @property
    def local_num_experts(self) -> int:
        return self.geometry.local_num_experts

    @property
    def resident(self) -> int:
        return int(self.id_of_slot.ge(0).sum().item())

    def _check_layer(self, layer_id: int) -> None:
        if not 0 <= layer_id < self.geometry.num_layers:
            raise ValueError(
                f"layer_id {layer_id} is outside [0, {self.geometry.num_layers})"
            )

    def _check_input_device(self, *tensors: torch.Tensor) -> None:
        for tensor in tensors:
            if not same_device(tensor.device, self.device):
                raise ValueError(
                    f"owner cache input is on {tensor.device}, expected {self.device}"
                )

    def _flat_id(self, layer_id: int, local_id: int) -> int:
        return layer_id * self.local_num_experts + local_id

    def _clear_slot(self, slot: int) -> int:
        old_flat = int(self.id_of_slot[slot].item())
        if old_flat < 0:
            return old_flat
        old_layer, old_local = divmod(old_flat, self.local_num_experts)
        self.slot_for_id[old_layer, old_local] = -1
        self.id_of_slot[slot] = -1
        self.usage[slot] = 0
        return old_flat

    def _admit(self, layer_id: int, local_ids: list[int]) -> tuple[list[int], list[int]]:
        """Admit one layer's unique local rows and return misses and evicted flat IDs."""
        self.step += 1
        step = self.step
        active_slots: set[int] = set()
        misses: list[int] = []
        for local_id in local_ids:
            slot = int(self.slot_for_id[layer_id, local_id].item())
            if slot < 0:
                misses.append(local_id)
            else:
                self.usage[slot] = step
                active_slots.add(slot)

        evicted: list[int] = []
        reserved = set(active_slots)
        for local_id in misses:
            candidates = [slot for slot in range(self.cache_size) if slot not in reserved]
            if not candidates:
                # A single layer can address at most local_num_experts and geometry guarantees
                # cache_size >= local_num_experts, so this is defensive rather than a normal
                # route. Failing is safer than evicting a hit from the same admission.
                raise RuntimeError("owner cache has no slot for an active local route")
            slot = min(candidates, key=lambda value: (int(self.usage[value].item()), value))
            old_flat = self._clear_slot(slot)
            if old_flat >= 0:
                evicted.append(old_flat)
            flat = self._flat_id(layer_id, local_id)
            self.id_of_slot[slot] = flat
            self.slot_for_id[layer_id, local_id] = slot
            self.usage[slot] = step
            reserved.add(slot)
        return misses, evicted

    def ensure_route(
        self, layer_id: int, weights: torch.Tensor, expert_ids: torch.Tensor
    ) -> OwnerCacheUpdate:
        """Admit a global route and rewrite owned entries to owner-local cache slots.

        The route weights remain globally normalized. Remote entries are masked to zero and
        use slot zero as a safe placeholder, so no ``-1`` sentinel can reach a future GEMM.
        """
        self._check_layer(layer_id)
        self._check_input_device(weights, expert_ids)
        route = self.geometry.partition_route(weights, expert_ids)
        local_flat_ids, owned = self.geometry.global_to_local_flat(layer_id, expert_ids)
        local_flat_ids = torch.where(
            owned,
            local_flat_ids,
            torch.zeros_like(local_flat_ids),
        )
        owned_local = route.local_ids.reshape(-1)[owned.reshape(-1)]
        unique_local = list(dict.fromkeys(int(value) for value in owned_local.tolist()))
        missing, evicted = self._admit(layer_id, unique_local)

        slots = self.slot_for_id[layer_id][route.local_ids.long()]
        safe_slots = torch.where(owned, slots, torch.zeros_like(slots))
        update = OwnerCacheUpdate(
            slot_ids=safe_slots,
            local_ids=route.local_ids,
            local_flat_ids=local_flat_ids,
            weights=route.weights,
            owned_mask=route.owned_mask,
            missing_local_ids=torch.tensor(
                missing, dtype=torch.int32, device=self.device
            ),
            evicted_flat_ids=torch.tensor(
                evicted, dtype=torch.int32, device=self.device
            ),
        )
        self.validate_invariants()
        return update

    def materialize_layer(self, layer_id: int, buffer_id: int = 0) -> torch.Tensor:
        """Materialize every local expert of one layer into a contiguous slot buffer."""
        self._check_layer(layer_id)
        if self.geometry.prefill_overlap:
            if buffer_id not in (0, 1):
                raise ValueError("owner prefill buffer_id must be 0 or 1")
        elif buffer_id != 0:
            raise ValueError("owner prefill overlap is disabled; buffer_id must be 0")

        first = buffer_id * self.local_num_experts
        target = list(range(first, first + self.local_num_experts))
        for local_id in range(self.local_num_experts):
            old_slot = int(self.slot_for_id[layer_id, local_id].item())
            if old_slot >= 0 and old_slot not in target:
                self._clear_slot(old_slot)
        for slot in target:
            self._clear_slot(slot)

        self.step += 1
        for local_id, slot in enumerate(target):
            flat = self._flat_id(layer_id, local_id)
            self.id_of_slot[slot] = flat
            self.slot_for_id[layer_id, local_id] = slot
            self.usage[slot] = self.step
        self.validate_invariants()
        return torch.tensor(target, dtype=torch.int32, device=self.device)

    def reset(self) -> None:
        self.slot_for_id.fill_(-1)
        self.id_of_slot.fill_(-1)
        self.usage.zero_()
        self.step = 0

    def validate_invariants(self) -> None:
        """Verify forward/reverse maps are a bijection for all resident entries."""
        self.geometry.validate_slot_maps(self.slot_for_id, self.id_of_slot)
        for layer_id in range(self.geometry.num_layers):
            for local_id in range(self.local_num_experts):
                slot = int(self.slot_for_id[layer_id, local_id].item())
                if slot < 0:
                    continue
                flat = self._flat_id(layer_id, local_id)
                if int(self.id_of_slot[slot].item()) != flat:
                    raise AssertionError(
                        f"owner cache forward map mismatch at layer={layer_id}, "
                        f"local_id={local_id}, slot={slot}"
                    )
        for slot, flat in enumerate(self.id_of_slot.tolist()):
            if flat < 0:
                continue
            layer, local = divmod(flat, self.local_num_experts)
            if int(self.slot_for_id[layer, local].item()) != slot:
                raise AssertionError(
                    f"owner cache reverse map mismatch at slot={slot}, flat_id={flat}"
                )


__all__ = [
    "ExpertOwnership",
    "OwnedRoute",
    "OwnerCacheGeometry",
    "OwnerCacheUpdate",
    "OwnerCacheAdapter",
]
