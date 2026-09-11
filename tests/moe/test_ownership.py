"""CPU contracts for the first TP+EP ownership boundary.

These tests intentionally stop before slot-cache admission: global router IDs, local bank rows,
and cache slots are separate namespaces until the EP runtime path is complete.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    iter_nvfp4_expert_pieces,
)
from freetoken.moe.ownership import (
    ExpertOwnership,
    OwnerCacheAdapter,
    OwnerCacheGeometry,
)


_GENERIC_RE = re.compile(
    r"^layer\.(?P<layer>\d+)\.expert\.(?P<expert>\d+)\."
    r"(?P<proj>gate|up|down)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_GENERIC_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_GENERIC_RE,
    proj_to_role={"gate": "gate", "up": "up", "down": "down"},
    layer_to_bank=lambda layer, _config: layer,
    desc="test NVFP4 experts",
)


def test_contiguous_ownership_covers_global_experts_once():
    owners = [ExpertOwnership(8, 2, rank) for rank in range(2)]
    assert [o.local_num_experts for o in owners] == [4, 4]
    assert [o.global_start for o in owners] == [0, 4]
    assert [o.global_end for o in owners] == [4, 8]
    assert [owners[0].owner(i) for i in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]

    for rank, owner in enumerate(owners):
        local, mask = owner.global_to_local(torch.arange(8, dtype=torch.int32))
        expected = list(range(4)) if rank == 0 else [-1] * 4 + list(range(4))
        if rank == 0:
            expected = list(range(4)) + [-1] * 4
        assert local.tolist() == expected
        assert int(mask.sum()) == 4
        assert owner.local_to_global(torch.arange(4, dtype=torch.int32)).tolist() == list(
            range(owner.global_start, owner.global_end)
        )


def test_ownership_rejects_invalid_geometry_and_ids():
    with pytest.raises(ValueError, match="not divisible"):
        ExpertOwnership(5, 2, 0)
    with pytest.raises(ValueError, match="outside"):
        ExpertOwnership(8, 2, 2)

    owner = ExpertOwnership(8, 2, 0)
    with pytest.raises(ValueError, match="global expert id"):
        owner.owner(8)
    with pytest.raises(ValueError, match="local expert IDs"):
        owner.local_to_global(torch.tensor([-1]))
    with pytest.raises(ValueError, match="router expert IDs"):
        owner.validate_global_ids(torch.tensor([0, 8]))


@pytest.mark.parametrize(
    "route_ids",
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 7],  # 5/5, including duplicates
        [0] * 10,  # rank 0 owns all entries
        [4] * 10,  # rank 1 owns all entries
    ],
)
def test_partition_route_masks_remote_entries_without_local_renormalization(route_ids):
    ids = torch.tensor([route_ids], dtype=torch.int32)
    weights = torch.arange(1, 11, dtype=torch.float32).reshape(1, 10) / 55
    owners = [ExpertOwnership(8, 2, rank) for rank in range(2)]
    routes = [owner.partition_route(weights, ids) for owner in owners]

    assert torch.equal(routes[0].weights + routes[1].weights, weights)
    assert torch.equal(routes[0].owned_mask | routes[1].owned_mask, torch.ones_like(ids, dtype=torch.bool))
    assert torch.equal(routes[0].owned_mask & routes[1].owned_mask, torch.zeros_like(ids, dtype=torch.bool))
    for route in routes:
        assert torch.all(route.local_ids >= 0)
        assert torch.all(route.local_ids < 4)
        assert torch.equal(route.weights[~route.owned_mask], torch.zeros_like(route.weights[~route.owned_mask]))


def test_partition_route_zero_local_entries_uses_safe_placeholder():
    owner = ExpertOwnership(8, 2, 0)
    ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.int32)
    weights = torch.full((1, 4), 0.25, dtype=torch.float32)
    route = owner.partition_route(weights, ids)
    assert route.owned_count == 0 and route.remote_count == 4
    assert torch.equal(route.local_ids, torch.zeros_like(ids))
    assert torch.equal(route.weights, torch.zeros_like(weights))


def test_partition_route_rejects_shape_and_dtype_mismatches():
    owner = ExpertOwnership(8, 2, 0)
    with pytest.raises(ValueError, match="same shape"):
        owner.partition_route(torch.ones(2), torch.zeros(1, dtype=torch.int32))
    with pytest.raises(TypeError, match="integer tensor"):
        owner.partition_route(torch.ones(1), torch.zeros(1, dtype=torch.float32))


def test_owner_cache_geometry_separates_global_local_and_flat_namespaces():
    geometry = OwnerCacheGeometry(
        global_num_experts=8,
        world_size=2,
        rank=1,
        num_layers=3,
        cache_size=8,
        prefill_overlap=True,
    )

    assert geometry.local_num_experts == 4
    assert (geometry.global_start, geometry.global_end) == (4, 8)
    ids = torch.tensor([[0, 4, 7]], dtype=torch.int32)
    local, owned = geometry.global_to_local(ids)
    assert local.tolist() == [[-1, 0, 3]]
    assert owned.tolist() == [[False, True, True]]
    flat, flat_owned = geometry.global_to_local_flat(2, ids)
    assert flat.tolist() == [[8, 8, 11]]
    assert torch.equal(flat_owned, owned)
    decoded_layer, decoded_local = geometry.flat_id_to_local(flat)
    assert decoded_layer.tolist() == [[2, 2, 2]]
    assert decoded_local.tolist() == [[0, 0, 3]]
    assert geometry.local_to_flat_id(2, torch.tensor([0, 3], dtype=torch.int32)).tolist() == [8, 11]
    all_flat = torch.arange(geometry.num_layers * geometry.local_num_experts, dtype=torch.int32)
    layers, local_rows = geometry.flat_id_to_local(all_flat)
    rebuilt = torch.cat([
        geometry.local_to_flat_id(layer, local_rows[layers == layer])
        for layer in range(geometry.num_layers)
    ])
    assert torch.equal(rebuilt, all_flat)
    assert layers.tolist() == [0] * 4 + [1] * 4 + [2] * 4
    with pytest.raises(ValueError, match="owner flat IDs"):
        geometry.flat_id_to_local(torch.tensor([12], dtype=torch.int32))

    route = geometry.partition_route(
        torch.tensor([[0.2, 0.3, 0.5]]), ids
    )
    assert route.local_ids.tolist() == [[0, 0, 3]]
    assert torch.allclose(route.weights, torch.tensor([[0.0, 0.3, 0.5]]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"global_num_experts": 7, "world_size": 2}, "not divisible"),
        ({"global_num_experts": 8, "world_size": 2, "rank": 2}, "outside"),
        ({"global_num_experts": 8, "world_size": 2, "cache_size": 3}, "smaller"),
        ({"global_num_experts": 8, "world_size": 2, "cache_size": 4, "prefill_overlap": True}, "2 \\* local"),
    ],
)
def test_owner_cache_geometry_rejects_invalid_capacity_or_owner(kwargs, message):
    params = {
        "global_num_experts": 8,
        "world_size": 2,
        "rank": 0,
        "num_layers": 2,
        "cache_size": 8,
        "prefill_overlap": False,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=message):
        OwnerCacheGeometry(**params)


def test_owner_cache_geometry_validates_legacy_binding_and_local_bank_shapes():
    geometry = OwnerCacheGeometry(8, 2, 0, num_layers=2, cache_size=6)
    geometry.validate_cache_binding(
        num_layers=2, num_experts=8, cache_size=6, prefill_overlap=False
    )
    with pytest.raises(ValueError, match="num_experts"):
        geometry.validate_cache_binding(
            num_layers=2, num_experts=4, cache_size=6, prefill_overlap=False
        )

    good_sources = {
        "gate_up": [torch.empty(4, 8, 16) for _ in range(2)],
        "down": [torch.empty(4, 16, 4) for _ in range(2)],
    }
    geometry.validate_source_banks(good_sources)
    with pytest.raises(ValueError, match="first dimension 4"):
        geometry.validate_source_banks(
            {"gate_up": [torch.empty(8, 8, 16) for _ in range(2)]}
        )

    geometry.validate_slot_maps(
        torch.full((2, 4), -1, dtype=torch.int32),
        torch.full((6,), -1, dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="slot_for_id shape"):
        geometry.validate_slot_maps(
            torch.full((2, 8), -1, dtype=torch.int32),
            torch.full((6,), -1, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="entries"):
        geometry.validate_slot_maps(
            torch.tensor([[6, -1, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32),
            torch.full((6,), -1, dtype=torch.int32),
        )


def test_offload_cache_owner_geometry_is_default_off_and_fail_fast_when_explicit():
    from freetoken.moe.offload_cache import OffloadMoeCache

    # Existing callers do not pass owner_geometry and retain the global cache geometry.
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=4,
        device=torch.device("cpu"),
    )
    assert cache.slot_for_id.shape == (1, 4)

    geometry = OwnerCacheGeometry(8, 2, 0, num_layers=1, cache_size=4)
    with pytest.raises(NotImplementedError, match="namespace mapping"):
        OffloadMoeCache(
            num_layers=1,
            num_experts=8,
            cache_size=4,
            device=torch.device("cpu"),
            owner_geometry=geometry,
        )


def test_owner_cache_adapter_rewrites_owned_routes_and_preserves_weights():
    geometry = OwnerCacheGeometry(8, 2, 1, num_layers=2, cache_size=4)
    cache = OwnerCacheAdapter(geometry)
    ids = torch.tensor([[0, 4, 7, 5, 4]], dtype=torch.int32)
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.15, 0.25]])

    update = cache.ensure_route(1, weights, ids)

    assert update.owned_mask.tolist() == [[False, True, True, True, True]]
    assert update.local_ids.tolist() == [[0, 0, 3, 1, 0]]
    assert update.local_flat_ids.tolist() == [[0, 4, 7, 5, 4]]
    assert torch.allclose(update.weights, torch.tensor([[0.0, 0.2, 0.3, 0.15, 0.25]]))
    assert update.missing_local_ids.tolist() == [0, 3, 1]
    assert torch.all(update.slot_ids[update.owned_mask] >= 0)
    assert torch.equal(update.slot_ids[~update.owned_mask], torch.zeros(1, dtype=torch.int32))
    assert cache.resident == 3


def test_owner_cache_adapter_remote_only_route_is_safe_and_does_not_admit():
    geometry = OwnerCacheGeometry(8, 2, 0, num_layers=1, cache_size=4)
    cache = OwnerCacheAdapter(geometry)
    ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.int32)
    weights = torch.full((1, 4), 0.25)

    update = cache.ensure_route(0, weights, ids)

    assert update.missing_local_ids.numel() == 0
    assert update.evicted_flat_ids.numel() == 0
    assert torch.equal(update.slot_ids, torch.zeros_like(ids))
    assert torch.equal(update.weights, torch.zeros_like(weights))
    assert cache.resident == 0
    assert cache.step == 1  # an empty owner route still advances one logical LRU call


@pytest.mark.parametrize("rank, local_id, remote_id", [(0, 0, 4), (1, 4, 0)])
def test_owner_cache_adapter_all_local_and_all_remote_have_no_cross_admission(
    rank, local_id, remote_id
):
    geometry = OwnerCacheGeometry(8, 2, rank, num_layers=1, cache_size=4)
    cache = OwnerCacheAdapter(geometry)
    weights = torch.full((1, 10), 0.1)

    local = cache.ensure_route(0, weights, torch.full((1, 10), local_id, dtype=torch.int32))
    assert local.missing_local_ids.tolist() == [0]
    assert cache.resident == 1

    remote = cache.ensure_route(0, weights, torch.full((1, 10), remote_id, dtype=torch.int32))
    assert remote.missing_local_ids.numel() == 0
    assert remote.evicted_flat_ids.numel() == 0
    assert torch.equal(remote.slot_ids, torch.zeros_like(remote.slot_ids))
    assert torch.equal(remote.weights, torch.zeros_like(remote.weights))
    assert cache.resident == 1


def test_owner_cache_adapter_lru_is_layer_local_but_pool_is_unified():
    geometry = OwnerCacheGeometry(4, 2, 0, num_layers=2, cache_size=2)
    cache = OwnerCacheAdapter(geometry)
    cache.ensure_route(0, torch.ones(1, 2), torch.tensor([[0, 1]], dtype=torch.int32))
    update = cache.ensure_route(1, torch.ones(1, 1), torch.tensor([[0]], dtype=torch.int32))

    assert update.evicted_flat_ids.tolist() == [0]
    assert cache.slot_for_id.tolist() == [[-1, 1], [0, -1]]
    cache.validate_invariants()


def test_owner_cache_adapter_materialize_uses_two_local_prefill_buffers():
    geometry = OwnerCacheGeometry(8, 2, 0, num_layers=2, cache_size=8, prefill_overlap=True)
    cache = OwnerCacheAdapter(geometry)
    assert cache.materialize_layer(0, buffer_id=0).tolist() == [0, 1, 2, 3]
    assert cache.materialize_layer(1, buffer_id=1).tolist() == [4, 5, 6, 7]
    assert cache.slot_for_id.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]

    cache.materialize_layer(0, buffer_id=0)
    assert cache.slot_for_id.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    with pytest.raises(ValueError, match="buffer_id"):
        cache.materialize_layer(1, buffer_id=2)


def test_owner_offload_cache_compacts_routes_before_legacy_admission(monkeypatch):
    from freetoken.moe.offload_cache import OwnerOffloadMoeCache

    geometry = OwnerCacheGeometry(8, 2, 1, num_layers=1, cache_size=4)
    cache = OwnerOffloadMoeCache(geometry, torch.device("cpu"))
    calls = []

    def fake_ensure(layer_id, local_ids):
        calls.append((layer_id, local_ids.clone()))
        assert layer_id == 0
        # The legacy kernel has already accepted a local-row route and rewrites it to slots.
        slot_by_local = {0: 2, 1: 1, 3: 0}
        slots = torch.tensor(
            [slot_by_local[int(local)] for local in local_ids.tolist()],
            dtype=torch.int32,
        )
        cache._cache.num_indices.fill_(3)
        cache._cache.src_indices[:3] = torch.tensor([0, 3, 1], dtype=torch.int32)
        cache._cache.evict_slots[:3] = torch.tensor([2, 1, 0], dtype=torch.int32)
        for local, slot in slot_by_local.items():
            cache._cache.slot_for_id[layer_id, local] = slot
            cache._cache.id_of_slot[slot] = local
        local_ids.copy_(slots)

    monkeypatch.setattr(cache._cache, "ensure_experts", fake_ensure)
    copied = []
    monkeypatch.setattr(cache._cache, "copy_missing", lambda: copied.append(True))

    ids = torch.tensor([[0, 4, 7, 5, 4]], dtype=torch.int32)
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.15, 0.25]])
    update = cache.ensure_route(0, weights, ids)

    assert len(calls) == 1
    assert calls[0][1].tolist() == [0, 3, 1, 0]
    assert update.slot_ids.tolist() == [[0, 2, 0, 1, 2]]
    assert update.local_ids.tolist() == [[0, 0, 3, 1, 0]]
    assert torch.allclose(
        update.weights, torch.tensor([[0.0, 0.2, 0.3, 0.15, 0.25]])
    )
    assert update.missing_local_ids.tolist() == [0, 3, 1]
    assert update.evicted_flat_ids.tolist() == []

    cache.copy_missing()
    assert copied == [True]
    with pytest.raises(RuntimeError, match="ensure_route"):
        cache.ensure_experts(0, torch.tensor([0], dtype=torch.int32))


def test_owner_offload_cache_remote_only_route_does_not_stage_copy():
    from freetoken.moe.offload_cache import OwnerOffloadMoeCache

    geometry = OwnerCacheGeometry(8, 2, 0, num_layers=1, cache_size=4)
    cache = OwnerOffloadMoeCache(geometry, torch.device("cpu"))
    ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.int32)
    weights = torch.full((1, 4), 0.25)

    update = cache.ensure_route(0, weights, ids)

    assert update.slot_ids.tolist() == [[0, 0, 0, 0]]
    assert update.weights.tolist() == [[0.0, 0.0, 0.0, 0.0]]
    assert update.missing_local_ids.numel() == 0
    assert cache._pending_owned is False


def _write_tiny_nvfp4_checkpoint(folder: Path, *, experts: int = 4) -> dict[str, torch.Tensor]:
    """Create one native-bank-shaped layer with distinct data per global expert."""
    H = I = 16  # keep every bank dimension divisible by the native 16-byte scale blocks
    tensors: dict[str, torch.Tensor] = {}
    for expert in range(experts):
        base = f"layer.0.expert.{expert}"
        for proj, out, inn in (("gate", I, H), ("up", I, H), ("down", H, I)):
            tensors[f"{base}.{proj}.weight"] = torch.full(
                (out, inn // 2), expert + (1 if proj == "up" else 11 if proj == "down" else 101),
                dtype=torch.uint8,
            )
            tensors[f"{base}.{proj}.weight_scale"] = torch.full(
                (out, inn // 16), expert + 1, dtype=torch.float8_e4m3fn
            )
            # Give every expert/projection a different global scale.  The test catches a
            # local-ID lookup here: rank 1 row 0 must receive expert 2's scale, not expert 0's.
            tensors[f"{base}.{proj}.weight_scale_2"] = torch.tensor(
                10.0 + expert * 3 + {"gate": 0, "up": 1, "down": 2}[proj],
                dtype=torch.float32,
            )

    shard = folder / "model.safetensors"
    save_file(tensors, str(shard))
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name for name in tensors}}),
        encoding="utf-8",
    )
    return tensors


def test_owner_reader_yields_local_rows_and_keeps_global_scales(tmp_path, monkeypatch):
    raw = _write_tiny_nvfp4_checkpoint(tmp_path)
    monkeypatch.setattr(
        "freetoken.models.nvfp4_banks.download_hf_weight", lambda _path: str(tmp_path)
    )

    config = SimpleNamespace(
        num_experts=4,
        hidden_size=16,
        moe_intermediate_size=16,
        num_moe_layers=1,
    )
    owner = ExpertOwnership(global_num_experts=4, world_size=2, rank=1)
    pieces = list(
        iter_nvfp4_expert_pieces(
            str(tmp_path),
            config,
            _GENERIC_SPEC,
            drop_page_cache=lambda _path: None,
            primary=False,
            ownership=owner,
        )
    )

    # The owner has global experts [2, 4), renumbered into LOCAL rows [0, 2): no full-E
    # read and no remote row may appear in the stream.
    assert [(layer, e0, e1) for layer, e0, e1, _ in pieces] == [(0, 0, 1), (0, 1, 2)]
    by_local = {e0: piece for _, e0, _, piece in pieces}

    for local, global_expert in enumerate((2, 3)):
        piece = by_local[local]
        for proj in ("gate", "up", "down"):
            assert torch.equal(
                piece[proj][0], raw[f"layer.0.expert.{global_expert}.{proj}.weight"]
            )
            assert torch.equal(
                piece[f"{proj}_scale"][0],
                raw[f"layer.0.expert.{global_expert}.{proj}.weight_scale"],
            )
            # Global scale lookup must use checkpoint ID 2/3 while the destination row is 0/1.
            expected = raw[f"layer.0.expert.{global_expert}.{proj}.weight_scale_2"].to(
                torch.float16
            )
            got = piece[f"{proj}_global"][0]
            assert torch.equal(got, expected.expand_as(got)), (local, global_expert, proj)

    # No remote expert row is present.
    assert not torch.equal(by_local[0]["gate"][0], raw["layer.0.expert.0.gate.weight"])


def test_owner_reader_rejects_ownership_geometry_mismatch(tmp_path, monkeypatch):
    _write_tiny_nvfp4_checkpoint(tmp_path)
    monkeypatch.setattr(
        "freetoken.models.nvfp4_banks.download_hf_weight", lambda _path: str(tmp_path)
    )
    config = SimpleNamespace(
        num_experts=4,
        hidden_size=16,
        moe_intermediate_size=16,
        num_moe_layers=1,
    )
    with pytest.raises(ValueError, match="global_num_experts"):
        list(
            iter_nvfp4_expert_pieces(
                str(tmp_path),
                config,
                _GENERIC_SPEC,
                drop_page_cache=lambda _path: None,
                primary=False,
                ownership=ExpertOwnership(global_num_experts=8, world_size=2, rank=0),
            )
        )
