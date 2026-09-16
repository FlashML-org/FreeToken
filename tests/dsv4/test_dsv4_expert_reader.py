"""The serial DSV4 expert reader: rank-local pieces, one shard's page cache at a time.

Under TP every rank reads the same checkpoint at once, on top of host expert banks that
already take most of the machine's RAM. The reader must therefore release each shard's
page cache before it opens the next one -- holding every shard until the last layer grows
the cache to the whole checkpoint, which does not fit beside the banks -- and it must
yield exactly the rank's ``shard_expert_piece`` of every tensor, whether it read the
rows straight off the file (gate/up) or cut a whole tensor (down).
"""

from __future__ import annotations

import json
import os

import pytest
import torch

import freetoken.distributed.info as info_mod
from freetoken.distributed import DistributedInfo
from freetoken.layers.quantization import QuantKind

safetensors_torch = pytest.importorskip("safetensors.torch")

H, I, E, L = 64, 128, 2, 2
SHAPES = {
    ("w1", "weight"): ((I, H // 2), torch.int8),
    ("w3", "weight"): ((I, H // 2), torch.int8),
    ("w2", "weight"): ((H, I // 2), torch.int8),
    ("w1", "scale"): ((I, H // 32), torch.uint8),
    ("w3", "scale"): ((I, H // 32), torch.uint8),
    ("w2", "scale"): ((H, I // 32), torch.uint8),
}
ROLE = {"w1": "gate", "w3": "up", "w2": "down"}


@pytest.fixture(autouse=True)
def _restore_tp():
    yield
    info_mod._TP_INFO = DistributedInfo(0, 1)


def _checkpoint(root) -> dict[str, torch.Tensor]:
    """One shard per layer, plus a trailing MTP-style layer and a dense tensor the reader skips."""
    full: dict[str, torch.Tensor] = {}
    weight_map: dict[str, str] = {}
    offset = 0
    for layer in range(L + 1):
        shard = f"model-{layer:05d}.safetensors"
        tensors = {}
        for e in range(E):
            for (proj, kind), (shape, dtype) in SHAPES.items():
                n = torch.Size(shape).numel()
                t = ((torch.arange(n) + offset) % 251).to(torch.uint8).view(dtype).reshape(shape)
                offset += n
                tensors[f"layers.{layer}.ffn.experts.{e}.{proj}.{kind}"] = t
        tensors[f"layers.{layer}.attn.wkv.weight"] = torch.zeros(4, 4)
        safetensors_torch.save_file(tensors, os.path.join(root, shard))
        weight_map.update({name: shard for name in tensors})
        if layer < L:
            full.update({k: v for k, v in tensors.items() if ".experts." in k})
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    os.makedirs(os.path.join(root, "inference"))
    with open(os.path.join(root, "inference", "config.json"), "w") as f:
        json.dump({"n_layers": L, "n_routed_experts": E, "dim": H, "moe_inter_dim": I}, f)
    return full


@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
def test_serial_reader_yields_rank_pieces_and_drops_each_shard_before_the_next(tmp_path, monkeypatch, tp, rank):
    import safetensors

    from freetoken.models.deepseek_v4 import weight

    full = _checkpoint(str(tmp_path))
    info_mod._TP_INFO = DistributedInfo(rank, tp)

    events: list[tuple[str, str]] = []
    real_open = safetensors.safe_open

    def recording_open(path, *a, **kw):
        events.append(("open", os.path.basename(path)))
        return real_open(path, *a, **kw)

    monkeypatch.setattr(weight.safetensors, "safe_open", recording_open)
    mapped_at_drop: list[str] = []

    def recording_drop(path):
        events.append(("drop", os.path.basename(path)))
        # Mapped pages survive POSIX_FADV_DONTNEED, so nothing may still map the shard
        # (a lingering slice handle or a zero-copy tensor view) when its cache is dropped.
        if os.path.exists("/proc/self/maps"):
            with open("/proc/self/maps") as maps:
                if any(line.rstrip().endswith(os.path.realpath(path)) for line in maps):
                    mapped_at_drop.append(os.path.basename(path))

    monkeypatch.setattr(weight, "drop_page_cache", recording_drop)

    pieces = {}
    for layer, e0, e1, piece in weight.iter_expert_pieces(str(tmp_path), None, QuantKind.MXFP4, parallel=False):
        assert e1 == e0 + 1
        assert (layer, e0) not in pieces
        pieces[(layer, e0)] = piece

    assert sorted(pieces) == [(layer, e) for layer in range(L) for e in range(E)]
    assert not mapped_at_drop, f"shards still mapped when their cache was dropped: {mapped_at_drop}"
    for (layer, e), piece in pieces.items():
        for (proj, kind), _ in SHAPES.items():
            role = ROLE[proj] + ("_scale" if kind == "scale" else "")
            src = full[f"layers.{layer}.ffn.experts.{e}.{proj}.{kind}"]
            want = weight.shard_expert_piece(role, src, rank=rank, tp_size=tp)
            assert torch.equal(piece[role][0], want), f"layer {layer} expert {e} {role}"

    # Every expert shard is opened exactly once, and its cache is dropped before the next
    # shard is opened -- never all at the end.
    opens = [name for ev, name in events if ev == "open"]
    assert opens == sorted({f"model-{layer:05d}.safetensors" for layer in range(L)})
    for name in opens:
        opened = events.index(("open", name))
        assert ("drop", name) in events[opened:], f"{name} cache never dropped after reading"
        nxt = [i for i, ev in enumerate(events) if ev[0] == "open" and i > opened]
        if nxt:
            assert events.index(("drop", name), opened) < nxt[0], f"{name} still cached when the next shard opened"
