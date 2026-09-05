"""TP sharding of the NVFP4 expert source banks: the two ranks' banks concatenated along the
intermediate axis must equal the unsharded placement."""

import torch
from freetoken.distributed.info import DistributedInfo
from freetoken.models import nvfp4_banks
from freetoken.models.nvfp4_banks import _alloc_nvfp4_host_banks, _Placer

E, H, INTER = 2, 64, 32


def _place(monkeypatch, rank, world):
    monkeypatch.setattr(
        nvfp4_banks, "get_tp_info", lambda: DistributedInfo(rank, world)
    )
    hb = _alloc_nvfp4_host_banks(1, E, H, INTER // world)
    banks = {name: [b.tensor for b in layers] for name, layers in hb.items()}
    placer = _Placer(banks, INTER)
    torch.manual_seed(0)
    for e in range(E):
        for role in ("gate", "up"):
            placer.put(
                0,
                e,
                role,
                "weight",
                torch.randint(0, 255, (INTER, H // 2), dtype=torch.uint8),
            )
            placer.put(
                0,
                e,
                role,
                "weight_scale",
                torch.randn(INTER, H // 16).to(torch.float8_e4m3fn),
                torch.tensor(0.5 + e, dtype=torch.float16),
            )
        placer.put(
            0,
            e,
            "down",
            "weight",
            torch.randint(0, 255, (H, INTER // 2), dtype=torch.uint8),
        )
        placer.put(
            0,
            e,
            "down",
            "weight_scale",
            torch.randn(H, INTER // 16).to(torch.float8_e4m3fn),
            torch.tensor(2.0 + e, dtype=torch.float16),
        )
    return {k: v[0] for k, v in banks.items()}


def test_rank_banks_concatenate_to_the_full_placement(monkeypatch):
    full = _place(monkeypatch, 0, 1)
    r0, r1 = _place(monkeypatch, 0, 2), _place(monkeypatch, 1, 2)
    n = INTER // 2
    for name in (
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global",
    ):  # rows: [gate I | up I]
        gate = torch.cat([r0[name][:, :n], r1[name][:, :n]], dim=1)
        up = torch.cat([r0[name][:, n:], r1[name][:, n:]], dim=1)
        assert torch.equal(
            torch.cat([gate, up], dim=1).view(torch.uint8), full[name].view(torch.uint8)
        ), name
    for name in ("down_packed", "down_scale"):  # columns
        assert torch.equal(
            torch.cat([r0[name], r1[name]], dim=2).view(torch.uint8),
            full[name].view(torch.uint8),
        ), name
    assert torch.equal(r0["down_global"], full["down_global"]) and torch.equal(
        r1["down_global"], full["down_global"]
    )
