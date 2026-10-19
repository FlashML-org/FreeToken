"""NVFP4 n-gram tables: synthetic-checkpoint loading and the pinned packed gather.

The loader tests run on CPU over real safetensors files in ``tmp_path``; the gather is
bitwise-checked against a torch reference that mirrors the checkpoint formula
(e2m1 code x block scale x global scale, fp32, stored bf16).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.models.qwen4_exp.weight import load_ple_table, ple_table_is_packed

from .common import requires_cuda

ROWS = 64
HEAD_DIM = 32
PACKED = HEAD_DIM // 2
GROUPS = HEAD_DIM // 16
INFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _synthetic_packed_checkpoint(tmp_path) -> tuple[torch.Tensor, torch.Tensor]:
    """Two table shards + a file with only the global scalar; returns (packed, block scales)."""
    gen = torch.Generator().manual_seed(11)
    codes = torch.randint(0, 16, (2 * ROWS, HEAD_DIM), generator=gen)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    scales = (torch.randn(2 * ROWS, GROUPS, generator=gen) * 0.3).to(torch.float8_e4m3fn)
    for shard, name in enumerate(("ple-a.safetensors", "ple-b.safetensors")):
        save_file(
            {
                f"{INFIX}.shard_{shard}.weight": packed[shard * ROWS : (shard + 1) * ROWS],
                f"{INFIX}.shard_{shard}.weight_scale": scales[shard * ROWS : (shard + 1) * ROWS],
            },
            str(tmp_path / name),
        )
    save_file({f"{INFIX}.weight_scale_2": torch.tensor(0.03125, dtype=torch.bfloat16)},
              str(tmp_path / "ple-c.safetensors"))
    return packed, scales


def _args():
    return SimpleNamespace(split_ngram_parts=2, ngram_head_dim=HEAD_DIM)


def test_nvfp4_packed_table_loads(tmp_path):
    packed, scales = _synthetic_packed_checkpoint(tmp_path)
    assert ple_table_is_packed(str(tmp_path))
    table = load_ple_table(str(tmp_path), _args(), pin=False)
    assert table.bank.tensor.dtype is torch.uint8
    assert tuple(table.bank.tensor.shape) == (2 * ROWS, PACKED)
    assert table.scale_bank is not None
    assert table.scale_bank.tensor.dtype is torch.float8_e4m3fn
    assert torch.equal(table.bank.tensor, packed)
    assert torch.equal(table.scale_bank.tensor.view(torch.uint8), scales.view(torch.uint8))
    assert float(table.weight_scale) == 0.03125


def test_nvfp4_table_missing_scales_rejected(tmp_path):
    packed, _ = _synthetic_packed_checkpoint(tmp_path)
    for shard in (0, 1):
        path = tmp_path / f"ple-{'ab'[shard]}.safetensors"
        save_file({f"{INFIX}.shard_{shard}.weight": packed[shard * ROWS : (shard + 1) * ROWS]},
                  str(path))
    with pytest.raises(ValueError, match="scale shards"):
        load_ple_table(str(tmp_path), _args(), pin=False)


def test_fp8_table_still_loads(tmp_path):
    gen = torch.Generator().manual_seed(5)
    rows = (torch.randn(2 * ROWS, HEAD_DIM, generator=gen) * 0.3).to(torch.float8_e4m3fn)
    for shard in (0, 1):
        save_file(
            {f"{INFIX}.shard_{shard}.weight": rows[shard * ROWS : (shard + 1) * ROWS]},
            str(tmp_path / f"ple-{'ab'[shard]}.safetensors"),
        )
    save_file({f"{INFIX}.weight_scale": torch.tensor(0.03125, dtype=torch.bfloat16)},
              str(tmp_path / "ple-c.safetensors"))
    assert not ple_table_is_packed(str(tmp_path))
    table = load_ple_table(str(tmp_path), _args(), pin=False)
    assert table.bank.tensor.dtype is torch.float8_e4m3fn
    assert table.scale_bank is None
    assert torch.equal(table.bank.tensor.view(torch.uint8), rows.view(torch.uint8))


@requires_cuda
def test_pinned_nvfp4_gather_matches_reference():
    from freetoken.models.qwen4_exp.ple import PinnedUVATable
    from freetoken.moe.host_banks import HostBank

    scale = 0.03125
    gen = torch.Generator().manual_seed(23)
    codes = torch.randint(0, 16, (ROWS, HEAD_DIM), generator=gen)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    scales = (torch.randn(ROWS, GROUPS, generator=gen) * 0.3).to(torch.float8_e4m3fn)

    bank = HostBank((ROWS, PACKED), torch.uint8)
    bank.tensor.copy_(packed)
    bank.pin()
    sbank = HostBank((ROWS, GROUPS), torch.float8_e4m3fn)
    sbank.tensor.copy_(scales)
    sbank.pin()
    table = PinnedUVATable(bank.tensor, scale, scales=sbank.tensor)

    lut = torch.tensor(_E2M1, dtype=torch.float32)
    vals = torch.empty(ROWS, HEAD_DIM, dtype=torch.float32)
    vals[:, 0::2] = lut[codes[:, 0::2].long()]
    vals[:, 1::2] = lut[codes[:, 1::2].long()]
    ref = (vals * (scales.float() * scale).repeat_interleave(16, dim=1)).to(torch.bfloat16)

    ids = torch.randint(0, ROWS, (37, 16), device="cuda")
    assert torch.equal(table.lookup(ids).reshape(-1, HEAD_DIM).cpu(),
                       ref[ids.reshape(-1).cpu()])

    table.prefetch(ids)
    assert torch.equal(table.lookup(ids).reshape(-1, HEAD_DIM).cpu(),
                       ref[ids.reshape(-1).cpu()])

    oob = ids.clone()
    oob[0] = torch.arange(-3, 5)
    out = table.lookup(oob)
    assert torch.equal(out[0, :3], torch.zeros(3, HEAD_DIM, dtype=torch.bfloat16, device="cuda"))
    assert torch.equal(out[0, 7], ref[oob[0, 7]].cuda())
