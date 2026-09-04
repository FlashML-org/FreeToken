import json
from types import SimpleNamespace

import pytest
import torch


def test_ftw_writes_auxiliary_banks_as_per_layer_checkpoint(tmp_path):
    from freetoken.checkpoint.convert import _write_auxiliary_ftw

    banks = SimpleNamespace(
        auxiliary_quant_format="q6_k_down",
        auxiliary_sources={
            "down": [
                torch.arange(6, dtype=torch.uint8).reshape(2, 3),
                torch.ones((2, 3), dtype=torch.uint8),
            ]
        },
        auxiliary_layer_ids=(7, 9),
    )

    metadata = _write_auxiliary_ftw(str(tmp_path), banks, shard_limit=4096)

    assert metadata == {
        "auxiliary_checkpoint": "auxiliary",
        "auxiliary_layer_ids": [7, 9],
    }
    index = json.loads((tmp_path / "auxiliary" / "freetoken_weight.json").read_text())
    assert index["quant_format"] == "q6_k_down"
    assert index["expert_bank_num_layers"] == 2
    assert [entry["name"] for entry in index["tensors"]] == [
        "down#L00000",
        "down#L00001",
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FTW expert banks require pinned GPU memory"
)
def test_ftw_loads_nested_auxiliary_banks(tmp_path):
    from freetoken.checkpoint.convert import _write_auxiliary_ftw
    from freetoken.checkpoint.ftw import (
        FTWWriter,
        layer_bank_entry_name,
        load_ftw_banks,
    )

    auxiliary = SimpleNamespace(
        auxiliary_quant_format="q6_k_down",
        auxiliary_sources={"down": [torch.arange(6, dtype=torch.uint8).reshape(2, 3)]},
        auxiliary_layer_ids=(9,),
    )
    metadata = _write_auxiliary_ftw(str(tmp_path), auxiliary, shard_limit=4096)
    writer = FTWWriter(str(tmp_path), shard_limit=4096)
    for layer_id in range(2):
        writer.add_tensor(
            layer_bank_entry_name("gate_up", layer_id),
            torch.full((2, 3), layer_id, dtype=torch.uint8),
            kind="experts_bank",
        )
    writer.finalize(
        {
            "quant_format": "q4_k_q5_k",
            "expert_bank_num_layers": 2,
            **metadata,
        }
    )

    banks = load_ftw_banks(str(tmp_path), num_layers=2)

    assert banks.auxiliary_quant_format == "q6_k_down"
    assert banks.auxiliary_layer_ids == (9,)
    assert len(banks.auxiliary_sources["down"]) == 1
    torch.testing.assert_close(
        banks.auxiliary_sources["down"][0], auxiliary.auxiliary_sources["down"][0]
    )
