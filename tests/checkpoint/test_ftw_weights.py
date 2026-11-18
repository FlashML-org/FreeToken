"""FTW replay: dropping entries by name before their bytes are read, and the vision-tower presence check."""

import torch

from freetoken.checkpoint.ftw import FTWReader, FTWWriter, ftw_tensor_names, iter_ftw_weights
from freetoken.models.weight import ftw_lacks_vision, load_weight


def _write_ftw(out_dir, names):
    writer = FTWWriter(str(out_dir))
    tensors = {name: torch.full((4, 8), float(i), dtype=torch.bfloat16) for i, name in enumerate(names)}
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor)
    writer.finalize({})
    return tensors


def test_keep_drops_entries_before_their_bytes_are_read(tmp_path, monkeypatch):
    tensors = _write_ftw(tmp_path, ["model.a.weight", "visual.b.weight", "model.c.weight"])
    read = []
    original = FTWReader.read_into

    def spy(self, dest, entry, **kwargs):
        read.append(entry["name"])
        return original(self, dest, entry, **kwargs)

    monkeypatch.setattr(FTWReader, "read_into", spy)
    got = dict(iter_ftw_weights(str(tmp_path), keep=lambda name: not name.startswith("visual.")))
    assert list(got) == ["model.a.weight", "model.c.weight"]
    assert read == ["model.a.weight", "model.c.weight"]
    for name, tensor in got.items():
        assert torch.equal(tensor, tensors[name])


def test_no_keep_replays_every_entry(tmp_path):
    tensors = _write_ftw(tmp_path, ["model.a.weight", "visual.b.weight"])
    got = dict(iter_ftw_weights(str(tmp_path)))
    assert list(got) == list(tensors)
    assert torch.equal(got["visual.b.weight"], tensors["visual.b.weight"])


def test_load_weight_text_only_skips_the_tower(tmp_path):
    names = ["model.a.weight", "visual.b.weight", "vision_tower.c.weight"]
    _write_ftw(tmp_path, names)
    cpu = torch.device("cpu")
    assert [n for n, _ in load_weight(str(tmp_path), cpu, include_vision=False)] == ["model.a.weight"]
    assert [n for n, _ in load_weight(str(tmp_path), cpu)] == names


def test_ftw_lacks_vision(tmp_path):
    _write_ftw(tmp_path / "text", ["model.a.weight"])
    _write_ftw(tmp_path / "vl", ["model.a.weight", "visual.b.weight"])
    assert ftw_lacks_vision(str(tmp_path / "text"))
    assert not ftw_lacks_vision(str(tmp_path / "vl"))
    assert not ftw_lacks_vision(str(tmp_path))
    assert ftw_tensor_names(str(tmp_path / "vl"), "weight") == ["model.a.weight", "visual.b.weight"]


def _write_nvfp4_expert_ftw(out_dir, *, with_alphas: bool):
    e, h, i = 4, 32, 16
    writer = FTWWriter(str(out_dir))
    writer.add_tensor("gate_up_packed", torch.ones((e, 2 * i, h // 2), dtype=torch.uint8), kind="experts_bank")
    writer.add_tensor("gate_up_scales", torch.ones((e, 2 * i, h // 16), dtype=torch.float8_e4m3fn), kind="experts_bank")
    writer.add_tensor("gate_up_global", torch.ones((e, 2 * i), dtype=torch.float32), kind="experts_bank")
    writer.add_tensor("down_packed", torch.ones((e, h, i // 2), dtype=torch.uint8), kind="experts_bank")
    writer.add_tensor("down_scales", torch.ones((e, h, i // 16), dtype=torch.float8_e4m3fn), kind="experts_bank")
    writer.add_tensor("down_global", torch.ones((e, h), dtype=torch.float32), kind="experts_bank")
    alphas = None
    if with_alphas:
        gu_alpha = torch.arange(1, e + 1, dtype=torch.float32)
        dn_alpha = torch.arange(1, e + 1, dtype=torch.float32) * 0.5
        writer.add_tensor("gate_up_alpha", gu_alpha, kind="experts_bank")
        writer.add_tensor("down_alpha", dn_alpha, kind="experts_bank")
        alphas = (gu_alpha, dn_alpha)
    writer.finalize({"quant_format": "nvfp4", "num_moe_layers": 1})
    return e, h, i, alphas


def test_nvfp4_ftw_w4a4_selection_and_alpha_replay(tmp_path, monkeypatch):
    import pytest
    from freetoken.checkpoint.ftw import load_ftw_banks_to_device
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.engine.engine import _adjust_ftw_quant_backend
    from freetoken.layers.moe import MoELayer
    from freetoken.layers.quantization import QuantBackend, QuantConfig, set_quant_backend
    from freetoken.layers.quantization.moe.nvfp4 import MarlinNvfp4MoEKernel, W4a4Nvfp4MoEKernel
    from freetoken.moe.expert_banks import attach_resident_banks

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    quant = QuantConfig.from_hf(
        {"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"]}}
    )
    with_alphas = str(tmp_path / "ftw_w4a4")
    no_alphas = str(tmp_path / "ftw_triton")
    e, h, i, alphas = _write_nvfp4_expert_ftw(with_alphas, with_alphas=True)
    _write_nvfp4_expert_ftw(no_alphas, with_alphas=False)

    def build_layer(ftw_dir: str, backend_spec: str | None) -> MoELayer:
        set_quant_backend(_adjust_ftw_quant_backend(ftw_dir, QuantBackend.parse(backend_spec)))
        return MoELayer(
            e, 2, h, i, allocate_experts=False, layer_id=0, quant_config=quant, prefix="model.layers.0.mlp.experts"
        )

    # Make marlin eligible too so auto must restrict candidates to (triton, w4a4).
    monkeypatch.setattr(MarlinNvfp4MoEKernel, "unusable_reason", lambda self, cfg: None)
    monkeypatch.setattr(MarlinNvfp4MoEKernel, "worth_it", lambda self, cfg: True)
    monkeypatch.setattr(W4a4Nvfp4MoEKernel, "worth_it", lambda self, cfg: True)
    assert build_layer(with_alphas, None).quant_method.kernel.name == "w4a4"

    monkeypatch.setattr(W4a4Nvfp4MoEKernel, "worth_it", lambda self, cfg: False)
    assert build_layer(with_alphas, None).quant_method.kernel.name == "triton"

    layer = build_layer(with_alphas, "moe.nvfp4=w4a4")
    assert layer.quant_method.kernel.name == "w4a4"
    banks = load_ftw_banks_to_device(with_alphas, num_layers=1, device=torch.device("cpu"))
    attach_resident_banks([layer], banks)
    assert layer._expert_alphas is not None
    assert torch.equal(layer._expert_alphas[0], alphas[0])
    assert torch.equal(layer._expert_alphas[1], alphas[1])

    monkeypatch.setattr(W4a4Nvfp4MoEKernel, "worth_it", lambda self, cfg: True)
    assert build_layer(no_alphas, None).quant_method.kernel.name == "triton"
    with pytest.raises(ValueError, match="gate_up_alpha.*down_alpha"):
        _adjust_ftw_quant_backend(no_alphas, QuantBackend.parse("moe.nvfp4=w4a4"))
    with pytest.raises(ValueError, match="packed for nvfp4 / triton"):
        _adjust_ftw_quant_backend(with_alphas, QuantBackend.parse("moe.nvfp4=marlin"))

