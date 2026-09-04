"""qwen4_exp weight loading against the real RadixArk/Qwen3.8-Flash-Next-NVFP4 checkpoint.

Set ``FREETOKEN_QWEN4EXP_MODEL`` to the local checkpoint directory to run these. Everything is
sampled except the PLE table, which is loaded and pinned in full once (~47.7 GiB) because that
is the only way to check the shard concatenation and the pin budget.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
from types import SimpleNamespace

import pytest
import safetensors
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kernel.aot_models import expert_bank_row_bytes
from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.weight import (
    _NVFP4_SOURCE_SPEC,
    _ZERO_CENTERED_NORM_SUFFIXES,
    iter_weights,
    load_ple_table,
)
from freetoken.moe.host_banks import HostResidency
from freetoken.utils import cached_load_hf_config

MODEL_PATH = os.environ.get("FREETOKEN_QWEN4EXP_MODEL")
pytestmark = [
    pytest.mark.needs_weights,
    pytest.mark.skipif(not MODEL_PATH, reason="FREETOKEN_QWEN4EXP_MODEL is not set"),
]

E, H, I = 512, 2560, 640
NUM_LAYERS = 48
PLE_LAYER = 1
PLE_SHARDS, PLE_ROWS_PER_SHARD, PLE_DIM = 128, 2_500_012, 160
PLE_BYTES = PLE_SHARDS * PLE_ROWS_PER_SHARD * PLE_DIM
EXPERT_LAYER_BYTES = E * sum(expert_bank_row_bytes("nvfp4", H, I).values())
LM = "model.language_model"


@pytest.fixture(scope="session", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


class _Reader:
    """Serves checkpoint tensors by their raw key, through the index shard map."""

    def __init__(self, folder: str):
        with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as fh:
            self._map = json.load(fh)["weight_map"]
        self._folder = folder
        self._handles: dict = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self._map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device="cpu"
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()


def _gdn_parts(layer: int) -> list[str]:
    return [f"{LM}.layers.{layer}.linear_attn.in_proj_{p}.weight" for p in ("qkv", "z", "b", "a")]


def _hc_parts(layer: int, hc: str) -> list[str]:
    return [f"{LM}.layers.{layer}.{hc}.input_mix_weight_down.weight",
            f"{LM}.layers.{layer}.{hc}.block_inject_weight.weight"]


def _qkv_parts(layer: int) -> list[str]:
    return [f"{LM}.layers.{layer}.self_attn.{p}_proj.weight" for p in ("q", "k", "v")]


# (model key, checkpoint keys, mode). "cat16" additionally requires the merged rows to be
# zero-padded up to a multiple of 16. Zero-centered norms are "same": (1+w) is a runtime op.
SAMPLES: tuple[tuple[str, list[str], str], ...] = (
    ("model.embed_tokens.weight", [f"{LM}.embed_tokens.weight"], "same"),
    ("lm_head.weight", ["lm_head.weight"], "same"),
    ("model.layers.0.linear_attn.in_proj.weight", _gdn_parts(0), "cat"),
    ("model.layers.46.linear_attn.in_proj.weight", _gdn_parts(46), "cat"),
    ("model.layers.0.linear_attn.conv1d.weight", [f"{LM}.layers.0.linear_attn.conv1d.weight"], "same"),
    ("model.layers.0.linear_attn.A_log", [f"{LM}.layers.0.linear_attn.A_log"], "same"),
    ("model.layers.0.linear_attn.dt_bias", [f"{LM}.layers.0.linear_attn.dt_bias"], "same"),
    ("model.layers.0.linear_attn.norm.weight", [f"{LM}.layers.0.linear_attn.norm.weight"], "same"),
    ("model.layers.0.linear_attn.out_proj.weight", [f"{LM}.layers.0.linear_attn.out_proj.weight"], "same"),
    ("model.layers.3.self_attn.qkv_proj.weight", _qkv_parts(3), "cat"),
    ("model.layers.47.self_attn.qkv_proj.weight", _qkv_parts(47), "cat"),
    ("model.layers.3.self_attn.o_proj.weight", [f"{LM}.layers.3.self_attn.o_proj.weight"], "same"),
    ("model.layers.3.self_attn.q_norm.weight", [f"{LM}.layers.3.self_attn.q_norm.weight"], "same"),
    ("model.layers.3.self_attn.k_norm.weight", [f"{LM}.layers.3.self_attn.k_norm.weight"], "same"),
    ("model.layers.3.self_attn.indexer.index_qk_proj.weight",
     [f"{LM}.layers.3.self_attn.indexer.index_qk_proj.weight"], "same"),
    ("model.layers.3.self_attn.indexer.q_layernorm.weight",
     [f"{LM}.layers.3.self_attn.indexer.q_layernorm.weight"], "same"),
    ("model.layers.47.self_attn.indexer.k_layernorm.weight",
     [f"{LM}.layers.47.self_attn.indexer.k_layernorm.weight"], "same"),
    ("model.layers.7.attn_hyper_connection.hc_norm.weight",
     [f"{LM}.layers.7.attn_hyper_connection.hc_norm.weight"], "same"),
    ("model.layers.7.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
     _hc_parts(7, "attn_hyper_connection"), "cat16"),
    ("model.layers.7.attn_hyper_connection.input_mix_weight_up.weight",
     [f"{LM}.layers.7.attn_hyper_connection.input_mix_weight_up.weight"], "same"),
    ("model.layers.47.mlp_hyper_connection.input_mix_weight_down_block_inject.weight",
     _hc_parts(47, "mlp_hyper_connection"), "cat16"),
    ("model.hyper_connection_mixer.hc_norm.weight",
     [f"{LM}.hyper_connection_mixer.hc_norm.weight"], "same"),
    ("model.hyper_connection_mixer.input_mix_weight_down.weight",
     [f"{LM}.hyper_connection_mixer.input_mix_weight_down.weight"], "same"),
    ("model.hyper_connection_mixer.input_mix_weight_up.weight",
     [f"{LM}.hyper_connection_mixer.input_mix_weight_up.weight"], "same"),
    (f"model.layers.{PLE_LAYER}.ple.key_proj.weight", [f"{LM}.layers.{PLE_LAYER}.ple.key_proj.weight"], "same"),
    (f"model.layers.{PLE_LAYER}.ple.value_proj.weight", [f"{LM}.layers.{PLE_LAYER}.ple.value_proj.weight"], "same"),
    (f"model.layers.{PLE_LAYER}.ple.norm_key.weight", [f"{LM}.layers.{PLE_LAYER}.ple.norm_key.weight"], "same"),
    (f"model.layers.{PLE_LAYER}.ple.norm_query.weight", [f"{LM}.layers.{PLE_LAYER}.ple.norm_query.weight"], "same"),
    (f"model.layers.{PLE_LAYER}.ple.norm_conv.weight", [f"{LM}.layers.{PLE_LAYER}.ple.norm_conv.weight"], "same"),
    (f"model.layers.{PLE_LAYER}.ple.conv1d.weight", [f"{LM}.layers.{PLE_LAYER}.ple.conv1d.weight"], "same"),
    (f"model.layers.{PLE_LAYER}.ple.ple_embedding.layer_multipliers",
     [f"{LM}.layers.{PLE_LAYER}.ple.ple_embedding.layer_multipliers"], "same"),
    (f"model.layers.{PLE_LAYER}.ple.ple_embedding.ngram_heads_offsets",
     [f"{LM}.layers.{PLE_LAYER}.ple.ple_embedding.ngram_heads_offsets"], "same"),
    (f"model.layers.{PLE_LAYER}.ple.ple_embedding.ngram_heads_vocab_sizes",
     [f"{LM}.layers.{PLE_LAYER}.ple.ple_embedding.ngram_heads_vocab_sizes"], "same"),
    ("model.layers.5.mlp.gate.weight", [f"{LM}.layers.5.mlp.gate.weight"], "same"),
    ("model.layers.5.mlp.shared_expert.gate_up_proj.weight",
     [f"{LM}.layers.5.mlp.shared_expert.gate_proj.weight",
      f"{LM}.layers.5.mlp.shared_expert.up_proj.weight"], "cat"),
    ("model.layers.5.mlp.shared_expert.down_proj.weight",
     [f"{LM}.layers.5.mlp.shared_expert.down_proj.weight"], "same"),
    ("model.layers.5.mlp.shared_expert_gate.weight",
     [f"{LM}.layers.5.mlp.shared_expert_gate.weight"], "same"),
)


@pytest.fixture(scope="module")
def reader() -> _Reader:
    r = _Reader(MODEL_PATH)
    yield r
    r.close()


@pytest.fixture(scope="module")
def dense_pass() -> tuple[list[str], dict[str, torch.Tensor]]:
    """One full iter_weights sweep: every emitted name, plus a clone of each sampled tensor.

    The zero-centered norms are kept too -- they are tiny, and checking all of them is the
    cheapest guard against the +1 creeping back into the load path."""
    wanted = {name for name, _raw, _mode in SAMPLES}
    names: list[str] = []
    sampled: dict[str, torch.Tensor] = {}
    for name, tensor in iter_weights(
        MODEL_PATH, torch.device("cpu"), include_moe_experts=True, include_non_moe=True
    ):
        names.append(name)
        if name in wanted or name.endswith(_ZERO_CENTERED_NORM_SUFFIXES):
            sampled[name] = tensor.clone()
    return names, sampled


def test_emitted_names_are_unique_and_complete(dense_pass):
    names, _sampled = dense_pass
    assert len(names) == len(set(names))
    assert len([n for n in names if n.endswith(".linear_attn.in_proj.weight")]) == 36
    assert len([n for n in names if n.endswith(".self_attn.qkv_proj.weight")]) == 12
    assert len([n for n in names
                if n.endswith(".input_mix_weight_down_block_inject.weight")]) == 2 * NUM_LAYERS
    assert len([n for n in names if ".ple." in n]) == 9
    assert {"model.embed_tokens.weight", "lm_head.weight",
            "model.hyper_connection_mixer.input_mix_weight_down.weight"} <= set(names)


@pytest.fixture(scope="module")
def model_state_dict_keys() -> set[str]:
    """Keys ``Qwen4ExpForCausalLM`` declares -- the authoritative target the loader must fill."""
    from freetoken.layers import rotary
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    config = parse_config(cached_load_hf_config(MODEL_PATH))
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))  # get_rope refuses to build on meta
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"):
            return set(Qwen4ExpForCausalLM(config).state_dict())
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()


def test_emitted_names_are_the_model_state_dict(dense_pass, model_state_dict_keys):
    names, _sampled = dense_pass
    # The routed NVFP4 experts come from the offload source banks, never from the dense pass.
    expected = {k for k in model_state_dict_keys
                if not k.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj"))}
    assert set(names) == expected


def test_every_zero_centered_norm_is_present_and_raw(dense_pass, reader):
    names, sampled = dense_pass
    zero_centered = [n for n in names if n.endswith(_ZERO_CENTERED_NORM_SUFFIXES)]
    # 2 HC per layer + the top-level mixer + 3 PLE norms + q/k_norm and indexer q/k per QSA layer
    assert len(zero_centered) == 2 * NUM_LAYERS + 1 + 3 + 4 * 12
    for name in zero_centered:
        assert torch.equal(sampled[name], reader.get(name.replace("model.", f"{LM}.", 1))), name


def test_no_mtp_visual_expert_or_table_tensor_is_loaded(dense_pass):
    names, _sampled = dense_pass
    for name in names:
        assert not name.startswith(("mtp.", "model.visual.", "visual."))
        assert ".mlp.experts." not in name
        assert "ngram_embedding" not in name
        assert not name.endswith((".weight_scale", ".weight_scale_2", ".input_scale"))


@pytest.mark.parametrize("name, raw_names, mode", SAMPLES, ids=[s[0] for s in SAMPLES])
def test_sampled_tensor_matches_the_checkpoint(dense_pass, reader, name, raw_names, mode):
    _names, sampled = dense_pass
    parts = [reader.get(raw) for raw in raw_names]
    got = sampled[name]
    if mode == "same":
        assert torch.equal(got, parts[0])
        assert got.dtype is parts[0].dtype
    else:
        rows = sum(p.shape[0] for p in parts)
        assert torch.equal(got[:rows], torch.cat(parts, dim=0))
        if mode == "cat16":
            assert got.shape[0] == rows + (-rows) % 16
            assert not got[rows:].any()
        else:
            assert got.shape[0] == rows


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinning needs CUDA")
def test_ple_table_loads_pinned_and_matches_the_checkpoint(reader):
    args = parse_config(cached_load_hf_config(MODEL_PATH)).qwen4_args
    table = load_ple_table(MODEL_PATH, args)
    assert table.bank.residency is HostResidency.PINNED
    assert table.bank.nbytes == PLE_BYTES
    assert abs(PLE_BYTES / 2**30 - 47.68) < 0.05
    assert table.tensor.shape == (PLE_SHARDS * PLE_ROWS_PER_SHARD, PLE_DIM)
    assert table.tensor.dtype is torch.float8_e4m3fn

    prefix = f"{LM}.layers.{PLE_LAYER}.ple.ple_embedding.ngram_embedding"
    assert torch.equal(table.weight_scale.reshape(1),
                       reader.get(f"{prefix}.weight_scale").reshape(1))
    rows = random.Random(0).sample(range(PLE_SHARDS * PLE_ROWS_PER_SHARD), 1000)
    by_shard: dict[int, list[int]] = {}
    for row in rows:
        by_shard.setdefault(row // PLE_ROWS_PER_SHARD, []).append(row)
    got = table.tensor.view(torch.uint8)
    for shard, shard_rows in by_shard.items():
        ref = reader.get(f"{prefix}.shard_{shard}.weight").view(torch.uint8)
        local = torch.tensor([r - shard * PLE_ROWS_PER_SHARD for r in shard_rows])
        assert torch.equal(got[torch.tensor(shard_rows)], ref[local])

    # The two pinned host allocations the engine must budget for.
    total = PLE_BYTES + NUM_LAYERS * EXPERT_LAYER_BYTES
    assert abs(total / 2**30 - 111.14) < 0.05


@pytest.fixture(scope="module")
def layer0_expert_banks():
    """The real NVFP4 source-bank loader, restricted to layer 0 (1.32 GiB instead of 63.5)."""
    if not torch.cuda.is_available():
        pytest.skip("expert bank pinning needs CUDA")
    spec = dataclasses.replace(
        _NVFP4_SOURCE_SPEC, layer_to_bank=lambda layer, config: 0 if layer == 0 else None
    )
    config = SimpleNamespace(num_experts=E, hidden_size=H, moe_intermediate_size=I,
                             num_moe_layers=1)
    return load_nvfp4_expert_source_banks(
        MODEL_PATH, config, spec, drop_page_cache=lambda path: None, primary=False
    )


@pytest.mark.slow
def test_sampled_experts_match_the_checkpoint(layer0_expert_banks, reader):
    banks = layer0_expert_banks
    for expert in random.Random(1).sample(range(E), 8):
        base = f"{LM}.layers.0.mlp.experts.{expert}"
        assert torch.equal(banks["gate_up_packed"][0][expert, :I],
                           reader.get(f"{base}.gate_proj.weight"))
        assert torch.equal(banks["gate_up_packed"][0][expert, I:],
                           reader.get(f"{base}.up_proj.weight"))
        assert torch.equal(banks["down_packed"][0][expert],
                           reader.get(f"{base}.down_proj.weight"))
        for proj, bank, rows in (("gate_proj", "gate_up_scale", slice(0, I)),
                                 ("up_proj", "gate_up_scale", slice(I, 2 * I)),
                                 ("down_proj", "down_scale", slice(None))):
            scale = reader.get(f"{base}.{proj}.weight_scale")
            assert torch.equal(banks[bank][0][expert][rows].reshape(-1).view(torch.uint8),
                               scale.reshape(-1).view(torch.uint8))
        gate_g = reader.get(f"{base}.gate_proj.weight_scale_2").to(torch.float16)
        assert torch.equal(banks["gate_up_global"][0][expert, :I], gate_g.reshape(1).expand(I))


def test_expert_bank_bytes_match_the_aot_row_table(layer0_expert_banks):
    measured = sum(t[0].numel() * t[0].element_size() for t in layer0_expert_banks.values())
    assert measured == EXPERT_LAYER_BYTES
    assert abs(NUM_LAYERS * EXPERT_LAYER_BYTES / 2**30 - 63.46) < 0.05


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dummy banks are pinned")
def test_dummy_expert_sources_have_the_real_bank_shapes(layer0_expert_banks):
    from freetoken.models.weight import _model_override, dummy_nvfp4_expert_sources
    from freetoken.models.register import get_model_spec

    spec = get_model_spec("Qwen4ExpForConditionalGeneration")
    # No dummy_* override, so --use-dummy-weight goes through the generic builders.
    for hook in ("dummy_nvfp4_expert_sources", "dummy_moe_expert_sources", "dummy_q4_0_expert_sources"):
        assert _model_override(spec, hook) is None

    config = SimpleNamespace(num_experts=E, hidden_size=H, moe_intermediate_size=I,
                             num_moe_layers=1)
    dummy = dummy_nvfp4_expert_sources(config)
    assert set(dummy) == set(layer0_expert_banks)
    for name, banks in dummy.items():
        assert banks[0].shape == layer0_expert_banks[name][0].shape
        assert banks[0].dtype is layer0_expert_banks[name][0].dtype
