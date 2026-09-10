"""What each checkpoint's quant config resolves to: the dialect, the method behind every layer, and agreement with the stored tensors.

Rows need the checkpoint's config.json locally (/mnt/nvme/models or the HF cache); absent ones skip.
"""

from __future__ import annotations

import glob
import json
import os
import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch

from freetoken.distributed.info import set_tp_info, try_get_tp_info
from freetoken.engine.config import EngineConfig
from freetoken.layers import set_rope_device
from freetoken.layers.quantization import (
    CompressedTensorsConfig,
    Fp8BlockConfig,
    KernelSelectionError,
    ModelOptConfig,
    Mxfp4Config,
    NoQuantConfig,
    QuantConfig,
    QuantKind,
)
from freetoken.layers.quantization.linear import (
    Fp8BlockLinearMethod,
    Fp8TensorLinearMethod,
    Mxfp8LinearMethod,
    Nvfp4LinearMethod,
    UnquantizedLinearMethod,
)
from freetoken.layers.quantization.moe import Fp8BlockMoEMethod, Mxfp4MoEMethod, Nvfp4MoEMethod, UnquantizedMoEMethod
from freetoken.models import create_model
from freetoken.models.gpt_oss.moe import GptOssMoELayer, GptOssOffloadMoELayer
from freetoken.models.register import checkpoint_quant_config, get_model_spec
from freetoken.utils.torch_utils import torch_dtype

MODELS = "/mnt/nvme/models"
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

BF16, FP8B, FP8T, MXFP8, NVFP4 = UnquantizedLinearMethod, Fp8BlockLinearMethod, Fp8TensorLinearMethod, Mxfp8LinearMethod, Nvfp4LinearMethod


def model_dir(name: str) -> str | None:
    path = os.path.join(MODELS, name.split("/")[-1])
    if os.path.isfile(os.path.join(path, "config.json")):  # a download in progress has the dir but no config yet
        return path
    snaps = glob.glob(os.path.join(HF_CACHE, "models--" + name.replace("/", "--"), "snapshots", "*", "config.json"))
    return os.path.dirname(snaps[0]) if snaps else None


def op_at(model, path: str):
    op = model
    for part in path.split("."):
        op = op.op_list[int(part)] if part.isdigit() else getattr(op, part)
    return op


def build_meta_model(path: str, strategy: str):
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    set_rope_device(torch.device("cpu"))
    config = EngineConfig(model_path=path, tp_info=try_get_tp_info(), dtype=torch.bfloat16, moe_strategy=strategy)
    from freetoken.engine.engine import _decode_target

    object.__setattr__(config.model_config, "moe_strategy", strategy)
    object.__setattr__(config.model_config, "decode_target", _decode_target(config))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        return create_model(config.model_config), config.model_config


def _layer_indices(model) -> dict[str, int | None]:
    """First layer of each kind, for the {moe} {dense} {lin} {full} {dsa} {kda} {ple} placeholders in probe paths."""
    layers = model.model.layers.op_list

    def first(pred):
        return next((i for i, layer in enumerate(layers) if pred(layer)), None)

    def has_experts(layer):
        if getattr(layer, "is_moe_layer", None) is not None:
            return layer.is_moe_layer
        block = next((getattr(layer, n) for n in ("mlp", "feed_forward", "block_sparse_moe", "ffn") if hasattr(layer, n)), None)
        return block is not None and hasattr(block, "experts")

    def is_linear_attn(layer):
        return getattr(layer, "_is_linear", False) or hasattr(layer, "linear_attn")

    attn = lambda layer: getattr(layer, "self_attn", None)
    ple = getattr(getattr(model, "_config", None), "qwen4_args", None)
    return {
        "moe": first(has_experts),
        "dense": first(lambda l: not has_experts(l)),
        "lin": first(is_linear_attn),
        "full": first(lambda l: not is_linear_attn(l)),
        "dsa": first(lambda l: getattr(attn(l), "indexer", None) is not None),
        "kda": first(lambda l: hasattr(attn(l), "in_proj")),
        "ple": ple.ple_layer_ids[0] if ple is not None else None,
    }


@dataclass(frozen=True)
class Case:
    name: str
    dialect: type
    methods: dict[str, Any]  # probe path -> method class, (class, kernel name), or None for a layer without a method
    strategy: str = "offload"
    raises: type | None = None
    skip_if: Callable[[], bool] | None = None
    env: dict[str, str] = field(default_factory=dict)
    check: Callable[[Any, dict], None] | None = None  # family extras beyond the method table


def _installed(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _qwen35_fp8(model, idx):
    assert model.model.layers.op_list[idx["lin"]].linear_attn._split_in_proj


def _qwen35_nvfp4(model, idx):
    assert model.model.layers.op_list[idx["lin"]].linear_attn._split_in_proj
    assert op_at(model, f"model.layers.{idx['lin']}.linear_attn.in_proj_qkvz").quant_method.scheme.has("input_scale")
    assert model.lm_head.weight.dtype is torch.uint8


def _no_split(model, idx):
    assert not model.model.layers.op_list[idx["lin"]].linear_attn._split_in_proj


def _dsv4(model, idx):
    block = op_at(model, "model.layers.2")
    for proj in (block.attn.wq_a, block.ffn.shared_experts.w1):
        assert proj.weight.dtype == torch.float8_e4m3fn and proj.weight_scale_inv.dtype == torch.float8_e8m0fnu
        assert proj.weight_scale_inv.shape == (proj.out_features // 128, proj.in_features // 128) and proj.bias is None
    assert set(block.attn.compressor.wkv.state_dict()) == {"weight"}
    experts = block.ffn.experts
    assert experts.limit == 10.0 and experts.quant_method.cpu_format == "ds_fp4" and not experts.has_bias
    H, I = 4096, 2048
    assert {r: (s.shape, s.dtype) for r, s in experts.quant_method.layout().items()} == {
        "gate_up": ((2 * I, H // 2), torch.uint8), "gate_up_scale": ((2 * I, H // 32), torch.float8_e8m0fnu),
        "down": ((H, I // 2), torch.uint8), "down_scale": ((H, I // 32), torch.float8_e8m0fnu),
    }
    keys = set(model.state_dict())
    assert "model.layers.2.attn.wq_a.weight_scale_inv" in keys and "model.layers.2.attn.wq_a.scale" not in keys
    assert "model.layers.2.attn.compressor.wkv.weight_scale_inv" not in keys
    assert "model.head.weight" in keys and not any(".ffn.experts." in k for k in keys)


def _gpt_oss(cls):
    def check(model, idx):
        experts = op_at(model, "model.layers.0.mlp.experts")
        assert type(experts) is cls
        assert experts.interleaved and experts.alpha == pytest.approx(1.702) and experts.limit == 7.0
        if cls is GptOssMoELayer:
            assert experts.gate_up_proj_blocks.dtype is torch.uint8 and experts.gate_up_proj_blocks.dim() == 4

    return check


def _m3_sparse(model, idx):
    layers = model.model.layers.op_list
    sparse = next((i for i, l in enumerate(layers) if getattr(l.self_attn, "is_sparse", False)), None)
    if sparse is not None:
        assert type(op_at(model, f"model.layers.{sparse}.self_attn.index_qk_proj").quant_method) is MXFP8


def _gemma_tied_head(model, idx):
    assert model.lm_head.quant_method is None


CASES = [
    Case("Qwen3.6-35B-A3B-FP8", Fp8BlockConfig, {
        "model.layers.{lin}.linear_attn.in_proj_qkvz": FP8B, "model.layers.{lin}.linear_attn.in_proj_ba": BF16,
        "model.layers.{lin}.linear_attn.out_proj": FP8B, "model.layers.{full}.self_attn.qkv_proj": FP8B,
        "model.layers.{full}.self_attn.o_proj": FP8B, "model.layers.{lin}.mlp.shared_expert.gate_up_proj": FP8B,
        "model.layers.{lin}.mlp.shared_expert.down_proj": FP8B, "model.layers.{lin}.mlp.experts": Fp8BlockMoEMethod,
        "model.layers.{lin}.mlp.gate": BF16, "model.layers.{lin}.mlp.shared_expert_gate": BF16, "lm_head": BF16,
    }, check=_qwen35_fp8),
    Case("Qwen3.6-35B-A3B-NVFP4", ModelOptConfig, {
        "model.layers.{lin}.linear_attn.in_proj_qkvz": FP8T, "model.layers.{lin}.linear_attn.in_proj_ba": BF16,
        "model.layers.{full}.self_attn.qkv_proj": FP8T, "model.layers.{lin}.mlp.shared_expert.gate_up_proj": NVFP4,
        "model.layers.{lin}.mlp.shared_expert.down_proj": NVFP4, "model.layers.{lin}.mlp.experts": Nvfp4MoEMethod,
        "lm_head": NVFP4,
    }, check=_qwen35_nvfp4),
    # resident NVFP4 experts have no kernel without vLLM's Marlin
    Case("Qwen3.6-35B-A3B-NVFP4", ModelOptConfig, {}, strategy="fused", raises=KernelSelectionError,
         skip_if=lambda: not _installed("flashinfer") or _installed("vllm")),
    Case("Qwen3.8-27B-NVFP4", ModelOptConfig, {
        "model.layers.{lin}.linear_attn.in_proj_qkvz": FP8T, "model.layers.{full}.self_attn.o_proj": FP8T,
        "model.layers.{lin}.mlp.gate_up_proj": NVFP4, "model.layers.{lin}.mlp.down_proj": NVFP4, "lm_head": NVFP4,
    }, strategy="auto"),
    Case("Qwen3.6-27B", NoQuantConfig, {
        "model.layers.{lin}.linear_attn.in_proj": BF16, "model.layers.{lin}.mlp.gate_up_proj": BF16, "lm_head": BF16,
    }, strategy="auto", check=_no_split),
    Case("Qwen3.6-35B-A3B", NoQuantConfig, {
        "model.layers.{lin}.mlp.experts": UnquantizedMoEMethod, "model.layers.{lin}.mlp.shared_expert.down_proj": BF16,
    }),
    Case("nvidia/Qwen3.6-27B-NVFP4", ModelOptConfig, {
        "model.layers.{lin}.linear_attn.in_proj_qkvz": FP8T, "model.layers.{lin}.linear_attn.in_proj_ba": BF16,
        "model.layers.{lin}.linear_attn.out_proj": FP8T, "model.layers.{full}.self_attn.qkv_proj": FP8T,
        "model.layers.{lin}.mlp.gate_up_proj": NVFP4, "model.layers.{lin}.mlp.down_proj": NVFP4, "lm_head": NVFP4,
    }, strategy="auto"),
    *[Case(name, Fp8BlockConfig, {
        "model.layers.{lin}.linear_attn.in_proj_qkvz": FP8B, "model.layers.{lin}.linear_attn.in_proj_ba": BF16,
        "model.layers.{lin}.linear_attn.out_proj": FP8B, "model.layers.{full}.self_attn.qkv_proj": FP8B,
        "model.layers.{full}.self_attn.o_proj": FP8B, "model.layers.{lin}.mlp.gate_up_proj": FP8B,
        "model.layers.{lin}.mlp.down_proj": FP8B, "lm_head": BF16,
    }, strategy="auto") for name in ("Qwen/Qwen3.8-27B-FP8", "Qwen/Qwen3.6-27B-FP8")],
    Case("RedHatAI/Muse-Glimmer-30B-NVFP4", CompressedTensorsConfig, {
        "model.layers.0.self_attn.qkvg_proj": NVFP4, "model.layers.0.self_attn.o_proj": NVFP4,
        "model.layers.0.mlp.gate_up_proj": NVFP4, "model.layers.0.mlp.down_proj": NVFP4, "lm_head": BF16,
    }, strategy="auto"),
    Case("nvidia/Gemma-4-31B-IT-NVFP4", ModelOptConfig, {
        "model.layers.0.feed_forward.shared_mlp.gate_up_proj": NVFP4, "model.layers.0.feed_forward.shared_mlp.down_proj": NVFP4,
        "model.layers.0.self_attn.qkv_proj": BF16, "model.layers.0.self_attn.o_proj": BF16,
    }, strategy="auto", check=_gemma_tied_head),
    Case("google/gemma-4-12B-it", NoQuantConfig, {
        "model.layers.0.feed_forward.shared_mlp.gate_up_proj": BF16, "model.layers.0.self_attn.qkv_proj": BF16,
    }, strategy="auto", check=_gemma_tied_head),
    Case("Qwen/Qwen3-4B", NoQuantConfig, {
        "model.layers.0.self_attn.qkv_proj": BF16, "model.layers.0.self_attn.o_proj": BF16,
        "model.layers.0.mlp.gate_up_proj": BF16, "model.layers.0.mlp.down_proj": BF16,
    }, strategy="auto"),
    *[Case(name, NoQuantConfig, {
        "model.layers.0.self_attn.qkv_proj": BF16, "model.layers.0.self_attn.o_proj": BF16,
        "model.layers.0.mlp.gate_up_proj": BF16, "model.layers.0.mlp.down_proj": BF16,
    }, strategy="auto") for name in ("Qwen/Qwen2.5-1.5B-Instruct", "unsloth/Llama-3.2-1B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3")],
    *[Case("Qwen/Qwen3-30B-A3B", NoQuantConfig, {
        "model.layers.0.mlp.experts": UnquantizedMoEMethod, "model.layers.0.mlp.gate": BF16,
        "model.layers.0.self_attn.qkv_proj": BF16, "lm_head": BF16,
    }, strategy=strategy) for strategy in ("fused", "offload")],
    Case("RadixArk/Qwen3.8-Flash-Next-NVFP4", ModelOptConfig, {
        "model.layers.{lin}.mlp.experts": Nvfp4MoEMethod, "model.layers.{lin}.linear_attn.in_proj": BF16,
        "model.layers.{lin}.linear_attn.out_proj": BF16, "model.layers.{full}.self_attn.qkv_proj": BF16,
        "model.layers.{full}.self_attn.o_proj": BF16, "model.layers.{lin}.mlp.shared_expert.gate_up_proj": BF16,
        "model.layers.{lin}.mlp.shared_expert.down_proj": BF16, "model.layers.{lin}.mlp.gate": BF16, "lm_head": BF16,
        # the indexer, hyper-connection and PLE projections ask the config too
        "model.layers.{full}.self_attn.indexer.index_qk_proj": BF16,
        "model.layers.{lin}.attn_hyper_connection.input_mix_weight_down_block_inject": BF16,
        "model.layers.{lin}.mlp_hyper_connection.input_mix_weight_up": BF16,
        "model.hyper_connection_mixer.input_mix_weight_down": BF16,
        "model.layers.{ple}.ple.key_proj": BF16, "model.layers.{ple}.ple.value_proj": BF16,
    }, check=_no_split),
    Case("deepseek-ai/DeepSeek-V4-Flash-0731", Fp8BlockConfig, {
        **{f"model.layers.2.{p}": (FP8B, "dsv4") for p in (
            "attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b", "attn.indexer.wq_b",
            "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3")},
        # the KV compressors and the indexer's scorer ship bf16; the fp8 config has no modules_to_not_convert
        **{f"model.layers.2.{p}": (BF16, "torch") for p in (
            "attn.compressor.wkv", "attn.compressor.wgate", "attn.indexer.weights_proj", "attn.indexer.compressor.wkv")},
        "model.head": (BF16, "torch"), "model.layers.2.ffn.experts": (Mxfp4MoEMethod, "triton"),
    }, check=_dsv4),
    Case("nvidia/MiniMax-M2.5-NVFP4", ModelOptConfig, {
        "model.layers.3.block_sparse_moe.experts": Nvfp4MoEMethod, "model.layers.3.self_attn.qkv_proj": BF16,
        "model.layers.3.self_attn.o_proj": BF16, "model.layers.3.block_sparse_moe.gate": BF16, "lm_head": BF16,
    }),
    Case("nvidia/MiniMax-M3-NVFP4", ModelOptConfig, {
        "model.layers.{moe}.block_sparse_moe.experts": Nvfp4MoEMethod,
        "model.layers.{moe}.block_sparse_moe.shared_experts.gate_up_proj": MXFP8,
        "model.layers.{moe}.block_sparse_moe.shared_experts.down_proj": MXFP8, "model.layers.{dense}.mlp.gate_up_proj": MXFP8,
        "model.layers.{moe}.self_attn.qkv_proj": MXFP8, "model.layers.{moe}.self_attn.o_proj": MXFP8, "lm_head": BF16,
    }, env={"FREETOKEN_M3_MAX_LAYERS": "8"}, check=_m3_sparse),
    Case("nvidia/Gemma-4-26B-A4B-NVFP4", ModelOptConfig, {
        "model.layers.0.feed_forward.experts": Nvfp4MoEMethod, "model.layers.0.feed_forward.shared_mlp.gate_up_proj": BF16,
        "model.layers.0.feed_forward.shared_mlp.down_proj": BF16, "model.layers.0.feed_forward.router.proj": BF16,
        "model.layers.0.self_attn.qkv_proj": BF16, "model.layers.0.self_attn.o_proj": BF16,
    }, check=_gemma_tied_head),
    Case("google/gemma-4-26B-A4B-it", NoQuantConfig, {
        "model.layers.0.feed_forward.experts": UnquantizedMoEMethod, "model.layers.0.feed_forward.shared_mlp.gate_up_proj": BF16,
    }),
    Case("RedHatAI/GLM-5.3-Flash-NVFP4", CompressedTensorsConfig, {
        "model.layers.{moe}.mlp.experts": Nvfp4MoEMethod, "model.layers.{moe}.mlp.shared_experts.gate_proj": BF16,
        "model.layers.{moe}.mlp.shared_experts.down_proj": BF16, "model.layers.{dense}.mlp.gate_proj": BF16,
        "model.layers.{moe}.self_attn.o_proj": BF16, "lm_head": BF16,
        # the indexer, the MLA kv_b_proj and the KDA projections ask the config too
        "model.layers.{dsa}.self_attn.indexer.wq_b": BF16, "model.layers.{dsa}.self_attn.kv_b_proj": BF16,
        "model.layers.{kda}.self_attn.in_proj": BF16, "model.layers.{kda}.self_attn.f_b_proj": BF16,
        "model.layers.{kda}.self_attn.g_b_proj": BF16,
    }),
    Case("zai-org/GLM-5.3-Flash", Fp8BlockConfig, {
        "model.layers.{moe}.mlp.experts": (Fp8BlockMoEMethod, "triton"), "model.layers.{moe}.mlp.shared_experts.gate_proj": FP8B,
        "model.layers.{moe}.mlp.shared_experts.down_proj": FP8B, "model.layers.{dense}.mlp.gate_proj": FP8B,
        "model.layers.{dsa}.self_attn.q_b_proj": FP8B, "model.layers.{dsa}.self_attn.kv_b_proj": BF16,
        "model.layers.{dsa}.self_attn.o_proj": FP8B, "model.layers.{kda}.self_attn.in_proj": BF16,
        "model.layers.{kda}.self_attn.o_proj": BF16, "lm_head": BF16,
    }),
    Case("nvidia/GLM-4.7-NVFP4", ModelOptConfig, {
        "model.layers.{moe}.mlp.experts": Nvfp4MoEMethod, "model.layers.{moe}.mlp.shared_experts.gate_proj": NVFP4,
        "model.layers.{moe}.mlp.shared_experts.down_proj": NVFP4, "model.layers.{moe}.mlp.gate": BF16, "lm_head": BF16,
    }),
    Case("nvidia/GLM-5.3-NVFP4", ModelOptConfig, {
        "model.layers.{moe}.mlp.experts": Nvfp4MoEMethod, "model.layers.{moe}.mlp.gate": BF16,
        "model.layers.{moe}.mlp.shared_experts.gate_proj": BF16, "model.layers.{moe}.self_attn.o_proj": BF16, "lm_head": BF16,
        "model.layers.{dsa}.self_attn.indexer.wq_b": BF16, "model.layers.{moe}.self_attn.kv_b_proj": BF16,
    }),
    *[Case("openai/gpt-oss-20b", Mxfp4Config, {
        "model.layers.0.mlp.experts": Mxfp4MoEMethod, "model.layers.0.self_attn.qkv_proj": BF16,
        "model.layers.0.self_attn.o_proj": BF16, "model.layers.0.mlp.router": BF16, "lm_head": BF16,
    }, strategy=strategy, check=_gpt_oss(cls)) for strategy, cls in (
        ("fused", GptOssMoELayer), ("offload", GptOssOffloadMoELayer), ("cpu", GptOssOffloadMoELayer), ("hybrid", GptOssOffloadMoELayer))],
    # hybrid decodes on the CPU executor too, so the expert kernel must have a CPU format
    Case("Qwen3.6-35B-A3B-NVFP4", ModelOptConfig, {"model.layers.{lin}.mlp.experts": (Nvfp4MoEMethod, "triton")}, strategy="hybrid"),
    Case("Qwen3.6-35B-A3B", NoQuantConfig, {"model.layers.{lin}.mlp.experts": (UnquantizedMoEMethod, "fused")}, strategy="hybrid"),
    Case("Qwen3.6-35B-A3B-FP8", Fp8BlockConfig, {}, strategy="hybrid", raises=KernelSelectionError),
]


@pytest.mark.parametrize("case", CASES, ids=[f"{c.name.split('/')[-1]}:{c.strategy}" for c in CASES])
def test_probed_layers_get_the_method_their_config_says(case: Case, monkeypatch):
    path = model_dir(case.name)
    if path is None:
        pytest.skip(f"{case.name} not present locally")
    if case.skip_if is not None and case.skip_if():
        pytest.skip("kernel availability differs on this host")
    for key, value in case.env.items():
        monkeypatch.setenv(key, value)
    if case.raises is not None:
        with pytest.raises(case.raises):
            build_meta_model(path, case.strategy)
        return
    model, model_config = build_meta_model(path, case.strategy)
    assert type(model_config.quant) is case.dialect
    idx = _layer_indices(model)
    for template, expected in case.methods.items():
        probe = template.format(**{k: v for k, v in idx.items() if v is not None})
        op = op_at(model, probe)
        if getattr(op, "prefix", ""):  # routers are built without one
            assert op.prefix == probe
        if expected is None:
            assert op.quant_method is None, probe
            continue
        cls, kernel = expected if isinstance(expected, tuple) else (expected, None)
        assert type(op.quant_method) is cls, f"{probe}: {type(op.quant_method).__name__}"
        if kernel is not None:
            assert op.quant_method.kernel.name == kernel, probe
    if case.check is not None:
        case.check(model, idx)


# --------------------------------------------------------------------------- config without local weights


def test_hub_ids_fetch_the_modelopt_sidecar(tmp_path, monkeypatch):
    """An old ModelOpt export keeps its quantization config only in hf_quant_config.json."""
    from huggingface_hub.utils import EntryNotFoundError

    sidecar = tmp_path / "hf_quant_config.json"
    sidecar.write_text(json.dumps({"producer": {"name": "modelopt", "version": "0.29.0"}, "quantization": {"quant_algo": "FP8", "exclude_modules": ["lm_head"]}}))

    def fake_download(repo_id, filename, **_):
        assert repo_id == "org/old-modelopt-fp8"
        if filename == "hf_quant_config.json":
            return str(sidecar)
        raise EntryNotFoundError(f"no {filename}")

    monkeypatch.setattr("freetoken.utils.hf.hf_hub_download", fake_download)
    quant = checkpoint_quant_config("org/old-modelopt-fp8", SimpleNamespace(architectures=["Qwen3MoeForCausalLM"]), get_model_spec("Qwen3MoeForCausalLM"))
    assert type(quant) is ModelOptConfig
    assert quant.scheme_for("model.layers.0.self_attn.q_proj").kind is QuantKind.FP8_TENSOR
    assert quant.scheme_for("lm_head") is None


def test_unsupported_dialects_fail_closed(tmp_path):
    hf_config = SimpleNamespace(architectures=["Qwen3MoeForCausalLM"], quantization_config={"quant_method": "gptq", "bits": 4})
    with pytest.raises(NotImplementedError, match="gptq"):
        checkpoint_quant_config(str(tmp_path), hf_config, get_model_spec("Qwen3MoeForCausalLM"))


# --------------------------------------------------------------------------- config against the stored tensors

# leaf names FreeToken builds as Linear / MoE layers; routers, norms, convs never ask for a scheme
_LINEAR_LEAVES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "w1", "w2", "w3",
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "in_proj_ba", "out_proj", "qkv_proj", "gate_up_proj",
    "lm_head", "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "wq_a", "wq_b", "wkv_a", "wkv_b", "wo",
    "wkv", "wo_a", "wo_b", "wgate", "weights_proj", "experts",
    "b_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "index_qk_proj", "key_proj", "value_proj",
    "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight",
}
_META_SUFFIXES = {"bias", "k_scale", "v_scale"}
_QUANT_SUFFIXES = {
    "weight_scale", "weight_scale_2", "weight_scale_inv", "weight_packed", "weight_global_scale",
    "input_scale", "input_scale_2", "input_global_scale", "scale",
    "gate_up_proj_blocks", "gate_up_proj_scales", "gate_up_proj_bias", "down_proj_blocks", "down_proj_scales", "down_proj_bias",
}
# unsupported dialects: from_hf must refuse them, nothing else is checked
_UNSUPPORTED = {"quark", "mxfp8", "fp_quant"}
# tables the model code reads itself; RadixArk lists the fp8 PLE table in ``ignore`` yet ships it quantized
_MODEL_READS = ("ple.ple_embedding",)
# non-Linear leaves whose parameters happen to be called scale
_NOT_LINEAR = {"router", "gate", "shared_expert_gate"}


def _weight_map(p: Path) -> dict[str, str] | None:
    """Tensor name -> shard file, from the index or from a single model.safetensors."""
    if (p / "model.safetensors.index.json").exists():
        return json.load(open(p / "model.safetensors.index.json"))["weight_map"]
    if (p / "model.safetensors").exists():
        return {name: "model.safetensors" for name in _safetensors_header(p / "model.safetensors") if name != "__metadata__"}
    return None


def _candidate_dirs() -> list[Path]:
    roots = glob.glob(os.path.join(HF_CACHE, "models--*/snapshots/*/")) + glob.glob(os.path.join(MODELS, "*/"))
    return [Path(d) for d in roots if (Path(d) / "config.json").exists() and _weight_map(Path(d)) is not None]


def _label(p: Path) -> str:
    m = re.search(r"models--([^/]+)/snapshots", str(p))
    name = m.group(1).replace("--", "/") if m else f"nvme:{p.name}"
    # an HF cache entry may hold the config and index without the shards; the tensor check then only has the names
    shards = set(_weight_map(p).values())
    return name if all((p / s).exists() for s in shards) else f"{name}[index-only]"


def _safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _tensor_info(p: Path, weight_map: dict[str, str]) -> dict[str, dict]:
    info: dict[str, dict] = {}
    for shard in sorted(set(weight_map.values())):
        if not (p / shard).exists():
            return {}
        for name, meta in _safetensors_header(p / shard).items():
            if name != "__metadata__":
                info[name] = meta
    return info


def _expected_kinds(suffixes: set[str], tensors: dict[str, dict], module: str) -> set[QuantKind]:
    s = suffixes - _META_SUFFIXES
    if not s & {"weight", "weight_packed", "gate_up_proj_blocks"} or not s & _QUANT_SUFFIXES:
        return {QuantKind.NONE}
    if "weight_packed" in s or "weight_scale_2" in s:
        return {QuantKind.NVFP4}
    if "gate_up_proj_blocks" in s:
        return {QuantKind.MXFP4}

    def dtype(suffix):
        return (tensors.get(f"{module}.{suffix}") or {}).get("dtype")

    def shape(suffix):
        return (tensors.get(f"{module}.{suffix}") or {}).get("shape")

    if "weight_scale_inv" in s:
        if tensors:
            return {QuantKind.MXFP8} if dtype("weight_scale_inv") == "U8" else {QuantKind.FP8_BLOCK}
        return {QuantKind.FP8_BLOCK, QuantKind.MXFP8}
    if "scale" in s:
        if tensors:
            return {QuantKind.FP8_BLOCK} if dtype("weight") == "F8_E4M3" else {QuantKind.MXFP4}
        return {QuantKind.FP8_BLOCK, QuantKind.MXFP4}
    if "weight_scale" in s:
        if tensors:
            if dtype("weight") == "U8":
                return {QuantKind.MXFP4}
            sh = shape("weight_scale") or []
            return {QuantKind.FP8_BLOCK} if len(sh) == 2 and sh[0] > 1 and sh[1] > 1 else {QuantKind.FP8_TENSOR}
        return {QuantKind.FP8_TENSOR, QuantKind.FP8_BLOCK, QuantKind.MXFP4}
    return {QuantKind.NONE}


def _probes(weight_map: dict[str, str]):
    mods: dict[str, set[str]] = defaultdict(set)
    for name in weight_map:
        module, _, suffix = name.rpartition(".")
        mods[module].add(suffix)
    for module, suffixes in mods.items():
        leaf = module.rpartition(".")[2]
        if leaf in _NOT_LINEAR or any(m in module for m in _MODEL_READS) or suffixes <= _META_SUFFIXES:
            continue
        if leaf in _LINEAR_LEAVES or suffixes & _QUANT_SUFFIXES:
            yield module, suffixes


DIRS = _candidate_dirs()


@pytest.mark.skipif(not DIRS, reason="no local checkpoint with config.json + safetensors")
@pytest.mark.parametrize("ckpt", DIRS, ids=_label)
def test_scheme_for_agrees_with_the_stored_tensors(ckpt: Path):
    cfg = json.load(open(ckpt / "config.json"))
    quant = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
    if str(quant.get("quant_method") or "").lower() in _UNSUPPORTED:
        with pytest.raises(NotImplementedError):
            QuantConfig.from_hf(cfg)
        return
    hfq = json.load(open(ckpt / "hf_quant_config.json")) if (ckpt / "hf_quant_config.json").exists() else None
    try:
        spec = get_model_spec((cfg.get("architectures") or [""])[0])
    except Exception:
        spec = None
    qc = QuantConfig.from_hf(cfg, hf_quant_config=hfq, unquantized=spec.unquantized_modules if spec else ())
    weight_map = _weight_map(ckpt)
    tensors = _tensor_info(ckpt, weight_map)
    mismatches, checked = [], 0
    for module, suffixes in _probes(weight_map):
        expected = _expected_kinds(suffixes, tensors, module)
        scheme = qc.scheme_for(module)
        kind = scheme.kind if scheme else QuantKind.NONE
        checked += 1
        if kind not in expected:
            mismatches.append(f"{module}: config says {kind}, tensors {sorted(suffixes)} say {sorted(k.value for k in expected)}")
    assert checked > 0
    assert not mismatches, f"{len(mismatches)}/{checked} modules disagree:\n  " + "\n  ".join(mismatches[:40])
