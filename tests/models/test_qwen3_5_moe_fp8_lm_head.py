"""qwen3_5_moe fp8 ``lm_head``: detection, layer choice, and the loader gate.

A compressed-tensors mixed-precision checkpoint can put ``lm_head`` in its fp8 group
(unsloth/Qwen3.6-35B-A3B-NVFP4-Fast). Three things have to agree for that to load: the config
detector reports ``"fp8"``, the model builds an ``Fp8LMHead``, and the dense loader keeps the
weight fp8 instead of dequantizing it. The loader gate is the subtle one -- emitting fp8 into
the bf16 ``ParallelLMHead`` fails with an unexpected ``lm_head.weight_scale`` key.

Runs off trimmed copies of the real ``quantization_config`` blocks.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.fp8_pertensor_linear import Fp8LMHead, Fp8PerTensorLinear
from freetoken.models.qwen3_5_moe.config import _lm_head_quant


def _quant_config(*, lm_head_in_targets: bool, strategy: str = "channel") -> dict:
    """Trimmed from unsloth/Qwen3.6-35B-A3B-NVFP4-Fast: fp8 group + NVFP4 expert group."""
    targets = [".self_attn.q_proj", ".linear_attn.in_proj_qkv"]
    if lm_head_in_targets:
        targets = targets + ["lm_head"]
    return {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {
                "targets": targets,
                "weights": {"num_bits": 8, "type": "float",
                            "strategy": strategy, "group_size": None},
            },
            "group_1": {
                "targets": ["re:.*mlp.experts.*"],
                "weights": {"num_bits": 4, "type": "float",
                            "strategy": "tensor_group", "group_size": 16},
            },
        },
    }


class _Cfg:
    def __init__(self, quantization_config):
        self.quantization_config = quantization_config


@pytest.mark.parametrize("strategy", ["tensor", "channel"])
def test_lm_head_in_the_fp8_group_is_detected(strategy):
    cfg = _Cfg(_quant_config(lm_head_in_targets=True, strategy=strategy))
    assert _lm_head_quant(cfg) == "fp8"


def test_lm_head_outside_the_fp8_group_is_not_claimed():
    """Only the group that targets ``lm_head`` may decide the head's dtype; a checkpoint whose
    fp8 group covers the attention projections alone must leave the head bf16."""
    cfg = _Cfg(_quant_config(lm_head_in_targets=False))
    assert _lm_head_quant(cfg) == "none"


def test_an_ignored_lm_head_is_not_claimed_even_when_a_group_targets_it():
    """compressed-tensors ``ignore`` wins over ``targets``. This is the common llm-compressor
    shape: a group matches broadly, then ``ignore`` carves modules back out. Such a head is
    bf16 on disk, so claiming it as fp8 would fail the load on the dtype check."""
    cfg = _Cfg(_quant_config(lm_head_in_targets=True))
    cfg.quantization_config["ignore"] = ["re:.*lm_head"]
    assert _lm_head_quant(cfg) == "none"


def test_an_unrelated_ignore_entry_does_not_suppress_the_fp8_head():
    cfg = _Cfg(_quant_config(lm_head_in_targets=True))
    cfg.quantization_config["ignore"] = ["re:.*mlp.experts.*", "model.embed_tokens"]
    assert _lm_head_quant(cfg) == "fp8"


def test_a_4bit_lm_head_group_is_not_read_as_fp8():
    cfg = _Cfg(_quant_config(lm_head_in_targets=True))
    cfg.quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] = 4
    assert _lm_head_quant(cfg) == "none"


def test_a_grouped_fp8_lm_head_is_not_read_as_per_tensor():
    """Fp8LMHead consumes a per-output-row scale; a block/group scale is a different layout
    and must not be routed here."""
    cfg = _Cfg(_quant_config(lm_head_in_targets=True, strategy="group"))
    cfg.quantization_config["config_groups"]["group_0"]["weights"]["group_size"] = 128
    assert _lm_head_quant(cfg) == "none"


def test_fp8_lm_head_buffers_match_the_per_output_row_contract():
    head = Fp8LMHead(num_embeddings=64, embedding_dim=8)
    assert isinstance(head, Fp8PerTensorLinear)
    assert head.weight.shape == (64, 8)
    assert head.weight.dtype == torch.float8_e4m3fn
    # Per output row, fp32 -- covers "channel" directly and "tensor" once broadcast.
    assert head.weight_scale.shape == (64,)
    assert head.weight_scale.dtype == torch.float32
    assert head.bias is None
