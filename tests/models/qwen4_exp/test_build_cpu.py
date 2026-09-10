"""The whole model must CONSTRUCT without a GPU.

Every other model test that builds a decoder layer is behind requires_cuda, so a constructor
signature mismatch between Qwen4ExpDecoderLayer and the ops it builds is invisible to a
CPU-only run -- it only shows up when a server boots. That happened: Qwen4ExpMoE did not
accept the `prefix` kwarg model.py passes, and a full CPU suite still went green.

Building on the meta device costs no memory and no GPU, so there is no reason not to.
"""

import torch
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers import set_rope_device
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

from .common import toy_hf_config


def _build():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cpu"))
    with torch.device("meta"):
        return Qwen4ExpForCausalLM(parse_config(toy_hf_config()))


def test_model_constructs_on_meta_without_cuda():
    model = _build()
    sd = model.state_dict()
    assert sd, "state dict is empty"
    # both layer families and the head must be present
    assert any(".self_attn." in k for k in sd), "no full-attention layer built"
    assert any(".linear_attn." in k for k in sd), "no GDN layer built"
    assert any(k.startswith("lm_head") for k in sd), "no lm_head built"


def test_every_layer_gets_its_own_prefixed_weights():
    # a prefix that is dropped or shared silently collapses layers onto one another
    sd = _build().state_dict()
    mlp_keys = {k for k in sd if ".mlp." in k}
    layers = {k.split(".layers.")[1].split(".")[0] for k in mlp_keys if ".layers." in k}
    assert len(layers) > 1, f"MoE weights landed on a single layer: {sorted(layers)}"
