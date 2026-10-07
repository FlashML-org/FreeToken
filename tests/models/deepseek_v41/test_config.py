"""Config parsing, the DSV41 layer-role table, registry dispatch and loader / model key parity."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from freetoken.layers import set_rope_device
from freetoken.layers.quantization import NoQuantConfig
from freetoken.distributed.info import set_tp_info, try_get_tp_info
from freetoken.models import create_model
from freetoken.models.deepseek_v41.args import DeepseekV41Args, Mode
from freetoken.models.deepseek_v41.config import parse_config
from freetoken.models.register import get_model_spec
from freetoken.utils.torch_utils import torch_dtype

from .common import ENGRAM_LAYERS, INDEX_SOURCES, KV_SOURCES, N_LAYERS, RATIOS, tiny_hf_config, write_tiny_checkpoint


def test_roles_follow_the_dsv41_mode_table():
    args = parse_config(tiny_hf_config()).dsv41_args
    modes = [r.mode for r in args.roles]
    assert modes == [Mode.WINDOW, Mode.WINDOW, Mode.FULL, Mode.REUSE, Mode.FULL, Mode.REINDEX]
    assert [r.kv_source for r in args.roles] == [None, None, 2, 2, 4, 4]
    assert args.roles[4].is_candidate_source and args.roles[5].uses_candidates and not args.roles[2].uses_candidates
    assert args.decoder_start_layer == 4 and args.backbone_kv_sources == (2, 4)
    assert args.freqs_params(0)[1] == 0 and args.freqs_params(2)[1] == 1024  # yarn only on compressing layers
    assert args.swa_decoder_replay == "bounded"


def test_shipping_layout_roles():
    """The real 40-layer layout: 3 encoder Full layers (ratio 2), the decoder Full at 20 building the
    candidate pool, Reindex at 24/28/32/36 inside it, Reuse elsewhere."""
    text = tiny_hf_config().text_config
    text.update(
        num_hidden_layers=40, compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        kv_source_layer_ids=[2, 8, 14, 20], index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20, num_nextn_predict_layers=3, engram_layer_ids=[1, 14], engram_num_embeddings=[10, 10],
    )
    args = DeepseekV41Args.from_hf(tiny_hf_config(**text))
    full = [r.layer_id for r in args.roles if r.mode is Mode.FULL]
    reindex = [r.layer_id for r in args.roles if r.mode is Mode.REINDEX]
    assert full == [2, 8, 14, 20] and reindex == [24, 28, 32, 36]
    assert sum(r.mode is Mode.REUSE for r in args.roles) == 40 - 2 - 4 - 4
    assert [r.layer_id for r in args.roles if r.uses_candidates] == reindex
    assert args.decoder_start_layer == 20


def test_bad_layouts_are_rejected():
    with pytest.raises(ValueError, match="no kv source"):
        DeepseekV41Args.from_hf(tiny_hf_config(kv_source_layer_ids=[4]))
    with pytest.raises(ValueError, match="reads kv source"):
        DeepseekV41Args.from_hf(tiny_hf_config(compress_ratios=[0, 0, 2, 1, 1, 1, 0]))
    with pytest.raises(ValueError, match="different kv source"):
        DeepseekV41Args.from_hf(tiny_hf_config(compress_ratios=[0, 0, 2, 2, 2, 2, 0], kv_source_layer_ids=[2, 4], candidate_source_layer_id=2, index_source_layer_ids=[2, 4, 5]))


def test_registry_and_kv_sources():
    spec = get_model_spec("DeepseekV41ForCausalLM")
    assert spec.module == "freetoken.models.deepseek_v41"
    from freetoken.attention import AttnType
    from freetoken.models.config import DSV4AttentionGroupConfig

    mc = parse_config(tiny_hf_config())
    args = mc.dsv41_args
    assert args.backbone_kv_sources == (2, 4) and tuple(r.ratio for r in args.roles) == tuple(RATIOS[:N_LAYERS])
    (group,) = mc.attention_groups
    assert type(group) is DSV4AttentionGroupConfig and group.variant == "v41"
    assert mc.kv_cache_group_specs()[0].attn_type is AttnType.DSV41 and mc.attn_type_for_layer(0) is AttnType.DSV41
    assert mc.num_layers == N_LAYERS and mc.dsv41_args is not None


def _meta_model(mc):
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    set_rope_device(torch.device("cpu"))
    mc = replace(mc, moe_strategy="offload", decode_target="gpu", quant=NoQuantConfig())
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        return create_model(mc)


def test_loader_names_every_model_parameter(tmp_path):
    from freetoken.models.deepseek_v41.weight import iter_weights

    write_tiny_checkpoint(str(tmp_path))
    model = _meta_model(parse_config(tiny_hf_config()))
    params = model.state_dict()
    loaded = dict(iter_weights(str(tmp_path), torch.device("cpu"), include_moe_experts=False))
    assert set(loaded) == set(params), (set(loaded) ^ set(params))
    for name, tensor in loaded.items():
        assert tuple(tensor.shape) == tuple(params[name].shape), name
    # routed experts, the DSpark stage, the vision tower and the image routing bias are not resident weights
    assert not any(".experts." in k or k.startswith(("mtp.", "vision.")) or k.endswith("bias_vl") for k in loaded)
    # per-role structure
    assert model.model.layers.op_list[3].attn.compressor is None and model.model.layers.op_list[3].attn.indexer is None
    assert model.model.layers.op_list[4].attn.compressor.ratio == 1 and model.model.layers.op_list[2].attn.compressor.ratio == 2
    assert [b.engram is not None for b in model.model.layers.op_list] == [L in ENGRAM_LAYERS for L in range(N_LAYERS)]
    assert all(L in INDEX_SOURCES for L in range(N_LAYERS) if model.model.layers.op_list[L].attn.indexer is not None)
    assert all(L in KV_SOURCES for L in range(N_LAYERS) if model.model.layers.op_list[L].attn.compressor is not None)


@pytest.mark.parametrize("missing_weight", [False, True])
def test_loader_closes_shards_when_expert_loading_stops(tmp_path, monkeypatch, missing_weight):
    from freetoken.models.deepseek_v41 import weight

    write_tiny_checkpoint(str(tmp_path))
    closed = []
    close, get = weight._ShardReader.close, weight._ShardReader.get

    def close_reader(reader):
        close(reader)
        closed.append(True)

    def read_weight(reader, name):
        if missing_weight and name.endswith(".w3.weight"):
            raise KeyError(name)
        return get(reader, name)

    monkeypatch.setattr(weight._ShardReader, "close", close_reader)
    monkeypatch.setattr(weight._ShardReader, "get", read_weight)
    tensors = weight.iter_weights(str(tmp_path), torch.device("cpu"))
    if missing_weight:
        with pytest.raises(KeyError):
            next(tensors)
    else:
        next(tensors)
        tensors.close()
    assert closed == [True]


def test_fp8_wo_a_layer_rejects_a_dequantized_ftw_layout_clearly():
    """An fp8-declared wo_a fed a state dict without ``wo_a_scale`` (an FTW converted before wo_a kept
    its payload) fails with a reconvert message, not a raw missing-key error; a bf16-declared layer fed
    the fp8 payload + scale dequantizes it on the way in."""
    from types import SimpleNamespace

    from freetoken.layers.quantization.scheme import fp8_block_scheme
    from freetoken.models.deepseek_v41.attention import DSV41Attention

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    args = parse_config(tiny_hf_config()).dsv41_args
    args.max_seq_len = 512
    # wo_a is fp8; the LinearReplicated modules stay unquantized (the bf16 quant method)
    plain = NoQuantConfig()
    quant = SimpleNamespace(
        scheme_for=lambda name: fp8_block_scheme("ue8m0", 32) if name.endswith(".wo_a") else None,
        get_quant_method=lambda layer, prefix: plain.get_quant_method(layer, prefix),
    )
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        fp8_layer = DSV41Attention(args, args.roles[2], quant_config=quant, prefix="attn")
        bf16_layer = DSV41Attention(args, args.roles[2], quant_config=None, prefix="attn")
    assert fp8_layer.wo_a.dtype == torch.float8_e4m3fn and bf16_layer.wo_a.dtype == torch.bfloat16
    rows, cols = fp8_layer.wo_a.shape
    with torch.device("meta"):
        state = {k: torch.empty_like(v) for k, v in fp8_layer.state_dict(prefix="attn").items()}
    del state["attn.wo_a_scale"]
    state["attn.wo_a"] = torch.empty(rows, cols, dtype=torch.bfloat16, device="meta")
    with pytest.raises(RuntimeError, match="reconvert"):
        fp8_layer.load_state_dict(dict(state), prefix="attn")
    # the bf16 layer given the fp8 payload + scale dequantizes (real tensors: the dequant runs)
    w = (torch.randn(rows, cols) * 0.05).to(torch.float8_e4m3fn)
    scale = torch.full((rows // 32, cols // 32), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    state2 = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in bf16_layer.state_dict(prefix="attn").items()}
    state2["attn.wo_a"], state2["attn.wo_a_scale"] = w, scale
    bf16_layer.load_state_dict(state2, prefix="attn")
    assert torch.equal(bf16_layer.wo_a, w.to(torch.bfloat16))
