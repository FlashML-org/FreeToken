"""Engine config resolution for DeepSeek-V4.1 on the tiny checkpoint: the DSV41 backend and pool are
picked, the page size is the window page, the replay knob lands on the model args, and the fp8
dialect (block 32, fp4 experts) reaches the quant layer with the matching activation block."""

from __future__ import annotations

import json
import os

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.scheduler.config import SchedulerConfig
from freetoken.engine.engine import _adjust_config
from freetoken.kvcache import resolve_pool_class
from freetoken.kvcache.dsv4.v41_pool import DSV41PagedKVCache

from .common import write_tiny_checkpoint


def _engine_config(path, **over):
    return SchedulerConfig(model_path=path, tp_info=DistributedInfo(rank=0, size=1), dtype=torch.bfloat16, **over)


@pytest.mark.parametrize("replay", ["bounded", "exact"])
def test_resolution_picks_dsv41_and_the_replay_knob(tmp_path, monkeypatch, replay):
    from freetoken.engine import engine

    monkeypatch.setattr(engine, "is_sm100_family", lambda: False)
    monkeypatch.setattr(engine, "is_sm90_family", lambda: True)
    write_tiny_checkpoint(str(tmp_path))
    config = _engine_config(str(tmp_path), attention_backend="auto", moe_strategy="offload", swa_decoder_replay=replay, max_seq_len_override=2048)
    _adjust_config(config)
    assert config.attention_backend == "dsv41_sparse"
    assert config.page_size == 128 and config.cache_type == "swa_radix"
    assert resolve_pool_class(config.model_config) is DSV41PagedKVCache
    args = config.model_config.dsv41_args
    assert args.swa_decoder_replay == replay and args.max_seq_len == 2048 and args.max_batch_size == config.max_running_req + 1
    assert config.max_extend_tokens == 8192  # the prefill chunk stays bounded (whole window pages)
    # the cache contract follows the replay mode: a bounded-mode prefix hit must leave the prompt's last
    # window to the prefill (the pool's prefix_replay_tokens); the history a resume reads stays one window
    from freetoken.kvcache.dsv4.v41_cost_model import dsv41_pool_sizes

    spec = config.model_config.kv_cache_group_specs()[0]
    pool = DSV41PagedKVCache(dsv41_pool_sizes(16, args, 1.0, 128), args, torch.device("cpu"))
    assert pool.prefix_replay_tokens == (128 if replay == "bounded" else 0)
    assert pool.sliding_window_size == 128 and spec.sliding_window == 128 and not spec.is_swa
    # bounded replay keeps the decoder's per-request window KV in private rings, off the shared pages
    assert pool.private_window_layer_ids == (tuple(range(args.decoder_start_layer, args.n_layers)) if replay == "bounded" else ())
    assert DSV41PagedKVCache.min_kv_tokens(config) // 128 == 8 + 3 * config.max_running_req + 2 * (config.max_running_req + 1) + 1


def test_fp8_block32_dialect_reaches_the_quant_layer(tmp_path):
    """A V4.1-style quantization_config on the tiny checkpoint: dense linears get the 32-block
    e8m0 scheme, routed experts the MXFP4 scheme with a 32-wide activation block."""
    from freetoken.distributed.info import set_tp_info, try_get_tp_info
    from freetoken.layers import OffloadMoELayer
    from freetoken.layers.quantization import Fp8BlockConfig, QuantKind
    from freetoken.layers.quantization.scheme import fp8_block_size

    write_tiny_checkpoint(str(tmp_path))
    cfg_path = os.path.join(str(tmp_path), "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"}
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    config = _engine_config(str(tmp_path), moe_strategy="offload")
    quant = config.model_config.quant
    assert type(quant) is Fp8BlockConfig and quant.block == 32
    dense = quant.scheme_for("model.layers.3.attn.wq_a")
    assert dense.kind is QuantKind.FP8_BLOCK and fp8_block_size(dense) == 32
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    experts = OffloadMoELayer(3, num_experts=4, top_k=2, hidden_size=64, intermediate_size=64, limit=10.0, quant_config=quant, prefix="model.layers.3.ffn.experts")
    assert experts.quant_method.scheme.kind is QuantKind.MXFP4 and experts.quant_method.cfg.act_block == 32
    # the family's bf16 modules stay unquantized although the fp8 config lists no exceptions
    for name in ("model.layers.2.attn.compressor.wkv", "model.layers.2.attn.indexer.wk", "model.layers.5.attn.indexer.weights_proj", "head"):
        assert quant.scheme_for(name) is None, name
    assert quant.scheme_for("model.layers.1.engram.wkv").kind is QuantKind.FP8_BLOCK
