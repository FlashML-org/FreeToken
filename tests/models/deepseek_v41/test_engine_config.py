"""Engine config resolution for DeepSeek-V4.1 on the tiny checkpoint: the CSA2 backend and pool are
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
from freetoken.kvcache.csa2_paged_pool import CSA2PagedKVCache

from .common import write_tiny_checkpoint


def _engine_config(path, **over):
    return SchedulerConfig(model_path=path, tp_info=DistributedInfo(rank=0, size=1), dtype=torch.bfloat16, **over)


@pytest.mark.parametrize("replay", ["bounded", "exact"])
def test_resolution_picks_csa2_and_the_replay_knob(tmp_path, monkeypatch, replay):
    from freetoken.engine import engine

    monkeypatch.setattr(engine, "is_sm100_family", lambda: False)
    monkeypatch.setattr(engine, "is_sm90_family", lambda: True)
    write_tiny_checkpoint(str(tmp_path))
    config = _engine_config(str(tmp_path), attention_backend="auto", moe_strategy="offload", decoder_replay=replay, max_seq_len_override=2048)
    _adjust_config(config)
    assert config.attention_backend == "csa2_sparse"
    assert config.page_size == 128 and config.cache_type == "swa_radix"
    assert resolve_pool_class(config.model_config) is CSA2PagedKVCache
    args = config.model_config.dsv41_args
    assert args.decoder_replay == replay and args.max_seq_len == 2048 and args.max_batch_size == config.max_running_req + 1
    assert config.max_extend_tokens == 8192  # the prefill chunk stays bounded (whole window pages)
    # the cache contract follows the replay mode: a bounded-mode prefix hit recomputes the window before
    # it, so the cache manager must match / lock / retain two windows of live history behind a hit
    geom = config.model_config.attention_groups[0].geometry
    spec = next(g for g in config.model_config.kv_cache_group_specs() if g.resume_history is not None)
    want = 256 if replay == "bounded" else 128
    assert geom.resume_history == want and spec.resume_history == want and spec.sliding_window == 128
    # bounded replay keeps the decoder's per-request window KV in private rings, off the shared pages
    assert geom.private_window_layer_ids == (tuple(range(args.decoder_start_layer, args.n_layers)) if replay == "bounded" else ())
    assert CSA2PagedKVCache.min_kv_tokens(config) // 128 == 8 + (2 * geom.resume_windows + 1) * config.max_running_req + 2 * (config.max_running_req + 1) + 1


def test_bounded_mode_cache_refuses_a_hit_with_one_live_window(tmp_path, monkeypatch):
    """SWARadixCache built from the resolved config demands two live windows behind a reusable
    position in bounded mode: with the page before a prompt's last window tombstoned, the match
    falls back to a shorter (safe) prefix instead of admitting a hit whose recompute has no keys."""
    from freetoken.engine import engine
    from freetoken.kvcache.swa_radix_cache import SWARadixCache
    from freetoken.scheduler.cache import CacheManager

    monkeypatch.setattr(engine, "is_sm100_family", lambda: False)
    monkeypatch.setattr(engine, "is_sm90_family", lambda: True)
    write_tiny_checkpoint(str(tmp_path))
    config = _engine_config(str(tmp_path), attention_backend="auto", moe_strategy="offload", decoder_replay="bounded", max_seq_len_override=2048)
    _adjust_config(config)
    resume = next(g for g in config.model_config.kv_cache_group_specs() if g.resume_history is not None).resume_history
    assert resume == 256
    ids = torch.arange(768, dtype=torch.int32)
    # exact mode's contract (one window) still admits the hit below; bounded mode's does not
    for window, admits in ((128, True), (resume, False)):
        cache = SWARadixCache(torch.device("cpu"), config.page_size, window)
        cache.insert(ids, torch.arange(768, dtype=torch.int32))
        assert cache.match_prefix(ids).cached_len == 768  # everything live: the whole prompt is reusable
        cache.trim_head_swa(ids, 640)  # the finish-time head trim: only [640, 768) stays swa-live
        got = cache.match_prefix(ids).cached_len
        assert (got == 768) is admits, (window, got)
        assert got % 128 == 0


def test_fp8_block32_dialect_reaches_the_quant_layer(tmp_path):
    """A V4.1-style quantization_config on the tiny checkpoint: dense linears get the 32-block
    e8m0 scheme, routed experts the MXFP4 scheme with a 32-wide activation block."""
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
    experts = quant.scheme_for("model.layers.3.ffn.experts")
    assert experts.kind is QuantKind.MXFP4 and experts.act_block(128) == 32
    # the family's bf16 modules stay unquantized although the fp8 config lists no exceptions
    for name in ("model.layers.2.attn.compressor.wkv", "model.layers.2.attn.indexer.wk", "model.layers.5.attn.indexer.weights_proj", "head"):
        assert quant.scheme_for(name) is None, name
    assert quant.scheme_for("model.layers.1.engram.wkv").kind is QuantKind.FP8_BLOCK
