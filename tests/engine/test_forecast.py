"""Pre-load memory forecast (engine/forecast.py): CPU-only, on synthetic header-only checkpoints.

The forecast must reuse the engine's own sizing functions, so each check prices the same
configuration through those functions directly and compares."""

import json
import struct
import time

import pytest
import torch

from freetoken.engine import forecast as fc_mod
from freetoken.engine.forecast import GiB, MiB, PreflightRefused, tensor_category

_DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}


def write_checkpoint(path, config: dict, tensors: dict[str, tuple[str, list[int]]]) -> str:
    """config.json plus one safetensors shard holding only its header (no tensor data)."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        nbytes = _DTYPE_BYTES[dtype]
        for d in shape:
            nbytes *= d
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
    return str(path)


TINY_QWEN3 = {
    "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 128,
    "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 2,
    "num_key_value_heads": 1, "head_dim": 64, "vocab_size": 256, "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6, "rope_theta": 10000, "hidden_act": "silu", "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}


# 2 slabs x 1 kv head x head_dim 64 x bf16 x 2 layers
KV_PER_TOKEN = 2 * 1 * 64 * 2 * 2


def tiny_qwen3_tensors() -> dict[str, tuple[str, list[int]]]:
    t = {"model.embed_tokens.weight": ("BF16", [256, 128]), "lm_head.weight": ("BF16", [256, 128]),
         "model.norm.weight": ("BF16", [128])}
    for i in range(2):
        p = f"model.layers.{i}."
        t.update({
            p + "self_attn.q_proj.weight": ("BF16", [128, 128]), p + "self_attn.k_proj.weight": ("BF16", [64, 128]),
            p + "self_attn.v_proj.weight": ("BF16", [64, 128]), p + "self_attn.o_proj.weight": ("BF16", [128, 128]),
            p + "self_attn.q_norm.weight": ("BF16", [64]), p + "self_attn.k_norm.weight": ("BF16", [64]),
            p + "mlp.gate_proj.weight": ("BF16", [256, 128]), p + "mlp.up_proj.weight": ("BF16", [256, 128]),
            p + "mlp.down_proj.weight": ("BF16", [128, 256]),
            p + "input_layernorm.weight": ("BF16", [128]), p + "post_attention_layernorm.weight": ("BF16", [128]),
        })
    return t


def _analyze(model_path, *flags, free_gib=8.0):
    from freetoken.server.info import analyze, split_argv
    from freetoken.server.args import parse_args

    opts, argv = split_argv([model_path, "--gpu-free-gib", str(free_gib), *flags])
    config, _ = parse_args(argv, prog="ft info")
    return analyze(config, opts)


@pytest.fixture
def tiny_qwen3(tmp_path):
    return write_checkpoint(tmp_path / "tiny-qwen3", TINY_QWEN3, tiny_qwen3_tensors())


def test_tensor_category_buckets():
    cases = {
        "model.layers.3.self_attn.qkv_proj.weight": "attention",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "linear_attention",
        "model.layers.1.mlp.experts.7.down_proj.weight": "experts",
        "model.layers.1.mlp.gate.weight": "mlp",
        "model.layers.1.mlp.shared_expert.up_proj.weight": "mlp",
        "model.embed_tokens.weight": "embeddings",
        "lm_head.weight_scale": "embeddings",
        "model.visual.blocks.0.attn.qkv.weight": "vision",
        "visual.merger.linear_fc1.weight": "vision",
        "mtp.layers.0.mlp.experts.0.gate_proj.weight": "mtp",
        "model.layers.0.input_layernorm.weight": "other",
    }
    for name, cat in cases.items():
        assert tensor_category(name) == cat, name


def test_forecast_matches_engine_sizing(tiny_qwen3):
    from freetoken.engine.engine import _startup_kv_budget
    from freetoken.kvcache.mha_pool import MHAKVCache

    r = _analyze(tiny_qwen3)
    config, f = r.config, r.forecast
    # weights: the meta model's parameters (bf16, untied lm_head) + the eager rope table
    params = sum(t.numel() * t.element_size() for t in _meta_model(config).state_dict().values())
    assert f.weights_bytes == params + r.inputs.weights.gpu["rope"]
    # KV price is the pool family's own kv_cost
    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(config)
    assert f.kv_bytes_per_token * config.page_size == cache_per_page == KV_PER_TOKEN * config.page_size
    # page count is the engine's startup solve over the forecast resident bytes
    free = r.gpu.free_before
    resident = f.weights_bytes + f.load_overhead
    expected = MHAKVCache.plan_num_pages(config, _startup_kv_budget(config.memory_ratio, free, free - resident))
    assert f.num_pages == expected
    assert f.kv_tokens == expected * config.page_size
    assert f.max_context == min(config.max_seq_len, f.kv_tokens)
    assert f.verdict == "fits", f.reasons


def _meta_model(config):
    from freetoken.models import create_model
    from freetoken.utils import torch_dtype

    with torch.device("meta"), torch_dtype(config.dtype):
        return create_model(config.model_config)


def test_num_pages_override_is_priced(tiny_qwen3):
    r = _analyze(tiny_qwen3, "--num-pages", "300")
    assert r.forecast.num_pages == 300
    assert r.forecast.kv_bytes == 300 * r.forecast.kv_bytes_per_token


def test_hopeless_config_does_not_fit_and_tips_recover(tiny_qwen3):
    # the budget cannot hold the load overhead estimate on top of the weights, but the weights
    # alone fit: tight, never refused on an estimate
    r = _analyze(tiny_qwen3, free_gib=0.65)
    assert r.forecast.verdict == "tight"
    assert "load overhead" in r.forecast.reasons[0]
    # a --num-pages request larger than the whole card is refused
    r = _analyze(tiny_qwen3, "--num-pages", str(10 * GiB // (KV_PER_TOKEN)), free_gib=1.0)
    assert r.forecast.verdict == "does not fit"
    assert "--num-pages" in r.forecast.reasons[0]


def test_tips_price_flag_changes(tiny_qwen3):
    r = _analyze(tiny_qwen3, "--memory-ratio", "0.8", free_gib=32)
    assert r.forecast.verdict == "fits"
    tip = next(t for t in r.tips if t.flags == ["--memory-ratio 0.95"])
    assert tip.forecast.kv_tokens > r.forecast.kv_tokens
    assert tip.forecast.memory_ratio == 0.95
    # the base config is left untouched by the what-if evaluation
    assert r.config.memory_ratio == 0.8


def test_tight_headroom_suggests_a_lower_memory_ratio(tiny_qwen3):
    r = _analyze(tiny_qwen3, "--memory-ratio", "0.97", free_gib=4)
    assert r.forecast.verdict == "tight"
    lower = [t for t in r.tips if t.flags[0] in ("--memory-ratio 0.95", "--memory-ratio 0.9")]
    assert lower and all(t.forecast.free_at_peak > r.forecast.free_at_peak for t in lower)
    assert not any(t.flags[0] == "--memory-ratio 0.99" for t in r.tips)


# ---------------------------------------------------------------------------------------------
# Preflight (Engine.__init__ calls run_preflight between the meta build and the weight load)
# ---------------------------------------------------------------------------------------------


def _resolved(model_path, *flags):
    from freetoken.distributed import set_tp_info
    from freetoken.engine.engine import _adjust_config
    from freetoken.layers import set_rope_device
    from freetoken.server.args import parse_args

    from freetoken.distributed import try_get_tp_info

    config, _ = parse_args(["--model", model_path, *flags])
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    _adjust_config(config)
    set_rope_device(torch.device("cpu"))
    return config, _meta_model(config)


def test_preflight_refuses_hopeless_config_fast(tiny_qwen3):
    config, model = _resolved(tiny_qwen3)
    start = time.monotonic()
    with pytest.raises(PreflightRefused, match="--skip-preflight"):
        fc_mod.run_preflight(config, model, free_before=MiB)
    assert time.monotonic() - start < 10


def test_preflight_lets_tight_config_through_with_warning(tiny_qwen3, monkeypatch):
    config, model = _resolved(tiny_qwen3)
    warnings = []
    monkeypatch.setattr(fc_mod.logger, "warning_rank0", lambda msg, *a, **k: warnings.append(msg))
    # weights fit, the load overhead estimate does not: the engine might still start
    weights = fc_mod.resident_weights(model, config).gpu_total
    fc = fc_mod.run_preflight(config, model, free_before=int((weights + 64 * MiB) / config.memory_ratio))
    assert fc is not None and fc.verdict == "tight"
    assert warnings and "TIGHT" in warnings[0]


def test_preflight_never_fails_on_its_own_errors(tiny_qwen3, monkeypatch):
    config, model = _resolved(tiny_qwen3)
    monkeypatch.setattr(fc_mod, "forecast_inputs", lambda *a: 1 / 0)
    assert fc_mod.run_preflight(config, model, free_before=GiB) is None


# ---------------------------------------------------------------------------------------------
# Hybrid GDN: the sibling state pool and the tips that shrink it
# ---------------------------------------------------------------------------------------------

TINY_QWEN3_5 = {
    "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text", "hidden_size": 128,
    "intermediate_size": 256, "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 64, "vocab_size": 256, "max_position_embeddings": 32768, "rms_norm_eps": 1e-6,
    "hidden_act": "silu", "tie_word_embeddings": False, "dtype": "bfloat16",
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "full_attention_interval": 4, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
    "linear_num_key_heads": 2, "linear_num_value_heads": 16, "linear_value_head_dim": 128,
    "attn_output_gate": True, "partial_rotary_factor": 0.25,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default", "partial_rotary_factor": 0.25},
}


@pytest.fixture
def tiny_hybrid(tmp_path):
    return write_checkpoint(tmp_path / "tiny-qwen3_5", TINY_QWEN3_5,
                            {"model.embed_tokens.weight": ("BF16", [256, 128])})


def test_hybrid_state_pool_is_the_engines(tiny_hybrid):
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots, state_pool_bytes

    r = _analyze(tiny_hybrid)
    config, f = r.config, r.forecast
    assert config.cache_type == "hybrid_radix"
    assert f.state_slots == _linear_pool_num_slots(config) == 4 * 4 + 8 + 1
    assert f.state_bytes == state_pool_bytes(config)
    # only the one full-attention layer holds paged KV
    assert f.kv_bytes_per_token == 2 * 1 * 64 * 2 * 1


def test_hybrid_no_fit_suggests_a_smaller_state_pool(tiny_hybrid):
    from freetoken.engine.forecast import forecast_inputs, suggest_tips

    config, model = _resolved(tiny_hybrid)
    inp = forecast_inputs(config, model, None)
    base = inp.forecast()
    # the weights fit, the 25-slot state pool does not
    inp.free_before = int((inp.weights.gpu_total + base.state_bytes - 4 * MiB) / config.memory_ratio)
    base = inp.forecast()
    assert base.verdict == "does not fit"
    tips, combo = suggest_tips(inp, base)
    by_flag = {t.flags[0]: t for t in tips}
    assert by_flag["--max-running-requests 1"].forecast.state_slots == 4 + 4 + 1
    assert by_flag["--cache-type naive"].forecast.state_slots == 4 + 1
    assert combo is not None and combo.forecast.verdict != "does not fit"
    assert config.max_running_req == 4 and config.cache_type == "hybrid_radix"


# ---------------------------------------------------------------------------------------------
# MoE offload: the expert slot cache comes from the engine's --moe-cache-auto plan
# ---------------------------------------------------------------------------------------------

TINY_QWEN3_MOE = {
    "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "hidden_size": 128,
    "intermediate_size": 256, "moe_intermediate_size": 64, "num_experts": 8, "num_experts_per_tok": 2,
    "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
    "vocab_size": 256, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "rope_theta": 10000,
    "hidden_act": "silu", "tie_word_embeddings": False, "torch_dtype": "bfloat16", "norm_topk_prob": True,
    "decoder_sparse_step": 1, "mlp_only_layers": [],
}


def test_moe_offload_slots_follow_the_auto_plan(tmp_path):
    from freetoken.engine.engine import plan_moe_cache_auto
    from freetoken.kvcache import resolve_pool_class

    path = write_checkpoint(tmp_path / "tiny-moe", TINY_QWEN3_MOE,
                            {"model.layers.0.mlp.experts.0.gate_proj.weight": ("BF16", [64, 128])})
    r = _analyze(path)
    config, f, inp = r.config, r.forecast, r.inputs
    assert config.moe_strategy == "offload" and config.moe_cache_auto
    # bf16 experts: gate+up+down rows of one expert
    assert f.per_expert_bytes == 3 * 64 * 128 * 2
    assert "experts" not in inp.weights.gpu and inp.weights.host["experts"] == 2 * 8 * f.per_expert_bytes
    size, pages, _ = plan_moe_cache_auto(
        config, resolve_pool_class(config.model_config), baseline_free=r.gpu.free_before,
        weights_bytes=f.weights_bytes + f.load_overhead, per_expert_bytes=f.per_expert_bytes,
        max_slots=inp.max_slots,
    )
    assert (f.moe_slots, f.num_pages) == (size, pages)


def test_free_memory_is_read_before_this_process_creates_a_cuda_context(tiny_qwen3, monkeypatch):
    """The arch probe in _adjust_config initializes CUDA on a GPU machine; a reading taken after
    it would count this process's own context twice."""
    from freetoken.engine import engine
    from freetoken.server import info

    calls = []
    real_gpu, real_adjust = info.gpu_info, engine._adjust_config
    monkeypatch.setattr(info, "gpu_info", lambda *a, **k: calls.append("gpu_info") or real_gpu(*a, **k))
    monkeypatch.setattr(engine, "_adjust_config", lambda *a, **k: calls.append("adjust") or real_adjust(*a, **k))
    _analyze(tiny_qwen3)
    assert calls[:2] == ["gpu_info", "adjust"]
