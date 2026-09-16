"""GLM-5.3-Flash tensor parallelism: the shard contract and the reader that feeds it.

Every split tensor must be exactly 1/tp of its TP=1 shape on one axis, and every other tensor must
keep its full shape: the MLA latent (q_a / kv_a), the DSA indexer, the KDA f_a|g_a bottleneck and
o_norm, norms, mHC and the router are replicated so every rank reads the same latent and picks the
same blocks. The reader is the only place the checkpoint is cut, so it is checked end to end: each
rank's tensors must match the rank model's declared shapes, and the ranks' slices must tile the
TP=1 tensors -- per part for the fused KDA projections, which are cut before they are fused.

CPU-only: shapes come from a meta-device build, weights from a tiny on-disk checkpoint.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

import freetoken.distributed.info as info_mod
from freetoken.distributed import DistributedInfo

HIDDEN, VOCAB, INTER = 64, 128, 96
HEADS, QLORA, KVLORA, NOPE, V = 4, 48, 32, 32, 32
KDA_H, KDA_D, KERNEL = 4, 128, 4
IDX_H, IDX_D, KPOOL = 16, 64, 4
P = KDA_H * KDA_D


def _text_config() -> dict:
    return {
        "hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": 2,
        "num_attention_heads": HEADS, "vocab_size": VOCAB, "hidden_act": "silu",
        "rms_norm_eps": 1e-5, "max_position_embeddings": 4096, "tie_word_embeddings": False,
        "q_lora_rank": QLORA, "kv_lora_rank": KVLORA, "qk_nope_head_dim": NOPE,
        "qk_rope_head_dim": 0, "v_head_dim": V, "mla_use_nope": True,
        "index_n_heads": IDX_H, "index_head_dim": IDX_D, "index_topk": 32,
        "indexer_types": ["full", "full"], "indexer_rope_interleave": True,
        "index_kpool": KPOOL, "index_kpool_compress": True, "index_kpool_always_select_tail": True,
        "linear_attn_config": {
            "num_heads": KDA_H, "head_dim": KDA_D, "short_conv_kernel_size": KERNEL, "gate_lower_bound": -5.0,
        },
        "layer_types": ["linear_attention", "deepseek_sparse_attention"],
        "mlp_layer_types": ["dense", "dense"],
        "first_k_dense_replace": 2,
        "mhc": True, "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
        "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 1,
        "moe_intermediate_size": 32, "norm_topk_prob": True, "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid", "n_group": 1, "topk_group": 1, "swiglu_limit": 10.0,
        "attention_bias": False, "model_type": "glm5_next_text",
    }


def _raw_config() -> dict:
    return {"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next", "text_config": _text_config()}


# model key suffix -> the axis a rank's slice is cut on; everything else is replicated
SPLIT = {
    "self_attn.q_b_proj.weight": 0, "self_attn.kv_b_proj.weight": 0, "self_attn.o_proj.weight": 1,
    "self_attn.in_proj.weight": 0, "self_attn.f_b_proj.weight": 0, "self_attn.g_b_proj.weight": 0,
    "self_attn.conv1d.weight": 0, "self_attn.A_log": 0, "self_attn.dt_bias": 0,
    "mlp.gate_proj.weight": 0, "mlp.up_proj.weight": 0, "mlp.down_proj.weight": 1,
    "embed_tokens.weight": 0, "lm_head.weight": 0,
}
# fused KDA tensors: (part sizes at TP=1) in fusion order
FUSED = {
    "model.layers.0.self_attn.in_proj.weight": (P, P, P, KDA_H),
    "model.layers.0.self_attn.conv1d.weight": (P, P, P),
}


def _set_tp(size: int, rank: int = 0) -> None:
    info_mod._TP_INFO = DistributedInfo(rank, size)


@pytest.fixture(autouse=True)
def _restore_tp():
    yield
    _set_tp(1)


def _config():
    from freetoken.models.glm5_next.config import parse_config
    from freetoken.utils.hf import RawConfigShim

    return parse_config(RawConfigShim(_raw_config()))


def _shapes(tp: int, rank: int = 0) -> dict[str, tuple[int, ...]]:
    from freetoken.models.glm5_next.model import Glm5NextForCausalLM

    _set_tp(tp, rank)
    with torch.device("meta"):
        model = Glm5NextForCausalLM(_config())
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def _split_axis(key: str) -> int | None:
    return next((axis for suffix, axis in SPLIT.items() if key.endswith(suffix)), None)


@pytest.mark.parametrize("tp", [2, 4])
def test_every_tensor_is_split_on_its_axis_or_replicated(tp):
    full, local = _shapes(1), _shapes(tp)
    assert local.keys() == full.keys()
    for key, shape in full.items():
        axis = _split_axis(key)
        want = list(shape)
        if axis is not None:
            want[axis] //= tp
        assert local[key] == tuple(want), f"{key}: TP={tp} shape {local[key]}, expected {tuple(want)}"


def _write_checkpoint(root: str) -> None:
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)
    full = _shapes(1)

    def rnd(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=g).to(dtype)

    ck = "model.language_model"
    t: dict[str, torch.Tensor] = {}
    for key, shape in full.items():
        if ".hc_" in key or key.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")) or ".mlp." in key:
            t[key.replace("model.", f"{ck}.", 1)] = rnd(*shape)
    a0 = f"{ck}.layers.0.self_attn"
    for name, shape in {
        "q_proj": (P, HIDDEN), "k_proj": (P, HIDDEN), "v_proj": (P, HIDDEN), "b_proj": (KDA_H, HIDDEN),
        "f_a_proj": (KDA_D, HIDDEN), "g_a_proj": (KDA_D, HIDDEN), "f_b_proj": (P, KDA_D), "g_b_proj": (P, KDA_D),
        "o_proj": (HIDDEN, P),
    }.items():
        t[f"{a0}.{name}.weight"] = rnd(*shape)
    for c in ("q", "k", "v"):
        t[f"{a0}.{c}_conv1d.weight"] = rnd(P, 1, KERNEL)
    t[f"{a0}.o_norm.weight"], t[f"{a0}.A_log"], t[f"{a0}.dt_bias"] = rnd(KDA_D), rnd(KDA_H), rnd(P)
    a1 = f"{ck}.layers.1.self_attn"
    for name, shape in {
        "q_a_proj": (QLORA, HIDDEN), "q_b_proj": (HEADS * NOPE, QLORA), "kv_a_proj_with_mqa": (KVLORA, HIDDEN),
        "kv_b_proj": (HEADS * (NOPE + V), KVLORA), "o_proj": (HIDDEN, HEADS * V),
        "indexer.wq_b": (IDX_H * IDX_D, QLORA), "indexer.wk": (IDX_D, HIDDEN), "indexer.weights_proj": (IDX_H, HIDDEN),
    }.items():
        t[f"{a1}.{name}.weight"] = rnd(*shape)
    t[f"{a1}.q_a_layernorm.weight"], t[f"{a1}.kv_a_layernorm.weight"] = rnd(QLORA), rnd(KVLORA)
    t[f"{a1}.indexer.k_norm.weight"], t[f"{a1}.indexer.k_norm.bias"] = rnd(IDX_D), rnd(IDX_D)
    t[f"{a1}.indexer.index_kpool_compress_gate"] = rnd(IDX_D, HIDDEN)
    t[f"{a1}.indexer.index_kpool_compress_ape"] = rnd(KPOOL, IDX_D)
    t[f"{ck}.embed_tokens.weight"], t[f"{ck}.norm.weight"], t["lm_head.weight"] = rnd(VOCAB, HIDDEN), rnd(HIDDEN), rnd(VOCAB, HIDDEN)

    save_file(t, os.path.join(root, "model-00001-of-00001.safetensors"))
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in t}}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(_raw_config(), f)


def _load(root: str, tp: int, rank: int) -> dict[str, torch.Tensor]:
    from freetoken.models.glm5_next.weight import iter_weights
    from freetoken.utils import hf

    _set_tp(tp, rank)
    hf.cached_load_hf_config.cache_clear() if hasattr(hf.cached_load_hf_config, "cache_clear") else None
    return dict(iter_weights(root, torch.device("cpu"), include_moe_experts=False, include_non_moe=True, include_vision=False))


@pytest.mark.parametrize("tp", [2, 4])
def test_reader_shards_match_the_rank_model_and_tile_the_full_tensors(tmp_path, tp):
    _write_checkpoint(str(tmp_path))
    full = _load(str(tmp_path), 1, 0)
    ranks = [_load(str(tmp_path), tp, r) for r in range(tp)]

    local_shapes = _shapes(tp)
    for r, weights in enumerate(ranks):
        assert weights.keys() == local_shapes.keys(), f"rank {r} yields a different key set than its model declares"
        for key, tensor in weights.items():
            assert tuple(tensor.shape) == local_shapes[key], f"rank {r} {key}: {tuple(tensor.shape)} vs {local_shapes[key]}"

    for key, whole in full.items():
        axis = _split_axis(key)
        if axis is None:
            assert all(torch.equal(w[key], whole) for w in ranks), f"{key} must be replicated"
            continue
        if key in FUSED:
            sizes = FUSED[key]
            local_sizes = [s // tp for s in sizes]
            parts = [torch.cat([w[key].split(local_sizes, dim=0)[i] for w in ranks], dim=0) for i in range(len(sizes))]
            assert torch.equal(torch.cat(parts, dim=0), whole), f"{key}: ranks do not tile each fused part"
        else:
            assert torch.equal(torch.cat([w[key] for w in ranks], dim=axis), whole), f"{key}: ranks do not tile axis {axis}"


@pytest.mark.parametrize("rank", range(4))
def test_nvfp4_expert_pieces_pack_into_the_ranks_bank(rank):
    """The Triton NVFP4 kernel's rank-local layout must hold exactly the reader's rank slices."""
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel
    from freetoken.models.nvfp4_banks import shard_nvfp4_piece

    tp, full_i, hidden = 4, 128, 64
    local_i, fp8 = full_i // tp, torch.float8_e4m3fn

    def payload(shape, offset, dtype=torch.uint8):
        raw = (torch.arange(torch.Size(shape).numel(), dtype=torch.int64) + offset) % 251
        return raw.to(torch.uint8).reshape(shape).view(dtype)

    src = {
        "gate": payload((full_i, hidden // 2), 1), "up": payload((full_i, hidden // 2), 17),
        "down": payload((hidden, full_i // 2), 33),
        "gate_scale": payload((full_i, hidden // 16), 49, fp8), "up_scale": payload((full_i, hidden // 16), 65, fp8),
        "down_scale": payload((hidden, full_i // 16), 81, fp8),
    }
    pieces = {role: shard_nvfp4_piece(role, t, rank=rank, tp_size=tp).unsqueeze(0) for role, t in src.items()}
    pieces.update({g: torch.full((1, 1), 0.5, dtype=torch.float16) for g in ("gate_global", "up_global", "down_global")})
    cfg = MoEConfig(num_experts=1, hidden=hidden, intermediate=full_i, top_k=2, tp_rank=rank, tp_size=tp, strategy="offload")
    kernel = TritonNvfp4MoEKernel()
    out = {role: torch.zeros((1, *spec.shape), dtype=spec.dtype) for role, spec in kernel.layout(cfg).items()}
    kernel.pack(pieces, cfg, out)

    lo = rank * local_i
    u8 = torch.uint8
    assert torch.equal(out["gate_up"][0, :local_i], src["gate"][lo:lo + local_i])
    assert torch.equal(out["gate_up"][0, local_i:], src["up"][lo:lo + local_i])
    assert torch.equal(out["down"][0], src["down"][:, lo // 2:(lo + local_i) // 2])
    assert torch.equal(out["gate_up_scale"][0, local_i:].view(u8), src["up_scale"][lo:lo + local_i].view(u8))
    assert torch.equal(out["down_scale"][0].view(u8), src["down_scale"][:, lo // 16:(lo + local_i) // 16].view(u8))
    assert out["gate_up_global"].shape == (1, 2 * local_i) and out["down_global"].shape == (1, hidden)
