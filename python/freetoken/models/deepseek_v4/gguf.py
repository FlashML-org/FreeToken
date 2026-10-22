"""DeepSeek-V4-Flash from a llama.cpp ``deepseek4`` GGUF (unsloth ``DeepSeek-V4-Flash-0731-GGUF``,
text only): config from the GGUF metadata, tensor names, and the dense state dict.

The GGUF geometry matches the HF checkpoint (43 MLA blocks with sliding-window sparse
attention, per-layer KV compressors and a Lightning Indexer on the ratio-4 blocks, mHC x4
residual streams, a 256-expert sqrtsoftplus MoE with hash routing on the first blocks and
a shared expert), so ``parse_gguf_config`` builds the family's own :class:`DeepseekV4Args`
from the GGUF KV metadata and reuses the shared ModelConfig construction
(:func:`deepseek_v4.config.model_config_from_args`).

The converter drops the MTP blocks and writes ``attention.compress_ratios`` as a single
placeholder 0, so the per-layer compressor layout is recovered from the tensor table: an
``indexer.*`` tensor means ratio 4, a compressor without an indexer means 128, unless its
``wkv`` carries the CSA overlap's double rows (also ratio 4). Routed experts serve from
the GGUF expert banks (``freetoken.models.gguf.experts``) like every GGUF family.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch

from freetoken.layers.quantization.ggml_types import is_quantized
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index, gguf_tensor_names
from freetoken.models.gguf.tensors import check_scales, fused, matrix
from freetoken.utils import init_logger

from .args import DeepseekV4Args
from .config import model_config_from_args

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

logger = init_logger(__name__)

# engine module leaf (after "model.layers.N.") -> GGUF tensor names (without ".weight")
_LAYER_LEAVES: dict[str, tuple[str, ...]] = {
    "attn.wq_a": ("attn_q_a",),
    "attn.wq_b": ("attn_q_b",),
    "attn.wkv": ("attn_kv",),
    "attn.wo_b": ("attn_output_b",),
    "attn.compressor.wkv": ("attn_compressor_kv",),
    "attn.compressor.wgate": ("attn_compressor_gate",),
    "attn.indexer.wq_b": ("indexer.attn_q_b",),
    "attn.indexer.weights_proj": ("indexer.proj",),
    "attn.indexer.compressor.wkv": ("indexer_compressor_kv",),
    "attn.indexer.compressor.wgate": ("indexer_compressor_gate",),
    "ffn.shared_experts.w1": ("ffn_gate_shexp",),
    "ffn.shared_experts.w3": ("ffn_up_shexp",),
    "ffn.shared_experts.w2": ("ffn_down_shexp",),
    # routed experts (GGUF expert banks), gate / up / down order
    "ffn.experts.0.w1": ("ffn_gate_exps",),
    "ffn.experts.0.w3": ("ffn_up_exps",),
    "ffn.experts.0.w2": ("ffn_down_exps",),
}
_TOP_LEAVES: dict[str, tuple[str, ...]] = {
    "model.embed": ("token_embd",),
    "model.head": ("output",),
}
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Engine module name -> the GGUF tensors (without ``.weight``) that store it; empty for modules the GGUF has no matrix for."""
    if name in _TOP_LEAVES:
        return _TOP_LEAVES[name]
    m = _LAYER_RE.match(name)
    if m is None:
        return ()
    return tuple(f"blk.{m.group(1)}.{leaf}" for leaf in _LAYER_LEAVES.get(m.group(2), ()))


def _compress_ratios(index: dict[str, GgufTensorInfo], head_dim: int, num_layers: int) -> tuple[int, ...]:
    """Per-layer compressor ratios from the tensor table, which every released file's
    ``attention.compress_ratios`` array (43 trunk entries ahead of 3 MTP zeros, which
    llama.cpp requires) confirms element for element.

    An ``indexer.proj`` tensor exists exactly on the ratio-4 blocks; a compressor without an
    indexer is a 128 block, unless its ``wkv`` has the CSA overlap's double rows (ratio 4)."""
    ratios = []
    for layer in range(num_layers):
        blk = f"blk.{layer}"
        if f"{blk}.indexer.proj.weight" in index:
            ratios.append(4)
            continue
        info = index.get(f"{blk}.attn_compressor_kv.weight")
        ratios.append(0 if info is None else 4 if info.ne[-1] == 2 * head_dim else 128)
    return tuple(ratios)


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    md = shim.metadata
    prefix = shim.model_type

    check_scales(shim.model_path)

    def kv(key: str, default=None):
        return md.get(f"{prefix}.{key}", default)

    def need(key: str):
        val = kv(key)
        if val is None:
            raise KeyError(f"missing GGUF metadata key {prefix}.{key}")
        return val

    names = gguf_tensor_names(shim.model_path)
    if any(n.startswith(("vision.", "aligner.")) for n in names):
        raise NotImplementedError(
            f"{shim.model_path}: deepseek4 GGUF with a vision tower is not served (the text-only 0731 release has none)"
        )

    # the clamp is per-layer in the GGUF (and the shared expert carries its own array); the model
    # takes one value, so anything non-uniform is refused instead of silently taking the first
    clamp = need("swiglu_clamp_exp")
    clamp_vals = list(clamp) if isinstance(clamp, list) else [clamp]
    clamp_shexp = kv("swiglu_clamp_shexp")
    shexp_vals = list(clamp_shexp) if isinstance(clamp_shexp, list) else [clamp_shexp] if clamp_shexp is not None else []
    if len(set(map(float, clamp_vals + shexp_vals))) != 1:
        raise NotImplementedError(f"{shim.model_path}: non-uniform swiglu_clamp_exp / swiglu_clamp_shexp is not served")

    from freetoken.third_party.gguf.constants import ExpertGatingFuncType

    if int(need("expert_gating_func")) != int(ExpertGatingFuncType.SQRTSOFTPLUS):
        raise NotImplementedError(f"{shim.model_path}: expert_gating_func other than sqrtsoftplus is not served")

    # every official DeepSeek-V4 config carries the same yarn block (V4-Flash, 0731, Pro,
    # Pro-0813: factor 16 over 65536, betas 32 / 1). The no-yarn files on HF lost the whole
    # block in conversion -- llama.cpp serves them unscaled, which is wrong for them -- so a
    # wholly missing block falls back to the family values and a partly present one refuses.
    scaling = {k: kv(f"rope.scaling.{k}") for k in
               ("type", "factor", "original_context_length", "yarn_beta_fast", "yarn_beta_slow")}
    if scaling["type"] not in (None, "yarn"):
        raise NotImplementedError(f"{shim.model_path}: rope scaling other than yarn is not served")
    missing = [k for k, v in scaling.items() if v is None]
    if missing:
        if len(missing) != len(scaling):
            raise ValueError(
                f"{shim.model_path}: rope.scaling is only partly present "
                f"(missing {', '.join(missing)}); the file is inconsistent"
            )
        logger.warning(
            f"{shim.model_path}: no rope.scaling block; using the family values "
            "(yarn, factor 16, original 65536, betas 32 / 1)"
        )
        scaling = {"type": "yarn", "factor": 16.0, "original_context_length": 65536,
                   "yarn_beta_fast": 32.0, "yarn_beta_slow": 1.0}

    head_dim = int(need("attention.key_length"))
    if int(need("attention.value_length")) != head_dim:
        raise NotImplementedError(f"{shim.model_path}: attention key and value lengths differ")
    if int(kv("attention.head_count_kv", 1) or 1) != 1:
        raise NotImplementedError(f"{shim.model_path}: more than one latent KV head is not served")

    num_layers = int(need("block_count"))
    index = gguf_tensor_index(shim.model_path)

    # REAP exports prune experts per layer: as a per-layer expert_count array, or with the
    # scalar kept but the expert tensors shrunk. llama.cpp builds every layer from one
    # n_expert and refuses both shapes; the model has no per-layer expert count to express.
    experts_kv = need("expert_count")
    if isinstance(experts_kv, list):
        raise NotImplementedError(
            f"{shim.model_path}: a per-layer expert_count array (REAP-style prune) is not served"
        )
    n_experts = int(experts_kv)
    for name, info in index.items():
        if name.endswith(".ffn_gate_exps.weight") and info.ne[-1] != n_experts:
            raise NotImplementedError(
                f"{shim.model_path}: {name} carries {info.ne[-1]} experts but expert_count says "
                f"{n_experts} (REAP-style per-layer prunes are not served)"
            )
    args = DeepseekV4Args(
        max_batch_size=1,
        vocab_size=int(shim.vocab_size),
        dim=int(need("embedding_length")),
        moe_inter_dim=int(need("expert_feed_forward_length")),
        n_layers=num_layers,
        n_hash_layers=int(kv("hash_layer_count", 0) or 0),
        n_heads=int(need("attention.head_count")),
        n_routed_experts=int(need("expert_count")),
        n_shared_experts=int(kv("expert_shared_count", 1) or 1),
        n_activated_experts=int(need("expert_used_count")),
        score_func="sqrtsoftplus",
        route_scale=float(need("expert_weights_scale")),
        swiglu_limit=float(clamp_vals[0]),
        q_lora_rank=int(need("attention.q_lora_rank")),
        head_dim=head_dim,
        rope_head_dim=int(need("rope.dimension_count")),
        norm_eps=float(need("attention.layer_norm_rms_epsilon")),
        o_groups=int(need("attention.output_group_count")),
        o_lora_rank=int(need("attention.output_lora_rank")),
        window_size=int(need("attention.sliding_window")),
        compress_ratios=_compress_ratios(index, head_dim, num_layers),
        compress_rope_theta=float(kv("attention.compress_rope_freq_base", 160000.0) or 160000.0),
        original_seq_len=int(scaling["original_context_length"]),
        rope_theta=float(need("rope.freq_base")),
        rope_factor=float(scaling["factor"]),
        beta_fast=float(scaling["yarn_beta_fast"]),
        beta_slow=float(scaling["yarn_beta_slow"]),
        index_n_heads=int(need("attention.indexer.head_count")),
        index_head_dim=int(need("attention.indexer.key_length")),
        index_topk=int(need("attention.indexer.top_k")),
        hc_mult=int(need("hyper_connection.count")),
        hc_sinkhorn_iters=int(need("hyper_connection.sinkhorn_iterations")),
        hc_eps=float(need("hyper_connection.epsilon")),
        n_mtp_layers=0,  # the converter drops the MTP blocks
        vision_n_layers=0,
    )
    config = model_config_from_args(args, max_position=int(need("context_length")))
    return replace(config, expert_quant="gguf")


# ----------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------

def _float(info: GgufTensorInfo) -> torch.Tensor:
    """A float tensor in torch order, dequantizing packed ggml rows: dynamic quants quantize
    tensors the module consumes as floats (the mHC mixers, the compressor gates, the router).
    Trailing unit dims in ``ne`` are dropped -- some writers ship tensors 4-D (llama.cpp's
    loaders try that shape first) and a trailing 1 carries no elements."""
    ne = info.ne
    while len(ne) > 1 and ne[-1] == 1:
        ne = ne[:-1]
    if not is_quantized(info.ggml_type):
        from freetoken.models.gguf.tensors import float_tensor

        return float_tensor(info).view(*reversed(ne))
    from freetoken.third_party.gguf import quants
    from freetoken.third_party.gguf.constants import GGMLQuantizationType

    flat = quants.dequantize(np.array(info.data()), GGMLQuantizationType(info.ggml_type))
    return torch.from_numpy(flat).view(*reversed(ne))


def _tid2eid(info: GgufTensorInfo) -> torch.Tensor:
    """The hash-route table as the module's int64 [vocab, top_k]."""
    ne = info.ne
    while len(ne) > 1 and ne[-1] == 1:
        ne = ne[:-1]
    raw = np.frombuffer(info.data(), dtype=np.int32).copy()
    return torch.from_numpy(raw).view(*reversed(ne)).to(torch.int64)


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The dense state dict from the GGUF, in the names the HF loader yields; the routed experts
    load as offload banks (``freetoken.models.gguf.experts``)."""
    if include_moe_experts:
        raise ValueError(
            "DeepSeek-V4 routed experts are served from the offload cache; "
            "run with --moe-strategy offload (include_moe_experts must be False)."
        )
    if not include_non_moe:
        return
    from freetoken.utils import cached_load_hf_config

    config = parse_gguf_config(cached_load_hf_config(model_path))
    args = config.dsv4_args
    index = gguf_tensor_index(model_path)
    hidden = config.hidden_size

    def t(name: str) -> GgufTensorInfo:
        return index[name if name in index else f"{name}.weight"]

    def bf16(name: str) -> torch.Tensor:
        return _float(t(name)).to(torch.bfloat16)

    yield from fused("model.embed", [t("token_embd")], [matrix(t("token_embd"))], hidden)
    yield from fused("model.head", [t("output")], [matrix(t("output"))], hidden)
    yield "model.norm.weight", bf16("output_norm")
    for part in ("fn", "base", "scale"):
        yield f"model.hc_head_{part}", _float(t(f"output_hc_{part}"))

    for layer in range(config.num_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        m = f"{pre}.attn"
        yield f"{pre}.attn_norm.weight", bf16(f"{blk}.attn_norm")
        yield f"{pre}.ffn_norm.weight", bf16(f"{blk}.ffn_norm")
        # mHC mixing tensors, fp32 on every layer
        for sub in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                yield f"{pre}.hc_{sub}_{part}", _float(t(f"{blk}.hc_{sub}_{part}"))

        for proj, src, k in (
            ("wq_a", "attn_q_a", hidden),
            ("wq_b", "attn_q_b", args.q_lora_rank),
            ("wkv", "attn_kv", hidden),
            ("wo_b", "attn_output_b", args.o_groups * args.o_lora_rank),
        ):
            info = t(f"{blk}.{src}")
            yield from fused(f"{m}.{proj}", [info], [matrix(info)], k)
        yield f"{m}.q_norm.weight", bf16(f"{blk}.attn_q_a_norm")
        yield f"{m}.kv_norm.weight", bf16(f"{blk}.attn_kv_a_norm")
        # wo_a feeds a bf16 einsum; the HF path dequantizes its fp8 the same way
        yield f"{m}.wo_a", _float(t(f"{blk}.attn_output_a")).to(torch.bfloat16)
        yield f"{m}.attn_sink", _float(t(f"{blk}.attn_sinks"))

        ratio = args.compress_ratios[layer]
        if ratio:
            compressors = ((f"{m}.compressor", f"{blk}.attn_compressor"),)
            if ratio == 4:
                compressors += ((f"{m}.indexer.compressor", f"{blk}.indexer_compressor"),)
            for c_dst, c_src in compressors:
                yield f"{c_dst}.ape", _float(t(f"{c_src}_ape"))
                # the compressors read wkv / wgate through a dense bf16 einsum, not the quant
                # method (the registry exempts them, like the HF spec), so a tensor the file
                # quantized dequantizes here
                for proj, suf in (("wkv", "kv"), ("wgate", "gate")):
                    yield f"{c_dst}.{proj}.weight", bf16(f"{c_src}_{suf}")
                yield f"{c_dst}.norm.weight", bf16(f"{c_src}_norm")
            if ratio == 4:
                for proj, src, k in (
                    ("indexer.wq_b", "indexer.attn_q_b", args.q_lora_rank),
                    ("indexer.weights_proj", "indexer.proj", hidden),
                ):
                    info = t(f"{blk}.{src}")
                    yield from fused(f"{m}.{proj}", [info], [matrix(info)], k)

        g = f"{pre}.ffn.gate"
        yield f"{g}.weight", bf16(f"{blk}.ffn_gate_inp")
        if layer < args.n_hash_layers:
            # a vision checkpoint also gives its hash layers a text bias; the text model has no slot for it
            yield f"{g}.tid2eid", _tid2eid(t(f"{blk}.ffn_gate_tid2eid"))
        else:
            yield f"{g}.bias", _float(t(f"{blk}.exp_probs_b.bias"))
        for proj, src in (("w1", "ffn_gate_shexp"), ("w3", "ffn_up_shexp"), ("w2", "ffn_down_shexp")):
            info = t(f"{blk}.{src}")
            k = hidden if proj != "w2" else args.moe_inter_dim
            yield from fused(f"{pre}.ffn.shared_experts.{proj}", [info], [matrix(info)], k)


__all__ = ["iter_gguf_weights", "parse_gguf_config", "to_gguf"]
