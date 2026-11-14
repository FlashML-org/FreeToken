"""DeepSeek-V4.1-Flash from a llama.cpp ``deepseek41`` GGUF (unsloth ``DeepSeek-V4.1-Flash-1017-GGUF``): config from the GGUF metadata, tensor names, the dense state dict and the Engram table.

The converter (conversion/deepseek41.py) keeps the geometry of the HF checkpoint and changes three things the loader undoes:
the Engram gate is stored as the product ``q_weight * k_weight`` (the model only ever uses the product), all Engram tables sit back to back in one
``per_layer_token_embd`` tensor, and the hash constants and the compressed token map are written as KVs (the runtime has no HF tokenizer to derive the map from).
The DSpark stage and the vision tower are separate files that this reader does not serve; ``exp_probs_b_vl`` in the text file stays unread with them.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.layers.quantization.ggml_types import row_bytes
from freetoken.models.gguf.config import routed_expert_count, uniform_swiglu_clamp
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index
from freetoken.models.gguf.tensors import check_scales, dequantized_tensor, fused, matrix

from .args import DeepseekV41Args, Mode
from .config import model_config_from_args
from .engram import EngramHash
from .engram_table import EngramRowSource

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

_TABLE = "per_layer_token_embd.weight"

# engine module leaf (after "model.layers.N.") -> GGUF tensor names (without ".weight")
_LAYER_LEAVES: dict[str, tuple[str, ...]] = {
    "attn.wq_a": ("attn_q_a",),
    "attn.wq_b": ("attn_q_b",),
    "attn.wkv": ("attn_kv",),
    "attn.wo_b": ("attn_output_b",),
    "attn.compressor.wkv": ("attn_compressor_kv",),
    "attn.indexer.wk": ("indexer.attn_k",),
    "attn.indexer.wq_b": ("indexer.attn_q_b",),
    "attn.indexer.weights_proj": ("indexer.proj",),
    "engram.wkv": ("engram_kv",),
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
    "head": ("output",),
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


def _args_from_metadata(shim: GgufConfigShim, index: dict[str, GgufTensorInfo]) -> DeepseekV41Args:
    md, prefix, path = shim.metadata, shim.model_type, shim.model_path

    def kv(key: str, default=None):
        return md.get(f"{prefix}.{key}", default)

    def need(key: str):
        val = kv(key)
        if val is None:
            raise KeyError(f"{path}: missing GGUF metadata key {prefix}.{key}")
        return val

    from freetoken.third_party.gguf.constants import ExpertGatingFuncType

    if int(need("expert_gating_func")) != int(ExpertGatingFuncType.SQRTSOFTPLUS):
        raise NotImplementedError(f"{path}: expert_gating_func other than sqrtsoftplus is not served")
    if int(kv("hash_layer_count", 0) or 0):
        raise NotImplementedError(f"{path}: hash-routed layers are not served on DeepSeek-V4.1")
    if need("rope.scaling.type") != "yarn":
        raise NotImplementedError(f"{path}: rope scaling other than yarn is not served")
    head_dim = int(need("attention.key_length"))
    if int(need("attention.value_length")) != head_dim:
        raise NotImplementedError(f"{path}: attention key and value lengths differ")
    if int(kv("attention.head_count_kv", 1) or 1) != 1:
        raise NotImplementedError(f"{path}: more than one latent KV head is not served")

    engram: dict = {}
    if (layers := kv("ple.layers")) is not None:
        ngram, heads = int(need("ple.ngram_size")), int(need("ple.heads_per_ngram"))
        cols = (ngram - 1) * heads
        sizes = [int(s) for s in need("ple.head_vocab_sizes")]
        engram = dict(
            engram_layer_ids=[int(i) for i in layers],
            engram_num_embeddings=[sum(sizes[i * cols : (i + 1) * cols]) for i in range(len(layers))],
            engram_max_ngram_size=ngram,
            # llama.cpp reads neither engram.vocab_size nor engram.compressed_vocab_size and some files leave them out;
            # the primes draw upward from one below this value, and the token map's ids are dense
            engram_vocab_size=sizes[0],
            engram_n_heads=heads,
            engram_head_dim=int(need("embedding_length_per_layer_input")),
            engram_pad_id=int(need("engram.pad_token_id")),
            engram_compressed_vocab_size=max(need("engram.token_map")) + 1,
        )
    return DeepseekV41Args(
        vocab_size=int(shim.vocab_size),
        dim=int(need("embedding_length")),
        moe_inter_dim=int(need("expert_feed_forward_length")),
        n_layers=int(need("block_count")),
        n_mtp_layers=0,  # the converter leaves the DSpark stage out
        n_heads=int(need("attention.head_count")),
        n_routed_experts=routed_expert_count(path, need("expert_count"), index),
        n_shared_experts=int(kv("expert_shared_count", 1) or 1),
        n_activated_experts=int(need("expert_used_count")),
        score_func="sqrtsoftplus",
        norm_topk_prob=bool(need("expert_weights_norm")),
        route_scale=float(need("expert_weights_scale")),
        swiglu_limit=uniform_swiglu_clamp(path, need("swiglu_clamp_exp"), need("swiglu_clamp_shexp")),
        q_lora_rank=int(need("attention.q_lora_rank")),
        head_dim=head_dim,
        rope_head_dim=int(need("rope.dimension_count")),
        norm_eps=float(need("attention.layer_norm_rms_epsilon")),
        o_groups=int(need("attention.output_group_count")),
        o_lora_rank=int(need("attention.output_lora_rank")),
        window_size=int(need("attention.sliding_window")),
        compress_ratios=need("attention.compress_ratios"),
        kv_source_layers=need("attention.kv_source_layers"),
        index_source_layers=need("attention.indexer.source_layers"),
        compress_rope_theta=float(need("attention.compress_rope_freq_base")),
        original_seq_len=int(need("rope.scaling.original_context_length")),
        rope_theta=float(need("rope.freq_base")),
        rope_factor=float(need("rope.scaling.factor")),
        beta_fast=int(need("rope.scaling.yarn_beta_fast")),
        beta_slow=int(need("rope.scaling.yarn_beta_slow")),
        index_n_heads=int(need("attention.indexer.head_count")),
        index_head_dim=int(need("attention.indexer.key_length")),
        index_topk=int(need("attention.indexer.top_k")),
        candidate_source_layer=int(kv("attention.indexer.candidate_source_layer", -1)),
        candidate_topk_blocks=int(kv("attention.indexer.candidate_top_k_blocks", 0)),
        candidate_block_size=int(kv("attention.indexer.candidate_block_size", 0)),
        hc_mult=int(need("hyper_connection.count")),
        hc_sinkhorn_iters=int(need("hyper_connection.sinkhorn_iterations")),
        hc_eps=float(need("hyper_connection.epsilon")),
        **engram,
    )


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    from dataclasses import replace

    check_scales(shim.model_path)
    index = gguf_tensor_index(shim.model_path)
    args = _args_from_metadata(shim, index)
    if not args.engram_layer_ids and (f"{shim.model_type}.engram.token_map" in shim.metadata or any(".engram_" in name for name in index)):
        raise NotImplementedError(f"{shim.model_path}: Engram tensors without {shim.model_type}.ple.layers, so not llama.cpp's deepseek41 layout (converted for an earlier branch?)")
    if args.engram_layer_ids:
        _check_engram(shim, args, index)
    config = model_config_from_args(args, max_position=int(shim.metadata[f"{shim.model_type}.context_length"]))
    return replace(config, expert_quant="gguf")


def _check_engram(shim: GgufConfigShim, args: DeepseekV41Args, index: dict[str, GgufTensorInfo]) -> None:
    """Refuse a file whose Engram hash or table this build would read differently, before any weight loads.

    The bucket primes and the multipliers derive from the config as on the HF path, so they are checked against the KVs llama.cpp hashes with."""
    path, arch = shim.model_path, shim.model_type
    hash = _engram_hash(shim.metadata, arch, args)
    for key, derived in (
        ("ple.head_offsets", _table_offsets(hash).flatten().tolist()),
        ("ple.head_vocab_sizes", hash.primes.flatten().tolist()),
        ("ple.layer_multipliers", hash.multipliers.flatten().tolist()),
    ):
        if [int(v) for v in shim.metadata[f"{arch}.{key}"]] != derived:
            raise ValueError(f"{path}: {arch}.{key} differs from the Engram hash this build derives from the config")
    # a metadata-only GGUF (an FTW dir's source_metadata.gguf) has no tensor table to check
    if not index:
        return
    if _TABLE not in index:
        raise NotImplementedError(f"{path}: no {_TABLE} tensor, the Engram tables llama.cpp's converter writes")
    # check_scales lets an NVFP4 tensor's .scale through, and the table reader would decode without it
    if _TABLE.replace(".weight", ".scale") in index:
        raise NotImplementedError(f"{path}: an Engram table with a per-tensor .scale is not served")
    head_dim, rows = index[_TABLE].ne
    if head_dim != args.engram_head_dim or rows < sum(hash.rows_per_layer):
        raise ValueError(f"{path}: {_TABLE} is {index[_TABLE].ne}, expected {args.engram_head_dim} values wide and at least {sum(hash.rows_per_layer)} rows")


def _engram_hash(metadata: dict, arch: str, args: DeepseekV41Args) -> EngramHash:
    return EngramHash(args, metadata[f"{arch}.engram.token_map"], args.engram_compressed_vocab_size)


def _layer_starts(hash: EngramHash) -> list[int]:
    """Each Engram layer's first row in the one table: the layers sit back to back."""
    return [sum(hash.rows_per_layer[:i]) for i in range(len(hash.rows_per_layer))]


def _table_offsets(hash: EngramHash) -> torch.Tensor:
    return hash.offsets + torch.tensor(_layer_starts(hash)).unsqueeze(1)


# ----------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The dense state dict from the GGUF, in the names the HF loader yields; the routed experts load as offload banks (``freetoken.models.gguf.experts``) and the Engram table streams from disk."""
    if include_moe_experts:
        raise ValueError("DeepSeek-V4.1 routed experts are served from the offload cache; run with --moe-strategy offload (include_moe_experts must be False).")
    if not include_non_moe:
        return
    from freetoken.utils import cached_load_hf_config

    config = parse_gguf_config(cached_load_hf_config(model_path))
    args: DeepseekV41Args = config.dsv41_args
    index = gguf_tensor_index(model_path)

    def t(name: str) -> GgufTensorInfo:
        return index[name if name in index else f"{name}.weight"]

    def bf16(name: str) -> torch.Tensor:
        return dequantized_tensor(t(name)).to(torch.bfloat16)

    def linear(dst: str, src: str, k: int) -> Iterator[tuple[str, torch.Tensor]]:
        info = t(src)
        yield from fused(dst, [info], [matrix(info)], k)

    yield from linear("model.embed", "token_embd", args.dim)
    yield from linear("head", "output", args.dim)
    yield "model.norm.weight", bf16("output_norm")
    for layer in range(args.n_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        m = f"{pre}.attn"
        role = args.roles[layer]
        yield f"{pre}.attn_norm.weight", bf16(f"{blk}.attn_norm")
        yield f"{pre}.ffn_norm.weight", bf16(f"{blk}.ffn_norm")
        for sub in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                yield f"{pre}.hc_{sub}_{part}", dequantized_tensor(t(f"{blk}.hc_{sub}_{part}"))

        yield from linear(f"{m}.wq_a", f"{blk}.attn_q_a", args.dim)
        yield f"{m}.q_norm.weight", bf16(f"{blk}.attn_q_a_norm")
        yield from linear(f"{m}.wq_b", f"{blk}.attn_q_b", args.q_lora_rank)
        yield from linear(f"{m}.wkv", f"{blk}.attn_kv", args.dim)
        yield f"{m}.kv_norm.weight", bf16(f"{blk}.attn_kv_a_norm")
        # wo_a feeds a grouped bf16 einsum, not the quant method
        yield f"{m}.wo_a", bf16(f"{blk}.attn_output_a")
        yield from linear(f"{m}.wo_b", f"{blk}.attn_output_b", args.o_groups * args.o_lora_rank)
        yield f"{m}.attn_sink", dequantized_tensor(t(f"{blk}.attn_sinks"))
        if role.mode is Mode.FULL:
            c = f"{blk}.attn_compressor"
            yield f"{m}.compressor.norm.weight", bf16(f"{c}_norm")
            if role.ratio > 1:
                # plain fp32 parameters behind a softmax pooling, not the quant method
                yield f"{m}.compressor.wkv", dequantized_tensor(t(f"{c}_kv"))
                yield f"{m}.compressor.wgate", dequantized_tensor(t(f"{c}_gate"))
            else:
                yield from linear(f"{m}.compressor.wkv", f"{c}_kv", args.dim)
            yield from linear(f"{m}.indexer.wk", f"{blk}.indexer.attn_k", args.head_dim)
            yield f"{m}.indexer.k_norm.weight", bf16(f"{blk}.indexer.k_norm")
        if role.has_indexer:
            yield from linear(f"{m}.indexer.wq_b", f"{blk}.indexer.attn_q_b", args.q_lora_rank)
            yield from linear(f"{m}.indexer.weights_proj", f"{blk}.indexer.proj", args.dim)

        g = f"{pre}.ffn.gate"
        yield f"{g}.weight", bf16(f"{blk}.ffn_gate_inp")
        yield f"{g}.bias", dequantized_tensor(t(f"{blk}.exp_probs_b.bias"))
        yield from linear(f"{pre}.ffn.shared_experts.w1", f"{blk}.ffn_gate_shexp", args.dim)
        yield from linear(f"{pre}.ffn.shared_experts.w3", f"{blk}.ffn_up_shexp", args.dim)
        yield from linear(f"{pre}.ffn.shared_experts.w2", f"{blk}.ffn_down_shexp", args.moe_inter_dim)
        if layer in args.engram_layer_ids:
            e = f"{pre}.engram"
            yield from linear(f"{e}.wkv", f"{blk}.engram_kv", args.engram_hash_cols * args.engram_head_dim)
            # the file holds the product q_weight * k_weight; the layer multiplies them back
            gate = dequantized_tensor(t(f"{blk}.engram_gate")).float()
            yield f"{e}.q_weight", gate
            yield f"{e}.k_weight", torch.ones_like(gate)


# ----------------------------------------------------------------------------------------------
# Engram
# ----------------------------------------------------------------------------------------------


def gguf_engram_tables(model_path: str, args: DeepseekV41Args) -> tuple[EngramHash, list[EngramRowSource]]:
    """The hash from the file's token map and one row source per Engram layer: each is a slice of the single ``per_layer_token_embd`` tensor.

    ``args`` comes from ``parse_gguf_config``, which has checked the hash constants and the table against the file."""
    from freetoken.models.gguf.reader import load_gguf_metadata

    md = load_gguf_metadata(model_path)
    hash = _engram_hash(md, md["general.architecture"], args)
    info = gguf_tensor_index(model_path)[_TABLE]
    nbytes = row_bytes(info.ggml_type, args.engram_head_dim)
    return hash, [
        EngramRowSource(info.path, info.offset + start * nbytes, layer_rows, args.engram_head_dim, None, info.ggml_type, nbytes)
        for start, layer_rows in zip(_layer_starts(hash), hash.rows_per_layer)
    ]


__all__ = ["gguf_engram_tables", "iter_gguf_weights", "parse_gguf_config", "to_gguf"]
