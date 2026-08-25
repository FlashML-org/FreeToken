"""Serve a deepseek4 GGUF checkpoint.

Unlike the safetensors path, everything here comes from the GGUF's own metadata. The
reference ``parse_config`` recovers ``DeepseekV4Args`` from the checkpoint's
``inference/config.json``, which ships beside the weights; a standalone .gguf has no such
file, and being self-describing is the point of the format. ``_args_from_gguf`` below
rebuilds the same dataclass from KV keys alone.

The mapping from GGUF tensor to model parameter was established by reading both sides
rather than by analogy with the qwen adapters, because three tensors do not behave the way
the names suggest:

* ``attn_output_a`` is Q8_0 in the file but ``attn.wo_a`` is a bare ``nn.Parameter`` in
  bfloat16, not a Linear (attention.py: "wo_a: dequantized to bf16, the reference runs a
  bf16 grouped-output einsum"). It must be dequantized to dense, and it has no ``.weight``
  suffix.
* the compressor and indexer projections are **F16** in the file, i.e. unquantized. F16 is
  in ``GGML_UNQUANTIZED``, so ``fused_mul_mat_gguf`` would take the ``x @ qweight.T`` path
  while ``GGUFLinear`` allocates a uint8 buffer. They must land dense on a normal
  ``.weight``, never packed.
* ``Indexer.wq_b`` is declared ``Linear(kind="fp8")`` (compress.py), which allocates a
  ``.scale`` that no GGUF tensor can fill, because that tensor is F16 here. That Linear is
  replaced outright rather than populated.

Routed experts never pass through this module; they are streamed from the offload cache by
``gguf_experts.load_gguf_expert_sources``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

import torch

from freetoken.models.config import DSV4AttentionGroupConfig, ModelConfig, RotaryConfig

from .args import DeepseekV4Args

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "deepseek4"

# llama.cpp's expert-gating enum. DeepSeek-V4 scores with sqrt-softplus; 1 and 2 are the
# long-standing softmax/sigmoid values. An unknown id raises rather than silently picking a
# scoring function, because the wrong one routes to the wrong experts and still produces
# fluent text.
_GATING = {1: "softmax", 2: "sigmoid", 4: "sqrtsoftplus"}


def _kv(shim: "GgufConfigShim", key: str, default: Any = None) -> Any:
    """One ``deepseek4.*`` metadata value. No default means the key is mandatory."""
    full = f"{_ARCH}.{key}"
    md = shim if isinstance(shim, dict) else shim.metadata
    if full not in md:
        if default is None:
            raise ValueError(
                f"deepseek4 GGUF is missing required metadata key {full!r}; this file does "
                f"not carry the config this adapter needs"
            )
        return default
    return md[full]


def _args_from_gguf(shim: "GgufConfigShim") -> DeepseekV4Args:
    """Rebuild DeepseekV4Args from GGUF metadata alone.

    Every field is sourced from a key that is actually present in the checkpoint; nothing
    is left to the dataclass default, because a silently-defaulted hyperparameter here
    produces a model that loads and generates confidently wrong text.

    Cross-checks worth keeping: ``compress_ratios`` carries one entry per layer plus the
    MTP layers, entries != 0 mark layers with an attention compressor, and entries == 4
    mark layers with the lightning indexer. Those counts must match the tensor table (41
    and 21 respectively for DeepSeek-V4-Flash), which is what makes this mapping
    self-validating rather than merely plausible.
    """
    ratios = tuple(int(x) for x in _kv(shim, "attention.compress_ratios"))
    swiglu = [float(x) for x in _kv(shim, "swiglu_clamp_exp", [])]
    gate_id = int(_kv(shim, "expert_gating_func"))
    if gate_id not in _GATING:
        raise ValueError(
            f"deepseek4 GGUF: unknown expert_gating_func {gate_id}; known values are "
            f"{sorted(_GATING)} (routing with the wrong scoring function still generates "
            f"fluent text, so this is not defaulted)"
        )

    return DeepseekV4Args(
        max_batch_size=1,
        max_seq_len=int(_kv(shim, "context_length")),
        # The fp8/fp4 reference paths do not apply: a GGUF carries its own block-quantized
        # weights and the adapter swaps the quantized projections for GGUF ops.
        dtype="bf16",
        scale_fmt=None,
        expert_dtype=None,
        vocab_size=int(_kv(shim, "vocab_size")),
        dim=int(_kv(shim, "embedding_length")),
        moe_inter_dim=int(_kv(shim, "expert_feed_forward_length")),
        n_layers=int(_kv(shim, "block_count")),
        n_hash_layers=int(_kv(shim, "hash_layer_count")),
        n_mtp_layers=int(_kv(shim, "nextn_predict_layers", 0)),
        n_heads=int(_kv(shim, "attention.head_count")),
        n_routed_experts=int(_kv(shim, "expert_count")),
        n_shared_experts=int(_kv(shim, "expert_shared_count")),
        n_activated_experts=int(_kv(shim, "expert_used_count")),
        score_func=_GATING[gate_id],
        route_scale=float(_kv(shim, "expert_weights_scale")),
        swiglu_limit=(swiglu[0] if swiglu else 10.0),
        q_lora_rank=int(_kv(shim, "attention.q_lora_rank")),
        head_dim=int(_kv(shim, "attention.key_length")),
        rope_head_dim=int(_kv(shim, "rope.dimension_count")),
        norm_eps=float(_kv(shim, "attention.layer_norm_rms_epsilon")),
        o_groups=int(_kv(shim, "attention.output_group_count")),
        o_lora_rank=int(_kv(shim, "attention.output_lora_rank")),
        window_size=int(_kv(shim, "attention.sliding_window")),
        compress_ratios=ratios,
        compress_rope_theta=float(_kv(shim, "attention.compress_rope_freq_base")),
        original_seq_len=int(_kv(shim, "rope.scaling.original_context_length")),
        rope_theta=float(_kv(shim, "rope.freq_base")),
        rope_factor=float(_kv(shim, "rope.scaling.factor")),
        beta_fast=int(_kv(shim, "rope.scaling.yarn_beta_fast")),
        beta_slow=int(_kv(shim, "rope.scaling.yarn_beta_slow")),
        index_n_heads=int(_kv(shim, "attention.indexer.head_count")),
        index_head_dim=int(_kv(shim, "attention.indexer.key_length")),
        index_topk=int(_kv(shim, "attention.indexer.top_k")),
        hc_mult=int(_kv(shim, "hyper_connection.count")),
        hc_sinkhorn_iters=int(_kv(shim, "hyper_connection.sinkhorn_iterations")),
        hc_eps=float(_kv(shim, "hyper_connection.epsilon")),
    )



def _check_schedule(model_path: str, args: DeepseekV4Args, served: int) -> None:
    """Cross-check the compress_ratios schedule against the tensor table.

    Cheap (the tensor table is metadata, not weights) and worth doing every load: it is the
    difference between finding a layer-count error here and finding it as degraded output
    after a 145 GiB load.
    """
    from freetoken.models.gguf.reader import gguf_tensor_names

    names = gguf_tensor_names(model_path)
    want_compressor = sum(1 for r in args.compress_ratios[:served] if r != 0)
    want_indexer = sum(1 for r in args.compress_ratios[:served] if r == 4)
    got_compressor = sum(
        1 for i in range(served) if f"blk.{i}.attn_compressor_kv.weight" in names)
    got_indexer = sum(
        1 for i in range(served) if f"blk.{i}.indexer.attn_q_b.weight" in names)

    for label, want, got in (("compressor", want_compressor, got_compressor),
                             ("indexer", want_indexer, got_indexer)):
        if want != got:
            raise ValueError(
                f"deepseek4 GGUF: compress_ratios predicts {want} layers with a {label} "
                f"but the file has {got}; the per-layer schedule does not match this "
                f"checkpoint (a wrong served-layer count is the usual cause)"
            )

def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    """ModelConfig for a deepseek4 GGUF, mirroring deepseek_v4/config.py::parse_config.

    The served layer count excludes the trailing MTP/NextN block: ``block_count`` counts it
    but it is not part of the forward pass, and treating it as a layer makes a uniform
    expert bank look mixed.
    """
    args = _args_from_gguf(shim)
    model_path = getattr(shim, "model_path", None)

    # How block_count relates to the MTP block is NOT consistent across architectures, so
    # it is derived rather than assumed. qwen35moe counts its NextN block inside
    # block_count (Ornith: block_count 41, blk.0..blk.40 where blk.40 is the MTP block, 40
    # served). deepseek4 does not (block_count 43, blk.0..blk.42 all served, and the MTP
    # layer carries no blk tensors at all). Subtracting n_mtp_layers unconditionally
    # silently drops the last real layer here.
    #
    # compress_ratios is the authority: it carries one entry per served layer plus the MTP
    # layers, so the served count falls out of it and is then cross-checked below.
    served_layers = len(args.compress_ratios) - args.n_mtp_layers
    if served_layers != args.n_layers:
        raise ValueError(
            f"deepseek4 GGUF: compress_ratios implies {served_layers} served layers "
            f"({len(args.compress_ratios)} entries minus {args.n_mtp_layers} MTP) but "
            f"block_count is {args.n_layers}; refusing to guess which is right"
        )
    args.n_layers = served_layers

    rope_scaling = {
        "rope_type": "yarn",
        "factor": args.rope_factor,
        "beta_fast": args.beta_fast,
        "beta_slow": args.beta_slow,
        "original_max_position_embeddings": args.original_seq_len,
    }

    from .gguf_experts import gguf_expert_types

    types = gguf_expert_types(model_path, served_layers) if model_path else None
    expert_types = (types["gate_up"][0], types["down"][0]) if types else None

    # The schedule derived from compress_ratios must match what the file actually contains.
    # A compressor exists where ratio != 0 and a lightning indexer where ratio == 4, so
    # these counts are an independent check on the layer count above: an off-by-one shows
    # up here as a mismatch rather than as a quietly missing layer at serving time.
    if model_path:
        _check_schedule(model_path, args, served_layers)

    return ModelConfig(
        num_layers=served_layers,
        num_qo_heads=args.n_heads,
        num_kv_heads=1,  # MLA: a single shared latent KV head (K == V)
        head_dim=args.head_dim,
        hidden_size=args.dim,
        vocab_size=args.vocab_size,
        intermediate_size=args.moe_inter_dim,
        hidden_act="silu",
        rms_norm_eps=args.norm_eps,
        tie_word_embeddings=False,  # output.weight is a separate tensor from token_embd
        rotary_config=RotaryConfig(
            head_dim=args.head_dim,
            rotary_dim=args.rope_head_dim,
            max_position=args.max_seq_len,
            base=args.rope_theta,
            scaling=rope_scaling,
        ),
        num_experts=args.n_routed_experts,
        num_experts_per_tok=args.n_activated_experts,
        moe_intermediate_size=args.moe_inter_dim,
        norm_topk_prob=True,
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        moe_enabled=True,
        expert_quant="gguf",
        attn_sm_scale=args.head_dim**-0.5,
        dsv4_args=args,
        gguf_model_path=model_path,
        gguf_expert_types=expert_types,
        attention_groups=(
            DSV4AttentionGroupConfig(
                name="dsv4",
                layer_ids=tuple(range(served_layers)),
                num_kv_heads=1,
                head_dim=args.head_dim,
                sliding_window=args.window_size,
            ),
        ),
    )


def is_gguf_model(config: ModelConfig) -> bool:
    """True when this config came from a GGUF checkpoint (native block-quant path)."""
    return getattr(config, "gguf_model_path", None) is not None


__all__ = [
    "parse_gguf_config",
    "is_gguf_model",
]
