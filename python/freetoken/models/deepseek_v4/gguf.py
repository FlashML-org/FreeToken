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

# ggml type -> the dtype its raw bytes represent (unquantized types only).
_UNQ_DTYPE = {0: torch.float32, 1: torch.float16, 30: torch.bfloat16}


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



class GGUFLinearNN(torch.nn.Module):
    """A GGUF-quantized Linear that DSV4's loader can actually fill.

    FreeToken's own ``layers.gguf.GGUFLinear`` is a ``BaseOP`` holding ``qweight`` as a
    plain tensor. That works for the qwen models, whose trees are built from BaseOP, but
    deepseek_v4 is raw ``nn.Module`` and loads via
    ``DeepseekV4ForCausalLM.load_state_dict``, which walks ``named_parameters()`` and
    demands a key for every one. A plain attribute is invisible there, and assigning a
    non-Module over a Module child raises outright.

    So the packed block bytes live in an ordinary ``nn.Parameter`` -- uint8, requires_grad
    False -- named ``weight`` to match the naming the rest of this model uses. The loader's
    ``.to(p.dtype)`` cast is then a no-op on uint8, and the tensor arrives byte-for-byte.
    """

    def __init__(self, in_features: int, out_features: int, quant_type: int,
                 bias: bool = False):
        super().__init__()
        from freetoken.models.gguf.dequant import row_bytes

        self.in_features = in_features
        self.out_features = out_features
        self._quant_type = int(quant_type)
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, row_bytes(in_features, self._quant_type),
                        dtype=torch.uint8),
            requires_grad=False,
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.layers.gguf import fused_mul_mat_gguf

        # fused_mul_mat_gguf takes [tokens, in_features] and treats dim 0 as the batch, so
        # leading dims must be folded and restored. F.linear -- which this replaces --
        # accepts any number of leading dims, and deepseek_v4 relies on that: its attention
        # passes 3-D tensors, and collapsing one silently reshapes q so the sparse-attention
        # kernel's `b, m, h, d = q.shape` unpack fails.
        shape = x.shape
        flat = x.reshape(-1, shape[-1]) if x.dim() != 2 else x
        out = fused_mul_mat_gguf(flat, self.weight, self._quant_type)
        if self.bias is not None:
            out = out + self.bias
        return out if x.dim() == 2 else out.view(*shape[:-1], out.shape[-1])


class GGUFEmbeddingNN(torch.nn.Module):
    """GGUF-quantized vocab embedding, as an nn.Module for the same reason as above.

    The table is never dequantized whole: only the looked-up rows are gathered in packed
    form and dequantized per lookup.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, quant_type: int):
        super().__init__()
        from freetoken.models.gguf.dequant import row_bytes

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._quant_type = int(quant_type)
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, row_bytes(embedding_dim, self._quant_type),
                        dtype=torch.uint8),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.models.gguf.dequant import GGML_UNQUANTIZED

        flat = x.flatten()
        rows = self.weight.index_select(0, flat)
        if self._quant_type in GGML_UNQUANTIZED:
            # Unquantized types are raw value bytes in the uint8 buffer; there is no
            # dequant kernel for them (ggml_dequantize rejects type 1 outright), so the
            # gathered rows are reinterpreted instead. DeepSeek-V4 ships token_embd as F16.
            from freetoken.models.deepseek_v4.gguf import _UNQ_DTYPE

            y = rows.view(_UNQ_DTYPE[self._quant_type]).to(torch.bfloat16)
        else:
            from freetoken.kernel.gguf import ggml_dequantize

            y = ggml_dequantize(rows, self._quant_type, flat.shape[0], self.embedding_dim,
                                torch.bfloat16)
        return y.view(*x.shape, self.embedding_dim)


def _dense(t, dtype: torch.dtype) -> torch.Tensor:
    """A GgufTensor as a dense tensor of its torch shape.

    Two paths, because neither covers everything. Unquantized types (F32/F16/BF16) are
    already values, so the packed bytes are simply reinterpreted -- no kernel needed, and
    it works without CUDA. Block-quantized types go through the vendored CUDA dequant:
    ``dequant.dequantize``'s pure-torch fallback only implements Q4_0 and Q6_K, and this
    checkpoint stores its attention projections and lm_head as Q8_0.
    """
    from freetoken.models.gguf.dequant import (
        BLOCK_SHAPE,
        GGML_BF16,
        GGML_F16,
        GGML_F32,
        GGML_UNQUANTIZED,
    )

    gt = int(t.ggml_type)
    raw = t.packed()
    if gt in GGML_UNQUANTIZED:
        view = {GGML_F32: torch.float32, GGML_F16: torch.float16,
                GGML_BF16: torch.bfloat16}[gt]
        return raw.reshape(-1).view(view).reshape(t.shape).to(dtype)

    from freetoken.kernel.gguf import ggml_dequantize

    block, type_size = BLOCK_SHAPE[gt]
    in_features = t.row_bytes // type_size * block
    out = ggml_dequantize(raw.cuda().contiguous(), gt, t.rows, in_features,
                          torch.bfloat16)
    return out.reshape(t.shape).to(dtype)


def _to_bf16(t) -> torch.Tensor:
    return _dense(t, torch.bfloat16)


def _to_f32(t) -> torch.Tensor:
    return _dense(t, torch.float32)


def _to_i64(t) -> torch.Tensor:
    """Read an I32 index table as int64.

    tid2eid is a routing table, not a weight: dequantizing it through a float path would
    round large token ids. Reinterpret the raw bytes instead.
    """
    return t.packed().reshape(-1).view(torch.int32).reshape(t.shape).to(torch.int64)


# suffix -> (destination template, kind). "packed" lands on a GGUFLinear's .weight;
# "bf16"/"f32" are dequantized onto an ordinary parameter. The destination is spelled out
# per tensor rather than derived from the name, because three of them do not follow the
# pattern the names imply (see the module docstring).
_LAYER_MAP: dict[str, tuple[str, str]] = {
    "attn_norm.weight":            ("attn_norm.weight", "f32"),
    "ffn_norm.weight":             ("ffn_norm.weight", "f32"),
    "attn_q_a.weight":             ("attn.wq_a.weight", "packed"),
    "attn_q_a_norm.weight":        ("attn.q_norm.weight", "f32"),
    "attn_q_b.weight":             ("attn.wq_b.weight", "packed"),
    "attn_kv.weight":              ("attn.wkv.weight", "packed"),
    "attn_kv_a_norm.weight":       ("attn.kv_norm.weight", "f32"),
    # wo_a is a bare nn.Parameter in bf16, NOT a Linear: no .weight, never packed.
    "attn_output_a.weight":        ("attn.wo_a", "bf16"),
    "attn_output_b.weight":        ("attn.wo_b.weight", "packed"),
    "attn_sinks.weight":           ("attn.attn_sink", "f32"),
    # compressor / indexer projections are F16 in the file. F16 is in GGML_UNQUANTIZED, so
    # GGUFLinear cannot hold them -- they must land dense on a normal .weight.
    "attn_compressor_kv.weight":   ("attn.compressor.wkv.weight", "bf16"),
    "attn_compressor_gate.weight": ("attn.compressor.wgate.weight", "bf16"),
    "attn_compressor_norm.weight": ("attn.compressor.norm.weight", "f32"),
    "attn_compressor_ape.weight":  ("attn.compressor.ape", "f32"),
    "indexer.attn_q_b.weight":     ("attn.indexer.wq_b.weight", "bf16"),
    "indexer.proj.weight":         ("attn.indexer.weights_proj.weight", "bf16"),
    "indexer_compressor_kv.weight":   ("attn.indexer.compressor.wkv.weight", "bf16"),
    "indexer_compressor_gate.weight": ("attn.indexer.compressor.wgate.weight", "bf16"),
    "indexer_compressor_norm.weight": ("attn.indexer.compressor.norm.weight", "f32"),
    "indexer_compressor_ape.weight":  ("attn.indexer.compressor.ape", "f32"),
    "hc_attn_base.weight":         ("hc_attn_base", "f32"),
    "hc_attn_fn.weight":           ("hc_attn_fn", "f32"),
    "hc_attn_scale.weight":        ("hc_attn_scale", "f32"),
    "hc_ffn_base.weight":          ("hc_ffn_base", "f32"),
    "hc_ffn_fn.weight":            ("hc_ffn_fn", "f32"),
    "hc_ffn_scale.weight":         ("hc_ffn_scale", "f32"),
    "ffn_gate_inp.weight":         ("ffn.gate.weight", "bf16"),
    "exp_probs_b.bias":            ("ffn.gate.bias", "f32"),
    # DeepSeek names the shared expert gate/up/down; Expert calls them w1/w3/w2.
    "ffn_gate_shexp.weight":       ("ffn.shared_experts.w1.weight", "packed"),
    "ffn_up_shexp.weight":         ("ffn.shared_experts.w3.weight", "packed"),
    "ffn_down_shexp.weight":       ("ffn.shared_experts.w2.weight", "packed"),
}

_GLOBAL_MAP: dict[str, tuple[str, str]] = {
    "output_norm.weight":    ("norm.weight", "f32"),
    # output.weight is Q8_0 but `head` is a bare bf16 nn.Parameter consumed by F.linear,
    # so it is dequantized rather than swapped. deepseek_v4/model.py already slices to the
    # last prefill position itself, so it needs no GGUFLMHead.
    "output.weight":         ("head", "bf16"),
    "output_hc_base.weight": ("hc_head_base", "f32"),
    "output_hc_fn.weight":   ("hc_head_fn", "f32"),
    "output_hc_scale.weight": ("hc_head_scale", "f32"),
}

# Routed experts are streamed from the offload cache, never yielded here.
_EXPERT_SUFFIXES = frozenset(
    {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"})


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every non-expert deepseek4 parameter."""
    import re

    from freetoken.models.gguf.reader import iter_gguf_tensors

    assert not include_moe_experts, (
        "deepseek4 GGUF keeps its routed experts in the offload cache; they are loaded by "
        "gguf_experts.load_gguf_expert_sources, not by iter_gguf_weights."
    )
    assert include_non_moe

    conv = {"packed": lambda t: t.packed(), "bf16": _to_bf16, "f32": _to_f32}

    for t in iter_gguf_tensors(model_path):
        name = t.name
        m = re.match(r"^blk\.(\d+)\.(.+)$", name)
        if m is None:
            dest = _GLOBAL_MAP.get(name)
            if dest is None:
                if name == "token_embd.weight":
                    yield "embed.weight", t.packed()
                    continue
                raise ValueError(
                    f"deepseek4 GGUF: unmapped global tensor {name!r}; this checkpoint does "
                    f"not match the layout this adapter expects"
                )
            path, kind = dest
            yield path, conv[kind](t)
            continue

        layer, suffix = int(m.group(1)), m.group(2)
        if suffix in _EXPERT_SUFFIXES:
            continue  # offload cache
        if suffix == "ffn_gate_tid2eid.weight":
            # Hash routing table on the first n_hash_layers layers; an index, not a weight.
            yield f"layers.{layer}.ffn.gate.tid2eid", _to_i64(t)
            continue
        dest = _LAYER_MAP.get(suffix)
        if dest is None:
            raise ValueError(
                f"deepseek4 GGUF: unmapped tensor {name!r}; this checkpoint does not match "
                f"the layout this adapter expects"
            )
        path, kind = dest
        yield f"layers.{layer}.{path}", conv[kind](t)


def _scan_quant_types(model_path: str) -> dict[tuple[int, str], int]:
    """(layer, suffix) -> ggml type, straight from the tensor table.

    A guessed type allocates a wrong-sized packed buffer, so nothing here has a default.
    Globals use layer -1.
    """
    import re

    from freetoken.models.gguf.reader import iter_gguf_tensors

    out: dict[tuple[int, str], int] = {}
    for t in iter_gguf_tensors(model_path):
        m = re.match(r"^blk\.(\d+)\.(.+)$", t.name)
        if m:
            out[(int(m.group(1)), m.group(2))] = int(t.ggml_type)
        else:
            out[(-1, t.name)] = int(t.ggml_type)
    return out


def convert_deepseek4_to_gguf(model, config: ModelConfig, *, model_path: str) -> None:
    """In place: swap deepseek4's quantized projections + embedding for native GGUF ops.

    Swapped to GGUFLinear (Q8_0 in the checkpoint): attention wq_a / wq_b / wkv / wo_b and
    the shared expert's w1 / w2 / w3.

    Deliberately NOT swapped:
      * ``attn.wo_a`` is a bare bf16 nn.Parameter, not a Linear; it is dequantized dense.
      * the compressor's wkv / wgate and the indexer's weights_proj are already
        ``Linear(kind="bf16")`` and their tensors are F16, so they take dense weights.
      * ``head`` is a bare bf16 nn.Parameter consumed by F.linear.

    Replaced rather than swapped: ``indexer.wq_b`` is declared ``Linear(kind="fp8")``,
    which allocates a ``.scale`` no GGUF tensor can fill because that tensor is F16 here.
    It becomes a bf16 Linear so ``.weight`` is bf16 and ``scale`` is None.
    """
    from .layers import Linear

    quant = _scan_quant_types(model_path)

    def qt(layer: int, suffix: str) -> int:
        key = (layer, suffix)
        if key not in quant:
            where = suffix if layer < 0 else f"blk.{layer}.{suffix}"
            raise ValueError(
                f"deepseek4 GGUF {model_path}: expected tensor {where} is absent, so its "
                f"quant type cannot be read; this checkpoint does not match the layout "
                f"this adapter expects"
            )
        return quant[key]

    def swap_linear(owner, attr: str, quant_type: int) -> None:
        lin = getattr(owner, attr)
        setattr(
            owner, attr,
            GGUFLinearNN(lin.in_features, lin.out_features, quant_type,
                         bias=getattr(lin, "bias", None) is not None),
        )

    # DeepseekV4ForCausalLM is an engine wrapper; the parameters live on the inner
    # Transformer, and state_dict() names them relative to it (no "_transformer." prefix),
    # which is what iter_gguf_weights emits.
    root = getattr(model, "_transformer", model)

    root.embed = GGUFEmbeddingNN(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        quant_type=qt(-1, "token_embd.weight"),
    )

    for layer_idx, layer in enumerate(root.layers):
        attn = layer.attn
        swap_linear(attn, "wq_a", qt(layer_idx, "attn_q_a.weight"))
        swap_linear(attn, "wq_b", qt(layer_idx, "attn_q_b.weight"))
        swap_linear(attn, "wkv", qt(layer_idx, "attn_kv.weight"))
        swap_linear(attn, "wo_b", qt(layer_idx, "attn_output_b.weight"))

        idx = getattr(attn, "indexer", None)
        if idx is not None:
            # F16 in the file, fp8 in the module: rebuild as bf16 so there is no orphan
            # .scale and F.linear is used instead of the block-fp8 GEMM.
            old = idx.wq_b
            idx.wq_b = Linear(old.in_features, old.out_features,
                              bias=getattr(old, "bias", None) is not None, kind="bf16")

        shexp = layer.ffn.shared_experts
        swap_linear(shexp, "w1", qt(layer_idx, "ffn_gate_shexp.weight"))
        swap_linear(shexp, "w3", qt(layer_idx, "ffn_up_shexp.weight"))
        swap_linear(shexp, "w2", qt(layer_idx, "ffn_down_shexp.weight"))


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "convert_deepseek4_to_gguf",
    "is_gguf_model",
]
