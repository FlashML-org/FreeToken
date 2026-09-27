"""Qwen3.8-Flash-Next (``qwen4exp``) GGUF adapter: build the FreeToken ``ModelConfig``
from GGUF KV metadata.

Mirrors ``qwen4_exp.config.parse_config`` (the HF path) but sources the geometry from
the ``qwen4exp.*`` KVs, with two GGUF-vs-HF differences handled here:

* ``ple.layers`` is already 0-based in GGUF (llama.cpp indexes ``ple_layer_arr[il]``
  directly); the HF config is 1-based and ``parse_config`` subtracts 1. So the ids go
  into ``Qwen4ExpArgs`` unchanged.
* The GGUF carries the resolved PLE ``ple.head_offsets`` / ``ple.head_vocab_sizes``
  rather than the HF ``ngram_vocab_size_base`` / ``split_ngram_parts``. The per-head
  sizes are the consecutive primes after ``ngram_vocab_size_base - 1`` (``ple.py``
  ``_nth_prime_after``), so ``ngram_vocab_size_base = head_vocab_sizes[0]`` reproduces
  them exactly; the table is one tensor, so ``split_ngram_parts = 1``.

The routed-expert quant is not a KV: it is read from the routed-expert tensor type so
``expert_quant`` / ``moe_weight_format`` route through the generic GGUF MoE path.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.gguf.dequant import GGML_BF16, GGML_F16, GGML_F32

from .config import Qwen4ExpArgs, ple_slot_states

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _expert_format(model_path: str) -> str:
    """GGUF expert format tag for the routed experts (per-role, e.g. 'iq4_xs+iq4_nl')."""
    from freetoken.moe.gguf_experts import gguf_expert_format

    return gguf_expert_format(model_path)


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata

    def g(key: str):
        val = m.get(f"qwen4exp.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key qwen4exp.{key}")
        return val

    num_layers = int(g("block_count"))
    hidden = int(g("embedding_length"))
    num_qo_heads = int(g("attention.head_count"))
    num_kv_heads = int(g("attention.head_count_kv"))
    head_dim = int(g("attention.key_length"))
    max_pos = int(g("context_length"))
    rms_eps = float(g("attention.layer_norm_rms_epsilon"))
    rope_base = float(g("rope.freq_base"))
    rotary_dim = int(g("rope.dimension_count"))

    # Every Nth layer (1-indexed) is full attention (QSA); the rest run GDN (linear).
    interval = int(g("full_attention_interval"))
    layer_types = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(num_layers)
    ]
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    # GDN geometry: key heads use the state size as their head dim; the value head dim is
    # the remaining inner width (llama-load-tensors qwen4exp: head_v_dim = d_inner / dt_rank).
    key_head_dim = int(g("ssm.state_size"))
    num_key_heads = int(g("ssm.group_count"))
    num_value_heads = int(g("ssm.time_step_rank"))
    inner = int(g("ssm.inner_size"))
    if inner % num_value_heads != 0:
        raise ValueError(f"qwen4exp: ssm.inner_size {inner} not divisible by time_step_rank {num_value_heads}")
    value_head_dim = inner // num_value_heads
    conv_kernel_dim = int(g("ssm.conv_kernel"))

    # QSA indexer. One shared index key, so one index kv head (llama-load-tensors: indexer_k_proj
    # is {n_embd, idx_head}). The compress ratio is per full layer and uniform in the releases.
    index_n_heads = int(g("attention.indexer.head_count"))
    index_head_dim = int(g("attention.indexer.key_length"))
    index_budget = int(g("attention.indexer.top_k"))
    ratios = [int(x) for x in (m.get("qwen4exp.attention.compress_ratios") or ())]
    nonzero = {r for r in ratios if r}
    if len(nonzero) > 1:
        raise ValueError(f"qwen4exp: non-uniform attention.compress_ratios {sorted(nonzero)}")
    # Some converters omit or zero this (a finetune whose HF config sets indexer_compress_ratio
    # 4 shipped all-zero); default to 4 so the full layers stay QSA, not MiniMax block-sparse.
    index_ratio = nonzero.pop() if nonzero else 4

    hc_count = int(g("hyper_connection.count"))
    hc_lowrank = int(g("hyper_connection.low_rank"))

    hidden_act = "silu"
    # The GDN output gate is sigmoid for qwen4exp (HF `output_gate_type`, and llama.cpp's
    # qwen4exp graph passes GGML_UNARY_OP_SIGMOID). The GGUF has no KV for it, and hidden_act
    # is the MoE activation, so default to the architecture's gate rather than hidden_act.
    output_gate = "sigmoid"
    # Text tokens go through mRoPE too (llama.cpp uses ggml_rope_multi with these sections),
    # so the sections must reach the rope rather than being dropped. GGUF stores 4 slots with
    # a trailing zero; the model uses a 3-section table.
    mrope_section = [int(x) for x in (m.get("qwen4exp.rope.dimension_sections") or ())]
    while mrope_section and mrope_section[-1] == 0:
        mrope_section.pop()
    mrope_section = tuple(mrope_section) if len(mrope_section) == 3 else None
    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_pos,
        base=rope_base,
        scaling=None,
        mrope_section=mrope_section,
        mrope_layout="interleaved" if mrope_section is not None else "contiguous",
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
        index_head_dim=index_head_dim,
        num_index_layers=len(full_ids),
        index_ratio=index_ratio,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        conv_kernel_dim=conv_kernel_dim,
        output_gate=output_gate,
    )
    groups = tuple(sorted((full_group, linear_group), key=lambda gp: gp.layer_ids[0]))

    # PLE n-gram embedding. Absent ple.layers leaves the module inert; the geometry is then
    # placeholder (never used, since no layer attaches a table).
    ple_layers = tuple(int(i) for i in (m.get("qwen4exp.ple.layers") or ()))
    if ple_layers:
        # ``embedding_length_per_layer_input`` is the *per-head* width (the n-gram table row,
        # e.g. 160); the concatenated embedding the PLE key_proj consumes is
        # num_ngram_heads * that (= n_embd), and ``Qwen4ExpArgs.ple_embed_dim`` is the total.
        ple_head_dim = int(g("embedding_length_per_layer_input"))
        ple_conv_kernel_size = int(g("ple.conv_kernel"))
        ngram_size = int(g("ple.ngram_size"))
        heads_per_ngram = int(g("ple.heads_per_ngram"))
        num_ngram_heads = (ngram_size - 1) * heads_per_ngram
        ple_embed_dim = ple_head_dim * num_ngram_heads
        ngram_vocab_size_base = int(g("ple.head_vocab_sizes")[0])
        eos = g("ple.eos_token_id")
    else:
        # no PLE layers: keep the args non-degenerate (num_ngram_heads >= 1) so nothing
        # divides by zero; the module is inert because ple_layer_ids is empty.
        ple_embed_dim, ple_conv_kernel_size = 1, 1
        ngram_size, heads_per_ngram, ngram_vocab_size_base = 2, 1, 1
        eos = m.get("tokenizer.ggml.eos_token_id", 0)
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    image_token_id = m.get("qwen4exp.ple.image_token_id")

    qwen4_args = Qwen4ExpArgs(
        hidden_size=hidden,
        hc_count=hc_count,
        hc_lowrank=hc_lowrank,
        ple_layer_ids=ple_layers,
        ple_embed_dim=ple_embed_dim,
        ple_conv_kernel_size=ple_conv_kernel_size,
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ngram_vocab_size_base=ngram_vocab_size_base,
        make_ngram_vocab_size_divisible_by=1,  # unused; the GGUF carries resolved sizes
        split_ngram_parts=1,  # the n-gram table is one GGUF tensor
        ngram_boundary_token_id=int(eos),
        index_n_heads=index_n_heads,
        index_kv_heads=1,
        index_head_dim=index_head_dim,
        index_budget=index_budget,
        index_ratio=index_ratio,
        image_token_id=int(image_token_id) if image_token_id is not None else None,
    )

    expert_quant = _expert_format(shim.model_path)

    # NOTE: a metadata-only FTW gguf carries no tensor table, so the plan is None and the
    # GGUF op-swap cannot run; direct .gguf loading is the supported path for now.
    gguf_types = _quant_plan(shim.model_path) if os.path.isfile(shim.model_path) else None

    # Some releases omit expert_shared_feed_forward_length; fall back to the routed width
    # when the checkpoint actually ships a shared expert, else 0.
    shexp_kv = m.get("qwen4exp.expert_shared_feed_forward_length")
    if shexp_kv is not None:
        shared_inter = int(shexp_kv)
    elif os.path.isfile(shim.model_path):
        from freetoken.models.gguf.reader import gguf_tensor_names

        has_shexp = any(
            n.endswith("ffn_gate_shexp.weight") for n in gguf_tensor_names(shim.model_path)
        )
        shared_inter = int(g("expert_feed_forward_length")) if has_shexp else 0
    else:
        shared_inter = int(g("expert_feed_forward_length"))  # metadata-only: assume present

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=0,
        hidden_act=hidden_act,
        rms_norm_eps=rms_eps,
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=full_rotary,
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        shared_expert_intermediate_size=shared_inter,
        norm_topk_prob=True,
        moe_enabled=True,
        use_qk_norm=True,
        model_type="qwen4_exp",
        architectures=list(shim.architectures),
        vision_config=None,
        image_token_id=qwen4_args.image_token_id,
        attention_groups=groups,
        expert_quant=expert_quant,
        moe_weight_format=expert_quant,
        qwen4_args=qwen4_args,
        slot_states=ple_slot_states(qwen4_args),
        gguf_quant_types=gguf_types,
    )


def gguf_shim(model_path: str):
    """Build a ``GgufConfigShim`` for a qwen4exp GGUF without the arch registry."""
    from freetoken.models.gguf.config import GgufConfigShim
    from freetoken.models.gguf.reader import gguf_tensor_names, load_gguf_metadata

    metadata = load_gguf_metadata(model_path)
    names = gguf_tensor_names(model_path)
    toks = metadata.get("tokenizer.ggml.tokens")
    if toks is None:
        raise ValueError(f"{model_path}: qwen4exp GGUF has no tokenizer.ggml.tokens to size the vocab")
    return GgufConfigShim(
        architectures=["Qwen4ExpGGUFForCausalLM"],
        model_path=model_path,
        model_type="qwen4exp",
        metadata=metadata,
        vocab_size=len(toks),
        tie_word_embeddings="output.weight" not in names,
    )


def _to_bf16(t):  # GgufTensor -> bf16 dense of its torch shape
    from freetoken.models.gguf.dequant import dequantize

    return dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16).reshape(t.shape)


def _to_fp32(t):  # GgufTensor -> fp32 dense of its torch shape
    from freetoken.models.gguf.dequant import dequantize

    return dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32).reshape(t.shape)


def _require_tp1(what: str) -> None:
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(
            f"qwen4exp GGUF {what} supports TP=1 only (packed GGUF layers are not TP-sharded)"
        )


# Per-layer scalar/norm tensors (gguf suffix -> freetoken layer-relative name), all tiny
# F32/BF16 dequantized to bf16.
_LAYER_SCALAR_MAP = {
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "hc_attn_norm.weight": "attn_hyper_connection.hc_norm.weight",
    "hc_ffn_norm.weight": "mlp_hyper_connection.hc_norm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ple_norm_key.weight": "ple.norm_key.weight",
    "ple_norm_query.weight": "ple.norm_query.weight",
    "ple_norm_conv.weight": "ple.norm_conv.weight",
}
_TOP_SCALAR_MAP = {
    "output_hc_norm.weight": "model.hyper_connection_mixer.hc_norm.weight",
}
# Routed experts -> offload banks (handled by moe/gguf_experts.py, not here).
_EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
_UNQUANTIZED_GGML = frozenset({GGML_F32, GGML_F16, GGML_BF16})


def _gdn_head_perm(num_key_heads: int, num_value_heads: int) -> torch.Tensor:
    """Index that reorders converter-ordered GDN value heads back to HF order.

    The GGUF converter writes per-value-head tensors in ``[num_v_per_k, num_k]`` order
    (``gguf[j] = hf[i]`` with ``j = i // R + K * (i % R)``); gathering with this perm
    turns GGUF rows back into ``hf[i]``.
    """
    per_k = num_value_heads // num_key_heads
    return torch.tensor([i // per_k + num_key_heads * (i % per_k) for i in range(num_value_heads)])


def _permute_head_blocks(x: torch.Tensor, perm: torch.Tensor, head_dim: int, dim: int) -> torch.Tensor:
    """Reorder ``dim`` of ``x`` (size ``len(perm)*head_dim``) by head, in place-free."""
    shape = list(x.shape)
    shape[dim : dim + 1] = [len(perm), head_dim]
    return x.reshape(shape).index_select(dim, perm).reshape(x.shape)


# Norms the model applies as (1+w) -- the converter folded the +1 into the stored weight.
_ZERO_CENTERED_NORMS = frozenset({
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "indexer.q_norm.weight",
    "indexer.k_norm.weight",
    "hc_attn_norm.weight",
    "hc_ffn_norm.weight",
    "output_hc_norm.weight",
    "ple_norm_key.weight",
    "ple_norm_query.weight",
    "ple_norm_conv.weight",
})

# (plan key, gguf suffix tuple) for each fused/single projection the op-swap can pack.
_PLAN_GROUPS = {
    "qkv": ("attn_q.weight", "attn_k.weight", "attn_v.weight"),
    "o": ("attn_output.weight",),
    "indexer": ("indexer.q_proj.weight", "indexer.k_proj.weight"),
    "inproj_qkvz": ("attn_qkv.weight", "attn_gate.weight"),
    "shexp_up": ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"),
    "shexp_down": ("ffn_down_shexp.weight",),
    "ple_key": ("ple_key.weight",),
    "ple_value": ("ple_value.weight",),
}


def _quant_plan(model_path: str) -> dict[str, int]:
    """Packed quant type per projection, or omitted when the parts cannot share a packed
    buffer (mixed types / unquantized). Mirrors ``iter_gguf_weights``' fusion decision so the
    op-swap builds exactly the ops the weight iterator names."""
    from freetoken.models.gguf.reader import iter_gguf_tensors

    types = {t.name: t.ggml_type for t in iter_gguf_tensors(model_path)}

    def packed(*names: str) -> int | None:
        ts = [types.get(n) for n in names]
        if any(x is None for x in ts):
            return None
        return ts[0] if len(set(ts)) == 1 and ts[0] not in _UNQUANTIZED_GGML else None

    plan: dict[str, int] = {}
    for key, names in (("embed", ("token_embd.weight",)), ("lm_head", ("output.weight",))):
        v = packed(*names)
        if v is not None:
            plan[key] = v
    layers = sorted({int(n.split(".")[1]) for n in types if n.startswith("blk.")})
    for layer in layers:
        for rel, suffixes in _PLAN_GROUPS.items():
            v = packed(*(f"blk.{layer}.{sfx}" for sfx in suffixes))
            if v is not None:
                plan[f"L{layer}.{rel}"] = v
    return plan


def _as_part(t) -> tuple[int, torch.Tensor]:
    return t.ggml_type, t.packed()


def _dequant_part(part: tuple[int, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    from freetoken.models.gguf.dequant import BLOCK_SHAPE, dequantize

    ggml_type, data = part
    block, size = BLOCK_SHAPE[ggml_type]
    out = dequantize(data.reshape(-1), ggml_type, dtype)
    return out.reshape(data.shape[0], data.shape[1] // size * block)


def _emit_group(base: str, rel: str, parts) -> tuple[str, torch.Tensor]:
    """Fuse projection parts along the output dim: packed ``.qweight`` only when every part
    shares one quant type, else a dense bf16 ``.weight`` (mixed checkpoints, e.g. QSA
    q=Q6_K with k/v=Q8_0, cannot share a packed row layout)."""
    types = {p[0] for p in parts}
    if len(types) == 1 and next(iter(types)) not in _UNQUANTIZED_GGML:
        return f"{base}.{rel}.qweight", torch.cat([p[1] for p in parts], dim=0)
    return f"{base}.{rel}.weight", torch.cat([_dequant_part(p, torch.bfloat16) for p in parts], dim=0)


def _emit_single(base: str, rel: str, t) -> tuple[str, torch.Tensor]:
    name = f"{base}.{rel}" if rel else base
    if t.ggml_type in _UNQUANTIZED_GGML:
        return f"{name}.weight", _to_bf16(t)
    return f"{name}.qweight", t.packed()


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(param_name, tensor)`` for every non-expert qwen4exp param.

    Quantized projections stay in their native packed block layout and are yielded as
    ``.qweight`` (uint8); norms / routers / recurrence params / HC mixes dequantize to
    bf16 (the HC merged down+inject GEMM and the GDN ``in_proj_ba`` are bf16 by design).
    Adjacent parts are fused along the output dim when their quant types agree (QSA
    q/k/v, indexer q/k, GDN qkv|z, shared-expert gate/up); a mixed group or unquantized
    parts become a dense bf16 ``.weight``. Routed experts are served from the offload
    cache and skipped here.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    assert not include_moe_experts, (
        "qwen4exp GGUF routed experts are served from the offload cache; "
        "load them with moe/gguf_experts.py, not iter_gguf_weights"
    )
    assert include_non_moe
    _require_tp1("weight loading")

    config = parse_gguf_config(gguf_shim(model_path))
    args = config.qwen4_args
    hc = args.hc_count
    lowrank = args.hc_lowrank
    pad = (-(lowrank + hc)) % 16
    # llama.cpp-style GDN conversion stores the value heads grouped by key head
    # ("[R, K]" order) and folds zero-centred norms as (1+w); the model wants HF order
    # and raw weights. See _gdn_head_perm / _ZERO_CENTERED_NORMS.
    linear = config.linear_attention_group()
    gdn_perm = _gdn_head_perm(linear.num_key_heads, linear.num_value_heads)
    gdn_head_dim = linear.value_head_dim
    key_dim = linear.num_key_heads * linear.key_head_dim

    qkv_buf: dict[int, dict[str, object]] = {}
    idx_buf: dict[int, dict[str, object]] = {}
    inproj_buf: dict[int, dict[str, object]] = {}
    hc_buf: dict[int, dict[str, torch.Tensor]] = {}
    shexp_buf: dict[int, dict[str, object]] = {}

    def layer_of(name: str) -> int:
        return int(name.split(".")[1])

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield _emit_single("model.embed_tokens", "", t)
            continue
        if name == "output.weight":
            yield _emit_single("lm_head", "", t)
            continue
        if name == "per_layer_token_embd.weight":
            continue  # PLE n-gram table: attached separately (load_host_tables)
        if name in _TOP_SCALAR_MAP:
            value = _to_fp32(t) - 1.0 if name in _ZERO_CENTERED_NORMS else _to_bf16(t)
            yield _TOP_SCALAR_MAP[name], value
            continue
        if name == "output_hc_down.weight":
            yield "model.hyper_connection_mixer.input_mix_weight_down.weight", _to_bf16(t)
            continue
        if name == "output_hc_up.weight":
            yield "model.hyper_connection_mixer.input_mix_weight_up.weight", _to_bf16(t)
            continue
        if not name.startswith("blk."):
            continue
        if any(name.endswith(sfx) for sfx in _EXPERT_SUFFIXES):
            continue  # routed experts -> offload banks

        layer = layer_of(name)
        suffix = name.split(".", 2)[2]
        base = f"model.layers.{layer}"

        if suffix in _LAYER_SCALAR_MAP:
            value = _to_fp32(t) - 1.0 if suffix in _ZERO_CENTERED_NORMS else _to_bf16(t)
            yield f"{base}.{_LAYER_SCALAR_MAP[suffix]}", value
            continue
        if suffix == "ssm_dt.bias":
            yield f"{base}.linear_attn.dt_bias", _to_fp32(t)[gdn_perm]  # model keeps dt_bias fp32
            continue
        if suffix == "ssm_a":
            # llama.cpp stores A = -exp(A_log) (gate = softplus(alpha+dt) * ssm_a); the
            # model keeps A_log and computes -exp(A_log), so invert back in fp32.
            a = _to_fp32(t)[gdn_perm]
            yield f"{base}.linear_attn.A_log", torch.log(-a)
            continue
        if suffix == "ffn_gate_inp_shexp.weight":
            # ggml ships it 1-D; the model's LinearReplicated(hidden, 1) weight is [1, hidden]
            yield f"{base}.mlp.shared_expert_gate.weight", _to_bf16(t).reshape(1, -1)
            continue

        # GDN (linear_attention) layer tensors. Value-head-indexed tensors come back in
        # converter order, so reorder them to HF order before fusing (packed rows reorder
        # fine because each row is a full-width packed row).
        if suffix == "attn_qkv.weight":
            packed = t.packed()
            inproj_buf.setdefault(layer, {})["qkv"] = (t.ggml_type, torch.cat([
                packed[: 2 * key_dim],
                _permute_head_blocks(packed[2 * key_dim :], gdn_perm, gdn_head_dim, 0),
            ], dim=0))
        elif suffix == "attn_gate.weight":
            inproj_buf.setdefault(layer, {})["z"] = (
                t.ggml_type, _permute_head_blocks(t.packed(), gdn_perm, gdn_head_dim, 0),
            )
        elif suffix == "ssm_alpha.weight":
            inproj_buf.setdefault(layer, {})["alpha"] = (t.ggml_type, t.packed()[gdn_perm])
        elif suffix == "ssm_beta.weight":
            inproj_buf.setdefault(layer, {})["beta"] = (t.ggml_type, t.packed()[gdn_perm])
        elif suffix == "ssm_out.weight":
            # input axis is the value dim, which packs across 128-wide heads, so dense
            yield f"{base}.linear_attn.out_proj.weight", _permute_head_blocks(
                _to_bf16(t), gdn_perm, gdn_head_dim, 1
            )
        elif suffix == "ssm_conv1d.weight":
            conv = _to_bf16(t).reshape(t.shape[0], 1, t.shape[1])
            yield f"{base}.linear_attn.conv1d.weight", torch.cat([
                conv[: 2 * key_dim],
                _permute_head_blocks(conv[2 * key_dim :], gdn_perm, gdn_head_dim, 0),
            ], dim=0)
        # QSA (full_attention) layer tensors.
        elif suffix == "attn_q.weight":
            qkv_buf.setdefault(layer, {})["q"] = _as_part(t)
        elif suffix == "attn_k.weight":
            qkv_buf.setdefault(layer, {})["k"] = _as_part(t)
        elif suffix == "attn_v.weight":
            qkv_buf.setdefault(layer, {})["v"] = _as_part(t)
        elif suffix == "attn_output.weight":
            yield _emit_single(f"{base}.self_attn", "o_proj", t)
        elif suffix == "indexer.q_proj.weight":
            idx_buf.setdefault(layer, {})["q"] = _as_part(t)
        elif suffix == "indexer.k_proj.weight":
            idx_buf.setdefault(layer, {})["k"] = _as_part(t)
        # Shared expert and PLE projections.
        elif suffix == "ffn_gate_shexp.weight":
            shexp_buf.setdefault(layer, {})["gate"] = _as_part(t)
        elif suffix == "ffn_up_shexp.weight":
            shexp_buf.setdefault(layer, {})["up"] = _as_part(t)
        elif suffix == "ffn_down_shexp.weight":
            yield _emit_single(f"{base}.mlp.shared_expert", "down_proj", t)
        elif suffix in ("ple_key.weight", "ple_value.weight"):
            proj = "key_proj" if suffix.startswith("ple_key") else "value_proj"
            yield _emit_single(f"{base}.ple", proj, t)
        elif suffix == "ple_conv1d.weight":
            yield f"{base}.ple.conv1d.weight", _to_bf16(t).reshape(t.shape[0], 1, t.shape[1])
        # Hyper-connection merged down + inject (bf16: the two parts may differ in GGUF type).
        elif suffix in ("hc_attn_down.weight", "hc_attn_inject.weight", "hc_attn_up.weight",
                        "hc_ffn_down.weight", "hc_ffn_inject.weight", "hc_ffn_up.weight"):
            hc_buf.setdefault(layer, {})[suffix] = _to_bf16(t)
        else:
            raise ValueError(f"unmapped qwen4exp GGUF tensor: {name}")

        # Emit fused groups once complete. Packed (.qweight) only when every part shares one
        # quant type; a mixed group (e.g. QSA q=Q6_K, k/v=Q8_0) falls back to a dense bf16
        # .weight, which the op-swap builds as a plain Linear.
        slots = qkv_buf.get(layer)
        if slots and {"q", "k", "v"} <= slots.keys():
            yield _emit_group(f"{base}.self_attn", "qkv_proj", [slots["q"], slots["k"], slots["v"]])
            del qkv_buf[layer]
        idx = idx_buf.get(layer)
        if idx and {"q", "k"} <= idx.keys():
            yield _emit_group(f"{base}.self_attn.indexer", "index_qk_proj", [idx["q"], idx["k"]])
            del idx_buf[layer]
        ip = inproj_buf.get(layer)
        if ip and {"qkv", "z", "alpha", "beta"} <= ip.keys():
            yield _emit_group(f"{base}.linear_attn", "in_proj_qkvz", [ip["qkv"], ip["z"]])
            # convert() always builds in_proj_ba dense (the b/a tensors are F32 in practice),
            # so emit it dense regardless of their type
            yield f"{base}.linear_attn.in_proj_ba.weight", torch.cat(
                [_dequant_part(ip["beta"], torch.bfloat16), _dequant_part(ip["alpha"], torch.bfloat16)], dim=0
            )
            del inproj_buf[layer]
        sh = shexp_buf.get(layer)
        if sh and {"gate", "up"} <= sh.keys():
            yield _emit_group(f"{base}.mlp.shared_expert", "gate_up_proj", [sh["gate"], sh["up"]])
            del shexp_buf[layer]
        h = hc_buf.get(layer)
        for owner in ("attn", "ffn"):
            down, inject, up = f"hc_{owner}_down.weight", f"hc_{owner}_inject.weight", f"hc_{owner}_up.weight"
            if h and {down, inject, up} <= h.keys():
                merged = torch.cat(
                    [h[down], h[inject], torch.zeros(pad, h[down].shape[1], dtype=h[down].dtype)],
                    dim=0,
                )
                rel = "attn_hyper_connection" if owner == "attn" else "mlp_hyper_connection"
                yield f"{base}.{rel}.input_mix_weight_down_block_inject.weight", merged
                yield f"{base}.{rel}.input_mix_weight_up.weight", h[up]
                for key in (down, inject, up):
                    del h[key]
        if h is not None and not h:
            del hc_buf[layer]

    # Derived n-gram hash buffers: the GGUF has no such tensors, but the loader only accepts
    # keys the reader yields, so emit them from the same constants convert() installs.
    if args.ple_layer_ids:
        from .ple import derive_ngram_hash_constants

        for ple_index, layer in enumerate(args.ple_layer_ids):
            mult, sizes, offsets = derive_ngram_hash_constants(
                vocab_size=config.vocab_size,
                ngram_size=args.ngram_size,
                num_ngram_heads=args.num_ngram_heads,
                ngram_vocab_size_base=args.ngram_vocab_size_base,
                ple_layer_index=ple_index,
            )
            pre = f"model.layers.{layer}.ple.ple_embedding"
            yield f"{pre}.layer_multipliers", torch.tensor(mult, dtype=torch.int64)
            yield f"{pre}.ngram_heads_vocab_sizes", torch.tensor(sizes, dtype=torch.int64)
            yield f"{pre}.ngram_heads_offsets", torch.tensor(offsets, dtype=torch.int64)

    assert not qkv_buf, f"incomplete QSA qkv groups: {sorted(qkv_buf)}"
    assert not idx_buf, f"incomplete indexer groups: {sorted(idx_buf)}"
    assert not inproj_buf, f"incomplete GDN in_proj groups: {sorted(inproj_buf)}"
    assert not hc_buf, f"incomplete hyper-connection groups: {sorted(hc_buf)}"
    assert not shexp_buf, f"incomplete shared-expert groups: {sorted(shexp_buf)}"


def is_gguf_model(config: ModelConfig) -> bool:
    """True when the config came from a GGUF with a readable tensor table (op-swap active)."""
    return getattr(config, "gguf_quant_types", None) is not None


def resolve_ple_source(engine_config) -> str:
    """The fp8 PLE n-gram table source (a HF repo id or a folder with ``model-plefp8-*``).

    HF checkpoints and FTW checkpoints with their own ``ple-table-*.safetensors`` resolve from
    ``model_path``; a GGUF (or an FTW without the side files) carries no fp8 table, so it must
    set ``--ple-source``.
    """
    source = getattr(engine_config, "ple_source", None)
    if source:
        return source
    import glob
    import os

    from freetoken.checkpoint.ftw import is_ftw_checkpoint
    from freetoken.models.gguf.reader import is_gguf_path

    model_path = engine_config.model_path
    if is_ftw_checkpoint(model_path) and glob.glob(
        os.path.join(model_path, "ple-table-*.safetensors")
    ):
        return model_path
    if is_gguf_path(model_path) or is_ftw_checkpoint(model_path):
        raise ValueError(
            "this checkpoint carries no fp8 PLE table; pass --ple-source <repo-or-dir> "
            "pointing at the original model-plefp8-* shards"
        )
    return model_path


def _swap(owner, attr: str, quant_type: int) -> None:
    from freetoken.layers.gguf import GGUFLinear

    lin = getattr(owner, attr)
    out_features, in_features = lin.weight.shape
    setattr(
        owner,
        attr,
        GGUFLinear(in_features, out_features, int(quant_type), has_bias=getattr(lin, "bias", None) is not None),
    )


def convert_qwen4_exp_to_gguf(model, config: ModelConfig) -> None:
    """In place: replace the projections the GGUF packs with native GGUF ops, leaving the
    rest dense bf16.

    Packed (one quant type across the group): token embedding, lm_head, QSA qkv / o /
    indexer, GDN ``in_proj_qkvz``, GDN ``ssm_out``, shared-expert gate_up / down, PLE
    key/value. Mixed-type or unquantized groups (e.g. QSA q=Q6_K with k/v=Q8_0) stay dense
    bf16, matching the ``.weight`` names ``iter_gguf_weights`` yields for them.

    GDN is built split (``in_proj_qkvz`` packed + ``in_proj_ba`` dense): the checkpoint
    quantizes qkv|z but ships b/a as F32, so they cannot share a packed buffer.
    """
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear, GGUFUntiedLMHead
    from freetoken.layers import LinearColParallelMerged

    plan = config.gguf_quant_types or {}
    inner = model.model

    embed_type = plan.get("embed")
    if embed_type is not None:
        inner.embed_tokens = GGUFEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            quant_type=int(embed_type),
        )
    lm_head_type = plan.get("lm_head")
    if lm_head_type is not None:
        model.lm_head = GGUFUntiedLMHead(
            config.hidden_size, config.vocab_size, int(lm_head_type), has_bias=False
        )

    for layer in inner.layers.op_list:
        layer_id = layer._layer_id
        head = getattr(layer, "self_attn", None)
        if head is not None:  # QSA (full_attention) layer
            for attr, key, out_features in (
                ("qkv_proj", "qkv", head.qo_attn_dim * 2 + head.kv_attn_dim * 2),
                ("o_proj", "o", None),
            ):
                qtype = plan.get(f"L{layer_id}.{key}")
                if qtype is None:
                    continue
                if out_features is None:
                    _swap(head, attr, qtype)
                else:
                    setattr(head, attr, GGUFLinear(config.hidden_size, out_features, int(qtype), has_bias=False))
            idx_type = plan.get(f"L{layer_id}.indexer")
            if idx_type is not None:
                idx = head.indexer
                setattr(
                    idx,
                    "index_qk_proj",
                    GGUFLinear(
                        config.hidden_size,
                        idx.num_heads * idx.head_dim + idx.num_kv_heads * idx.head_dim,
                        int(idx_type),
                        has_bias=False,
                    ),
                )
        else:  # GDN (linear_attention) layer: split into packed qkvz + dense ba
            g = layer.linear_attn
            g.in_proj_ba = LinearColParallelMerged(
                config.hidden_size, [g.num_v_heads, g.num_v_heads], has_bias=False,
                quant_config=None, prefix="",
            )
            qkvz_type = plan.get(f"L{layer_id}.inproj_qkvz")
            if qkvz_type is not None:
                g.in_proj_qkvz = GGUFLinear(
                    config.hidden_size, g.conv_dim + g.value_dim, int(qkvz_type), has_bias=False
                )
            else:
                g.in_proj_qkvz = LinearColParallelMerged(
                    config.hidden_size, [g.conv_dim, g.value_dim], has_bias=False,
                    quant_config=None, prefix="",
                )
            g._split_in_proj = True
            if hasattr(g, "in_proj"):
                del g.in_proj
            out_type = plan.get(f"L{layer_id}.ssm_out")
            if out_type is not None:
                _swap(g, "out_proj", out_type)

        shared = layer.mlp.shared_expert
        up_type = plan.get(f"L{layer_id}.shexp_up")
        if up_type is not None:
            _swap(shared, "gate_up_proj", up_type)
        down_type = plan.get(f"L{layer_id}.shexp_down")
        if down_type is not None:
            _swap(shared, "down_proj", down_type)

        if layer.ple is not None:
            key_type = plan.get(f"L{layer_id}.ple_key")
            if key_type is not None:
                _swap(layer.ple, "key_proj", key_type)
            value_type = plan.get(f"L{layer_id}.ple_value")
            if value_type is not None:
                _swap(layer.ple, "value_proj", value_type)

    # The n-gram hash buffers normally come from checkpoint tensors; a GGUF has none, so
    # derive them (they reproduce the GGUF's ple.* values exactly, see the config test).
    if inner.ple_layers:
        from .ple import derive_ngram_hash_constants

        for ple in inner.ple_layers:
            a = ple.args
            mult, sizes, offsets = derive_ngram_hash_constants(
                vocab_size=config.vocab_size,
                ngram_size=a.ngram_size,
                num_ngram_heads=a.num_ngram_heads,
                ngram_vocab_size_base=a.ngram_vocab_size_base,
                ple_layer_index=ple.ple_index,
            )
            emb = ple.ple_embedding
            emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
            emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
            emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))


__all__ = [
    "parse_gguf_config",
    "gguf_shim",
    "iter_gguf_weights",
    "is_gguf_model",
    "convert_qwen4_exp_to_gguf",
    "resolve_ple_source",
]
