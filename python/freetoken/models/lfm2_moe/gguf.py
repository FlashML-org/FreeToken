"""LFM2-MoE GGUF adapter: the ModelConfig from GGUF metadata, native K-quant weights.

llama.cpp's ``lfm2moe`` writer renames every HF key but keeps the tensor *roles*, so this
module is the mapping layer between the two: :func:`parse_gguf_config` derives the same
``ModelConfig`` ``lfm2_moe.config.parse_config`` would (llama.cpp marks the short-conv layers
with ``attention.head_count_kv == 0``), :func:`iter_gguf_weights` renames the ggml keys onto
the model's attribute paths while leaving the quantized projections in their packed block
layout, and :func:`convert_lfm2_moe_to_gguf` swaps those projections for the native GGUF ops.

A K-quant file mixes types per tensor (Q4_K attention and gate/up experts, Q6_K v_proj and
down experts in Q4_K_M), so every per-tensor type travels in the config's ``gguf_tensor_types``
rather than being a family constant, and the routed experts load as native K-quant banks
(``load_gguf_k_expert_sources``) that the borrowed ggml MoE kernels read directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    ShortConvGroupConfig,
)
from freetoken.models.gguf.dequant import (
    GGML_NAME,
    GGML_Q4_0,
    GGML_Q4_K,
    GGML_Q6_K,
    GGML_Q8_0,
    padded_row_bytes,
)

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim

# ggml types the expert banks may be packed with: every type this repo models (BLOCK_SHAPE)
# that the borrowed grouped-MoE kernels dispatch on. Q4_K/Q6_K are what llama.cpp's K-quant
# recipes produce; the others cover a Q4_0/Q8_0 requantization of the same checkpoint.
_EXPERT_TYPES = (GGML_Q4_0, GGML_Q8_0, GGML_Q4_K, GGML_Q6_K)


def _tensor_types(shim: "GgufConfigShim") -> dict[str, int]:
    """``{checkpoint tensor name: ggml type}`` from the source file's tensor table."""
    from freetoken.models.gguf.reader import GGUF_TENSOR_TYPES_KV, gguf_tensor_names, iter_gguf_tensors

    if gguf_tensor_names(shim.model_path):
        return {t.name: t.ggml_type for t in iter_gguf_tensors(shim.model_path)}
    # A metadata-only GGUF (an FTW dir's source_metadata.gguf) has no tensor table; the
    # converter carries the type map as a KV for exactly this reason.
    packed = shim.metadata.get(GGUF_TENSOR_TYPES_KV)
    if not packed:
        raise ValueError(
            f"{shim.model_path}: this LFM2-MoE GGUF carries no tensor table (a converted FTW "
            f"dir) and no {GGUF_TENSOR_TYPES_KV!r} KV with the per-tensor quant types; "
            "reconvert the checkpoint with the current freetoken.checkpoint.convert"
        )
    return {name: int(kind) for name, kind in (entry.rsplit("|", 1) for entry in packed)}


def _layer_ids(kv_per_layer: list[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(conv_ids, full_ids)`` from the GGUF's per-layer ``attention.head_count_kv``.

    llama.cpp writes 0 heads for the short-conv layers and the attention head count for the
    full ones, which is the only layer-type signal the file carries.
    """
    conv_ids = tuple(i for i, kv in enumerate(kv_per_layer) if kv == 0)
    full_ids = tuple(i for i, kv in enumerate(kv_per_layer) if kv > 0)
    if not full_ids:
        raise ValueError("lfm2moe GGUF has no full-attention layer")
    return conv_ids, full_ids


def _expert_bank_bytes(
    types: dict[str, int],
    *,
    first_moe_layer: int,
    num_layers: int,
    num_experts: int,
    hidden: int,
    inter: int,
) -> int:
    """Exact expert-bank host bytes: the packed rows the loader writes, padding included."""
    total = 0
    for layer in range(first_moe_layer, num_layers):
        gate, down = expert_types_of_layer(types, layer)
        gate_up_bytes, _ = padded_row_bytes(hidden, gate)
        down_bytes, _ = padded_row_bytes(inter, down)
        total += num_experts * (2 * inter * gate_up_bytes + hidden * down_bytes)
    return total


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata

    def g(key: str):
        val = m.get(f"lfm2moe.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key lfm2moe.{key}")
        return val

    num_layers = int(g("block_count"))
    hidden = int(g("embedding_length"))
    num_qo_heads = int(g("attention.head_count"))
    head_dim = hidden // num_qo_heads
    kv_per_layer = [int(v) for v in g("attention.head_count_kv")]
    if len(kv_per_layer) != num_layers:
        raise ValueError(
            f"lfm2moe.attention.head_count_kv has {len(kv_per_layer)} entries, block_count {num_layers}"
        )
    conv_ids, full_ids = _layer_ids(kv_per_layer)
    num_kv_heads = kv_per_layer[full_ids[0]]

    gating_func = int(g("expert_gating_func"))
    if gating_func != 2:  # llama.cpp: 1 = softmax, 2 = sigmoid
        raise ValueError(
            f"lfm2moe.expert_gating_func {gating_func} is not sigmoid (2); the LFM2-MoE router "
            "is sigmoid-routed"
        )

    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=head_dim,
        max_position=int(g("context_length")),
        base=float(g("rope.freq_base")),
        scaling=None,  # llama.cpp writes no rope scaling for these releases (rope_type default)
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=rotary,
    )
    conv_group = ShortConvGroupConfig(
        name="conv",
        layer_ids=conv_ids,
        conv_dim=hidden,
        conv_kernel_dim=int(g("shortconv.l_cache")),
        # llama.cpp writes no shortconv bias tensor and every release sets conv_bias false
        conv_bias=False,
    )
    groups = tuple(
        sorted((g_ for g_ in (full_group, conv_group) if g_.layer_ids), key=lambda g_: g_.layer_ids[0])
    )

    types = _tensor_types(shim)
    first_k_dense_replace = int(g("leading_dense_block_count"))
    num_experts = int(g("expert_count"))
    moe_inter = int(g("expert_feed_forward_length"))

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=int(g("feed_forward_length")),
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=rotary,
        num_experts=num_experts,
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=moe_inter,
        norm_topk_prob=True,  # Lfm2MoeConfig default; llama.cpp writes no key for it
        use_expert_bias=True,  # every release ships the F32 exp_probs_b bias
        first_k_dense_replace=first_k_dense_replace,
        moe_enabled=True,
        model_type=shim.model_type,
        architectures=list(shim.architectures),
        attention_groups=groups,
        # native GGUF K-quant experts: the file's own packed rows, read by the borrowed ggml
        # MoE kernels, with the per-layer types carried in layer_tags
        expert_quant="gguf_k",
        moe_weight_format="gguf_k",
        gguf_tensor_types=tuple(sorted(types.items())),
        expert_bank_bytes=_expert_bank_bytes(
            types,
            first_moe_layer=first_k_dense_replace,
            num_layers=num_layers,
            num_experts=num_experts,
            hidden=hidden,
            inter=moe_inter,
        ),
    )


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken model params.
# --------------------------------------------------------------------------------------

# Layer tensor mapping: GGUF key after ``blk.N.`` -> (model path after ``model.layers.N``,
# role). "qweight" keeps the packed block rows (the native GGUF ops read them), "dense" is a
# small F32 tensor dequantized to bf16, "depthwise" is the short conv's F32 kernel, "bias32"
# is the router's F32 expert bias (kept fp32: it shifts the top-k selection).
_LAYER_MAP: dict[str, tuple[str, str]] = {
    "attn_norm.weight": ("operator_norm.weight", "dense"),
    "ffn_norm.weight": ("ffn_norm.weight", "dense"),
    "attn_q.weight": ("self_attn.q_proj", "qweight"),
    "attn_k.weight": ("self_attn.k_proj", "qweight"),
    "attn_v.weight": ("self_attn.v_proj", "qweight"),
    "attn_output.weight": ("self_attn.out_proj", "qweight"),
    "attn_q_norm.weight": ("self_attn.q_layernorm.weight", "dense"),
    "attn_k_norm.weight": ("self_attn.k_layernorm.weight", "dense"),
    "shortconv.conv.weight": ("conv.conv.weight", "depthwise"),
    "shortconv.in_proj.weight": ("conv.in_proj", "qweight"),
    "shortconv.out_proj.weight": ("conv.out_proj", "qweight"),
    "ffn_gate.weight": ("feed_forward.w1", "qweight"),
    "ffn_up.weight": ("feed_forward.w3", "qweight"),
    "ffn_down.weight": ("feed_forward.w2", "qweight"),
    "ffn_gate_inp.weight": ("feed_forward.gate.weight", "dense"),
    "exp_probs_b.bias": ("feed_forward.expert_bias", "bias32"),
}
# Tensors the loader must see: two per-layer norms plus the mixer and feed-forward of the
# layer's kind (checked after the pass so a renamed or missing tensor cannot load silently).
_EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")


def _to_bf16(t) -> torch.Tensor:
    """Dequantize a GgufTensor (F32/F16/Q*) to a dense bf16 tensor of its torch shape."""
    from freetoken.models.gguf.dequant import dequantize

    return dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16).reshape(t.shape)


def _require_tp1(what: str) -> None:
    """GGUF quant layers / expert banks are not sharded: reject TP>1 with a clear error
    instead of failing later on a confusing shape mismatch."""
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(
            f"lfm2_moe GGUF {what} currently supports TP=1 only "
            "(GGUF quant layers and expert banks are not tensor-parallel sharded)."
        )


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(param_name, tensor)`` for every non-expert LFM2-MoE param.

    Quantized projections (attention q/k/v/o, the short-conv in/out projections, the dense
    MLP) and the token embedding stay in their native packed block layout and are yielded as
    ``.qweight`` (uint8); norms, the depthwise conv kernel and the router dequantize to bf16,
    and the router's expert bias stays fp32. Routed experts are served from the offload cache
    (:func:`load_gguf_k_expert_sources`), so they are skipped here.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    assert not include_moe_experts, (
        "lfm2_moe GGUF stores the routed experts as native K-quant banks and only supports the "
        "offload backend; they load through load_gguf_k_expert_sources()."
    )
    assert include_non_moe
    _require_tp1("weight loading")

    seen: dict[int, set[str]] = {}
    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name in ("token_embd_norm.weight", "output_norm.weight"):
            yield "model.embedding_norm.weight", _to_bf16(t)
            continue
        if name == "output.weight":  # untied head (no released LFM2-MoE GGUF ships one)
            yield "lm_head.qweight", t.packed()
            continue
        if name == "rope_freqs.weight":
            continue  # rope frequencies are recomputed in-engine
        if not name.startswith("blk."):
            raise ValueError(f"unmapped lfm2moe GGUF tensor: {name}")
        if any(name.endswith(sfx) for sfx in _EXPERT_SUFFIXES):
            continue  # routed experts -> the offload banks
        layer = int(name.split(".")[1])
        suffix = name.split(".", 2)[2]
        entry = _LAYER_MAP.get(suffix)
        if entry is None:
            raise ValueError(f"unmapped lfm2moe GGUF tensor: {name}")
        rel, role = entry
        base = f"model.layers.{layer}"
        seen.setdefault(layer, set()).add(suffix)
        if role == "qweight":
            yield f"{base}.{rel}.qweight", t.packed()
        elif role == "depthwise":
            # ggml stores [K, conv_dim] -> torch [conv_dim, K]; the op wants [conv_dim, 1, K]
            yield f"{base}.{rel}", _to_bf16(t).reshape(t.shape[0], 1, t.shape[1])
        elif role == "bias32":
            from freetoken.models.gguf.dequant import dequantize

            assert t.ggml_type == 0, f"{name} is {GGML_NAME[t.ggml_type]}, expected F32"
            yield f"{base}.{rel}", dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32)
        else:
            yield f"{base}.{rel}", _to_bf16(t)

    _check_layers_complete(seen)


def _check_layers_complete(seen: dict[int, set[str]]) -> None:
    """Every layer must yield its norms, mixer and feed-forward: a missing tensor here would
    leave a module on its uninitialized allocation."""
    missing: list[str] = []
    conv_suffixes = {k for k, v in _LAYER_MAP.items() if v[0].startswith("conv.")}
    attn_suffixes = {k for k, v in _LAYER_MAP.items() if v[0].startswith("self_attn.")}
    for layer, suffixes in sorted(seen.items()):
        norms = {"attn_norm.weight", "ffn_norm.weight"}
        mixer = conv_suffixes if suffixes & conv_suffixes else attn_suffixes
        ffn = (
            {"ffn_gate_inp.weight", "exp_probs_b.bias"}
            if suffixes & {"ffn_gate_inp.weight", "exp_probs_b.bias"}
            else {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}
        )
        want = norms | mixer | ffn
        if suffixes != want:
            missing.append(
                f"layer {layer}: missing {sorted(want - suffixes)}, unexpected {sorted(suffixes - want)}"
            )
    if missing:
        raise ValueError("incomplete lfm2moe GGUF checkpoint: " + "; ".join(missing))


# --------------------------------------------------------------------------------------
# Model layer swap: dense bf16 Linear/Embedding -> native GGUF-quant ops.
# --------------------------------------------------------------------------------------


def is_gguf_model(config: ModelConfig) -> bool:
    """True when the model was parsed from a GGUF checkpoint (native-quant path)."""
    return getattr(config, "moe_weight_format", None) == "gguf_k"


def _qtype(types: dict[str, int], key: str) -> int:
    try:
        return types[key]
    except KeyError:
        raise ValueError(f"GGUF type table has no {key!r}; the checkpoint is not an LFM2-MoE GGUF") from None


def convert_lfm2_moe_to_gguf(model, config: ModelConfig) -> None:
    """In place: replace the projections the GGUF quantizes with native GGUF ops.

    Quantized in the checkpoint -> swapped: every attention projection, the short-conv
    in/out projections, the dense MLP of the leading layers, and the token embedding (also
    the tied LM head). Left dense bf16/fp32: the router gate and its expert bias, every
    RMSNorm, the depthwise conv kernel, and the routed experts (offload banks).
    """
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear, GGUFLinearHead, GGUFTiedLMHead

    types = dict(config.gguf_tensor_types)

    def swap(owner, attr: str, key: str) -> None:
        lin = getattr(owner, attr)
        out_features, in_features = lin.weight.shape
        setattr(
            owner,
            attr,
            GGUFLinear(
                in_features,
                out_features,
                _qtype(types, key),
                has_bias=lin.bias is not None,
            ),
        )

    inner = model.model
    embed_type = _qtype(types, "token_embd.weight")
    embed = GGUFEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        quant_type=embed_type,
    )
    inner.embed_tokens = embed

    for layer_id, layer in enumerate(inner.layers.op_list):
        blk = f"blk.{layer_id}"
        if hasattr(layer, "self_attn"):
            for attr, key in (
                ("q_proj", "attn_q.weight"),
                ("k_proj", "attn_k.weight"),
                ("v_proj", "attn_v.weight"),
                ("out_proj", "attn_output.weight"),
            ):
                swap(layer.self_attn, attr, f"{blk}.{key}")
        else:
            swap(layer.conv, "in_proj", f"{blk}.shortconv.in_proj.weight")
            swap(layer.conv, "out_proj", f"{blk}.shortconv.out_proj.weight")
        if layer_id >= config.first_k_dense_replace:
            continue
        for attr, key in (("w1", "ffn_gate.weight"), ("w3", "ffn_up.weight"), ("w2", "ffn_down.weight")):
            swap(layer.feed_forward, attr, f"{blk}.{key}")

    if config.tie_word_embeddings:
        model.lm_head = GGUFTiedLMHead(embed, embed_type)
    else:
        # an untied head has its own packed rows (no released LFM2-MoE GGUF ships one)
        out_type = _qtype(types, "output.weight")
        model.lm_head = GGUFLinearHead(
            config.hidden_size, config.vocab_size, out_type, has_bias=False
        )


# --------------------------------------------------------------------------------------
# Routed-expert host banks (native K-quant) for the offload cache.
# --------------------------------------------------------------------------------------


def expert_types_of_layer(types: dict[str, int], layer: int) -> tuple[int, int]:
    """``(gate_up, down)`` ggml types of one *checkpoint* layer's routed experts.

    The banks fuse ``ffn_gate_exps`` and ``ffn_up_exps`` into one gate_up bank, so both must
    share a row geometry, and the borrowed ggml MoE kernels dispatch on Q4_K / Q6_K only.
    """
    gate = _qtype(types, f"blk.{layer}.ffn_gate_exps.weight")
    up = _qtype(types, f"blk.{layer}.ffn_up_exps.weight")
    if gate != up:
        raise ValueError(
            f"blk.{layer}: gate experts are {GGML_NAME[gate]} but up experts are "
            f"{GGML_NAME[up]}; the fused gate_up bank needs one row geometry"
        )
    down = _qtype(types, f"blk.{layer}.ffn_down_exps.weight")
    bad = [GGML_NAME[t] for t in (gate, down) if t not in _EXPERT_TYPES]
    if bad:
        raise ValueError(
            f"blk.{layer}: the expert banks serve {[GGML_NAME[t] for t in _EXPERT_TYPES]} "
            f"rows, got {bad}"
        )
    return gate, down


def expert_layer_types(config: ModelConfig) -> list[tuple[int, int]]:
    """Per-MoE-layer ``(gate_up, down)`` ggml types from the config's type table.

    The banks are indexed by MoE-layer id: the leading dense layers own no experts.
    """
    types = dict(config.gguf_tensor_types)
    return [
        expert_types_of_layer(types, i + config.first_k_dense_replace)
        for i in range(config.num_moe_layers)
    ]


def _bank_shapes(
    config: ModelConfig, types: list[tuple[int, int]]
) -> list[dict[str, tuple[tuple[int, ...], torch.dtype]]]:
    """Per-MoE-layer ``{bank: (shape, dtype)}``, rows padded to a 16-byte multiple."""
    experts, hidden = config.num_experts, config.hidden_size
    inter = config.moe_intermediate_size
    shapes = []
    for gate_up_type, down_type in types:
        gate_up_bytes, _ = padded_row_bytes(hidden, gate_up_type)
        down_bytes, _ = padded_row_bytes(inter, down_type)
        shapes.append(
            {
                "gate_up": ((experts, 2 * inter, gate_up_bytes), torch.uint8),
                "down": ((experts, hidden, down_bytes), torch.uint8),
            }
        )
    return shapes


def load_gguf_k_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> tuple[dict[str, list[torch.Tensor]], list[tuple[int, int]]]:
    """Per-layer host banks of the routed experts' native K-quant block rows.

    Returns ``(banks, layer_tags)``: ``gate_up`` is one ``[E, 2I, row_bytes]`` tensor per layer
    (the layer's ``ffn_gate_exps`` rows followed by its ``ffn_up_exps`` rows, the order the
    ggml MoE kernel's activation split expects) and ``down`` one ``[E, H, row_bytes]``;
    each row holds the file's own packed blocks, so nothing is dequantized at load. Rows are
    padded to a 16-byte multiple with whole super-blocks (zero blocks, exactly zero
    contribution), because the offload cache moves rows with 16-byte-granular kernels while a
    Q6_K row of 1536 elements is 1260 bytes. ``layer_tags`` is the per-layer
    ``(gate_up_type, down_type)`` pair, the only place a mixed K-quant file's type is known.

    ``layer_sink=None`` (serving): pin each layer's two banks as they complete via an
    internally-owned :class:`PinPipeline` (or, on a CUDA-less host, allocate the mmap banks but
    never pin -- the CPU executor reads them pageable). ``layer_sink`` given (converter): the
    completion tracker fires into it instead, so the returned tensors are only valid until the
    sink releases them.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import HostBank, LayerCompletionTracker, PinPipeline

    _require_tp1("expert banks")
    tags = expert_layer_types(config)
    shapes = _bank_shapes(config, tags)
    experts, hidden = config.num_experts, config.hidden_size
    inter = config.moe_intermediate_size
    banks: dict[str, list[torch.Tensor]] = {}
    hb: dict[str, list[HostBank] | HostBank] = {}
    for name in ("gate_up", "down"):
        hb[name] = [HostBank(shapes[i][name][0], torch.uint8) for i in range(len(tags))]
        banks[name] = [b.tensor for b in hb[name]]

    first = config.first_k_dense_replace

    def _load(sink) -> None:
        # gate + up + down = three writes complete a layer (gate_up lands in two halves)
        tracker = LayerCompletionTracker(3, hb, sink) if sink is not None else None
        written = {name: [0] * len(tags) for name in banks}
        for t in iter_gguf_tensors(model_path):
            if not t.name.startswith("blk."):
                continue
            suffix = t.name.split(".", 2)[2]
            if suffix not in _EXPERT_SUFFIXES:
                continue
            layer = int(t.name.split(".")[1]) - first
            if not 0 <= layer < len(tags):
                raise ValueError(f"{t.name}: expert tensor outside the MoE layer range")
            packed = t.packed()  # [E * rows, row_bytes] of the file's own blocks
            if suffix == "ffn_down_exps.weight":
                dst = banks["down"][layer]
                # the file's rows are exactly packed; the bank's rows are padded to a 16-byte
                # multiple, so the copy view takes the file's row width against the bank stride
                src_bytes = t.row_bytes
                dst.as_strided(
                    (experts, hidden, src_bytes), (dst.stride(0), dst.stride(1), 1)
                ).copy_(packed.reshape(experts, hidden, src_bytes))
                written["down"][layer] += 1
            else:
                dst = banks["gate_up"][layer]
                src_bytes = t.row_bytes
                row0 = 0 if suffix.startswith("ffn_gate") else inter
                view = dst.as_strided(
                    (experts, inter, src_bytes),
                    (dst.stride(0), dst.stride(1), 1),
                    storage_offset=row0 * dst.stride(1),
                )
                view.copy_(packed.reshape(experts, inter, src_bytes))
                written["gate_up"][layer] += 1
            if tracker is not None:
                tracker.note(layer)
        for name, counts in written.items():
            bad = [l for l, n in enumerate(counts) if n != (2 if name == "gate_up" else 1)]
            if bad:
                raise ValueError(f"{name} banks were not filled for layers {bad}")

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)  # CUDA-less: mmap banks stay pageable, never pinned
    return banks, tags


def dummy_gguf_k_expert_sources(
    config: ModelConfig,
) -> tuple[dict[str, list[torch.Tensor]], list[tuple[int, int]]]:
    """Random K-quant expert banks shaped like :func:`load_gguf_k_expert_sources` output."""
    from freetoken.moe.host_banks import HostBank, pin_banks

    tags = expert_layer_types(config)
    shapes = _bank_shapes(config, tags)
    hb: dict[str, list[HostBank] | HostBank] = {
        name: [HostBank(shapes[i][name][0], torch.uint8) for i in range(len(tags))]
        for name in ("gate_up", "down")
    }
    banks = {name: [b.tensor for b in per_layer] for name, per_layer in hb.items()}
    for per_layer in banks.values():
        for t in per_layer:
            t.view(torch.uint8).random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)
    return banks, tags


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "convert_lfm2_moe_to_gguf",
    "is_gguf_model",
    "expert_layer_types",
    "load_gguf_k_expert_sources",
    "dummy_gguf_k_expert_sources",
]
