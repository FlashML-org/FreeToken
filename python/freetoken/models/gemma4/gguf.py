"""Gemma 4 GGUF adapter: build the FreeToken ``ModelConfig`` from GGUF metadata.

The GGUF checkpoint's geometry is identical to the HF gemma4 model (hybrid
SWA/full attention with ``k_eq_v`` full layers, 128 routed experts + shared MLP,
final-logit softcap), so this produces the *same* ``ModelConfig`` as
``gemma4.config.parse_config`` -- only the source is GGUF KV metadata instead of a
HF config object. transformers' own GGUF->config conversion is rejected by the
gemma4 strict dataclass (per-layer ``num_key_value_heads`` array), so we read the
metadata directly. ``expert_quant`` is set to ``"q4_0"`` to route the routed experts
through the native-Q4_0 offload-cache path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)
from freetoken.models.gguf.dequant import GGML_Q4_0, GGML_Q6_K, dequantize, row_bytes

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _full_rotary_dim(shim: "GgufConfigShim", full_head_dim: int) -> int:
    """Rotated width of the full-attention layers, recovered from ``rope_freqs.weight``.

    llama.cpp writes ``rope.dimension_count = full head_dim`` and carries gemma4's
    partial_rotary_factor as a divisor tensor instead: ``[1.0] * n_rot//2 + [1e30] * rest``,
    which drives the unrotated tail's frequencies to zero (ggml ops.cpp ``theta / ff``).
    Count the untouched entries to get the rotated width back.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    for t in iter_gguf_tensors(shim.model_path):
        if t.name == "rope_freqs.weight":  # F32, so the packed bytes ARE the values
            freqs = t.packed().reshape(-1).view(torch.float32)
            return int((freqs == 1.0).sum().item()) * 2
    # A metadata-only GGUF (an FTW dir's source_metadata.gguf) has no tensor table. Every
    # released gemma4 sets partial_rotary_factor 0.25 on the full layers, and the converter
    # asserts rope_type == "proportional" (llama.cpp conversion/gemma.py).
    return full_head_dim // 4


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata

    def g(key: str):
        val = m.get(f"gemma4.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key gemma4.{key}")
        return val

    def g_opt(key, default=None):
        val = m.get(f"gemma4.{key}")
        return default if val is None else val

    # Dense (E-series E2B/E4B) vs MoE (12B/26B): dense GGUFs carry no expert metadata.
    num_experts = int(g_opt("expert_count", 0) or 0)
    is_moe = num_experts > 0

    num_layers = int(g("block_count"))
    hidden = int(g("embedding_length"))
    num_qo_heads = int(g("attention.head_count"))
    kv_per_layer = g("attention.head_count_kv")  # per-layer list OR a single scalar
    # True -> sliding-window (SWA) layer, False -> full attention. Flatten single-element
    # wrappers the reader may leave on each entry.
    def _flat_bool(x):
        while isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        return bool(x)

    swa_pattern = [_flat_bool(x) for x in g("attention.sliding_window_pattern")]
    assert len(swa_pattern) == num_layers, "sliding_window_pattern length != block_count"

    swa_layer_ids = tuple(i for i, is_swa in enumerate(swa_pattern) if is_swa)
    full_layer_ids = tuple(i for i, is_swa in enumerate(swa_pattern) if not is_swa)

    swa_head_dim = int(g("attention.key_length_swa"))
    full_head_dim = int(g("attention.key_length"))

    def _kv_for(layer_ids):
        if isinstance(kv_per_layer, (list, tuple)):
            return int(kv_per_layer[layer_ids[0]] if layer_ids else kv_per_layer[0])
        return int(kv_per_layer)

    swa_kv = _kv_for(swa_layer_ids)
    full_kv = _kv_for(full_layer_ids)

    # feed_forward_length is a per-layer list on the E-series (KV-shared tail is 2x wide);
    # take the base width and flag double-wide when any layer is larger. Older MoE GGUFs
    # store a single scalar.
    _ffn = g("feed_forward_length")
    _ffn = [int(x) for x in _ffn] if isinstance(_ffn, (list, tuple)) else [int(_ffn)]
    intermediate = min(_ffn)
    use_double_wide_mlp = max(_ffn) > intermediate
    # Per-Layer Embeddings + KV-layer sharing (E-series only; absent on the MoE line).
    ple_hidden = int(g_opt("embedding_length_per_layer_input", 0) or 0)
    num_kv_shared_layers = int(g_opt("attention.shared_kv_layers", 0) or 0)
    # Full layers reuse K as V only when the GGUF ships no separate value projection
    # (older MoE gemma4); the E-series declares value_length and ships attn_v.
    full_k_eq_v = g_opt("attention.value_length") is None

    max_pos = int(g("context_length"))
    full_rotary = RotaryConfig(
        head_dim=full_head_dim,
        rotary_dim=_full_rotary_dim(shim, full_head_dim),
        max_position=max_pos,
        base=float(g("rope.freq_base")),
        # gemma4's full layers are partial-rotary ("proportional"): the tail dims must come
        # out unrotated. llama.cpp expresses that with rope.dimension_count == full head_dim
        # plus a rope_freqs divisor tensor, so a plain "default" rope here would rotate dims
        # that have to be identity (conversion/gemma.py generate_extra_tensors).
        scaling={"rope_type": "proportional"},
    )
    swa_rotary = RotaryConfig(
        head_dim=swa_head_dim,
        rotary_dim=int(g("rope.dimension_count_swa")),
        max_position=max_pos,
        base=float(g("rope.freq_base_swa")),
        scaling=None,
    )

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=full_kv,
        head_dim=full_head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        hidden_act="gelu_tanh",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=full_rotary,
        intermediate_size=intermediate,
        num_experts=num_experts,
        num_experts_per_tok=int(g_opt("expert_used_count", 0) or 0),
        moe_intermediate_size=int(g_opt("expert_feed_forward_length", 0) or 0),
        norm_topk_prob=True,
        model_type="gemma4",
        architectures=list(shim.architectures),
        moe_enabled=is_moe,
        expert_quant="q4_0" if is_moe else "none",
        moe_weight_format="q4_0",
        use_qk_norm=True,
        attn_sm_scale=1.0,
        final_logit_softcapping=float(g("final_logit_softcapping")),
        embedding_scale=float(hidden) ** 0.5,
        # Gemma-4 E-series (dense): Per-Layer Embeddings + KV-layer sharing + double-wide MLP.
        per_layer_hidden_size=ple_hidden,
        per_layer_vocab_size=int(shim.vocab_size) if ple_hidden else 0,
        num_kv_shared_layers=num_kv_shared_layers,
        use_double_wide_mlp=use_double_wide_mlp,
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_layer_ids,
                num_kv_heads=full_kv,
                head_dim=full_head_dim,
                rotary_config=full_rotary,
                # Older MoE gemma4 GGUFs ship no v_proj (k reused as v); the E-series ships attn_v.
                k_eq_v=full_k_eq_v,
            ),
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=swa_layer_ids,
                num_kv_heads=swa_kv,
                head_dim=swa_head_dim,
                rotary_config=swa_rotary,
                sliding_window=int(g("attention.sliding_window")),
            ),
        ),
    )


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken gemma4 module params.
# --------------------------------------------------------------------------------------

# Per-layer 1:1 norm/scale tensors (gguf suffix -> freetoken module-relative name). All
# are tiny F32 tensors dequantized to bf16.
_LAYER_SCALAR_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "pre_ffw_norm_2.weight": "feed_forward.pre_feedforward_layernorm_2.weight",
    "post_ffw_norm.weight": "feed_forward.post_feedforward_layernorm.weight",
    "post_ffw_norm_1.weight": "feed_forward.post_feedforward_layernorm_1.weight",
    "post_ffw_norm_2.weight": "feed_forward.post_feedforward_layernorm_2.weight",
    "layer_output_scale.weight": "feed_forward.layer_scalar",
    # E-series Per-Layer-Embedding step (post_norm = the post-PLE RMSNorm).
    "post_norm.weight": "post_per_layer_input_norm.weight",
    "ffn_gate_inp.weight": "feed_forward.router.proj.weight",
    "ffn_gate_inp.scale": "feed_forward.router.scale",
    "ffn_down_exps.scale": "feed_forward.router.per_expert_scale",
}
# Expert weight tensors (handled by the offload bank loader, not iter_gguf_weights).
_EXPERT_SUFFIXES = ("ffn_gate_up_exps.weight", "ffn_down_exps.weight")


def _to_bf16(t) -> torch.Tensor:
    """Dequantize a GgufTensor (F32/F16/Q*) to a dense bf16 tensor of its torch shape."""
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16)
    return flat.reshape(t.shape)


def _require_tp1(what: str) -> None:
    """GGUF quant layers / expert banks are not sharded; reject TP>1 with a clear
    error instead of failing later on a confusing shape mismatch (mirrors the HF
    gemma4 loader's TP=1 restriction)."""
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(
            f"gemma4 GGUF {what} currently supports TP=1 only "
            "(GGUF quant layers and expert banks are not tensor-parallel sharded)."
        )


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every non-expert gemma4 param.

    Quantized projections (attention q/k/v/o, shared-MLP gate/up/down) and the
    embedding stay in their native packed block layout and are yielded as ``.qweight``
    (uint8); norms / router / per-layer scalars dequantize to bf16. q/k/v and gate/up
    are fused by concatenating packed rows along the output dim (same input dim ->
    same ``row_bytes``). Routed experts are served from the offload cache, so they are
    skipped here (asserts the offload contract like the other MoE models).
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    config = parse_gguf_config(cached_load_hf_config(model_path))
    # MoE gemma4 keeps its Q4_0 experts in the offload cache (include_moe_experts must be
    # False). Dense E-series has no experts, so the flag is irrelevant there.
    assert not (config.moe_enabled and include_moe_experts), (
        "gemma4 MoE GGUF experts are served from the offload cache "
        "(include_moe_experts must be False)."
    )
    assert include_non_moe
    _require_tp1("weight loading")

    # Full-attention layers ship no attn_v (k reused as v); SWA layers do. Knowing which
    # is which lets us emit the fused qkv as soon as its parts are present.
    k_eq_v_layers = {
        lid
        for lid in range(config.num_layers)
        if isinstance(config.attention_group_for_layer(lid), FullAttentionGroupConfig)
        and config.attention_group_for_layer(lid).k_eq_v
    }
    # E-series KV-shared tail layers carry q only (k/v reused from another layer).
    kv_shared_layers = {
        lid for lid in range(config.num_layers) if config.is_kv_shared_layer(lid)
    }

    # Per-layer fusion buffers: layer -> {slot: packed[out, row_bytes]}.
    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}
    gate_up_buf: dict[int, dict[str, torch.Tensor]] = {}

    def layer_of(name: str) -> int:
        return int(name.split(".")[1])

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()  # Q6_K packed table
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(t)
            continue
        if name == "rope_freqs.weight":
            continue  # rope frequencies recomputed in-engine
        # E-series Per-Layer Embeddings (top-level tensors). The big per-layer table stays a
        # packed Q6_K GGUFEmbedding on GPU (~1.9 GB); the small projection/norm dequant to bf16.
        if name == "per_layer_token_embd.weight":
            yield "model.embed_tokens_per_layer.qweight", t.packed()
            continue
        if name == "per_layer_model_proj.weight":
            yield "model.per_layer_model_projection.weight", _to_bf16(t)
            continue
        if name == "per_layer_proj_norm.weight":
            yield "model.per_layer_projection_norm.weight", _to_bf16(t)
            continue
        if not name.startswith("blk."):
            continue
        if any(name.endswith(sfx) for sfx in _EXPERT_SUFFIXES):
            continue  # routed experts -> offload banks

        layer = layer_of(name)
        suffix = name.split(".", 2)[2]  # after "blk.N."
        base = f"model.layers.{layer}"

        # E-series KV-shared layers: reuse another layer's K/V -> the model has no k/v/k_norm
        # here. Drop those tensors and yield q_proj standalone (not fused into qkv).
        if layer in kv_shared_layers:
            if suffix in ("attn_k.weight", "attn_v.weight", "attn_k_norm.weight"):
                continue
            if suffix == "attn_q.weight":
                yield f"{base}.self_attn.q_proj.qweight", t.packed()
                continue

        # E-series Per-Layer-Embedding projections (tiny Q4_0 -> bf16 LinearReplicated).
        if suffix == "inp_gate.weight":
            yield f"{base}.per_layer_input_gate.weight", _to_bf16(t)
            continue
        if suffix == "proj.weight":
            yield f"{base}.per_layer_projection.weight", _to_bf16(t)
            continue

        if suffix in _LAYER_SCALAR_MAP:
            rel = _LAYER_SCALAR_MAP[suffix]
            tensor = _to_bf16(t)
            if rel.endswith("layer_scalar"):
                tensor = tensor.reshape(1)
            yield f"{base}.{rel}", tensor
            continue

        # Quantized projections: keep packed; fuse q/k/v and gate/up.
        if suffix == "attn_q.weight":
            qkv_buf.setdefault(layer, {})["q"] = t.packed()
        elif suffix == "attn_k.weight":
            qkv_buf.setdefault(layer, {})["k"] = t.packed()
        elif suffix == "attn_v.weight":
            qkv_buf.setdefault(layer, {})["v"] = t.packed()
        elif suffix == "ffn_gate.weight":
            gate_up_buf.setdefault(layer, {})["gate"] = t.packed()
        elif suffix == "ffn_up.weight":
            gate_up_buf.setdefault(layer, {})["up"] = t.packed()
        elif suffix == "attn_output.weight":
            yield f"{base}.self_attn.o_proj.qweight", t.packed()
        elif suffix == "ffn_down.weight":
            yield f"{base}.feed_forward.shared_mlp.down_proj.qweight", t.packed()
        else:
            raise ValueError(f"unmapped gemma4 GGUF tensor: {name}")

        # Emit fused qkv once all parts are present: k_eq_v full layers reuse k as v
        # (no attn_v tensor), SWA layers wait for their separate attn_v.
        slots = qkv_buf.get(layer)
        if slots is not None and "q" in slots and "k" in slots:
            if layer in k_eq_v_layers:
                v = slots["k"]
            elif "v" in slots:
                v = slots["v"]
            else:
                v = None
            if v is not None:
                yield f"{base}.self_attn.qkv_proj.qweight", torch.cat(
                    [slots["q"], slots["k"], v], dim=0
                )
                del qkv_buf[layer]
        gu = gate_up_buf.get(layer)
        if gu is not None and "gate" in gu and "up" in gu:
            yield f"{base}.feed_forward.shared_mlp.gate_up_proj.qweight", torch.cat(
                [gu["gate"], gu["up"]], dim=0
            )
            del gate_up_buf[layer]

    assert not qkv_buf, f"incomplete qkv groups: {sorted(qkv_buf)}"
    assert not gate_up_buf, f"incomplete gate_up groups: {sorted(gate_up_buf)}"


# --------------------------------------------------------------------------------------
# Model layer swap: dense bf16 Linear/Embedding -> native GGUF-quant ops.
# --------------------------------------------------------------------------------------


def is_gguf_model(config: ModelConfig) -> bool:
    """True when the model was parsed from a GGUF checkpoint (native-quant path)."""
    return getattr(config, "moe_weight_format", None) == "q4_0"


class GGUFTiedLMHead:
    """Tied LM head over a native Q6_K embedding table (logits via ggml matmul).

    Holds only a reference to the GGUF embedding (no params of its own -> empty
    state_dict), mirroring ``ParallelLMHead`` with ``tie_word_embeddings``. TP=1 only.
    """

    def __init__(self, embedding, quant_type: int):
        self._embedding = embedding
        self._quant_type = quant_type

    def state_dict(self, *, prefix: str = "", result=None):
        return result if result is not None else {}

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False):
        state_dict.pop(f"{prefix}.weight", None)
        state_dict.pop(f"{prefix}.bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx
        from freetoken.layers.gguf import fused_mul_mat_gguf

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fused_mul_mat_gguf(x, self._embedding.qweight, self._quant_type)


def convert_gemma4_to_gguf(model, config: ModelConfig) -> None:
    """In place: replace gemma4's dense projections + embedding with native GGUF ops.

    Quantized in the checkpoint -> swapped: attention qkv/o, shared-MLP gate_up/down
    (all Q4_0) and the token embedding (Q6_K, also the tied LM head). Left as dense
    bf16 (F32 in the GGUF): the router gate, all RMSNorms, the per-layer scalars, and
    the routed experts (served from the offload cache).
    """
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear

    def swap_linear(owner, attr, quant_type=GGML_Q4_0):
        lin = getattr(owner, attr)
        out_features, in_features = lin.weight.shape
        setattr(
            owner,
            attr,
            GGUFLinear(in_features, out_features, quant_type, has_bias=lin.bias is not None),
        )

    inner = model.model
    embed = GGUFEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        quant_type=GGML_Q6_K,
        embed_scale=config.embedding_scale,
    )
    inner.embed_tokens = embed

    # E-series: the per-layer-embedding table is a packed Q6_K GGUFEmbedding on GPU (~1.9 GB)
    # instead of the bf16 host-RAM CpuPerLayerEmbedding used for the safetensors path.
    if getattr(inner, "_has_ple", False):
        inner.embed_tokens_per_layer = GGUFEmbedding(
            num_embeddings=config.per_layer_vocab_size,
            embedding_dim=config.num_layers * config.per_layer_hidden_size,
            quant_type=GGML_Q6_K,
            embed_scale=float(config.per_layer_hidden_size) ** 0.5,
        )

    for layer_id, layer in enumerate(inner.layers.op_list):
        # KV-shared layers carry a standalone q_proj (no fused qkv); others carry qkv_proj.
        swap_linear(layer.self_attn, "q_proj" if config.is_kv_shared_layer(layer_id) else "qkv_proj")
        swap_linear(layer.self_attn, "o_proj")
        swap_linear(layer.feed_forward.shared_mlp, "gate_up_proj")
        swap_linear(layer.feed_forward.shared_mlp, "down_proj")

    if config.tie_word_embeddings:
        model.lm_head = GGUFTiedLMHead(embed, GGML_Q6_K)


# --------------------------------------------------------------------------------------
# Routed-expert host banks (native Q4_0) for the offload cache.
# --------------------------------------------------------------------------------------

def _q4_0_expert_specs(config: ModelConfig) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    E = config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    return {
        "gate_up": ((E, 2 * I, row_bytes(H, GGML_Q4_0)), torch.uint8),
        "down": ((E, H, row_bytes(I, GGML_Q4_0)), torch.uint8),
    }


def load_q4_0_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Per-layer host banks of the routed experts' native Q4_0 block bytes.

    ``gate_up`` is one ``[E, 2I, H//32*18]`` tensor per layer and ``down`` one
    ``[E, H, I//32*18]`` per layer (independent :class:`HostBank` allocations) -- each
    expert's packed rows verbatim from the GGUF (no dequant), whole layers arriving in
    one shot (gate_up + down = 2 writes/layer), so the offload cache streams whole
    experts to the ggml MoE kernels.

    ``layer_sink=None`` (serving): pin each layer's two banks as they complete via an
    internally-owned :class:`PinPipeline` (or, on a CUDA-less host, allocate the mmap
    banks but never pin -- the CPU executor reads them pageable). ``layer_sink`` given
    (converter): the completion tracker fires into it instead -- nothing here is pinned,
    and the sink may release banks it has written out, so the returned tensors are only
    valid until then (the caller owns that tradeoff).
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    _require_tp1("expert banks")
    L, E = config.num_layers, config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    h_bytes, i_bytes = row_bytes(H, GGML_Q4_0), row_bytes(I, GGML_Q4_0)
    hb = alloc_layer_banks(_q4_0_expert_specs(config), L)  # lazy anon mmaps (unpinned)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    seen_gu, seen_dn = set(), set()

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(2, hb, sink) if sink is not None else None  # gate_up + down
        for t in iter_gguf_tensors(model_path):
            if not t.name.startswith("blk."):
                continue
            layer = int(t.name.split(".")[1])
            if t.name.endswith("ffn_gate_up_exps.weight"):
                banks["gate_up"][layer].copy_(t.packed().reshape(E, 2 * I, h_bytes))
                seen_gu.add(layer)
            elif t.name.endswith("ffn_down_exps.weight"):
                banks["down"][layer].copy_(t.packed().reshape(E, H, i_bytes))
                seen_dn.add(layer)
            else:
                continue
            if tracker is not None:
                tracker.note(layer)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)  # CUDA-less: mmap banks stay pageable, never pinned

    want = set(range(L))
    assert seen_gu == want and seen_dn == want, (
        f"missing Q4_0 expert layers: gate_up {sorted(want - seen_gu)}, "
        f"down {sorted(want - seen_dn)}"
    )
    return banks


def dummy_q4_0_expert_sources(config: ModelConfig) -> dict[str, list[torch.Tensor]]:
    """Random Q4_0 expert banks shaped like ``load_q4_0_expert_sources`` output."""
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    L = config.num_layers
    hb = alloc_layer_banks(_q4_0_expert_specs(config), L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for t in banks["gate_up"] + banks["down"]:
        t.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)  # match the other dummies: pin-after-fill (no-op mmap fill on CPU-only)
    return banks


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "convert_gemma4_to_gguf",
    "is_gguf_model",
    "load_q4_0_expert_sources",
    "dummy_q4_0_expert_sources",
]
