"""Qwen3.5-MoE GGUF adapter: build the FreeToken ``ModelConfig`` and stream weights
from a llama.cpp ``qwen35moe`` checkpoint.

The geometry is identical to the HF qwen3_5_moe model (hybrid GDN/full attention on a
``full_attention_interval`` stride, 256 routed experts + a shared expert, NextN/MTP head),
so this produces the *same* ``ModelConfig`` as ``qwen3_5_moe.config.parse_config`` -- only
the source is GGUF KV metadata instead of a HF config object.

Tensor-name mapping is the inverse of llama.cpp's ``gguf-py/gguf/tensor_mapping.py``.
The one non-obvious part is the GDN projections. llama.cpp's *qwen3.5* mapping splits
what qwen3next fused::

    attn_qkv    <- model.layers.{i}.linear_attn.in_proj_qkv
    attn_gate   <- model.layers.{i}.linear_attn.in_proj_z
    ssm_beta    <- model.layers.{i}.linear_attn.in_proj_b
    ssm_alpha   <- model.layers.{i}.linear_attn.in_proj_a

FreeToken's HF loader already knows how to put those back together -- see ``_PT_FP8_FUSE``
and ``_PT_BF16_FUSE`` in ``weight.py``, which fuse ``(in_proj_qkv, in_proj_z) ->
in_proj_qkvz`` and ``(in_proj_b, in_proj_a) -> in_proj_ba`` in that order. We emit the
same fused buffers here so the model code sees one representation regardless of source.

Verified against vcruz305/Ornith-1.5-35B-A3B-GGUF (IQ3_M), whose metadata gives
block_count=41 (40 decoder layers + 1 NextN block), embedding_length=2048,
head_count=16, head_count_kv=2, key_length=value_length=256, expert_count=256,
expert_used_count=8, expert_feed_forward_length=512, full_attention_interval=4,
ssm.conv_kernel=4, ssm.state_size=128, ssm.group_count=16, ssm.time_step_rank=32,
ssm.inner_size=4096. Those are self-consistent: the packed ``attn_qkv`` output width of
8192 is exactly q(16*128) + k(16*128) + v(32*128), i.e. num_k_heads == group_count and
num_v_heads == time_step_rank, both with head_dim == state_size == 128.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

import torch

# Verify that LinearGatedDeltaGroupConfig is available for isinstance checks
# (imported above in the config module import)

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.gguf.dequant import (
    GGML_UNQUANTIZED as GGML_UNQUANTIZED_SET,
    GGML_IQ3_S,
    GGML_NAME,
    GGML_Q4_K,
    GGML_Q6_K,
    dequantize,
    row_bytes,
)

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "qwen35moe"


def _kv(shim: "GgufConfigShim", key: str, default: Any = None) -> Any:
    """Read ``<arch>.<key>`` from the GGUF metadata.

    The prefix is the checkpoint's own ``general.architecture``: "qwen35moe" for the MoE
    variant, "qwen35" for the dense one (e.g. Qwen3.8-27B). Same geometry keys either way.
    """
    val = shim.metadata.get(f"{shim.model_type}.{key}", default)
    if val is None and default is None:
        raise ValueError(
            f"GGUF {shim.model_path}: missing required key {shim.model_type}.{key}"
        )
    return val


def _uniform_expert_types(model_path: str, num_layers: int) -> tuple[int, int] | None:
    """``(gate_up, down)`` ggml types of the routed-expert banks, or None if not uniform.

    The offload slot pool is one allocation per bank shared by every layer, and
    ``moe_vec.cuh`` addresses it as ``expert * nrows * (ncols / qk)`` with no padding
    allowance -- so a bank whose type varies by layer cannot be served. We return None
    rather than raising here because ``parse_gguf_config`` also runs for metadata-only
    inspection; ``expert_banks._gguf_banks`` is where the load actually fails, with the
    offending layers named. (llama.cpp's *_M mixes hit this: Ornith IQ3_M splits
    ffn_down_exps across Q4_K and IQ3_S, while IQ3_S / IQ3_XXS are uniform.)
    """
    from .gguf_experts import gguf_expert_types

    try:
        types = gguf_expert_types(model_path, num_layers)
    except Exception:
        return None
    gate_up, down = set(types["gate_up"]), set(types["down"])
    if len(gate_up) != 1 or len(down) != 1:
        return None
    return (next(iter(gate_up)), next(iter(down)))


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    block_count = int(_kv(shim, "block_count"))
    # llama.cpp appends the NextN/MTP block to the decoder stack. FreeToken serves
    # text-only without speculative decoding, so the MTP block is not a decoder layer.
    nextn = int(_kv(shim, "nextn_predict_layers", 0))
    num_layers = block_count - nextn

    hidden_size = int(_kv(shim, "embedding_length"))
    num_qo_heads = int(_kv(shim, "attention.head_count"))
    num_kv_heads = int(_kv(shim, "attention.head_count_kv"))
    head_dim = int(_kv(shim, "attention.key_length"))
    rms_eps = float(_kv(shim, "attention.layer_norm_rms_epsilon"))
    rope_base = float(_kv(shim, "rope.freq_base"))
    rotary_dim = int(_kv(shim, "rope.dimension_count"))
    max_pos = int(_kv(shim, "context_length"))

    # Dense variants (qwen35, e.g. Qwen3.8-27B) carry no expert_* keys at all: every
    # decoder layer gets a plain SwiGLU MLP sized by feed_forward_length instead of the
    # routed block plus shared expert.
    num_experts = int(_kv(shim, "expert_count", 0))
    experts_per_tok = int(_kv(shim, "expert_used_count", 0))
    moe_inter = int(_kv(shim, "expert_feed_forward_length", 0))
    shared_inter = int(_kv(shim, "expert_shared_feed_forward_length", 0))
    dense_inter = int(_kv(shim, "feed_forward_length", 0))
    moe_enabled = num_experts > 0

    # GDN geometry. state_size is the per-head dim; group_count is the number of k heads
    # and time_step_rank the number of v heads (see module docstring for the arithmetic
    # that pins this down against the packed attn_qkv width).
    conv_kernel = int(_kv(shim, "ssm.conv_kernel"))
    state_size = int(_kv(shim, "ssm.state_size"))
    num_k_heads = int(_kv(shim, "ssm.group_count"))
    num_v_heads = int(_kv(shim, "ssm.time_step_rank"))
    inner_size = int(_kv(shim, "ssm.inner_size"))
    if num_v_heads * state_size != inner_size:
        raise ValueError(
            f"GGUF {shim.model_path}: ssm.time_step_rank({num_v_heads}) * "
            f"ssm.state_size({state_size}) != ssm.inner_size({inner_size}); the GDN head "
            "layout assumed by this adapter does not hold for this checkpoint"
        )

    # llama.cpp writes the stride, not a per-layer list: layer i is full attention when
    # (i + 1) % interval == 0. For Ornith (interval=4, 40 layers) that is 3,7,...,39.
    interval = int(_kv(shim, "full_attention_interval"))
    full_ids = tuple(i for i in range(num_layers) if (i + 1) % interval == 0)
    linear_ids = tuple(i for i in range(num_layers) if i not in set(full_ids))

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_pos,
        base=rope_base,
        scaling=None,
    )
    groups = tuple(
        sorted(
            (
                FullAttentionGroupConfig(
                    name="full",
                    layer_ids=full_ids,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    rotary_config=full_rotary,
                ),
                LinearGatedDeltaGroupConfig(
                    name="linear",
                    layer_ids=linear_ids,
                    num_key_heads=num_k_heads,
                    num_value_heads=num_v_heads,
                    key_head_dim=state_size,
                    value_head_dim=state_size,
                    conv_kernel_dim=conv_kernel,
                    output_gate=True,
                ),
            ),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=shim.vocab_size,
        intermediate_size=0 if moe_enabled else dense_inter,
        hidden_act="silu",
        rms_norm_eps=rms_eps,
        tie_word_embeddings=shim.tie_word_embeddings,
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=experts_per_tok,
        moe_intermediate_size=moe_inter,
        shared_expert_intermediate_size=shared_inter,
        norm_topk_prob=True,
        moe_enabled=moe_enabled,
        use_qk_norm=True,
        model_type=shim.model_type,
        architectures=list(shim.architectures),
        vision_config=None,
        image_token_id=None,
        attention_groups=groups,
        # Only the MoE variant has offload expert banks; a dense checkpoint must not
        # advertise expert_quant="gguf" or the engine would go looking for banks that do
        # not exist. is_gguf_model() keys on gguf_model_path instead, so the op swap still
        # runs for both.
        expert_quant="gguf" if moe_enabled else "none",
        gguf_expert_types=(
            _uniform_expert_types(shim.model_path, num_layers) if moe_enabled else None
        ),
        gguf_model_path=shim.model_path,
        weight_block_size=None,
        attn_quant="gguf",
        dense_quant="gguf",
        lm_head_quant="gguf",
    )


# --------------------------------------------------------------------------------------
# Tensor-name mapping (inverse of llama.cpp gguf-py/gguf/tensor_mapping.py for qwen3.5)
# --------------------------------------------------------------------------------------

# Per-layer 1:1 renames that need no reshaping or fusing.
_LAYER_MAP: dict[str, str] = {
    # shared by both layer kinds
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    # full-attention layers. attn_q/attn_k/attn_v are deliberately absent: the model has
    # no q_proj/k_proj/v_proj attributes -- Qwen3_5Attention builds one merged qkv_proj
    # (_qkv_split = [8192, 512, 512]) -- so they are fused by iter_gguf_weights, not renamed.
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    # GDN (linear-attention) layers
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    # MoE router + shared expert. ffn_gate_shexp / ffn_up_shexp are absent for the same
    # reason as attn_q/k/v: _SharedExpert has a single merged gate_up_proj, so they are
    # fused rather than renamed.
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
}

# Suffixes that are PARTS of a merged projection: never renamed 1:1, always combined by
# iter_gguf_weights into the merged buffer the model actually declares. Listed here so
# gguf_name_to_freetoken can report them as "handled elsewhere" (None) instead of
# inventing a parameter name that does not exist on the module.
#
# The merged targets and their concat orders:
#   self_attn.qkv_proj   <- attn_q, attn_k, attn_v          (_qkv_split [8192, 512, 512])
#   linear_attn.in_proj  <- attn_qkv, attn_gate, ssm_beta, ssm_alpha
#                           (_in_proj_split [conv_dim, value_dim, n_v, n_v])
#   mlp.shared_expert.gate_up_proj <- ffn_gate_shexp, ffn_up_shexp
#
# NOTE the GDN target is ``in_proj``, not ``in_proj_qkvz``/``in_proj_ba``: gdn.py only
# splits those two out on the fp8 branch (``self._fp8``), and a GGUF checkpoint sets
# attn_quant="gguf", so the single fused in_proj is what exists.
_MERGED_PARTS: frozenset[str] = frozenset({
    "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
})

# Routed-expert stacks: [num_experts, out, in] packed blocks, handled by the offload
# expert-bank loader rather than yielded as ordinary parameters.
_EXPERT_SUFFIXES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)

_GLOBAL_MAP: dict[str, str] = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def gguf_name_to_freetoken(name: str, num_layers: int) -> str | None:
    """Map one llama.cpp tensor name to its FreeToken parameter name.

    Returns ``None`` for anything this function does not rename 1:1 -- the NextN/MTP
    block, the routed-expert stacks (read directly by the expert-bank loader), and the
    parts of a merged projection (combined by :func:`iter_gguf_weights`, which is the
    only place that knows the concat order and the per-part quant types). Callers that
    want full coverage accounting should treat ``None`` as "handled elsewhere", not
    "unmapped": returning an invented ``q_proj``/``gate_proj`` name for a fusion part
    would name an attribute the module does not have.
    """
    if name in _GLOBAL_MAP:
        return _GLOBAL_MAP[name]
    if not name.startswith("blk."):
        return None
    _, idx, suffix = name.split(".", 2)
    layer = int(idx)
    if layer >= num_layers:
        return None  # the trailing NextN/MTP block: served text-only, no speculation
    if suffix.startswith("nextn."):
        return None
    if suffix in _EXPERT_SUFFIXES:
        return None
    if suffix in _MERGED_PARTS:
        return None  # fused by iter_gguf_weights into the merged buffer
    mapped = _LAYER_MAP.get(suffix)
    if mapped is None:
        return None
    return f"model.layers.{layer}.{mapped}"


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken qwen35moe module params.
# --------------------------------------------------------------------------------------


def _scan_quant_types(model_path: str) -> dict[tuple[int, str], int]:
    """Scan GGUF tensor table once and return {(layer, suffix): ggml_type}.

    This allows us to detect which groups are mixed-quant without hardcoding.
    Quant levels (IQ2_*, IQ3_XXS, IQ3_M) may mix differently.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    quant_types = {}
    for t in iter_gguf_tensors(model_path):
        if not t.name.startswith("blk."):
            # Globals (token_embd.weight, output.weight, output_norm.weight) keyed under
            # layer -1 so the swap can size the embedding and lm_head from the file too.
            quant_types[(-1, t.name)] = t.ggml_type
            continue
        _, idx, suffix = t.name.split(".", 2)
        layer = int(idx)
        quant_types[(layer, suffix)] = t.ggml_type
    return quant_types


# NOTE on the Gemma-style (1 + weight) RMSNorm convention: do NOT apply it here.
#
# HF Qwen3.5 stores these norm weights as raw ``w`` with an effective scale of ``1 + w``,
# and FreeToken's GemmaRMSNorm multiplies by the raw buffer -- so weight.py adds 1.0 at
# load time (_is_gemma_norm / _GEMMA_NORM_SUFFIXES). llama.cpp's converter ALREADY folds
# the +1 into the tensors it writes, so a GGUF checkpoint arrives pre-shifted and adding
# it again double-counts.
#
# Measured on Ornith-1.5-35B IQ3_S (mean, min, max):
#   blk.3.attn_norm.weight             0.920  0.701  1.660
#   blk.3.post_attention_norm.weight   1.135  0.004  1.279
#   blk.3.attn_q_norm.weight           1.326  0.684  1.883
#   output_norm.weight                 2.640  0.763  3.484
# Raw w would centre near 0.0X; these centre near 1, i.e. already 1+w.



# --------------------------------------------------------------------------------------
# V-head order: llama.cpp writes TILED, FreeToken (like HF) wants GROUPED
# --------------------------------------------------------------------------------------
# When num_k_heads != num_v_heads, llama.cpp's converter reorders every tensor that indexes
# the V-head dimension (conversion/qwen.py, _LinearAttentionVReorderBase) so ggml_repeat can
# replace an interleaved repeat:
#
#   HF / FreeToken (grouped by K head):  [G0_v0, G0_v1, G1_v0, G1_v1, ...]
#   GGUF            (tiled for ggml):    [G0_v0, G1_v0, ..., G0_v1, G1_v1, ...]
#
# FreeToken's GDN pairs K head k with V heads [k*R, (k+1)*R), i.e. grouped -- the safetensors
# loader applies no reorder because HF is already grouped. So a GGUF checkpoint must be
# un-tiled on load or every K/V pairing in all 30 GDN layers is wrong. That is invisible to
# shape, dtype and magnitude checks: activations stay healthy and the text is nonsense.
#
# Ornith: num_k_heads=16, num_v_heads=32 -> num_v_per_k=2, head_v_dim=128.


def _ungroup_v(t: torch.Tensor, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int):
    """Tiled -> grouped along ``dim``: the inverse of llama.cpp's _reorder_v_heads.

    The forward transform views the axis as [K, R, D] and swaps K/R. The inverse is the same
    operation with the two counts exchanged: view as [R, K, D] and swap back.
    """
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape)
    view = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1:]
    out = t.reshape(*view)
    perm = list(range(len(view)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return out.permute(*perm).contiguous().reshape(*shape)


def _ungroup_packed_rows(packed: torch.Tensor, num_k_heads: int, num_v_per_k: int, head_dim: int):
    """Un-tile whole ROWS of a packed [out, row_bytes] tensor.

    Safe on quantized data: each output row is an independent run of blocks over the input
    dim, so permuting rows never splits a block. (Permuting COLUMNS would -- see ssm_out.)
    """
    return _ungroup_v(packed, 0, num_k_heads, num_v_per_k, head_dim)


def _to_bf16(t) -> torch.Tensor:
    """A (1 + weight) norm: dequantize then add 1, matching weight.py's load-time shift."""
    return _to_bf16(t) + 1.0


def _to_bf16(t) -> torch.Tensor:
    """Dequantize a GgufTensor (F32/F16/Q*) to a dense bf16 tensor of its torch shape."""
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16)
    return flat.reshape(t.shape)


def _dequant_any(t) -> torch.Tensor:
    """Dequantize a GgufTensor of ANY ggml type to dense bf16, via the CUDA kernel.

    The pure-torch ``dequantize`` in models/gguf/dequant.py implements only Q4_0 and Q6_K
    (it is the reference/test path). ``ggml_dequantize`` covers all 19 quant types, so use
    it for the one tensor that genuinely has to be materialized dense -- ssm_out, whose
    columns need un-tiling. Round-trips through the GPU; the result is a CPU tensor so the
    normal load path places it.
    """
    from freetoken.kernel.gguf import ggml_dequantize

    if t.ggml_type in GGML_UNQUANTIZED_SET:
        return _to_bf16(t)
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{t.name}: needs dense dequantization of ggml type "
            f"{GGML_NAME.get(t.ggml_type, t.ggml_type)}, which only the CUDA kernel "
            f"implements, but no CUDA device is available"
        )
    out_f, in_f = t.shape[0], t.shape[1]
    packed = t.packed().reshape(out_f, row_bytes(in_f, t.ggml_type)).cuda()
    return ggml_dequantize(packed, t.ggml_type, out_f, in_f, torch.bfloat16).cpu()


def _to_f32(t) -> torch.Tensor:
    """Dequantize a GgufTensor (F32/F16/Q*) to a dense float32 tensor of its torch shape."""
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32)
    return flat.reshape(t.shape)


def _require_tp1(what: str) -> None:
    """GGUF quant layers / expert banks are not sharded; reject TP>1 with a clear
    error instead of failing later on a confusing shape mismatch."""
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(
            f"qwen35moe GGUF {what} currently supports TP=1 only "
            "(GGUF quant layers and expert banks are not tensor-parallel sharded)."
        )


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every non-expert qwen35moe param.

    Quantized projections (attention qkv/o, linear_attn in/out, shared-MLP gate_up/down)
    stay in their native packed block layout and are yielded as ``.qweight`` (uint8) or
    ``.qweight_<i>`` for mixed-quant groups; norms and gates dequantize to bf16. q/k/v,
    attn_qkv/gate/beta/alpha, and gate/up are fused by concatenating packed rows or
    materializing parts separately (GGUFMergedLinear for mixed quants). Routed experts
    are served from the offload cache (asserts the offload contract like the other MoE
    models).

    A_log and dt_bias stay float32 (gdn.py keeps recurrence-gating params in fp32).
    conv1d.weight is bf16 (model dtype) reshaped to [conv_dim, 1, kernel].
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    # Only the MoE variant keeps its routed experts out of this iterator; they come from
    # the offload cache instead. A dense qwen35 checkpoint has no routed experts at all, so
    # the engine legitimately asks for "everything" and the assert must not fire.
    config_moe = int(_kv(cached_load_hf_config(model_path), "expert_count", 0)) > 0
    assert not (config_moe and include_moe_experts), (
        "qwen35moe GGUF keeps its routed experts in the offload cache; they are loaded by "
        "the expert-bank loader, not by iter_gguf_weights."
    )
    assert include_non_moe
    _require_tp1("weight loading")

    # Parse config to determine which layers are full-attention vs GDN.
    config = parse_gguf_config(cached_load_hf_config(model_path))
    full_layer_ids = {
        lid
        for lid in range(config.num_layers)
        if isinstance(config.attention_group_for_layer(lid), FullAttentionGroupConfig)
    }

    # Get GDN group to extract attn_qkv_size and conv_kernel for conv1d reshape.
    gdn_group = None
    for group in config.attention_groups:
        if isinstance(group, LinearGatedDeltaGroupConfig):
            gdn_group = group
            break
    # attn_qkv_size = q + k + v = num_k_heads*state_size*2 + num_v_heads*state_size
    gdn_attn_qkv_size = (
        2 * gdn_group.num_key_heads * gdn_group.key_head_dim + gdn_group.num_value_heads * gdn_group.value_head_dim
        if gdn_group
        else 8192
    )
    gdn_conv_kernel = gdn_group.conv_kernel_dim if gdn_group else 4
    # V-head un-tiling geometry (see _ungroup_v). Only needed when the GDN has fewer K
    # heads than V heads, which is exactly when llama.cpp reorders.
    _vK = gdn_group.num_key_heads if gdn_group else 0
    _vN = gdn_group.num_value_heads if gdn_group else 0
    _vD = gdn_group.value_head_dim if gdn_group else 0
    _vR = (_vN // _vK) if _vK else 1
    _untile = bool(_vK and _vN and _vK != _vN)
    _qk_rows = 2 * gdn_group.num_key_heads * gdn_group.key_head_dim if gdn_group else 0

    # Scan quant types once to determine which fusion groups are mixed-quant.
    quant_map = _scan_quant_types(model_path)

    # Per-layer fusion buffers: layer -> {slot: packed[out, row_bytes]}.
    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}  # full-attn qkv
    in_proj_buf: dict[int, dict[str, torch.Tensor]] = {}  # GDN in_proj (qkv+gate+beta+alpha)
    gate_up_buf: dict[int, dict[str, torch.Tensor]] = {}  # shared_expert gate_up (MoE)
    dense_mlp_buf: dict[int, dict[str, torch.Tensor]] = {}  # mlp gate_up (dense qwen35)

    def layer_of(name: str) -> int:
        return int(name.split(".")[1])

    for t in iter_gguf_tensors(model_path):
        name = t.name
        layer = layer_of(name) if name.startswith("blk.") else None

        # Global tensors
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()  # IQ3_S packed table
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(t)
            continue
        if name == "output.weight":
            # The untied LM head, packed (Q6_K here). Ornith ships output.weight, so
            # tie_word_embeddings is False and lm_head is a real GGUFLinear that must be
            # filled; when a checkpoint omits it the head aliases the embedding table and
            # there is nothing to yield.
            if not config.tie_word_embeddings:
                yield "lm_head.qweight", t.packed()
            continue
        if not name.startswith("blk."):
            continue

        # Skip block 40 (NextN/MTP, dropped) and nextn.* tensors.
        if layer >= config.num_layers:
            continue
        if "nextn." in name:
            continue

        # Skip routed-expert stacks (offload banks).
        if any(name.endswith(sfx) for sfx in _EXPERT_SUFFIXES):
            continue

        suffix = name.split(".", 2)[2]  # after "blk.N."
        base = f"model.layers.{layer}"

        # Scalar/norm tensors: dequant to bf16 or stay F32.
        # norms, mlp.gate, shared_expert_gate -> bf16
        # A_log, dt_bias -> float32
        # conv1d.weight -> float32, reshaped
        if suffix == "attn_norm.weight":
            yield f"{base}.input_layernorm.weight", _to_bf16(t)
            continue
        if suffix == "post_attention_norm.weight":
            yield f"{base}.post_attention_layernorm.weight", _to_bf16(t)
            continue
        if suffix == "ffn_gate_inp.weight":
            yield f"{base}.mlp.gate.weight", _to_bf16(t)
            continue
        if suffix == "ffn_gate_inp_shexp.weight":
            # llama.cpp stores the single-output shared-expert gate as a 1-D [hidden]
            # vector; _SharedExpert's LinearReplicated(hidden, 1) declares [1, hidden].
            yield f"{base}.mlp.shared_expert_gate.weight", _to_bf16(t).reshape(1, -1)
            continue
        if suffix == "ssm_norm.weight":
            yield f"{base}.linear_attn.norm.weight", _to_bf16(t)
            continue
        if suffix == "ssm_a":
            # llama.cpp writes this tensor already transformed: it holds A = -exp(A_log),
            # not A_log. Measured on Ornith IQ3_S every value is negative, spanning
            # [-70.11, -0.0189]. gdn.py computes
            #     g = -A_log.exp() * softplus(a + dt_bias)
            # so handing it A directly gives -exp(-10.6) ~ -2.5e-5 and the recurrent decay
            # gate collapses to zero in all 30 GDN layers -- healthy activation magnitudes,
            # incoherent text. Invert the transform so -exp(A_log) reproduces the stored A.
            a = _to_f32(t)
            if _untile:
                a = _ungroup_v(a, 0, _vK, _vR, 1)
            if not bool((a < 0).all()):
                raise ValueError(
                    f"{name}: expected llama.cpp's pre-transformed A = -exp(A_log) (all "
                    f"negative); got min={float(a.min())} max={float(a.max())}, which would "
                    f"make log(-A) NaN"
                )
            yield f"{base}.linear_attn.A_log", torch.log(-a)
            continue
        if suffix == "ssm_dt.bias":
            dt = _to_f32(t)
            if _untile:
                dt = _ungroup_v(dt, 0, _vK, _vR, 1)
            yield f"{base}.linear_attn.dt_bias", dt
            continue
        if suffix == "ssm_conv1d.weight":
            # F32 in the file, but _DepthwiseConv1d allocates at the model dtype: gdn.py
            # exempts only A_log / dt_bias from the downcast, so this one is bf16. Reshape
            # to [conv_dim, 1, kernel] -- gdn.py's _conv_weight() does .squeeze(1).
            w = _to_bf16(t).reshape(gdn_attn_qkv_size, gdn_conv_kernel)
            if _untile:
                # channels are [q | k | v]; only the V block is tiled
                qk, v = w[:_qk_rows], w[_qk_rows:]
                w = torch.cat([qk, _ungroup_v(v, 0, _vK, _vR, _vD)], dim=0)
            w = w.reshape(gdn_attn_qkv_size, 1, gdn_conv_kernel)
            yield f"{base}.linear_attn.conv1d.weight", w
            continue
        if suffix == "attn_q_norm.weight":
            yield f"{base}.self_attn.q_norm.weight", _to_bf16(t)
            continue
        if suffix == "attn_k_norm.weight":
            yield f"{base}.self_attn.k_norm.weight", _to_bf16(t)
            continue

        # Dense variant (qwen35): a plain SwiGLU MLP per layer instead of the routed block.
        # Qwen3_5DenseMLP subclasses _SharedExpert, so the targets are mlp.gate_up_proj
        # (gate|up fused) and mlp.down_proj. Same placement rule as the shared expert below:
        # these appear on both layer kinds, so they must be consumed before the layer-kind
        # branch whose `else: continue` drops anything it does not recognise.
        if suffix in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
            if suffix == "ffn_down.weight":
                yield f"{base}.mlp.down_proj.qweight", t.packed()
            else:
                dense_mlp_buf.setdefault(layer, {})[
                    "gate" if suffix == "ffn_gate.weight" else "up"
                ] = t.packed()
                d = dense_mlp_buf[layer]
                if "gate" in d and "up" in d:
                    types = [
                        quant_map.get((layer, "ffn_gate.weight")),
                        quant_map.get((layer, "ffn_up.weight")),
                    ]
                    if len(set(types)) == 1:
                        yield f"{base}.mlp.gate_up_proj.qweight", torch.cat(
                            [d["gate"], d["up"]], dim=0
                        )
                    else:
                        yield f"{base}.mlp.gate_up_proj.qweight_0", d["gate"]
                        yield f"{base}.mlp.gate_up_proj.qweight_1", d["up"]
                    del dense_mlp_buf[layer]
            continue

        # Shared expert (present on every layer, both kinds) -- must be handled BEFORE the
        # per-layer-kind branch below, whose `else: continue` swallows any suffix it does
        # not recognize.
        if suffix in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"):
            if suffix == "ffn_gate_shexp.weight":
                gate_up_buf.setdefault(layer, {})["gate"] = t.packed()
            elif suffix == "ffn_up_shexp.weight":
                gate_up_buf.setdefault(layer, {})["up"] = t.packed()
            else:
                yield f"{base}.mlp.shared_expert.down_proj.qweight", t.packed()

            # Emit the fused gate_up once both parts have arrived.
            gu = gate_up_buf.get(layer)
            if gu is not None and "gate" in gu and "up" in gu:
                types = [
                    quant_map.get((layer, "ffn_gate_shexp.weight")),
                    quant_map.get((layer, "ffn_up_shexp.weight")),
                ]
                if len(set(types)) == 1:
                    yield f"{base}.mlp.shared_expert.gate_up_proj.qweight", torch.cat(
                        [gu["gate"], gu["up"]], dim=0
                    )
                else:
                    yield f"{base}.mlp.shared_expert.gate_up_proj.qweight_0", gu["gate"]
                    yield f"{base}.mlp.shared_expert.gate_up_proj.qweight_1", gu["up"]
                del gate_up_buf[layer]
            continue

        # Quantized projections: keep packed; fuse per layer.
        # Full-attention: qkv from q, k, v
        if layer in full_layer_ids:
            if suffix == "attn_q.weight":
                qkv_buf.setdefault(layer, {})["q"] = t.packed()
            elif suffix == "attn_k.weight":
                qkv_buf.setdefault(layer, {})["k"] = t.packed()
            elif suffix == "attn_v.weight":
                qkv_buf.setdefault(layer, {})["v"] = t.packed()
            elif suffix == "attn_output.weight":
                yield f"{base}.self_attn.o_proj.qweight", t.packed()
            else:
                continue  # unmapped for full-attn layers

            # Emit fused qkv once all three parts are present.
            slots = qkv_buf.get(layer)
            if slots is not None and "q" in slots and "k" in slots and "v" in slots:
                # Determine if this is a mixed-quant group.
                types = [
                    quant_map.get((layer, "attn_q.weight")),
                    quant_map.get((layer, "attn_k.weight")),
                    quant_map.get((layer, "attn_v.weight")),
                ]
                if len(set(types)) == 1:
                    # Uniform quant: fuse via torch.cat along dim 0.
                    yield f"{base}.self_attn.qkv_proj.qweight", torch.cat(
                        [slots["q"], slots["k"], slots["v"]], dim=0
                    )
                else:
                    # Mixed quant: emit GGUFMergedLinear format.
                    yield f"{base}.self_attn.qkv_proj.qweight_0", slots["q"]
                    yield f"{base}.self_attn.qkv_proj.qweight_1", slots["k"]
                    yield f"{base}.self_attn.qkv_proj.qweight_2", slots["v"]
                del qkv_buf[layer]

        # GDN layers: in_proj from attn_qkv, attn_gate, ssm_beta, ssm_alpha
        # and out_proj
        else:
            if suffix == "attn_qkv.weight":
                w = t.packed()
                if _untile:
                    # rows are [q | k | v]; only the V rows are tiled. Row permutation is
                    # safe on packed data -- each row is its own run of blocks.
                    w = torch.cat(
                        [w[:_qk_rows], _ungroup_packed_rows(w[_qk_rows:], _vK, _vR, _vD)], dim=0
                    )
                in_proj_buf.setdefault(layer, {})["qkv"] = w
            elif suffix == "attn_gate.weight":
                w = t.packed()
                if _untile:
                    w = _ungroup_packed_rows(w, _vK, _vR, _vD)
                in_proj_buf.setdefault(layer, {})["gate"] = w
            elif suffix in ("ssm_beta.weight", "ssm_alpha.weight"):
                w = t.packed()
                if _untile:
                    # one row per V head -> head_dim 1
                    w = _ungroup_packed_rows(w, _vK, _vR, 1)
                in_proj_buf.setdefault(layer, {})[
                    "beta" if suffix.startswith("ssm_beta") else "alpha"
                ] = w
            elif suffix == "ssm_out.weight":
                # out_proj consumes the V dimension along its COLUMNS, and llama.cpp tiled
                # those columns. A column permutation cannot be done on packed data -- a
                # 128-wide head straddles the 256-element quant blocks -- so this one tensor
                # is dequantized to dense bf16. Cost: out*in*2 bytes per GDN layer
                # (2048*4096*2 = 16 MiB, ~503 MiB over 30 layers). convert_qwen35_to_gguf
                # therefore leaves linear_attn.out_proj as a dense Linear.
                w = _dequant_any(t)
                if _untile:
                    w = _ungroup_v(w, 1, _vK, _vR, _vD)
                yield f"{base}.linear_attn.out_proj.weight", w
            else:
                continue  # unmapped for GDN layers

            # Emit fused in_proj once all four parts are present.
            slots = in_proj_buf.get(layer)
            if (
                slots is not None
                and "qkv" in slots
                and "gate" in slots
                and "beta" in slots
                and "alpha" in slots
            ):
                # Determine if this is a mixed-quant group.
                types = [
                    quant_map.get((layer, "attn_qkv.weight")),
                    quant_map.get((layer, "attn_gate.weight")),
                    quant_map.get((layer, "ssm_beta.weight")),
                    quant_map.get((layer, "ssm_alpha.weight")),
                ]
                if len(set(types)) == 1:
                    # Uniform quant: fuse via torch.cat along dim 0.
                    yield f"{base}.linear_attn.in_proj.qweight", torch.cat(
                        [
                            slots["qkv"],
                            slots["gate"],
                            slots["beta"],
                            slots["alpha"],
                        ],
                        dim=0,
                    )
                else:
                    # Mixed quant: emit GGUFMergedLinear format.
                    yield f"{base}.linear_attn.in_proj.qweight_0", slots["qkv"]
                    yield f"{base}.linear_attn.in_proj.qweight_1", slots["gate"]
                    yield f"{base}.linear_attn.in_proj.qweight_2", slots["beta"]
                    yield f"{base}.linear_attn.in_proj.qweight_3", slots["alpha"]
                del in_proj_buf[layer]

    # Verify no fusion buffers are incomplete.
    assert not qkv_buf, f"incomplete full-attn qkv groups: {sorted(qkv_buf)}"
    assert not in_proj_buf, f"incomplete GDN in_proj groups: {sorted(in_proj_buf)}"
    assert not gate_up_buf, f"incomplete shared_expert gate_up groups: {sorted(gate_up_buf)}"
    assert not dense_mlp_buf, f"incomplete dense mlp gate_up groups: {sorted(dense_mlp_buf)}"


def is_gguf_model(config: ModelConfig) -> bool:
    """True when this config came from a GGUF checkpoint (native block-quant path).

    Keys on gguf_model_path rather than expert_quant: the dense qwen35 variant has no
    expert banks and therefore no expert_quant="gguf", but still needs its dense ops
    swapped for GGUF ops.
    """
    return getattr(config, "gguf_model_path", None) is not None


def convert_qwen35_to_gguf(model, config: ModelConfig, *, model_path: str) -> None:
    """In place: replace qwen35moe's dense projections + embedding with native GGUF ops.

    Quantized in the checkpoint -> swapped: attention qkv/o (mixed-quant), linear_attn
    in/out, shared-MLP gate_up/down, and the token embedding (IQ3_S, also the lm_head if
    tied). Left as dense bf16 (F32 in the GGUF): all RMSNorms, the router gate
    (ffn_gate_inp), the per-layer shared_expert_gate, conv1d.weight, A_log, dt_bias,
    and the routed experts (served from the offload cache).

    The per-layer quant types are read from the GGUF file, not hardcoded, to support
    different quant levels (IQ3_M, IQ3_XXS, IQ2_*, etc.) which may mix differently.
    """
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear, gguf_merged_or_plain

    # Scan quant types to drive layer swaps.
    quant_map = _scan_quant_types(model_path)

    # Determine full-attention vs GDN layers.
    full_layer_ids = {
        lid
        for lid in range(config.num_layers)
        if isinstance(config.attention_group_for_layer(lid), FullAttentionGroupConfig)
    }

    # Split widths come from the config, not constants: Ornith-1.5 is 16 q heads / 2 kv /
    # 32 v heads, Qwen3.8-27B is 24 / 4 / 48. Hardcoding either breaks the other.
    _qkv_split = [
        config.num_qo_heads * config.head_dim * 2,   # q is gated, hence *2
        config.num_kv_heads * config.head_dim,
        config.num_kv_heads * config.head_dim,
    ]
    _g = config.linear_attention_group()
    _in_proj_split = (
        [
            2 * _g.num_key_heads * _g.key_head_dim + _g.num_value_heads * _g.value_head_dim,
            _g.num_value_heads * _g.value_head_dim,
            _g.num_value_heads,
            _g.num_value_heads,
        ]
        if _g is not None
        else []
    )

    def qt(layer: int, suffix: str) -> int:
        """The ggml type of one tensor, straight from the file.

        No default: a guessed type silently allocates a wrong-sized packed buffer, and the
        only symptom is garbage output. (An earlier version defaulted o_proj to Q4_K, which
        happened to match IQ3_M and mis-sized every full-attention o_proj on IQ3_S.)
        """
        key = (layer, suffix)
        if key not in quant_map:
            raise ValueError(
                f"GGUF {model_path}: expected tensor "
                f"{suffix if layer < 0 else f'blk.{layer}.{suffix}'} is absent, so its quant "
                f"type cannot be read; this checkpoint does not match the qwen35moe layout "
                f"this adapter expects"
            )
        return quant_map[key]

    def swap_linear(owner, attr, quant_type: int):
        """Replace a dense Linear with the GGUFLinear its packed weight will land in."""
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
        quant_type=qt(-1, "token_embd.weight"),
    )
    inner.embed_tokens = embed

    for layer_idx, layer in enumerate(inner.layers.op_list):
        if layer_idx in full_layer_ids:
            # qkv_proj: q | k | v. Mixed in every Ornith quant level (v is a K-quant while
            # q/k are I-quants), so this is normally the GGUFMergedLinear path.
            layer.self_attn.qkv_proj = gguf_merged_or_plain(
                config.hidden_size,
                _qkv_split,
                [
                    qt(layer_idx, "attn_q.weight"),
                    qt(layer_idx, "attn_k.weight"),
                    qt(layer_idx, "attn_v.weight"),
                ],
                has_bias=False,
            )
            swap_linear(layer.self_attn, "o_proj", qt(layer_idx, "attn_output.weight"))
        else:
            # in_proj: qkv | z | b | a, matching gdn.py's
            # _in_proj_split = [conv_dim, value_dim, num_v_heads, num_v_heads].
            layer.linear_attn.in_proj = gguf_merged_or_plain(
                config.hidden_size,
                _in_proj_split,
                [
                    qt(layer_idx, "attn_qkv.weight"),
                    qt(layer_idx, "attn_gate.weight"),
                    qt(layer_idx, "ssm_beta.weight"),
                    qt(layer_idx, "ssm_alpha.weight"),
                ],
                has_bias=False,
            )
            # linear_attn.out_proj is deliberately NOT swapped: its columns index the
            # V-head dimension, which llama.cpp tiled, and un-tiling columns needs dense
            # values (a 128-wide head straddles the quant blocks). iter_gguf_weights yields
            # it as dense bf16 ".weight", so the constructed Linear must stay dense.

        if not config.moe_enabled:
            # Dense qwen35: one SwiGLU MLP per layer (Qwen3_5DenseMLP), no routed experts
            # and no shared expert.
            I = config.intermediate_size
            layer.mlp.gate_up_proj = gguf_merged_or_plain(
                config.hidden_size,
                [I, I],
                [qt(layer_idx, "ffn_gate.weight"), qt(layer_idx, "ffn_up.weight")],
                has_bias=False,
            )
            swap_linear(layer.mlp, "down_proj", qt(layer_idx, "ffn_down.weight"))
            continue

        # Shared expert: gate|up fuse when they share a type (they do in every quant level
        # seen so far); down is independent and does vary (Q4_K on IQ3_M's first layers).
        I = config.shared_expert_intermediate_size
        layer.mlp.shared_expert.gate_up_proj = gguf_merged_or_plain(
            config.hidden_size,
            [I, I],
            [qt(layer_idx, "ffn_gate_shexp.weight"), qt(layer_idx, "ffn_up_shexp.weight")],
            has_bias=False,
        )
        swap_linear(
            layer.mlp.shared_expert, "down_proj", qt(layer_idx, "ffn_down_shexp.weight")
        )

    if config.tie_word_embeddings:
        from freetoken.models.gemma4.gguf import GGUFTiedLMHead

        model.lm_head = GGUFTiedLMHead(embed, qt(-1, "token_embd.weight"))
    else:
        # NOT swap_linear: a plain GGUFLinear would compute logits for every prefill
        # position, and [tokens, vocab] is the largest tensor in the model. See GGUFLMHead.
        from freetoken.layers.gguf import GGUFLMHead

        head = model.lm_head
        out_features, in_features = head.weight.shape
        model.lm_head = GGUFLMHead(
            in_features, out_features, qt(-1, "output.weight"),
            has_bias=head.bias is not None,
        )


__all__ = [
    "parse_gguf_config",
    "gguf_name_to_freetoken",
    "iter_gguf_weights",
    "convert_qwen35_to_gguf",
    "is_gguf_model",
    "_MERGED_PARTS",
    "_EXPERT_SUFFIXES",
]
