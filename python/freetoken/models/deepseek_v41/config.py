"""Engine configuration for the mixed FP8/NVFP4 DeepSeek-V4.1 checkpoint."""

from __future__ import annotations

from freetoken.layers.quantization import QuantConfig, QuantKind, QuantScheme, WeightDesc
from freetoken.layers.quantization.scheme import nvfp4_scheme
from freetoken.models.config import DSV41AttentionGroupConfig, ModelConfig, RotaryConfig

from .args import as_dict, load_args


class DeepseekV41QuantConfig(QuantConfig):
    """Native block-32 projections and W4A16 NVFP4 routed experts."""

    dialect = "deepseek_v41"
    STORAGE = {
        QuantKind.FP8_BLOCK: {"weight": "weight", "weight_scale_inv": "scale"},
        QuantKind.NVFP4: {"weight": "weight", "weight_scale": "weight_scale",
                          "weight_global": "weight_scale_2"},
    }

    def __init__(self, args):
        super().__init__()
        self._fp8 = QuantScheme(QuantKind.FP8_BLOCK, WeightDesc("e4m3", (32, 32), "e8m0"),
                                {"weight", "weight_scale_inv"})
        self._nvfp4 = nvfp4_scheme(input_scale=False)
        self._experts = {f"layers.{i}.ffn.experts" for i in range(args.n_layers)}
        self._projections = set()
        for i in range(args.n_layers):
            prefix = f"layers.{i}"
            self._projections.update(f"{prefix}.attn.{name}" for name in ("wq_a", "wq_b", "wkv", "wo_b"))
            self._projections.update(f"{prefix}.ffn.shared_experts.{name}" for name in ("w1", "w2", "w3"))
            if i in args.index_source_layers:
                self._projections.add(f"{prefix}.attn.indexer.wq_b")
            if i in args.engram_layer_ids:
                self._projections.add(f"{prefix}.engram.wkv")

    def scheme_for_name(self, name):
        prefix, _, suffix = name.partition(".experts")
        if prefix + ".experts" in self._experts:
            if not suffix or (len(parts := suffix.split(".")) == 3
                              and parts[1].isdigit() and parts[2] in {"w1", "w2", "w3"}):
                return self._nvfp4
        return self._fp8 if name in self._projections else None


def checkpoint_quant_config(hf_config):
    return DeepseekV41QuantConfig(parse_config(hf_config).dsv41_args)


def parse_config(hf_config) -> ModelConfig:
    raw = as_dict(hf_config)
    args = load_args(hf_config)
    quant = as_dict(raw.get("quantization_config"))
    layers = as_dict(quant.get("quantized_layers"))
    algo = str(quant.get("moe_quant_algo", "")).lower()
    if not algo and layers:
        algos = {str(as_dict(v).get("quant_algo", "")).lower() for k, v in layers.items()
                 if k.startswith("layers.") and k.endswith(".ffn.experts")}
        algo = "nvfp4" if algos == {"nvfp4"} else ""
    if not algo and str(quant.get("expert_dtype", "")).lower() == "nvfp4":
        algo = "nvfp4"
    if (algo != "nvfp4" or int(quant.get("group_size", 16)) != 16
            or int(quant.get("expert_block_size", 16)) != 16):
        raise ValueError("DeepSeek-V4.1 requires NVFP4 routed experts with group_size=16")
    # ModelOpt leaves expert_dtype='fp4' in its converted NVFP4 config.
    if str(quant.get("expert_dtype", "nvfp4")).lower() not in {"nvfp4", "fp4"}:
        raise ValueError("DeepSeek-V4.1 requires NVFP4 routed experts")
    if str(quant.get("expert_scale_fmt", "e4m3")).lower() != "e4m3":
        raise ValueError("DeepSeek-V4.1 NVFP4 routed experts require E4M3 block scales")
    if quant.get("expert_global_scale", True) is not True:
        raise ValueError("DeepSeek-V4.1 NVFP4 routed experts require a global scale")
    for layer in range(args.n_layers):
        entry = layers.get(f"layers.{layer}.ffn.experts")
        if layers and (entry is None or str(as_dict(entry).get("quant_algo", "")).lower() != "nvfp4"
                       or int(as_dict(entry).get("group_size", 16)) != 16):
            raise ValueError(f"Missing or incompatible NVFP4 quantization for backbone layer {layer}")
    if tuple(quant.get("weight_block_size", (32, 32))) != (32, 32):
        raise ValueError("DeepSeek-V4.1 resident FP8 projections require 32x32 weight blocks")
    if quant.get("scale_fmt", "ue8m0") != "ue8m0":
        raise ValueError("DeepSeek-V4.1 resident FP8 projections require UE8M0 scales")
    text = as_dict(raw.get("text_config", raw))
    max_position = int(text.get("max_position_embeddings", args.original_seq_len * args.rope_factor))
    return ModelConfig(
        num_layers=args.n_layers, num_qo_heads=args.n_heads, num_kv_heads=1,
        head_dim=args.head_dim, hidden_size=args.dim, vocab_size=args.vocab_size,
        intermediate_size=args.moe_inter_dim, rms_norm_eps=args.norm_eps,
        rotary_config=RotaryConfig(args.head_dim, args.rope_head_dim, max_position,
                                  args.rope_theta, {
                                      "rope_type": "yarn", "factor": args.rope_factor,
                                      "beta_fast": args.beta_fast, "beta_slow": args.beta_slow,
                                      "original_max_position_embeddings": args.original_seq_len,
                                  }),
        hidden_act="swiglu_clamp", tie_word_embeddings=False,
        num_experts=args.n_routed_experts, num_experts_per_tok=args.n_activated_experts,
        moe_intermediate_size=args.moe_inter_dim, norm_topk_prob=args.norm_topk_prob,
        model_type="deepseek_v41", architectures=["DeepseekV41ForCausalLM"],
        moe_strategy="offload", moe_enabled=True, expert_quant="nvfp4",
        weight_block_size=(32, 32), n_shared_experts=args.n_shared_experts,
        shared_expert_intermediate_size=args.moe_inter_dim,
        routed_scaling_factor=args.route_scale, swiglu_limit=args.swiglu_limit,
        hidden_act_alpha=1.0, attn_sm_scale=args.head_dim ** -0.5,
        vision_config=raw.get("vision_config") if args.vision_enabled else None,
        image_token_id=args.image_token_id if args.vision_enabled else None,
        dsv41_args=args,
        attention_groups=(DSV41AttentionGroupConfig(
            name="dsv41", layer_ids=tuple(range(args.n_layers)), num_kv_heads=1,
            head_dim=args.head_dim, sliding_window=args.window_size,
        ),),
    )


__all__ = ["parse_config", "checkpoint_quant_config", "DeepseekV41QuantConfig"]
