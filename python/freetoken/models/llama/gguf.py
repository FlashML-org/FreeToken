""" Llama GGUF adapter: build the FreeToken ``ModelConfig`` from GGUF metadata.
"""
from torch import Tensor
from typing import Iterator
from freetoken.models.config import ModelConfig,RotaryConfig
from freetoken.models.gguf.config import GgufConfigShim
from freetoken.models.gguf.reader import iter_gguf_tensors


def parse_gguf_config(shim:GgufConfigShim) -> ModelConfig:
    """Parse a GGUF config shim into a FreeToken ``ModelConfig``."""

    gguf_metadata_llama = shim.metadata
    # model architecture
    arch=shim.model_type
    
    hidden_size = gguf_metadata_llama.get(f"{arch}.embedding_length")
    num_of_heads = gguf_metadata_llama.get(f"{arch}.attention.head_count")
    # head_dim and rotary_dim 
    head_dim = hidden_size//num_of_heads
    
    dense_ffn_size = gguf_metadata_llama.get(f"{arch}.feed_forward_length")
    
    expert_count = gguf_metadata_llama.get(f"{arch}.expert_count", 1)
     
    # The GGUF shim is a minimal HF-config-like dict with the fields we need.
    return ModelConfig(
    #   directly, from shim
        model_type = shim.model_type,
        architectures = shim.architectures,
        vocab_size = shim.vocab_size,
        tie_word_embeddings = shim.tie_word_embeddings,
        head_dim = head_dim,
        hidden_size = hidden_size,      
      
    
# from mapping GGUF keys

        num_layers = gguf_metadata_llama.get(f"{arch}.block_count"),
        intermediate_size = gguf_metadata_llama.get(f"{arch}.feed_forward_length"),                                 
        
        num_qo_heads = num_of_heads,                       
        num_kv_heads = gguf_metadata_llama.get(f"{arch}.attention.head_count_kv"),                                       
        rms_norm_eps = gguf_metadata_llama.get(f"{arch}.attention.layer_norm_rms_epsilon"),                              
                                                         
    # Rotary Config                           
        rotary_config = RotaryConfig(                      
            head_dim = gguf_metadata_llama.get(f"{arch}.rope.dimension_count",head_dim)   ,                           
            rotary_dim = gguf_metadata_llama.get(f"{arch}.rope.dimension_count",head_dim)   ,      #identical to head_dim                
            max_position = min(gguf_metadata_llama.get(f"{arch}.context_length", 8192), 8192),                                      
            base = gguf_metadata_llama.get(f"{arch}.rope.freq_base"),                                           
            scaling = None                                 
        ),      
        
        hidden_act = "silu",   #llama standard activation                            
        num_experts = expert_count,   #for MoE              
        num_experts_per_tok = gguf_metadata_llama.get(f"{arch}.expert_used_count", 1), 
        norm_topk_prob = False, 
         
        # If it's a dense model, this just falls back to the dense size.    
        moe_intermediate_size = gguf_metadata_llama.get(f"{arch}.expert_feed_forward_length",dense_ffn_size),                       
        
        moe_weight_format = "q4_0", #4bit symmetric    
        moe_enabled = expert_count > 1,
        
        shared_expert_intermediate_size= dense_ffn_size if expert_count> 1 else 0 
         
    )

from freetoken.models.gguf.reader import iter_gguf_tensors
from freetoken.models.gguf.dequant import dequantize
import torch

def _to_bf16(t) -> torch.Tensor:
    """Dequantize a GgufTensor (F32/F16) to a dense bf16 tensor for LayerNorms."""
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16)
    return flat.reshape(t.shape)

_LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_output.weight": "self_attn.o_proj.qweight",
    "ffn_down.weight": "mlp.down_proj.qweight",
}

def iter_gguf_weights(
        model_path: str,
        device,
        *,
        include_moe_experts: bool,
        include_non_moe: bool
        ) -> Iterator[tuple[str, Tensor]]:
    """Iterate over GGUF weights, yielding (name, tensor) pairs for the model's parameters."""
    
    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}
    gate_up_buf: dict[int, dict[str, torch.Tensor]] = {}

    for t in iter_gguf_tensors(model_path):
        name = t.name
        
        is_moe_expert = "exps.weight" in name
        if is_moe_expert and not include_moe_experts:
            continue
        if not is_moe_expert and not include_non_moe:
            continue
            
        # Global Standalone Tensors
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(t)
            continue
        if name == "output.weight":
            yield "lm_head.qweight", t.packed()
            continue
        if name == "rope_freqs.weight":
            continue  # Recomputed dynamically in the engine
            
        if not name.startswith("blk."):
            continue
            
        # Block Tensors
        layer = int(name.split(".")[1])
        suffix = name.split(".", 2)[2]
        base = f"model.layers.{layer}"
        
        # Mapped Standalones
        if suffix in _LAYER_MAP:
            mapped_name = _LAYER_MAP[suffix]
            # Norms must be dequantized to bf16; projections stay packed as Q4_0
            if "norm" in suffix:
                yield f"{base}.{mapped_name}", _to_bf16(t)
            else:
                yield f"{base}.{mapped_name}", t.packed()
            continue
            
        # Fusable Tensors
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
            
    # MoE and Shared Expert Tensors 
        # The Router Gate
        elif suffix == "ffn_gate_inp.weight":
            yield f"{base}.router.weight", t.packed()
        
        # Shared Expert (shexp)
        elif suffix == "ffn_down_shexp.weight":
            yield f"{base}.shared_expert.down_proj.qweight", t.packed()
        elif suffix == "ffn_gate_shexp.weight":
            gate_up_buf.setdefault(layer, {})["gate_shexp"] = t.packed()
        elif suffix == "ffn_up_shexp.weight":
            gate_up_buf.setdefault(layer, {})["up_shexp"] = t.packed()
            
        # Routed Experts (exps)
        elif suffix == "ffn_down_exps.weight":
            yield f"{base}.mlp.down_proj.qweight", t.packed()
        elif suffix == "ffn_gate_exps.weight":
            gate_up_buf.setdefault(layer, {})["gate_exps"] = t.packed()
        elif suffix == "ffn_up_exps.weight":
            gate_up_buf.setdefault(layer, {})["up_exps"] = t.packed()
            
        # Trigger Fusion
        slots = qkv_buf.get(layer)
        if slots and "q" in slots and "k" in slots and "v" in slots:
            yield f"{base}.self_attn.qkv_proj.qweight", torch.cat(
                [slots["q"], slots["k"], slots["v"]], dim=0
            )
            del qkv_buf[layer]
            
        gu = gate_up_buf.get(layer)
        if gu:
            if "gate" in gu and "up" in gu:
                yield f"{base}.mlp.gate_up_proj.qweight", torch.cat(
                    [gu["gate"], gu["up"]], dim=0
                )
                del gu["gate"]
                del gu["up"]
                
            if "gate_shexp" in gu and "up_shexp" in gu:
                yield f"{base}.shared_expert.gate_up_proj.qweight", torch.cat(
                    [gu["gate_shexp"], gu["up_shexp"]], dim=0
                )
                del gu["gate_shexp"]
                del gu["up_shexp"]
                
            if "gate_exps" in gu and "up_exps" in gu:
                yield f"{base}.mlp.gate_up_proj.qweight", torch.cat(
                    [gu["gate_exps"], gu["up_exps"]], dim=0
                )
                del gu["gate_exps"]
                del gu["up_exps"]

def convert_llama_to_gguf(model, config:ModelConfig) -> None:
    """Convert a FreeToken Llama model to GGUF format in-place.

    This is a no-op for non-GGUF models, and raises an error if the model is not a Llama.
    """
    # to prevent the circular imports 
    from freetoken.layers.gguf import GGUFEmbedding,GGUFLinear                                             
    from freetoken.models.gguf.dequant import GGML_Q4_0, GGML_Q6_K    
    
    # helper functino for swapping layers
    def swap_linear(owner,attr_name, quant_type=GGML_Q4_0, has_bias=False):
        old_layer=getattr(owner,attr_name)
        out_features, in_features= old_layer.weight.shape
        # replace the attribute with the custom GGUF kernal
        setattr(
            owner,
            attr_name,
            GGUFLinear(in_features, out_features, quant_type)
        )
    inner_model=model.model
    inner_model.embed_tokens=GGUFEmbedding(
        num_embeddings=config.vocab_size,            
        embedding_dim=config.hidden_size,            
        quant_type=GGML_Q6_K,                        
        embed_scale=None
    )
    
    for layer in inner_model.layers.op_list:                 
        swap_linear(layer.self_attn, "qkv_proj")     
        swap_linear(layer.self_attn, "o_proj")       
        
        # If it's a Dense model, swap the standard MLP
        if not hasattr(layer, "router"):
            swap_linear(layer.mlp, "gate_up_proj")       
            swap_linear(layer.mlp, "down_proj")   
            
        # If it's a Shared-Expert MoE model, swap the Shared Expert MLP
        if getattr(layer, "shared_expert", None) is not None:
            swap_linear(layer.shared_expert, "gate_up_proj")       
            swap_linear(layer.shared_expert, "down_proj")   
    
    # swap the LM head
    if not config.tie_word_embeddings:
        swap_linear(model,"lm_head",quant_type=GGML_Q6_K)

def is_gguf_model(config) -> bool:
    """Check if the model config is for a GGUF model."""
    return getattr(config, "moe_weight_format", None) == "q4_0"
