"""DeepSeek-V4.1-Flash: Causal Encoder-Decoder MoE with Compressed Sparse Attention 2.

40 layers (20 encoder + 20 decoder), every layer a sliding window (128) plus -- from layer 2 -- global
sparse attention over cross-layer-shared compressed KV in packed fp4 (Full / Reindex / Reuse
modes, a hierarchical candidate pool in the decoder), single-pass mHC residual streams, Engram n-gram
memory at layers 1 and 14 streamed from disk, one shared + 384 routed MXFP4 experts per layer served
from the offload cache. Images use the DeepSeek ViT; DSpark speculative decoding is not served.
"""

from .config import parse_config
from .model import DeepseekV41ForCausalLM
from .weight import ftw_side_files, iter_expert_pieces, iter_weights, iter_vision_weights

__all__ = ["DeepseekV41ForCausalLM", "parse_config", "iter_weights", "iter_vision_weights", "iter_expert_pieces", "ftw_side_files"]
