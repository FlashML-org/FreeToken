"""Qwen3.5-family MTP draft: the checkpoint's own ``mtp.*`` next-token layer, built from the target's layer classes."""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING

import torch

from freetoken.layers import GemmaPlusOneRMSNorm, GemmaPlusOneRMSNormFused, LinearReplicated, OPList
from freetoken.layers.quantization import finalize_quant
from freetoken.layers.base import BaseOP
from freetoken.models.blocks import embed_input_ids
from freetoken.models.config import FullAttentionGroupConfig
from freetoken.utils import torch_dtype

from .model import Qwen3_5DecoderLayer
from .weight import iter_mtp_expert_pieces, iter_mtp_weights

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def draft_config(config: ModelConfig) -> ModelConfig:
    """One full-attention layer with the target's geometry; its MoE is resident on the GPU whatever the target's strategy."""
    groups = [g for g in config.attention_groups if isinstance(g, FullAttentionGroupConfig)]
    group = groups[0] if groups else config.default_full_attention_group()
    return dataclasses.replace(
        config,
        num_layers=1,
        attention_groups=(dataclasses.replace(group, layer_ids=(0,)),),
        moe_strategy="fused",
        decode_target="gpu",
    )


class Qwen3_5MTPDraft(BaseOP):
    """``h' = layer(fc(cat(norm_e(embed(token)), norm_h(h))))``: the next token's hidden from a token and the hidden before it."""

    # the draft MTPProposer picks from the target's backends
    attention_backend = None
    draft_config = staticmethod(draft_config)

    @staticmethod
    def relay_width(config: ModelConfig) -> int:
        return config.hidden_size

    @staticmethod
    def check_source(path: str, target_path: str) -> None:
        from freetoken.models.gguf.reader import is_gguf_path
        from freetoken.models.loader import safetensors_weight_map

        if is_gguf_path(path):
            raise ValueError(f"the Qwen3.5 MTP draft loads from safetensors, not from the GGUF {path}")
        if os.path.isdir(path) and not any(name.startswith("mtp.") for name in safetensors_weight_map(path)):
            raise ValueError(f"{path} has no mtp.* tensors")

    def __init__(self, config: ModelConfig, target_model) -> None:
        self.config = draft_config(config)
        hidden = config.hidden_size
        self.fc = LinearReplicated(2 * hidden, hidden, has_bias=False, quant_config=config.quant, prefix="mtp.fc")
        self.pre_fc_norm_embedding = GemmaPlusOneRMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaPlusOneRMSNorm(hidden, eps=config.rms_norm_eps)
        self.layers = OPList([Qwen3_5DecoderLayer(self.config, 0, prefix="mtp.layers.0")])
        self.norm = GemmaPlusOneRMSNormFused(hidden, eps=config.rms_norm_eps)
        # the target's own module; an underscore name keeps it out of this module's state dict
        self._embed_tokens = target_model.model.embed_tokens

    @classmethod
    def load(cls, config: ModelConfig, target_model, model_path: str, device: torch.device, dtype: torch.dtype) -> Qwen3_5MTPDraft:
        with torch.device("meta"), torch_dtype(dtype):
            draft = cls(config, target_model)
        expected = draft.state_dict()
        loaded = {}
        for name, tensor in iter_mtp_weights(model_path, device):
            if name not in expected:
                raise ValueError(f"{model_path}: unexpected MTP tensor {name!r}")
            loaded[name] = tensor.to(device=device, dtype=expected[name].dtype)
        if missing := sorted(set(expected) - set(loaded)):
            raise ValueError(f"{model_path}: MTP tensors missing: {missing}")
        draft.load_state_dict(loaded)
        if draft.config.moe_enabled:
            from freetoken.moe.expert_banks import attach_resident_banks, build_expert_banks

            # packed by the layer's own expert method, as the target's resident layers are
            moe = draft.layers.op_list[0].mlp.experts
            pieces = iter_mtp_expert_pieces(model_path)
            banks = build_expert_banks(moe.quant_method, 1, pieces, device=device, resident=True)
            attach_resident_banks([moe], banks)
        # not the root: it reaches the target's embedding, which the target finalized already
        finalize_quant(draft.fc)
        finalize_quant(draft.layers)
        return draft

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._embed_tokens.forward(input_ids)

    def embed_batch(self, batch) -> torch.Tensor:
        """The target batch's token embeddings with its multimodal rows merged in, as the target embeds them."""
        return embed_input_ids(self._embed_tokens, batch.input_ids, batch)

    def forward_hidden(self, embeds: torch.Tensor, prev_hidden: torch.Tensor) -> torch.Tensor:
        """Draft hidden of each row from the embedding of its token and the hidden of the position before it."""
        x = self.fc.forward(torch.cat([self.pre_fc_norm_embedding.forward(embeds), self.pre_fc_norm_hidden.forward(prev_hidden)], dim=-1))
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, _ = self.norm.forward(x, residual)
        return x

    def head_input(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden
