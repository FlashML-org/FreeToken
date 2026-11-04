"""Qwen3.5-family MTP draft: the checkpoint's own ``mtp.*`` next-token layer, built from the target's layer classes."""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING

import torch

from freetoken.layers import GemmaPlusOneRMSNorm, GemmaPlusOneRMSNormFused, LinearReplicated, OPList
from freetoken.layers.quantization import finalize_quant
from freetoken.layers.base import BaseOP
from freetoken.models.config import FullAttentionGroupConfig
from freetoken.utils import torch_dtype

from .model import Qwen3_5DecoderLayer
from .weight import iter_mtp_expert_pieces, iter_mtp_weights

if TYPE_CHECKING:
    from freetoken.layers.embedding import VocabParallelEmbedding
    from freetoken.models.config import ModelConfig


def draft_config(config: ModelConfig) -> ModelConfig:
    """One full-attention layer with the target's geometry."""
    groups = [g for g in config.attention_groups if isinstance(g, FullAttentionGroupConfig)]
    group = groups[0] if groups else config.default_full_attention_group()
    return dataclasses.replace(
        config,
        num_layers=1,
        attention_groups=(dataclasses.replace(group, layer_ids=(0,)),),
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
        from freetoken.models.gguf.reader import gguf_architecture, gguf_tensor_index, is_gguf_path, load_gguf_metadata
        from freetoken.models.loader import safetensors_weight_map

        if is_gguf_path(path):
            if not is_gguf_path(target_path):
                raise ValueError(f"the Qwen3.5 MTP draft loads from safetensors mtp.* tensors, not from the GGUF {path}")
            from .gguf import nextn_block_index

            blk = nextn_block_index(path)
            index = gguf_tensor_index(path)
            if f"blk.{blk}.nextn.eh_proj.weight" not in index:
                raise ValueError(f"{path}: blk.{blk} carries no nextn tensors")
            if os.path.abspath(path) != os.path.abspath(target_path):
                # a separate draft file must be an mtp-*.gguf sidecar: only the MTP block, same
                # arch; anything else is another trunk read under this target's shapes
                blocks = {name.split(".")[1] for name in index if name.startswith("blk.")}
                if blocks != {str(blk)} or gguf_architecture(path) != gguf_architecture(target_path):
                    raise ValueError(
                        f"the MTP draft of a GGUF target is the trunk's own nextn block or an "
                        f"mtp-*.gguf sidecar of the same model, not {path}"
                    )
                # the same arch string spans sizes (qwen35 covers the 9B and the 27B); a wrong-size
                # sidecar would otherwise die on a bare shape assert deep in load_state_dict
                side_hidden = load_gguf_metadata(path).get(f"{gguf_architecture(path)}.embedding_length")
                target_hidden = load_gguf_metadata(target_path).get(f"{gguf_architecture(target_path)}.embedding_length")
                if side_hidden is not None and target_hidden is not None and int(side_hidden) != int(target_hidden):
                    raise ValueError(
                        f"{path}: an mtp-*.gguf sidecar of a model with hidden size {int(side_hidden)}, but the "
                        f"target {target_path} has hidden size {int(target_hidden)}; the sidecar must be the "
                        f"MTP layer of the same model as the target"
                    )
        elif is_gguf_path(target_path):
            # a folder draft would build under the trunk's GGUF quant names
            raise ValueError(f"a GGUF target's MTP draft is the trunk's own nextn block, not {path}")
        elif os.path.isdir(path) and not any(name.startswith("mtp.") for name in safetensors_weight_map(path)):
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
        from freetoken.models.gguf.reader import is_gguf_path

        # a GGUF trunk keeps the draft in its own ggml types, mapped by the target config's own to_gguf;
        # an mtp-*.gguf sidecar carries a different mix, so its table names the draft's types
        gguf = is_gguf_path(model_path)
        if gguf:
            from .gguf import nextn_quant

            config = dataclasses.replace(config, quant=nextn_quant(model_path))
        with torch.device("meta"), torch_dtype(dtype):
            draft = cls(config, target_model)
        expected = draft.state_dict()
        loaded = {}
        if gguf:
            from .gguf import iter_nextn_weights

            weights = iter_nextn_weights(model_path, draft.config)
        else:
            weights = iter_mtp_weights(model_path, device)
        for name, tensor in weights:
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
            if gguf:
                from .gguf import nextn_expert_banks

                banks = nextn_expert_banks(model_path, device)
            else:
                pieces = iter_mtp_expert_pieces(model_path)
                banks = build_expert_banks(moe.quant_method, 1, pieces, device=device, resident=True)
            attach_resident_banks([moe], banks)
        # not the root: it reaches the target's embedding, which the target finalized already
        finalize_quant(draft.fc)
        finalize_quant(draft.layers)
        return draft

    @property
    def embed_tokens(self) -> VocabParallelEmbedding:
        return self._embed_tokens

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
