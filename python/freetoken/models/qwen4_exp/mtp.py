"""Qwen3.8-Flash-Next MTP draft: the checkpoint's own ``mtp.*`` layer, one QSA decoder layer over the hyper-connection streams.

Draft row c reads the target's residual streams before the final mix, ``R_c [hc_count*hidden]``, and the token after it::

    e     = fc_embedding(norm_e(embed(x_{c+1})))
    R_in  = fc_hidden(norm_h(R_c)) per stream + e      # norm_h: one RMS statistic over all hc_count*hidden
    R_out = layer(R_in)                                # QSA attention + routed MoE, no PLE
    logits = lm_head(mixer(R_out))

``R_out`` relays to the next draft step. The weights come from the checkpoint's ``mtp.*`` tensors, or from a Strata MTP pack (a GGUF with the same names, stacked ggml experts).
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.core import get_global_ctx
from freetoken.layers import GemmaPlusOneRMSNorm, LinearReplicated, OPList
from freetoken.layers.base import BaseOP
from freetoken.layers.quantization import finalize_quant
from freetoken.models.blocks import embed_input_ids
from freetoken.models.config import FullAttentionGroupConfig
from freetoken.utils import torch_dtype

from .hc import GatedResidual
from .model import Qwen4ExpDecoderLayer
from .weight import iter_mtp_weights

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


_PACK_EXPERT_RE = re.compile(r"^(mtp\.layers\.0\.mlp\.experts)\.0\.(gate|up|down)_proj$")


def _pack_tensors(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Module name -> the pack's tensors: the HF name itself, but stacked experts with gate and up in one tensor."""
    m = _PACK_EXPERT_RE.match(name)
    if m is None:
        return (name,)
    return (f"{m[1]}.{'down_proj' if m[2] == 'down' else 'gate_up_proj'}",)


def _iter_pack_weights(model_path: str, config: ModelConfig) -> Iterator[tuple[str, torch.Tensor]]:
    """The dense tensors of a Strata MTP pack, named and fused as Qwen4ExpMTPDraft names them; its norms are stored zero-centered, as the layers read them."""
    from freetoken.models.gguf.reader import gguf_tensor_index
    from freetoken.models.gguf.tensors import float_tensor, fused, matrix

    index = gguf_tensor_index(model_path)
    hidden, width = config.hidden_size, config.qwen4_args.hc_count * config.hidden_size

    def t(name: str):
        return index[f"mtp.{name}.weight"]

    def linear(name: str, parts: tuple[str, ...], k: int, pad_to: int = 0):
        infos = [t(p) for p in parts]
        return fused(name, infos, [matrix(i) for i in infos], k, pad_to)

    sa, mlp = "layers.0.self_attn", "layers.0.mlp"
    for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", f"{sa}.q_norm", f"{sa}.k_norm", f"{sa}.indexer.q_layernorm",
                 f"{sa}.indexer.k_layernorm", f"{mlp}.gate", f"{mlp}.shared_expert_gate", "hyper_connection_mixer.hc_norm"):
        yield f"{name}.weight", float_tensor(t(name))
    yield from linear("fc_embedding", ("fc_embedding",), hidden)
    yield from linear("fc_hidden", ("fc_hidden",), hidden)
    yield from linear(f"{sa}.qkv_proj", tuple(f"{sa}.{p}_proj" for p in "qkv"), hidden)
    yield from linear(f"{sa}.o_proj", (f"{sa}.o_proj",), int(t(f"{sa}.o_proj").ne[0]))
    yield from linear(f"{sa}.indexer.index_qk_proj", (f"{sa}.indexer.index_qk_proj",), hidden)
    yield from linear(f"{mlp}.shared_expert.gate_up_proj", (f"{mlp}.shared_expert.gate_proj", f"{mlp}.shared_expert.up_proj"), hidden)
    yield from linear(f"{mlp}.shared_expert.down_proj", (f"{mlp}.shared_expert.down_proj",), config.shared_expert_intermediate_size)
    for hc in ("layers.0.attn_hyper_connection", "layers.0.mlp_hyper_connection"):
        down = (f"{hc}.input_mix_weight_down", f"{hc}.block_inject_weight")
        yield from linear(f"{hc}.input_mix_weight_down_block_inject", down, width, pad_to=16)
        yield from linear(f"{hc}.input_mix_weight_up", (f"{hc}.input_mix_weight_up",), config.qwen4_args.hc_lowrank)
        yield f"{hc}.hc_norm.weight", float_tensor(t(f"{hc}.hc_norm"))
    yield from linear("hyper_connection_mixer.input_mix_weight_down", ("hyper_connection_mixer.input_mix_weight_down",), width)
    yield from linear("hyper_connection_mixer.input_mix_weight_up", ("hyper_connection_mixer.input_mix_weight_up",), config.qwen4_args.hc_lowrank)


def _pack_expert_banks(model_path: str, device: torch.device):
    """The pack's stacked experts as the resident banks of bank layer 0, in their packed ggml rows."""
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.gguf.experts import resident_expert_bank
    from freetoken.models.gguf.reader import gguf_tensor_index
    from freetoken.moe.expert_banks import ExpertBanks

    index = gguf_tensor_index(model_path)
    sources = {role: [resident_expert_bank(index[f"mtp.layers.0.mlp.experts.{role}_proj"], device)] for role in ("gate_up", "down")}
    return ExpertBanks("gguf", sources, kind=QuantKind.GGUF, kernel="ggml", layout=None)


def _pack_quant(model_path: str):
    from freetoken.layers.quantization import GGUFConfig, NameMap
    from freetoken.models.gguf.reader import gguf_tensor_index
    from freetoken.models.register import get_model_spec

    packed = get_model_spec("Qwen4ExpForConditionalGeneration").packed_modules_mapping
    types = {name: info.ggml_type for name, info in gguf_tensor_index(model_path).items()}
    return GGUFConfig(types, _pack_tensors, name_map=NameMap(packed=packed))


def draft_config(config: ModelConfig) -> ModelConfig:
    """The target config cut to one QSA layer (layer id 0) without PLE; its MoE is resident on the GPU whatever the target's strategy."""
    group = next(g for g in config.attention_groups if isinstance(g, FullAttentionGroupConfig) and g.index_ratio > 1)
    return dataclasses.replace(
        config,
        num_layers=1,
        attention_groups=(dataclasses.replace(group, layer_ids=(0,), num_index_layers=1),),
        qwen4_args=dataclasses.replace(config.qwen4_args, ple_layer_ids=()),
        slot_states=(),
        vision_config=None,
        moe_strategy="fused",
        decode_target="gpu",
    )


class Qwen4ExpMTPDraft(BaseOP):
    attention_backend = "qsa_sparse"
    reads_gguf_pack = True
    draft_config = staticmethod(draft_config)

    @staticmethod
    def relay_width(config: ModelConfig) -> int:
        return config.qwen4_args.hc_count * config.hidden_size

    def __init__(self, config: ModelConfig, target_model, quant=None) -> None:
        draft = draft_config(config)
        # a Strata pack carries its own ggml types
        self.config = draft = draft if quant is None else dataclasses.replace(draft, quant=quant)
        hidden = draft.hidden_size
        self.hc_count = draft.qwen4_args.hc_count
        self.fc_embedding = LinearReplicated(hidden, hidden, has_bias=False, quant_config=draft.quant, prefix="mtp.fc_embedding")
        self.fc_hidden = LinearReplicated(hidden, hidden, has_bias=False, quant_config=draft.quant, prefix="mtp.fc_hidden")
        self.pre_fc_norm_embedding = GemmaPlusOneRMSNorm(hidden, eps=draft.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaPlusOneRMSNorm(self.hc_count * hidden, eps=draft.rms_norm_eps)
        self.layers = OPList([Qwen4ExpDecoderLayer(draft, 0, prefix="mtp.layers.0")])
        self.hyper_connection_mixer = GatedResidual(draft, use_combine=False, prefix="mtp.hyper_connection_mixer")
        # the target's own module; an underscore name keeps it out of this module's state dict
        self._embed_tokens = target_model.model.embed_tokens

    @classmethod
    def load(cls, config: ModelConfig, target_model, model_path: str, device: torch.device, dtype: torch.dtype) -> Qwen4ExpMTPDraft:
        from freetoken.moe.expert_banks import attach_resident_banks, build_expert_banks
        from freetoken.moe.expert_pieces import iter_extra_layer_expert_pieces

        pack = model_path.endswith(".gguf")
        with torch.device("meta"), torch_dtype(dtype):
            draft = cls(config, target_model, _pack_quant(model_path) if pack else None)
        expected = draft.state_dict()
        loaded = {}
        weights = _iter_pack_weights(model_path, draft.config) if pack else iter_mtp_weights(model_path, device)
        for name, tensor in weights:
            if name not in expected:
                raise ValueError(f"{model_path}: unexpected MTP tensor {name!r}")
            loaded[name] = tensor.to(device=device, dtype=expected[name].dtype)
        if missing := sorted(set(expected) - set(loaded)):
            raise ValueError(f"{model_path}: MTP tensors missing: {missing}")
        draft.load_state_dict(loaded)
        # packed by the layer's own expert method, as the target's resident layers are
        moe = draft.layers.op_list[0].mlp.experts
        if pack:
            banks = _pack_expert_banks(model_path, device)
        else:
            pieces = iter_extra_layer_expert_pieces(model_path, "mtp.layers.0.mlp.experts", device)
            banks = build_expert_banks(moe.quant_method, 1, pieces, device=device, resident=True)
        attach_resident_banks([moe], banks)
        # not the root: it reaches the target's embedding, which the target finalized already
        for module in (draft.fc_embedding, draft.fc_hidden, draft.layers, draft.hyper_connection_mixer):
            finalize_quant(module)
        return draft

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._embed_tokens.forward(input_ids)

    def embed_batch(self, batch) -> torch.Tensor:
        """The target batch's token embeddings with its multimodal rows merged in, as the target embeds them."""
        return embed_input_ids(self._embed_tokens, batch.input_ids, batch)

    def forward_hidden(self, embeds: torch.Tensor, prev_hidden: torch.Tensor) -> torch.Tensor:
        """``R_out`` of each row from the embedding of its token and the residual streams of the position before it."""
        rows, hidden = embeds.shape
        e = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(embeds))
        h = self.pre_fc_norm_hidden.forward(prev_hidden)
        r = self.fc_hidden.forward(h.view(rows * self.hc_count, hidden)).view(rows, self.hc_count, hidden)
        r += e.unsqueeze(1)
        return self.layers.op_list[0].forward(r.view(rows, -1), get_global_ctx().batch)

    def head_input(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.hyper_connection_mixer.mix(hidden)[0]


__all__ = ["Qwen4ExpMTPDraft", "draft_config"]
