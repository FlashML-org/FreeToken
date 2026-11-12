"""Gemma 4 MTP draft: the ``gemma4-assistant`` sidecar's own four layers, built from the target's layer classes.

One draft row (llama.cpp's ``gemma4-assistant.cpp``): the target's sqrt-scaled token embedding
concat the target's post-final-norm hidden -> ``pre_projection`` to draft width -> four
KV-shared decoder layers that read the target's cache and store nothing -> ``output_norm``.
The draft's own tied table gives the logits and ``post_projection`` returns the recurrent
hidden in target width; unlike the qwen drafts there is no enorm/hnorm and no logit softcap.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.layers.quantization import finalize_quant
from freetoken.utils import torch_dtype

from .model import Gemma4DecoderLayer

if TYPE_CHECKING:
    from freetoken.layers.embedding import VocabParallelEmbedding
    from freetoken.models.config import ModelConfig

# every released assistant is 4 layers; check_source rejects another count
DRAFT_LAYERS = 4


class Gemma4MTPDraft(BaseOP):
    """``norm(layers(pre_projection(cat(target_embed(token), h))))``: the next token's draft hidden."""

    @staticmethod
    def relay_width(config: ModelConfig) -> int:
        # what a row reads and relays is h_nextn: post_projection's output, in target width
        return config.hidden_size

    @staticmethod
    def check_source(path: str, target_path: str) -> None:
        from freetoken.models.gguf.reader import gguf_architecture, gguf_tensor_index, is_gguf_path, load_gguf_metadata

        from .gguf import ASSISTANT_ARCH

        if not is_gguf_path(path):
            raise ValueError(f"the Gemma 4 MTP draft is a {ASSISTANT_ARCH} GGUF sidecar, and {path} is not a .gguf file")
        arch = gguf_architecture(path)
        if arch != ASSISTANT_ARCH:
            raise ValueError(f"{path}: a Gemma 4 MTP draft is a {ASSISTANT_ARCH} GGUF, not {arch!r}")
        md = load_gguf_metadata(path)
        block_count = int(md[f"{ASSISTANT_ARCH}.block_count"])
        nextn = int(md.get(f"{ASSISTANT_ARCH}.nextn_predict_layers", 0))
        if block_count != nextn:
            raise ValueError(f"{path}: block_count {block_count} != nextn_predict_layers {nextn}")
        if block_count != DRAFT_LAYERS:
            raise ValueError(f"{path}: {block_count} assistant layers, but this draft reads exactly {DRAFT_LAYERS}")

        if is_gguf_path(target_path):
            t_md = load_gguf_metadata(target_path)
            target_hidden = int(t_md[f"{gguf_architecture(target_path)}.embedding_length"])
            tokens = t_md.get("tokenizer.ggml.tokens")
            if tokens is None:
                raise ValueError(f"{target_path}: no tokenizer.ggml.tokens to match the draft's vocab against")
            target_vocab = len(tokens)
        else:
            from freetoken.utils import cached_load_hf_config

            hf = cached_load_hf_config(target_path)
            text = getattr(hf, "text_config", None) or hf
            target_hidden, target_vocab = int(text.hidden_size), int(text.vocab_size)
        out = int(md[f"{ASSISTANT_ARCH}.embedding_length_out"])
        if out != target_hidden:
            raise ValueError(
                f"{path}: an assistant for a hidden-{out} backbone, but the target {target_path} is hidden {target_hidden}; "
                "the sidecar must be the assistant of the same release as the target"
            )
        draft_vocab = int(gguf_tensor_index(path)["token_embd.weight"].ne[1])
        if draft_vocab != target_vocab:
            raise ValueError(f"{path}: the draft head covers {draft_vocab} tokens, but the target {target_path} has {target_vocab}")

    def __init__(self, config: ModelConfig, target_model) -> None:
        # ``config`` describes the draft itself (gguf.assistant_config): the target's
        # widths differ everywhere, including 2 * hidden for the fusion below
        self.config = config
        hidden = config.hidden_size
        target_hidden = target_model.model.embed_tokens.embedding_dim
        self.pre_projection = LinearReplicated(
            2 * target_hidden, hidden, has_bias=False, quant_config=config.quant, prefix="pre_projection"
        )
        self.layers = OPList([Gemma4DecoderLayer(config, layer_id, prefix=f"layers.{layer_id}") for layer_id in range(config.num_layers)])
        self.norm = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_projection = LinearReplicated(
            hidden, target_hidden, has_bias=False, quant_config=config.quant, prefix="post_projection"
        )
        # the draft's own tied table, output-only: input embeddings are the target's
        self.vocab = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=hidden, quant_config=config.quant, prefix="vocab"
        )
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=hidden,
            tie_word_embeddings=True,
            tied_embedding=self.vocab,
            prefix="lm_head",
        )
        # the target's own module; an underscore name keeps it out of this module's state dict
        self._embed_tokens = target_model.model.embed_tokens

    @classmethod
    def load(cls, config: ModelConfig, target_model, model_path: str, device: torch.device, dtype: torch.dtype) -> Gemma4MTPDraft:
        from freetoken.models.gguf.reader import is_gguf_path

        from .gguf import ASSISTANT_ARCH, assistant_config, assistant_quant, iter_assistant_weights

        if not is_gguf_path(model_path):
            raise ValueError(f"the Gemma 4 MTP draft loads from a {ASSISTANT_ARCH} GGUF sidecar, not {model_path}")

        draft_model_config = dataclasses.replace(assistant_config(model_path), quant=assistant_quant(model_path))
        with torch.device("meta"), torch_dtype(dtype):
            draft = cls(draft_model_config, target_model)
        expected = draft.state_dict()
        loaded = {}
        for name, tensor in iter_assistant_weights(model_path, draft_model_config):
            if name not in expected:
                raise ValueError(f"{model_path}: unexpected MTP tensor {name!r}")
            loaded[name] = tensor.to(device=device, dtype=expected[name].dtype)
        if missing := sorted(set(expected) - set(loaded)):
            raise ValueError(f"{model_path}: MTP tensors missing: {missing}")
        draft.load_state_dict(loaded)
        # not the root: it reaches the target's embedding, which the target finalized already
        finalize_quant(draft.pre_projection)
        finalize_quant(draft.post_projection)
        finalize_quant(draft.layers)
        finalize_quant(draft.vocab)
        return draft

    @property
    def embed_tokens(self) -> VocabParallelEmbedding:
        return self._embed_tokens

    def forward_hidden(self, embeds: torch.Tensor, prev_hidden: torch.Tensor) -> torch.Tensor:
        """The draft hidden of each row: ``embeds`` is the target's (sqrt-scaled) embedding of its token,
        ``prev_hidden`` the target hidden in target width."""
        x = self.pre_projection.forward(torch.cat([embeds, prev_hidden], dim=-1))
        for layer in self.layers.op_list:
            x = layer.forward(x, None, None)
        return self.norm.forward(x)

    def draft_logits(self, normed: torch.Tensor) -> torch.Tensor:
        """The draft samples its own tied table, uncapped; the target's head is never used."""
        return self.lm_head.logits(normed)

    def recurrent_hidden(self, normed: torch.Tensor) -> torch.Tensor:
        """The ``prev_hidden`` the next chain step reads, back in target width."""
        return self.post_projection.forward(normed)
