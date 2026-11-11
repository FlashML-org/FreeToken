from __future__ import annotations

from typing import Dict

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.utils import div_ceil, init_logger, nvtx_annotate

from .base import BaseOP
from .quantization import LayerKind, QuantConfig, quant_method_for

logger = init_logger(__name__)


class VocabParallelEmbedding(BaseOP):
    """The vocab table, sharded over TP: its ``quant_method`` owns the table, looks rows up, and serves a tied LM head's product."""

    quant_layer_kind = LayerKind.EMBEDDING

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float | None = None,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.embedding_dim = embedding_dim
        self.prefix = prefix
        # the LM head subclass is a linear and builds its own method
        self.quant_method = None
        if self.quant_layer_kind is LayerKind.EMBEDDING:
            self.quant_method = quant_method_for(quant_config, self, prefix)
            self.quant_method.create_weights(self)
        # Gemma scales embeddings by sqrt(hidden_size). The scale is materialized in
        # the weight dtype (bf16) to match HF, which downcasts the scalar. The GPU
        # scalar is built lazily (model __init__ runs on the meta device) and cached
        # so it is not reallocated inside a captured CUDA graph.
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.quant_method.embed(self, x)
        if self.tp_size > 1:
            y = self._comm.all_reduce(y)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(
                    self._embed_scale, dtype=y.dtype, device=y.device
                )
            y = y * self._embed_scale_t
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """The head is a linear layer over the vocab shard: its weights come from ``quant_method``
    unless they are tied to the input embedding."""

    quant_layer_kind = LayerKind.LINEAR

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.has_bias = bias
        self.prefix = prefix
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings
        if tied_embedding is not None and tied_embedding.quant_method.online:
            # the head multiplies by the whole table every step, which an online table would decode in full each time
            logger.info("--online-quant leaves %s as stored: the LM head is tied to it", tied_embedding.prefix)
            tied_embedding.quant_method = quant_method_for(None, tied_embedding, tied_embedding.prefix)
            tied_embedding.quant_method.create_weights(tied_embedding)
        self.in_features = embedding_dim
        self.out_features = self.num_embeddings_tp
        self.output_sizes = (self.num_embeddings_tp,)
        if tied_embedding is None:
            self.quant_method = quant_method_for(quant_config, self, prefix)
            self.quant_method.create_weights(self)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        # set by keep_rows: a head over a subset of the rows, and the token id of each of its rows
        self.kept_rows: tuple[ParallelLMHead, torch.Tensor] | None = None

    def finalize(self) -> None:
        if self.quant_method is not None:
            self.quant_method.finalize(self)

    def keep_rows(self, ids: torch.Tensor, quant_config: QuantConfig | None) -> None:
        """Set ``kept_rows`` to a head with its own weights over only the rows ``ids``, repeated up to a multiple of 128 rows.

        Call after the weights load and before ``finalize_quant``: a kernel's finalize can rewrite the weights so that rows cannot be picked, and ``finalize_quant`` also finalizes the new head.
        """
        # scaled_mm needs N % 16, b12x N % 8, and the sm_120 GEMV's fast path N % 128; a repeated row maps to the same token
        ids = ids[torch.arange(div_ceil(ids.numel(), 128) * 128) % ids.numel()]
        rows = ids.numel()
        with torch.device("meta"):
            head = ParallelLMHead(rows, self.embedding_dim, self.has_bias, quant_config=quant_config, prefix=self.prefix)
        full = self.state_dict()
        ids = ids.to(next(iter(full.values())).device)
        for name, want in head.state_dict().items():
            have = full[name]
            if want.dim() == 0:
                kept = have.clone()
            elif want.shape[0] >= rows and have.shape[1:] == want.shape[1:] and have.shape[0] - self.num_embeddings == want.shape[0] - rows:
                kept = have.index_select(0, ids)
                if want.shape[0] > rows:
                    # ggml's kernels read past the last row and need zeros there
                    kept = torch.cat([kept, kept.new_zeros(want.shape[0] - rows, *kept.shape[1:])])
            else:
                raise NotImplementedError(f"{self.prefix}.{name} is not stored by row (shape {list(have.shape)}), so the head cannot keep a subset of its rows")
            setattr(head, name, kept)
        self.kept_rows = head, ids

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor, *, export_hidden: bool = True) -> torch.Tensor:
        """``export_hidden=False``: the model hands its own final hidden to ``batch.features``."""
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if export_hidden and (features := batch.features) is not None:
            features.final_hidden(x)
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices
        return self.logits(x)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """``num_embeddings`` logits for each row of ``x``; reads no batch state."""
        if self.tied_embedding is not None:
            logits = self.tied_embedding.quant_method.logits(self.tied_embedding, x, self.bias)
        else:
            logits = self.quant_method.apply(self, x)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        # keyed on rows, not batch size: a verify batch has several rows per request
        if input_shape[0] == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]