"""GLM-5.3-Flash MTP (multi-token prediction) block -- the checkpoint's layer 45.

DeepSeek-style draft: ``x = eh_proj([enorm(embed(t_next)) ; hnorm(h_main)])`` runs through
ONE decoder block and predicts the token after ``t_next`` through ``shared_head.norm`` +
the shared ``lm_head``. CHECKPOINT FACT: layer 45 ships NO hc_* weights -- the MTP block
is a PLAIN pre-norm residual block (single stream), unlike the mHC main stack. Its MLA/DSA
attention appends to its OWN slab row in the shared paged pool (layer 45 was added to the
full attention group) at the same positions/page rows as the main stack.

The MTP MoE experts (288, ~3.9G) must be VRAM-resident (add 45 to
FREETOKEN_GLM5_RESIDENT_LAYERS): the offload bank layout only covers layers < 45.
"""

from __future__ import annotations

import torch

from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.models.config import ModelConfig


class _SharedHead(BaseOP):
    def __init__(self, dim: int, eps: float):
        self.norm = RMSNorm(size=dim, eps=eps)


class _MtpLayer(BaseOP):
    """Plain pre-norm residual decoder block (no mHC -- layer 45 has no hc weights)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        from .attention import FullAttention
        from .moe import Glm5SparseBlock

        self.self_attn = FullAttention(config, layer_id)
        self.mlp = Glm5SparseBlock(config, layer_id)
        self.input_layernorm = RMSNorm(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            size=config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn.forward(self.input_layernorm.forward(x))
        return x + self.mlp.forward(self.post_attention_layernorm.forward(x))


class Glm5MtpBlock(BaseOP):
    def __init__(self, config: ModelConfig):
        a = config.glm5_args
        assert a.mtp_layer_id in a.resident_layer_ids, (
            "MTP needs its experts VRAM-resident: add 45 to FREETOKEN_GLM5_RESIDENT_LAYERS"
        )
        d = config.hidden_size
        self.enorm = RMSNorm(size=d, eps=a.norm_eps)
        self.hnorm = RMSNorm(size=d, eps=a.norm_eps)
        self.eh_proj = LinearReplicated(2 * d, d, has_bias=False)
        self.layer = _MtpLayer(config, a.mtp_layer_id)
        self.shared_head = _SharedHead(d, a.norm_eps)

    def draft_logits(
        self, h_pre_norm: torch.Tensor, tok_embed: torch.Tensor, lm_head
    ) -> torch.Tensor:
        """One draft step: ``h_pre_norm`` [T, D] (main stack's collapsed pre-final-norm
        hidden at these positions), ``tok_embed`` [T, D] (embedding of the token the main
        model just emitted there). Returns logits [T, V] predicting the FOLLOWING token.
        Runs under the CURRENT batch ctx (same positions/out_loc -> layer-45 slab)."""
        return self.draft_step(h_pre_norm, tok_embed, lm_head)[0]

    def draft_step(
        self, h_pre_norm: torch.Tensor, tok_embed: torch.Tensor, lm_head
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """draft_logits plus the block's output hidden [T, D] (pre shared-head norm) --
        the ``h`` input of the NEXT chained draft (DeepSeek-style depth-1 chaining)."""
        x = torch.cat(
            [self.enorm.forward(tok_embed), self.hnorm.forward(h_pre_norm)], dim=-1
        )
        x = self.eh_proj.forward(x)
        x = self.layer.forward(x)
        return lm_head.forward(self.shared_head.norm.forward(x)), x


__all__ = ["Glm5MtpBlock"]
