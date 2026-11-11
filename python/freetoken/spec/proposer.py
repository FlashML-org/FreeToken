"""The draft side of speculative decoding: what every proposer declares and implements, and the engine state it may touch."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, ClassVar, List, NamedTuple

import torch

if TYPE_CHECKING:
    from freetoken.core import Batch, Req
    from freetoken.models.blocks import FeatureSink
    from freetoken.models.config import SlotStateSpec


def check_draft_path(path: str) -> None:
    """Fail at config time on a local ``--speculative-draft-model-path`` that does not exist; a HF repo id is "name" or "org/name"."""
    local = os.path.isabs(path) or path.startswith(".") or path.count("/") > 1 or path.endswith(".gguf")
    if local and not os.path.exists(path):
        raise ValueError(f"--speculative-draft-model-path {path} does not exist")


class SpecProposal(NamedTuple):
    """One step's drafts and, when they were sampled, the distribution each was drawn from, kept on its support; None when the drafts are argmaxes."""

    draft_token_ids: torch.Tensor                       # [bs, k] int64 target-vocab ids
    draft_support_ids: torch.Tensor | None = None       # [bs, k, K] the token ids where q can be nonzero
    draft_support_probs: torch.Tensor | None = None     # [bs, k, K] fp32 q over them, summing to 1


class DraftContext(NamedTuple):
    """Built by SpecRunner.attach; the only engine state the draft side gets. A rebuild replaces it."""

    page_table: torch.Tensor
    num_pages: int
    device: torch.device
    stream: torch.cuda.Stream
    # the engine's padding request: its table row points at the dummy page
    dummy_req: Req
    # the tensors of the proposer's slot_states, ``[num_slots, *shape]`` each, on the target's GDN slots
    slot_states: dict[str, torch.Tensor]


class Proposer:
    """Draft side of one speculative method; holds no engine reference."""

    # True: the draft's KV slot i depends on token i+1, so the prefix cache keys position i by (t_i, t_{i+1})
    radix_lookahead: ClassVar[bool] = False

    @classmethod
    def adjust_config(cls, config, override) -> None:
        """Raise if this proposer cannot serve the config; the base rejects TP > 1."""
        if config.tp_info.size > 1:
            raise ValueError(
                "speculative decoding needs --tp 1 for now: without a cross-rank agreement on the drafts, "
                "near-tied draft argmaxes can differ between ranks and the ranks diverge"
            )

    @classmethod
    def prepare_target(cls, config, model) -> None:
        """Called with the target's weights loaded, before its kernels finalize them."""

    @classmethod
    def num_steps(cls, config) -> tuple[int, int]:
        """(default, max) draft tokens per step; called after ``adjust_config``."""
        raise NotImplementedError

    @classmethod
    def budget(cls, config) -> tuple[int, int]:
        """(fixed bytes, draft KV bytes per target KV token), from the config alone."""
        return 0, 0

    @classmethod
    def slot_states(cls, config) -> tuple[SlotStateSpec, ...]:
        """Per-request draft state that rides the target's GDN slots and is copied and restored with them; from the config alone. The proposer writes its own prefill track snapshot."""
        return ()

    def __init__(self, config, target_model) -> None:
        self.config = config
        self.k = config.speculative_num_steps

    def attach(self, ctx: DraftContext) -> None:
        """Build the draft KV pool, backends and buffers; called again with a new context after a rebuild."""
        self.ctx = ctx

    def detach(self, *, keep_kv: bool) -> None:
        """Free draft graphs and drop the context before a rebuild swaps the pools; draft KV on the target's slots goes too unless keep_kv says the target KV pool stays."""
        self.ctx = None

    def propose(self, reqs: List[Req], rows: torch.Tensor, pending: torch.Tensor) -> SpecProposal:
        """k drafts per request after its pending token; ``rows`` holds the requests' table_idx and ``pending`` their pending tokens, on the device."""
        raise NotImplementedError

    def feature_sink(self, batch: Batch) -> FeatureSink | None:
        """What the target forward of a prefill or verify batch hands the draft."""
        return None

    def post_verify(self, batch: Batch, out: torch.Tensor, emitted: torch.Tensor) -> None:
        """Advance the draft on the verify's features; ``batch`` is the padded verify batch."""

    def on_prefill(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None:
        pass

    def capture_graphs(self, bs_list: List[int], pool) -> None:
        pass

    def release(self, req: Req) -> None:
        pass
