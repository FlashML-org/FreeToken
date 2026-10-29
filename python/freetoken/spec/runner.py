"""SpecRunner: the target side of speculative decoding and the order of one step; the Proposer it holds is the draft side."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from freetoken.engine.sample import make_device_tensor
from freetoken.engine.spec_protocol import SpecForwardOutput

from .proposer import DraftContext, Proposer
from .rejection import reject

if TYPE_CHECKING:
    from freetoken.core import Batch, Req
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.models.blocks import FeatureSink


def proposer_class(config) -> type[Proposer]:
    from freetoken.models.register import _load_attr
    from freetoken.spec import PROPOSERS

    module, attr = PROPOSERS[config.speculative_algorithm].split(":")
    return _load_attr(module, attr)


class SpecRunner:
    """Implements engine.spec_protocol.SpecRuntime; the engine loads it by name and never imports freetoken.spec."""

    def __init__(self, config, engine) -> None:
        self.config = config
        self.device = engine.device
        self.k = config.speculative_num_steps
        self.proposer = proposer_class(config)(config, engine.model)
        self.radix_lookahead = self.proposer.radix_lookahead
        self.engine = None

    @classmethod
    def adjust_config(cls, config, override) -> None:
        proposer_cls = proposer_class(config)
        if config.speculative_num_steps == 0:
            override("speculative_num_steps", proposer_cls.default_k)
        if not 1 <= config.speculative_num_steps <= proposer_cls.max_k:
            raise ValueError(
                f"--speculative-num-steps must be in [1, {proposer_cls.max_k}] for "
                f"{config.speculative_algorithm}, got {config.speculative_num_steps}"
            )
        proposer_cls.adjust_config(config, override)

    @classmethod
    def budget(cls, config) -> tuple[int, int]:
        return proposer_class(config).budget(config)

    def attach(self, engine) -> None:
        """Second construction phase, after the pools and before graph capture."""
        self.engine = engine
        self._host = torch.zeros(self.config.max_running_req + 1, self.k + 2, dtype=torch.int64, pin_memory=True)
        self.proposer.attach(DraftContext(
            page_table=engine.page_table, num_pages=engine.num_pages, device=engine.device,
            stream=engine.stream, dummy_req=engine.dummy_req,
        ))

    def detach(self, *, keep_kv: bool) -> None:
        self.proposer.detach(keep_kv=keep_kv)

    def capture_graphs(self, bs_list: List[int], pool) -> None:
        self.proposer.capture_graphs(bs_list, pool)

    def feature_sink(self, batch: Batch) -> FeatureSink | None:
        return self.proposer.feature_sink(batch)

    def on_prefill(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None:
        self.proposer.on_prefill(batch, next_tokens_gpu)

    def propose_into(self, reqs: List[Req], token_pool: torch.Tensor) -> None:
        """Draft k tokens per request into token_pool after its pending token at cached_len."""
        rows = make_device_tensor([r.table_idx for r in reqs], torch.int64, self.device)
        drafts = self.proposer.propose(reqs, rows)
        cols = make_device_tensor([r.cached_len for r in reqs], torch.int64, self.device).unsqueeze(1)
        token_pool[rows.unsqueeze(1), cols + 1 + torch.arange(self.k, device=self.device)] = drafts.to(token_pool.dtype)

    def verify_step(self, batch: Batch, sample_args: BatchSamplingArgs) -> SpecForwardOutput:
        engine = self.engine
        bs, width = batch.size, self.k + 1
        # each request's rows read its pending token then its drafts
        drafts = batch.input_ids[: bs * width].view(bs, width)[:, 1:].to(torch.int64)
        batch.features = self.feature_sink(batch)
        logits = engine.forward_logits(batch)[: bs * width]
        out, emitted = reject(logits, drafts, sample_args, engine.sampler)
        host = self._host[:bs]
        host.copy_(torch.cat([out, emitted.unsqueeze(1)], dim=1), non_blocking=True)
        copy_done = torch.cuda.Event()
        copy_done.record(engine.stream)
        # after the D2H is queued: the drain waits for the copy only, while the draft's work overlaps it
        self.proposer.post_verify(batch, out, emitted)
        return SpecForwardOutput(out, host[:, :width], copy_done, host[:, width])

    def release(self, req: Req) -> None:
        self.proposer.release(req)
