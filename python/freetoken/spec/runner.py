"""SpecRunner: the target side of speculative decoding and the order of one step; the Proposer it holds is the draft side."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from freetoken.engine.spec_protocol import SpecForwardOutput

from .proposer import DraftContext, Proposer
from .rejection import RejectionSampler

if TYPE_CHECKING:
    from freetoken.core import Batch, Req
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.models.blocks import FeatureSink
    from freetoken.models.config import SlotStateSpec

_AUTO = object()


def proposer_class(config) -> type[Proposer]:
    from freetoken.models.register import _load_attr
    from freetoken.spec import PROPOSERS

    module, attr = PROPOSERS[config.speculative_algorithm].split(":")
    return _load_attr(module, attr)


class SpecRunner:
    """Implements engine.spec_protocol.SpecRuntime; the engine loads it by name and never imports freetoken.spec."""

    def __init__(self, config, engine, proposer: Proposer | None = _AUTO) -> None:
        self.config = config
        self.device = engine.device
        self.proposer_cls = proposer_class(config)
        self.k = config.speculative_num_steps
        self.rows_per_req = config.decode_rows_per_req
        self.radix_lookahead = self.proposer_cls.radix_lookahead
        self.proposer = self.proposer_cls(config, engine.model) if proposer is _AUTO else proposer
        self.engine = None

    @classmethod
    def default_k(cls, config) -> int:
        return proposer_class(config).default_k

    @classmethod
    def validate(cls, config) -> None:
        proposer_cls = proposer_class(config)
        if not 1 <= config.speculative_num_steps <= proposer_cls.max_k:
            raise ValueError(
                f"--speculative-num-steps must be in [1, {proposer_cls.max_k}] for "
                f"{config.speculative_algorithm}, got {config.speculative_num_steps}"
            )
        proposer_cls.validate(config)

    @classmethod
    def budget(cls, config) -> tuple[int, int]:
        return proposer_class(config).budget(config)

    @classmethod
    def slot_states(cls, config) -> tuple[SlotStateSpec, ...]:
        return proposer_class(config).slot_states(config)

    def attach(self, engine) -> None:
        """Second construction phase, after the pools and before graph capture."""
        self.engine = engine
        table_rows = self.config.max_running_req + 1
        self.rejection = RejectionSampler()
        self._out = torch.zeros(table_rows, self.rows_per_req, dtype=torch.int64, device=self.device)
        self._host = torch.zeros(table_rows, self.rows_per_req + 1, dtype=torch.int64, pin_memory=True)
        if self.proposer is not None:
            pool = engine.linear_state_pool
            self.proposer.attach(DraftContext(
                page_table=engine.page_table, num_pages=engine.num_pages, device=engine.device,
                stream=engine.stream, dummy_req=engine.dummy_req,
                slot_states=dict(pool.slot_states) if pool is not None else {},
            ))

    def detach(self, *, keep_kv: bool) -> None:
        if self.proposer is not None:
            self.proposer.detach(keep_kv=keep_kv)

    def _device_ints(self, values: List[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int64, pin_memory=self.device.type == "cuda").to(self.device, non_blocking=True)

    def propose_into(self, reqs: List[Req], token_pool: torch.Tensor) -> None:
        """Draft k tokens per request into token_pool after its pending token at cached_len."""
        cs = [r.cached_len for r in reqs]
        rows = self._device_ints([r.table_idx for r in reqs])
        self._proposal = self.proposer.propose(reqs, cs, rows, token_pool)
        cols = self._device_ints(cs).unsqueeze(1) + 1 + torch.arange(self.k, device=self.device)
        token_pool[rows.unsqueeze(1), cols] = self._proposal.tokens.to(token_pool.dtype)

    def verify_step(self, batch: Batch, sample_args: BatchSamplingArgs, token_pool: torch.Tensor) -> SpecForwardOutput:
        engine = self.engine
        bs, width = batch.size, self.rows_per_req
        batch.verify = engine.verify_info()
        batch.features = self.verify_sink(batch.padded_size * width)
        logits = engine.forward_logits(batch)[: bs * width]
        drafts = self._proposal.tokens[:bs].to(torch.int64)
        out, emitted = self.rejection(logits, drafts, sample_args, engine.sampler)
        self._out[self._device_ints([r.table_idx for r in batch.reqs])] = out
        host = self._host[:bs]
        host.copy_(torch.cat([out, emitted.unsqueeze(1)], dim=1), non_blocking=True)
        copy_done = torch.cuda.Event()
        copy_done.record(engine.stream)
        # after the D2H is queued: the drain waits for the copy only, while the draft's work overlaps it
        if self.proposer is not None:
            self.proposer.post_verify(batch, out, emitted)
        return SpecForwardOutput(out, host[:, :width], copy_done, host[:, width])

    def commit(self, rows: List[int], reqs: List[Req], entry_cached: List[int], counts: List[int],
               slots: List[int], token_pool: torch.Tensor) -> None:
        """Commit each request's first n verified tokens: target state, then the pending token at cached_len."""
        if not reqs:
            return
        self.engine.commit_verify(rows, reqs, slots, counts, self._out)
        table = self._device_ints([r.table_idx for r in reqs])
        n = self._device_ints(counts)
        token_pool[table, self._device_ints(entry_cached) + n] = self._out[table, n - 1].to(token_pool.dtype)

    def release(self, req: Req) -> None:
        if self.proposer is not None:
            self.proposer.release(req)

    def prefill_sink(self, batch: Batch) -> FeatureSink | None:
        return self.proposer.prefill_sink(batch) if self.proposer is not None else None

    def on_prefill(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None:
        if self.proposer is not None:
            self.proposer.on_prefill(batch, next_tokens_gpu)

    def verify_sink(self, rows: int) -> FeatureSink | None:
        return self.proposer.verify_sink(rows) if self.proposer is not None else None

    def capture_graphs(self, bs_list: List[int], pool) -> None:
        if self.proposer is not None:
            self.proposer.capture_graphs(bs_list, pool)
