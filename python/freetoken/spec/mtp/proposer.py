"""MTPProposer: drafts with the checkpoint's own MTP layer, one draft token per chained forward."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, List, Protocol

import torch

from freetoken.core import Batch, get_global_ctx
from freetoken.engine.sample import make_device_tensor
from freetoken.spec.proposer import DraftContext, Proposer
from freetoken.spec.relay import HiddenRelay

if TYPE_CHECKING:
    from freetoken.core import Req
    from freetoken.models.config import ModelConfig


class MTPDraft(Protocol):
    """The class a family registers as ``ModelSpec.spec_drafts["mtp"]``."""

    # the draft's own config, from draft_config
    config: ModelConfig
    # None: the proposer picks from the target's backends
    attention_backend: ClassVar[str | None]

    @staticmethod
    def draft_config(config: ModelConfig) -> ModelConfig: ...

    @staticmethod
    def relay_width(config: ModelConfig) -> int:
        """Width of the hidden a draft row reads and relays to the next step."""
        ...

    @staticmethod
    def check_source(path: str, target_path: str) -> None:
        """Raise ValueError if ``path`` cannot give this draft its weights; called before the target loads."""
        ...

    @classmethod
    def load(cls, config: ModelConfig, target_model, model_path: str, device: torch.device, dtype: torch.dtype) -> MTPDraft: ...

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def embed_batch(self, batch: Batch) -> torch.Tensor: ...

    def forward_hidden(self, embeds: torch.Tensor, prev_hidden: torch.Tensor) -> torch.Tensor: ...

    def head_input(self, hidden: torch.Tensor) -> torch.Tensor:
        """What the target's LM head reads from a draft hidden."""
        ...


def with_resident_moe(config: ModelConfig) -> ModelConfig:
    """``config`` with its MoE resident on the GPU, the strategy every draft builds with whatever the target's."""
    return dataclasses.replace(config, moe_strategy="fused", decode_target="gpu")


@dataclass
class _DraftReq:
    """The request fields the attention backends and rope read, for a draft batch over positions [cached_len, device_len)."""

    cached_len: int
    device_len: int
    table_idx: int
    mrope_delta: int = 0
    mrope_positions_full: torch.Tensor | None = None

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len


class _PrefillSink:
    """Every prefill row's final hidden, and each request's [cached_len, device_len) when the batch was formed."""

    wanted_layers = frozenset()

    def __init__(self, spans: List[tuple[int, int]]) -> None:
        self.spans = spans
        self.hidden: torch.Tensor | None = None

    def final_hidden(self, x: torch.Tensor) -> None:
        self.hidden = x

    def layer_output(self, layer_id: int, x: torch.Tensor, num_streams: int = 1) -> None:
        pass


class _VerifySink:
    """Copies the verify rows' final hidden into a persistent buffer, so a captured verify graph writes it too."""

    wanted_layers = frozenset()

    def __init__(self, buffer: torch.Tensor) -> None:
        self.buffer = buffer

    def final_hidden(self, x: torch.Tensor) -> None:
        self.buffer[: x.shape[0]].copy_(x)

    def layer_output(self, layer_id: int, x: torch.Tensor, num_streams: int = 1) -> None:
        pass


class MTPProposer(Proposer):
    default_k = 3
    max_k = 8
    radix_lookahead = True

    @classmethod
    def adjust_config(cls, config, override) -> None:
        import os

        from freetoken.checkpoint.ftw import is_ftw_checkpoint

        super().adjust_config(config, override)
        arch = config.hf_config.architectures[0]
        if "mtp" not in dict(config.model_spec.spec_drafts):
            raise ValueError(f"MTP is not supported for {arch} yet")
        if config.use_dummy_weight:
            raise ValueError("MTP reads its draft from the checkpoint, so it cannot run with --dummy-weight")
        path = config.speculative_draft_model_path or config.model_path
        # checked here because a draft that cannot load otherwise fails only after the whole target has loaded
        # a HF repo id is "name" or "org/name"; anything else names a local path
        local = os.path.isabs(path) or path.startswith(".") or path.count("/") > 1 or path.endswith(".gguf")
        if config.speculative_draft_model_path and local and not os.path.exists(path):
            raise ValueError(f"--speculative-draft-model-path {path} does not exist")
        # FTW keeps only what the target's iter_weights yields, which drops mtp.*
        if is_ftw_checkpoint(path):
            raise ValueError("MTP cannot load its draft from an FTW checkpoint yet")
        cls._draft_class(config).check_source(path, config.model_path)

    @classmethod
    def _draft_class(cls, config) -> type[MTPDraft]:
        from freetoken.models.register import _load_attr

        return _load_attr(*dict(config.model_spec.spec_drafts)["mtp"].split(":"))

    @classmethod
    def budget(cls, config) -> tuple[int, int]:
        from freetoken.kvcache.base import spec_kv_bytes_per_token

        draft_cls = cls._draft_class(config)
        relay_bytes = draft_cls.relay_width(config.model_config) * config.dtype.itemsize
        rows = config.max_running_req + 1
        # relay hidden and carry, and the verify hiddens
        fixed = rows * (config.speculative_num_steps + 3) * relay_bytes
        groups = draft_cls.draft_config(with_resident_moe(config.model_config)).kv_cache_group_specs()
        return fixed, sum(spec_kv_bytes_per_token(group, config) for group in groups)

    def __init__(self, config, target_model) -> None:
        super().__init__(config, target_model)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = config.dtype
        path = config.speculative_draft_model_path or config.model_path
        from freetoken.layers.moe import OffloadMoELayer, iter_moe_layers

        draft_cls = self._draft_class(config)
        self.draft = draft_cls.load(with_resident_moe(config.model_config), target_model, path, self.device, config.dtype)
        # an offloaded or shared MoE layer here would read the target's experts as its own, silently
        target_moe = {id(m) for m in iter_moe_layers(target_model)}
        if any(isinstance(m, OffloadMoELayer) or id(m) in target_moe for m in iter_moe_layers(self.draft)):
            raise RuntimeError(f"{draft_cls.__name__} must build its own MoE layers, resident on the GPU")
        self.lm_head = target_model.lm_head
        self.mrope = config.model_config.model_is_mrope
        self._prefill_state: dict[int, tuple[int, int]] = {}  # table row -> (uid, first position not drafted yet)
        self._prefill_sink: _PrefillSink | None = None
        self.kv = None

    # ------------------------------------------------------------------ lifecycle

    def attach(self, ctx: DraftContext) -> None:
        from freetoken.attention import create_attention_backend
        from freetoken.kvcache import create_kvcache_pool

        super().attach(ctx)
        cfg = self.draft.config
        rows = self.config.max_running_req + 1
        # the draft's own layers on the target's slot ids: same page size, same pages plus the dummy page padded rows write to
        # a rebuild that keeps the target KV pool keeps this one, which the prefix cache still indexes
        if self.kv is None:
            self.kv = create_kvcache_pool(
                model_config=cfg, num_pages=ctx.num_pages + 1, page_size=self.config.page_size, dtype=self.dtype,
                device=ctx.device, num_req_slots=rows, num_speculative_tokens=self.k,
            )
        width = self.draft.relay_width(self.config.model_config)
        self.relay = HiddenRelay(rows, width, self.dtype, ctx.device)
        self.verify_hidden = torch.zeros(rows * (self.k + 1), width, dtype=self.dtype, device=ctx.device)
        # verify row of each request's last accepted position; padded rows keep stale but in-range values
        self.picked_rows = torch.arange(rows, dtype=torch.int64, device=ctx.device) * (self.k + 1)
        self.extend_graphs = self.chain_graphs = None
        parts = [p.strip() for p in self.config.attention_backend.split(",")]
        fixed = self.draft.attention_backend
        # backends bind the context's KV pool when they are built
        with get_global_ctx().override(kv_cache=self.kv):
            self.extend_backend = create_attention_backend(fixed or ("fa" if "fa" in parts else "triton"), cfg)
            self.chain_backend = create_attention_backend(fixed or "triton", cfg)

    def detach(self, *, keep_kv: bool) -> None:
        if self.extend_graphs is not None:
            self.extend_graphs.destroy_cuda_graphs()
        self.extend_graphs = self.chain_graphs = None
        if not keep_kv:
            self.kv = None
        self.relay = self.verify_hidden = self.picked_rows = None
        self.extend_backend = self.chain_backend = None
        self._prefill_state.clear()
        super().detach(keep_kv=keep_kv)

    def release(self, req: Req) -> None:
        self._prefill_state.pop(req.table_idx, None)

    # ------------------------------------------------------------------ draft forwards

    def _draft_batch(self, reqs: List[_DraftReq], phase: str, rows_per_req: int | None, backend) -> Batch:
        batch = Batch(reqs=reqs, phase=phase, rows_per_req=rows_per_req)
        batch.padded_reqs = reqs
        host = torch.cat([torch.arange(r.cached_len, r.device_len, dtype=torch.int32) for r in reqs])
        batch.positions = host.pin_memory().to(self.device, non_blocking=True)
        if self.mrope:
            batch.mrope_positions = self._mrope_positions(reqs).pin_memory().to(self.device, non_blocking=True)
        rows = torch.cat([torch.full((r.extend_len,), r.table_idx, dtype=torch.int64) for r in reqs])
        rows = rows.pin_memory().to(self.device, non_blocking=True)
        batch.out_loc = self.ctx.page_table[rows, batch.positions.to(torch.int64)]
        backend.prepare_metadata(batch)
        return batch

    @staticmethod
    def _mrope_positions(reqs: List[_DraftReq]) -> torch.Tensor:
        """[3, n] rope rows of the draft positions, by the target's rule: a prompt position's precomputed column, else position + delta."""
        cols = []
        for r in reqs:
            pos = torch.arange(r.cached_len, r.device_len, dtype=torch.int32)
            col = (pos + r.mrope_delta).unsqueeze(0).expand(3, -1).clone()
            full = r.mrope_positions_full
            if full is not None:
                inside = pos < full.shape[1]
                col[:, inside] = full[:, pos[inside].long()]
            cols.append(col)
        return torch.cat(cols, dim=1)

    def _forward(self, batch: Batch, backend, embeds: torch.Tensor, prev_hidden: torch.Tensor) -> torch.Tensor:
        with get_global_ctx().forward_batch(batch, kv_cache=self.kv, attn_backend=backend, hide_target_state=True):
            return self.draft.forward_hidden(embeds, prev_hidden)

    def _draft_token(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head.logits(self.draft.head_input(hidden)).argmax(dim=-1)

    def _view(self, req, cached_len: int, device_len: int) -> _DraftReq:
        return _DraftReq(cached_len, device_len, req.table_idx, getattr(req, "mrope_delta", 0), getattr(req, "mrope_positions_full", None))

    # ------------------------------------------------------------------ one step

    def capture_graphs(self, bs_list: List[int], pool) -> None:
        from freetoken.engine.graph import GraphRunner

        from .chain_graph import ChainGraphs

        if not bs_list:
            return
        ctx = self.ctx
        self.extend_graphs = GraphRunner(
            stream=ctx.stream,
            device=ctx.device,
            forward=self._extend_forward,
            attn_backend=self.extend_backend,
            cuda_graph_bs=bs_list,
            cuda_graph_max_bs=max(bs_list),
            free_memory=0,
            # the table width, not the request limit: verify widens a request k rows past the limit
            max_seq_len=ctx.page_table.shape[1],
            dummy_req=ctx.dummy_req,
            mrope=self.mrope,
            rows_per_req=self.k + 1,
            phase="verify",
            ctx_override=dict(kv_cache=self.kv, attn_backend=self.extend_backend, hide_target_state=True),
            pool=pool,
        )
        if self.k > 1 and ChainGraphs.supports(self.chain_backend):
            self.chain_graphs = ChainGraphs(self, bs_list, self.extend_graphs.graph_pool)

    def propose(self, reqs: List[Req], rows: torch.Tensor) -> torch.Tensor:
        cs = [r.cached_len for r in reqs]
        if self.chain_graphs is not None and (drafts := self.chain_graphs.replay(reqs, cs)) is not None:
            return drafts
        token = self.relay.d1[rows]
        hidden = self.relay.hidden[rows]
        drafts = [token]
        for j in range(1, self.k):
            # chain step j drafts at position C+j-1 with the previous step's draft hidden standing in for the target's
            views = [self._view(r, c + j - 1, c + j) for r, c in zip(reqs, cs)]
            batch = self._draft_batch(views, "decode", 1, self.chain_backend)
            hidden = self._forward(batch, self.chain_backend, self.draft.embed(token), hidden)
            token = self._draft_token(hidden)
            drafts.append(token)
        return torch.stack(drafts, dim=1)

    def feature_sink(self, batch: Batch) -> _PrefillSink | _VerifySink:
        if batch.phase == "verify":
            return _VerifySink(self.verify_hidden)
        self._prefill_sink = _PrefillSink([(r.cached_len, r.device_len) for r in batch.reqs])
        return self._prefill_sink

    def _extend_forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft hiddens of a verify-shaped batch; returns each request's row at its last accepted position and that row's argmax."""
        input_ids = get_global_ctx().batch.input_ids
        n = input_ids.shape[0]
        hidden = self.draft.forward_hidden(self.draft.embed(input_ids), self.verify_hidden[:n])
        picked = hidden[self.picked_rows[: n // (self.k + 1)]]
        return picked, self._draft_token(picked)

    def post_verify(self, batch: Batch, out: torch.Tensor, emitted: torch.Tensor) -> None:
        """Redraft positions C..C+k from the target's verify hiddens, then relay each request's row at its last accepted position."""
        k1 = self.k + 1
        bs, padded = batch.size, batch.padded_size
        tokens = torch.zeros(padded, k1, dtype=torch.int64, device=self.device)
        tokens[:bs] = out
        views = [self._view(r, r.cached_len, r.device_len) for r in batch.padded_reqs]
        draft = Batch(reqs=views[:bs], phase="verify", rows_per_req=k1)
        draft.padded_reqs = views
        draft.input_ids = tokens.view(-1)
        draft.positions = batch.positions
        draft.mrope_positions = batch.mrope_positions
        draft.out_loc = batch.out_loc
        draft.active_table_idx = batch.active_table_idx
        self.picked_rows[:bs] = torch.arange(bs, device=self.device) * k1 + emitted
        self.extend_backend.prepare_metadata(draft)
        graphs = self.extend_graphs
        if graphs is not None and graphs.can_use_cuda_graph(draft) and padded in graphs.graph_map:
            picked, d1 = graphs.replay(draft)
        else:
            with get_global_ctx().forward_batch(draft, kv_cache=self.kv, attn_backend=self.extend_backend, hide_target_state=True):
                picked, d1 = self._extend_forward()
            picked, d1 = picked[:bs], d1[:bs]
        table = make_device_tensor([r.table_idx for r in batch.reqs], torch.int64, self.device)
        self.relay.hidden[table] = picked
        self.relay.d1[table] = d1

    def on_prefill(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None:
        """Draft the chunk's positions: row p pairs token p+1 with the target hidden at p; a final chunk adds the sampled token's row and relays it."""
        sink, self._prefill_sink = self._prefill_sink, None
        hidden = sink.hidden
        # image rows take the embeddings the target used
        embeds = self.draft.embed_batch(batch)
        views, emb, prev, finals = [], [], [], []
        offset = drafted = 0
        for i, (req, (s, e)) in enumerate(zip(batch.reqs, sink.spans)):
            rows = hidden[offset : offset + e - s]
            chunk_embeds = embeds[offset : offset + e - s]
            offset += e - s
            row = req.table_idx
            state = self._prefill_state.get(row)
            # a continuation chunk resumes where the last one stopped; a first chunk starts at its prefix hit, whose draft KV the tree already holds
            frontier = state[1] if state is not None and state[0] == req.uid else s
            final = e == req.prompt_len
            end = e if final else e - 1
            if frontier < end:
                # draft row p embeds the token at p+1
                emb.append(chunk_embeds[frontier + 1 - s : min(end + 1, e) - s])
                if final:
                    emb.append(self.draft.embed(next_tokens_gpu[i : i + 1]))
                if frontier == s - 1:
                    # cloned: the carry of this row is overwritten below, before the draft forward reads it
                    prev += [self.relay.carry[row : row + 1].clone(), rows[: end - s]]
                else:
                    prev.append(rows[frontier - s : end - s])
                views.append(self._view(req, frontier, end))
                drafted += end - frontier
                if final:
                    finals.append((drafted - 1, row))
            self.relay.carry[row] = rows[-1]
            self._prefill_state[row] = (req.uid, max(frontier, end))
        if not views:
            return
        draft = self._draft_batch(views, "prefill", None, self.extend_backend)
        out = self._forward(draft, self.extend_backend, torch.cat(emb), torch.cat(prev))
        if finals:
            idx = make_device_tensor([i for i, _ in finals], torch.int64, self.device)
            table = make_device_tensor([r for _, r in finals], torch.int64, self.device)
            picked = out[idx]
            self.relay.hidden[table] = picked
            self.relay.d1[table] = self._draft_token(picked)
