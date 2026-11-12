"""SharedKVMTPProposer: drafts with an assistant checkpoint that keeps no KV of its own, reading the target's paged slabs through a read-only view. A family selects it through its spec_drafts behind the mtp CLI algorithm; it is never a --spec-algo name."""

from __future__ import annotations

from itertools import accumulate
from typing import TYPE_CHECKING, List, Protocol

import torch

from freetoken.core import Batch, get_global_ctx
from freetoken.engine.sample import make_device_tensor
from freetoken.spec.mtp.proposer import _DraftReq, _PrefillSink, _VerifySink, with_resident_moe
from freetoken.spec.mtp.shared_kv import ReadOnlyTargetKVView
from freetoken.spec.proposer import DraftContext, Proposer, SpecProposal, check_draft_path

if TYPE_CHECKING:
    from freetoken.core import Req
    from freetoken.layers.embedding import VocabParallelEmbedding
    from freetoken.models.config import ModelConfig


class SharedKVMTPDraft(Protocol):
    """The class a family registers as ``ModelSpec.spec_drafts["shared_kv_mtp"]``. It owns its vocab table and stores no KV."""

    @staticmethod
    def relay_width(config: ModelConfig) -> int:
        """Width of the hidden a draft row reads and relays to the next step."""
        ...

    @staticmethod
    def check_source(path: str, target_path: str) -> None:
        """Raise ValueError if ``path`` cannot give this draft its weights; called before the target loads."""
        ...

    @classmethod
    def load(cls, config: ModelConfig, target_model, model_path: str, device: torch.device, dtype: torch.dtype) -> SharedKVMTPDraft: ...

    @property
    def embed_tokens(self) -> VocabParallelEmbedding:
        """The target's token embedding, which the draft shares."""
        ...

    def forward_hidden(self, embeds: torch.Tensor, prev_hidden: torch.Tensor) -> torch.Tensor:
        """Every draft layer reads the target's paged slabs through the view and stores nothing."""
        ...

    def draft_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Logits from a vocab table the draft owns."""
        ...

    def recurrent_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """A forwarded draft hidden back in the relay width."""
        ...


class SharedKVMTPProposer(Proposer):
    # the draft adds no KV slot keyed by the next token, and the SWA radix cache asserts lookahead away
    radix_lookahead = False

    @classmethod
    def num_steps(cls, config) -> tuple[int, int]:
        return 3, 8

    @classmethod
    def adjust_config(cls, config, override) -> None:
        from freetoken.checkpoint.ftw import is_ftw_checkpoint

        super().adjust_config(config, override)
        if config.use_dummy_weight:
            raise ValueError("MTP reads its draft from the checkpoint, so it cannot run with --dummy-weight")
        path = config.speculative_draft_model_path or config.model_path
        # checked here because a draft that cannot load otherwise fails only after the whole target has loaded
        if config.speculative_draft_model_path:
            check_draft_path(path)
        # FTW keeps only what the target's iter_weights yields, which drops mtp.*
        if is_ftw_checkpoint(path):
            raise ValueError("MTP cannot load its draft from an FTW checkpoint yet")
        cls._draft_class(config).check_source(path, config.model_path)
        # refused here rather than at construction so every bad combination fails in the spec-config gate
        if config.model_config.model_is_mrope:
            raise ValueError("a read-only MTP draft has no mrope position mapping yet")

    @classmethod
    def _draft_class(cls, config) -> type[SharedKVMTPDraft]:
        from freetoken.models.register import _load_attr

        return _load_attr(*dict(config.model_spec.spec_drafts)["shared_kv_mtp"].split(":"))

    @classmethod
    def budget(cls, config) -> tuple[int, int]:
        relay_bytes = cls._draft_class(config).relay_width(config.model_config) * config.dtype.itemsize
        rows = config.max_running_req + 1
        # the seed hidden and the verify hiddens; the draft reads the target's slabs and allocates no KV of its own
        return rows * (config.speculative_num_steps + 2) * relay_bytes, 0

    @classmethod
    def target_may_be_windowed(cls, config) -> bool:
        from freetoken.kvcache import resolve_pool_class
        from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

        # the draft reads a strict subset of a verify row's KV, so the window frees cannot release a slot
        # it still reads; the gate reads the target only, so the DSV4 window tiers keep the refusal
        return resolve_pool_class(config.model_config) is HybridSWAKVCache

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
        self._prefill_sink: _PrefillSink | None = None
        self.kv = None

    # ------------------------------------------------------------------ lifecycle

    def attach(self, ctx: DraftContext) -> None:
        from freetoken.attention import create_attention_backend

        super().attach(ctx)
        cfg = self.draft.config
        rows = self.config.max_running_req + 1
        # a rebuild swaps the target pool, so the read-only view is rebuilt with every attach
        self.kv = ReadOnlyTargetKVView(ctx.kv_cache, self.draft.config, self.config.model_config)
        width = self.draft.relay_width(self.config.model_config)
        # the target hidden that seeds the next proposal, one row per table row
        self.seed_hidden = torch.zeros(rows, width, dtype=self.dtype, device=ctx.device)
        self.verify_hidden = torch.zeros(rows * (self.k + 1), width, dtype=self.dtype, device=ctx.device)
        # backends bind the context's KV pool when they are built; read-only draft forwards are
        # decode-shaped only, so one triton backend serves both roles
        with get_global_ctx().override(kv_cache=self.kv):
            self.chain_backend = create_attention_backend("triton", cfg)

    def detach(self, *, keep_kv: bool) -> None:
        self.kv = None  # a view over the old pool is dead once a rebuild swaps it, keep_kv included
        self.seed_hidden = self.verify_hidden = None
        self.chain_backend = None
        super().detach(keep_kv=keep_kv)

    # ------------------------------------------------------------------ draft forwards

    def _draft_batch_ro(self, table_idxs: List[int], qs: List[int]) -> Batch:
        """Decode-shaped batch for a read-only draft: row i sits at q_i and gathers ``[0, q_i)``, its own
        slot rolled back or not yet computed; the draft writes none (llama.cpp ``common_speculative_impl_draft_mtp::draft``)."""
        reqs = [_DraftReq(q, q + 1, t) for t, q in zip(table_idxs, qs)]
        batch = Batch(reqs=reqs, phase="decode", rows_per_req=1)
        batch.padded_reqs = reqs
        batch.positions = make_device_tensor(qs, torch.int32, self.device)
        indptr = torch.tensor([0] + list(accumulate(qs)), dtype=torch.int32).pin_memory().to(self.device, non_blocking=True)
        indices = torch.cat([self.ctx.page_table[t, :q] for t, q in zip(table_idxs, qs)])
        from freetoken.attention.triton import TritonMetadata

        batch.attn_metadata = TritonMetadata(
            cu_seqlens_q_gpu=torch.arange(len(qs) + 1, device=self.device, dtype=torch.int32),
            indptr=indptr,
            indices=indices,
            q_to_req=torch.arange(len(qs), device=self.device, dtype=torch.int32),
            q_positions=batch.positions,
            is_decode=True,
            prefix_lens=batch.positions.clone(),
            max_q_len=1,
            swa_indices=self.kv.translate_loc_from_full_to_swa(indices) if getattr(self.kv, "swa_paged", False) else None,
        )
        return batch

    def _forward(self, batch: Batch, backend, embeds: torch.Tensor, prev_hidden: torch.Tensor) -> torch.Tensor:
        with get_global_ctx().forward_batch(batch, kv_cache=self.kv, attn_backend=backend, hide_target_state=True):
            return self.draft.forward_hidden(embeds, prev_hidden)

    def _draft_token(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.draft.draft_logits(hidden).argmax(dim=-1)

    def _relayed(self, out: torch.Tensor) -> torch.Tensor:
        """A forwarded draft hidden as the next step's prev_hidden and relay value; sampling always takes the pre-relay out."""
        return self.draft.recurrent_hidden(out)

    # ------------------------------------------------------------------ one step

    def capture_graphs(self, bs_list: List[int], pool) -> None:
        # read-only drafts stay eager until token-identity is proven; capture is mechanical after that
        return

    def propose(self, reqs: List[Req], rows: torch.Tensor, pending: torch.Tensor) -> SpecProposal:
        cs = [r.cached_len for r in reqs]
        table_idxs = [r.table_idx for r in reqs]
        token = pending
        hidden = self.seed_hidden[rows]
        # one decode-shaped batch serves every step: the draft stores no KV, so only the carried
        # token and hidden move and every step reads the same [0, cached_len) window.
        batch = self._draft_batch_ro(table_idxs, cs)
        drafts = []
        for _ in range(self.k):
            out = self._forward(batch, self.chain_backend, self.draft.embed_tokens.forward(token), hidden)
            token = self._draft_token(out)
            hidden = self._relayed(out)
            drafts.append(token)
        return SpecProposal(torch.stack(drafts, dim=1))

    def feature_sink(self, batch: Batch) -> _PrefillSink | _VerifySink:
        if batch.phase == "verify":
            return _VerifySink(self.verify_hidden)
        self._prefill_sink = _PrefillSink([(r.cached_len, r.device_len) for r in batch.reqs])
        return self._prefill_sink

    def post_verify(self, batch: Batch, out: torch.Tensor, emitted: torch.Tensor) -> None:
        """Stash each request's target hidden at its last accepted position; propose runs it as step 0."""
        k1 = self.k + 1
        bs = batch.size
        # the accepted row's target hidden is the next draft's prev_hidden; the pending token
        # out[emitted] rides the token pool to cached_len, so nothing here needs the host
        pick = torch.arange(bs, device=self.device) * k1 + emitted[:bs]
        table = make_device_tensor([r.table_idx for r in batch.reqs], torch.int64, self.device)
        self.seed_hidden[table] = self.verify_hidden[pick]

    def on_prefill(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None:
        """Stash each request's target hidden at its last prompt position; propose runs the sampled token as step 0."""
        sink, self._prefill_sink = self._prefill_sink, None
        hidden = sink.hidden
        # only the row that predicted the sampled token survives: it is the seed step's prev_hidden,
        # and propose already owns everything after it
        table, last = [], []
        offset = 0
        for req, (s, e) in zip(batch.reqs, sink.spans):
            offset += e - s
            table.append(req.table_idx)
            last.append(offset - 1)
        t = make_device_tensor(table, torch.int64, self.device)
        i = make_device_tensor(last, torch.int64, self.device)
        self.seed_hidden[t] = hidden[i]
