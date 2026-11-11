"""SharedKVMTPProposer: drafts with an assistant checkpoint that keeps no KV of its own, reading the target's paged slabs through a read-only view. A family selects it through its spec_drafts behind the mtp CLI algorithm; it is never a --spec-algo name."""

from __future__ import annotations

from itertools import accumulate
from typing import TYPE_CHECKING, ClassVar, List, Protocol

import torch

from freetoken.core import Batch, get_global_ctx
from freetoken.engine.sample import make_device_tensor
from freetoken.spec.mtp.proposer import _DraftReq, _PrefillSink, _VerifySink, with_resident_moe
from freetoken.spec.mtp.relay import HiddenRelay
from freetoken.spec.mtp.shared_kv import ReadOnlyTargetKVView
from freetoken.spec.proposer import DraftContext, Proposer, SpecProposal, check_draft_path

if TYPE_CHECKING:
    from freetoken.core import Req
    from freetoken.layers.embedding import VocabParallelEmbedding
    from freetoken.models.config import ModelConfig


class SharedKVMTPDraft(Protocol):
    """The class a family registers as ``ModelSpec.spec_drafts["shared_kv_mtp"]``. It owns its vocab table and stores no KV."""

    # None: the proposer picks the single triton backend
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
    def load(cls, config: ModelConfig, target_model, model_path: str, device: torch.device, dtype: torch.dtype) -> SharedKVMTPDraft: ...

    @property
    def embed_tokens(self) -> VocabParallelEmbedding:
        """The target's token embedding, which the draft shares."""
        ...

    def forward_hidden(self, embeds: torch.Tensor, prev_hidden: torch.Tensor, *, shared_kv=None) -> torch.Tensor:
        """The proposer passes no ``shared_kv``: every draft layer reads the target's paged slabs through the view and stores nothing."""
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
        # relay hidden and carry, and the verify hiddens; the draft reads the target's slabs and allocates no KV of its own
        return rows * (config.speculative_num_steps + 3) * relay_bytes, 0

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
        self._prefill_state: dict[int, tuple[int, int]] = {}  # table row -> (uid, first position not drafted yet)
        self._prefill_sink: _PrefillSink | None = None
        self.kv = None

    # ------------------------------------------------------------------ lifecycle

    def attach(self, ctx: DraftContext) -> None:
        from freetoken.attention import create_attention_backend

        super().attach(ctx)
        cfg = self.draft.draft_config(with_resident_moe(self.config.model_config))
        rows = self.config.max_running_req + 1
        # a rebuild swaps the target pool, so the read-only view is rebuilt with every attach
        self.kv = ReadOnlyTargetKVView(ctx.kv_cache, self.draft.config, self.config.model_config)
        width = self.draft.relay_width(self.config.model_config)
        self.relay = HiddenRelay(rows, width, self.dtype, ctx.device)
        self.verify_hidden = torch.zeros(rows * (self.k + 1), width, dtype=self.dtype, device=ctx.device)
        fixed = self.draft.attention_backend
        # backends bind the context's KV pool when they are built; read-only draft forwards are
        # decode-shaped only, so one triton backend serves both roles
        with get_global_ctx().override(kv_cache=self.kv):
            self.extend_backend = self.chain_backend = create_attention_backend(fixed or "triton", cfg)

    def detach(self, *, keep_kv: bool) -> None:
        self.kv = None  # a view over the old pool is dead once a rebuild swaps it, keep_kv included
        self.relay = self.verify_hidden = None
        self.extend_backend = self.chain_backend = None
        self._prefill_state.clear()
        super().detach(keep_kv=keep_kv)

    def release(self, req: Req) -> None:
        self._prefill_state.pop(req.table_idx, None)

    # ------------------------------------------------------------------ draft forwards

    def _draft_batch_ro(self, table_idxs: List[int], qs: List[int], backend) -> Batch:
        """Decode-shaped batch for a read-only draft: row i sits at position q_i and gathers exactly
        ``[0, q_i)`` - its own slot holds rolled-back or not-yet-computed data, and the draft writes
        none (llama.cpp ``common/speculative.cpp:1849``, ``:1933-1936``)."""
        reqs = [_DraftReq(q, q + 1, t) for t, q in zip(table_idxs, qs)]
        batch = Batch(reqs=reqs, phase="decode", rows_per_req=1)
        batch.padded_reqs = reqs
        batch.positions = make_device_tensor(qs, torch.int32, self.device)
        rows = make_device_tensor(table_idxs, torch.int64, self.device)
        # the slot this row would write is never written; the clamp keeps a full-context final row in range
        batch.out_loc = self.ctx.page_table[rows, batch.positions.to(torch.int64).clamp(max=self.ctx.page_table.shape[1] - 1)]
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
        token = self.relay.d1[rows]
        hidden = self.relay.hidden[rows]
        drafts = [token]
        table_idxs = [r.table_idx for r in reqs]
        for j in range(1, self.k):
            # a shared-memory draft decodes every step at position C, reading only the target's committed KV [0, C)
            batch = self._draft_batch_ro(table_idxs, cs, self.chain_backend)
            out = self._forward(batch, self.chain_backend, self.draft.embed_tokens.forward(token), hidden)
            hidden = self._relayed(out)
            token = self._draft_token(out)
            drafts.append(token)
        return SpecProposal(torch.stack(drafts, dim=1))

    def feature_sink(self, batch: Batch) -> _PrefillSink | _VerifySink:
        if batch.phase == "verify":
            return _VerifySink(self.verify_hidden)
        self._prefill_sink = _PrefillSink([(r.cached_len, r.device_len) for r in batch.reqs])
        return self._prefill_sink

    def post_verify(self, batch: Batch, out: torch.Tensor, emitted: torch.Tensor) -> None:
        """Relay each request's row at its last accepted position."""
        k1 = self.k + 1
        bs = batch.size
        # only the picked row is ever consumed: the pending token's row at the next free position,
        # C+emitted+1 (cached_len is still pre-drain), reading [0, C+emitted+1) through the prev_hidden row
        em = emitted[:bs]
        table_idxs = [r.table_idx for r in batch.reqs]
        qs = [r.cached_len + e + 1 for r, e in zip(batch.reqs, em.tolist())]
        pick = torch.arange(bs, device=self.device) * k1 + em
        tokens = out[torch.arange(bs, device=self.device), em]
        ro = self._draft_batch_ro(table_idxs, qs, self.extend_backend)
        hidden = self._forward(ro, self.extend_backend, self.draft.embed_tokens.forward(tokens), self.verify_hidden[pick])
        table = make_device_tensor(table_idxs, torch.int64, self.device)
        self.relay.hidden[table] = self._relayed(hidden)
        self.relay.d1[table] = self._draft_token(hidden)

    def on_prefill(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None:
        """Draft the chunk's positions: row p pairs token p+1 with the target hidden at p; a final chunk adds the sampled token's row and relays it."""
        sink, self._prefill_sink = self._prefill_sink, None
        hidden = sink.hidden
        # a read-only draft consumes only the final row of a finishing chunk; intermediate rows'
        # outputs feed nothing (carry and prev are target hiddens), so compute that row alone
        emb, prev, finals = [], [], []
        offset = 0
        for i, (req, (s, e)) in enumerate(zip(batch.reqs, sink.spans)):
            rows = hidden[offset : offset + e - s]
            offset += e - s
            row = req.table_idx
            state = self._prefill_state.get(row)
            frontier = state[1] if state is not None and state[0] == req.uid else s
            final = e == req.prompt_len
            end = e if final else e - 1
            if final and frontier < end:
                # the sampled token's row at position end, over the target's committed KV [0, end)
                emb.append(self.draft.embed_tokens.forward(next_tokens_gpu[i : i + 1]))
                prev.append(rows[-1:])
                finals.append((row, end))
            self.relay.carry[row] = rows[-1]
            self._prefill_state[row] = (req.uid, max(frontier, end))
        if not finals:
            return
        ro = self._draft_batch_ro([r for r, _ in finals], [p for _, p in finals], self.extend_backend)
        out = self._forward(ro, self.extend_backend, torch.cat(emb), torch.cat(prev))
        table = make_device_tensor([r for r, _ in finals], torch.int64, self.device)
        self.relay.hidden[table] = self._relayed(out)
        self.relay.d1[table] = self._draft_token(out)
