"""A minimal in-process engine for the tiny DeepSeek-V4.1 model: builds the DSV41 pool, backend and
context by hand (no scheduler), loads a synthetic bf16 or FP8/MXFP4 checkpoint with resident or offloaded experts, and drives
ragged prefill / batched decode batches through ``model.forward()``. Shared by the forward smoke test
and the reference-parity test."""

from __future__ import annotations

from dataclasses import replace

import torch

from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.distributed.info import set_tp_info, try_get_tp_info
from freetoken.engine.engine import _materialize_loaded_weight_state_dict
from freetoken.kvcache.dsv4.v41_cost_model import dsv41_pool_sizes
from freetoken.kvcache.dsv4.v41_pool import DSV41PagedKVCache
from freetoken.layers import set_rope_device
from freetoken.layers.quantization import NoQuantConfig
from freetoken.layers.quantization.method import finalize_quant
from freetoken.models import create_model
from freetoken.models.deepseek_v41.config import parse_config
from freetoken.models.deepseek_v41.engram import ZeroEngramTable
from freetoken.models.deepseek_v41.weight import iter_weights
from freetoken.utils.hf import cached_load_hf_config
from freetoken.utils.torch_utils import torch_dtype

P = 128


class TinyEngine:
    def __init__(self, checkpoint: str, *, max_seq_len: int = 1024, max_running_req: int = 2, swa_decoder_replay: str = "exact", engram_table=None, quantized: bool = False):
        self.device = torch.device("cuda")
        self.checkpoint = checkpoint
        if try_get_tp_info() is None:
            set_tp_info(0, 1)
        set_rope_device(self.device)
        hf_config = cached_load_hf_config(checkpoint)
        mc = parse_config(hf_config)
        args = mc.dsv41_args
        args.max_seq_len = max_seq_len
        args.max_batch_size = max_running_req + 1
        args.swa_decoder_replay = swa_decoder_replay
        self.args = args
        quant = NoQuantConfig()
        if quantized:
            from freetoken.models.register import checkpoint_quant_config, get_model_spec

            quant = checkpoint_quant_config(checkpoint, hf_config, get_model_spec("DeepseekV41ForCausalLM"))
        self.config = replace(mc, moe_strategy="offload" if quantized else "fused", decode_target="gpu", quant=quant)
        mc = self.config
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            self.model = create_model(self.config)
        state = _materialize_loaded_weight_state_dict(
            self.model.state_dict(), iter_weights(checkpoint, self.device, include_moe_experts=False), device=self.device,
        )
        self.model.load_state_dict(state)
        finalize_quant(self.model)
        from freetoken.moe.expert_banks import load_expert_banks

        method = self.model.model.layers.op_list[0].ffn.experts.quant_method
        if quantized:
            from freetoken.moe.offload_cache import OffloadMoeCache, attach_offload_moe_cache

            self.banks = load_expert_banks(
                checkpoint, mc, method=method, device=self.device, dtype=torch.bfloat16, parallel=False,
            )
            self.expert_cache = OffloadMoeCache(
                num_layers=mc.num_moe_layers, num_experts=mc.num_experts,
                cache_size=mc.num_moe_layers * mc.num_experts, device=self.device,
                quant_format=self.banks.quant_format, layout=method.layout(),
            )
            self.expert_cache.set_bank_sources(self.banks.sources, layer_residency=self.banks.layer_residency)
            attach_offload_moe_cache(self.model, self.expert_cache)
        else:  # resident experts load as banks, not through the state dict (as in Engine._load_resident_experts)
            from freetoken.layers import iter_moe_layers
            from freetoken.moe.expert_banks import attach_resident_banks

            attach_resident_banks(list(iter_moe_layers(self.model)), load_expert_banks(
                checkpoint, mc, method=method, device=self.device, dtype=torch.bfloat16, parallel=False, resident=True,
            ))
        for layer in self.model.engram_layers():
            layer.attach_table(engram_table if engram_table is not None else ZeroEngramTable(layer.width, self.device))

        self.max_running_req = max_running_req
        num_pages = max_seq_len // P
        self.pool = DSV41PagedKVCache(dsv41_pool_sizes(num_pages + 1, args, 1.0, P), args, self.device, n_scratch=max_running_req + 1)
        self.pool._init_paged_state(max_running_req, radix=False)
        self.page_table = torch.zeros(max_running_req + 1, max_seq_len, dtype=torch.int32, device=self.device)
        self.page_table[max_running_req].fill_(num_pages * P)  # the dummy row -> the reserved tail page
        self.pool.attach_page_table(self.page_table)
        try:
            ctx = get_global_ctx()
        except AssertionError:
            ctx = Context(page_size=P)
            set_global_ctx(ctx)
        from freetoken.attention.dsv41_sparse import DSV41SparseAttnBackend

        self.ctx = ctx
        ctx.page_table = self.page_table
        ctx.kv_cache = self.pool  # the backend reads the pool's device from the context
        self.backend = DSV41SparseAttnBackend(self.config)
        self._next_page = 0
        self._bind()

    def _bind(self) -> None:
        """Point the process-wide ``Context`` at this engine's page table, pool and backend. The
        context is a singleton the model reads through ``get_global_ctx()``, so a test that holds two
        engines must rebind before each forward; ``prefill`` / ``decode`` do this, and the model
        re-resolves its backend when the binding changes."""
        ctx = self.ctx
        if (
            getattr(ctx, "attn_backend", None) is self.backend
            and getattr(ctx, "kv_cache", None) is self.pool
            and getattr(ctx, "page_table", None) is self.page_table
        ):
            return
        ctx.page_table = self.page_table
        ctx.kv_cache = self.pool
        ctx.attn_backend = self.backend
        self.model.mark_for_rebind()

    # ----- requests -----------------------------------------------------------------------
    def new_request(self, table_idx: int, tokens: list[int], max_new: int = 64) -> Req:
        """A request with pages for ``len(tokens) + max_new`` positions bound on row ``table_idx``."""
        n_pages = -(-(len(tokens) + max_new) // P)
        first = self._next_page
        self._next_page += n_pages
        locs = torch.arange(first * P, (first + n_pages) * P, device=self.device)
        self.page_table[table_idx, : n_pages * P] = locs.to(torch.int32)
        self.pool.alloc_swa(locs)
        return Req(input_ids=torch.tensor(tokens, dtype=torch.int32), table_idx=table_idx, cached_len=0, output_len=max_new,
                   uid=table_idx, sampling_params=SamplingParams(), cache_handle=None)

    def new_request_on_prefix(self, table_idx: int, donor: Req, tokens: list[int], max_new: int = 64) -> Req:
        """A request whose first ``hit`` positions alias ``donor``'s pages (what a radix prefix hit does
        through the page table) and whose remaining positions get pages of their own. ``hit`` is what the
        scheduler admits: the donor's prompt, capped by ``match_req`` (the last token, and the pool's
        ``prefix_replay_tokens`` under bounded replay) and page-aligned. The caller prefills the rest."""
        limit = len(tokens) - 1
        if self.pool.prefix_replay_tokens:
            limit = min(limit, max(0, len(tokens) - self.pool.prefix_replay_tokens))
        hit = min(donor.device_len, limit) // P * P
        assert tokens[:hit] == donor.input_ids[:hit].tolist(), "the prefix must match the donor's prompt"
        self.page_table[table_idx, :hit] = self.page_table[donor.table_idx, :hit]
        n_pages = -(-(len(tokens) + max_new - hit) // P)
        first = self._next_page
        self._next_page += n_pages
        locs = torch.arange(first * P, (first + n_pages) * P, device=self.device)
        self.page_table[table_idx, hit : hit + n_pages * P] = locs.to(torch.int32)
        self.pool.alloc_swa(locs)
        return Req(input_ids=torch.tensor(tokens, dtype=torch.int32), table_idx=table_idx, cached_len=hit, output_len=max_new,
                   uid=table_idx, sampling_params=SamplingParams(), cache_handle=None)

    def prefill(self, reqs: list[Req]) -> torch.Tensor:
        """Run each request's ``[cached_len, device_len)`` tokens; returns ``[B, vocab]`` logits."""
        batch = Batch(reqs=reqs, phase="prefill")
        batch.padded_reqs = list(reqs)
        batch.input_ids = torch.cat([r.input_ids[r.cached_len : r.device_len] for r in reqs]).to(self.device)
        batch.positions = torch.cat([torch.arange(r.cached_len, r.device_len) for r in reqs]).to(self.device)
        self._bind()
        self.backend.prepare_metadata(batch)
        with self.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, False):
            return self.model.forward().float()

    def decode(self, reqs: list[Req], tokens: list[int]) -> torch.Tensor:
        """Feed one new token per request (appended at ``device_len - 1``); returns ``[B, vocab]``."""
        for r, t in zip(reqs, tokens):
            r.append_host(torch.tensor([t], dtype=torch.int32))
            r.complete_one()
        batch = Batch(reqs=reqs, phase="decode")
        batch.padded_reqs = list(reqs)
        batch.input_ids = torch.tensor(tokens, dtype=torch.int32, device=self.device)
        batch.positions = torch.tensor([r.device_len - 1 for r in reqs], device=self.device)
        batch.active_table_idx = torch.tensor([r.table_idx for r in reqs], device=self.device)
        self._bind()
        self.backend.prepare_metadata(batch)
        with self.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, False):
            return self.model.forward().float()

    def finish_prefill(self, reqs: list[Req]) -> None:
        for r in reqs:
            r.cached_len = r.device_len
