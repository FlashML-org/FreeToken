"""Shared fixtures for the qwen4_exp tests: toy configs, hash constants, the QSA pool and backend.

The geometry is the shipping one everywhere it matters for QSA (head_dim 256, index head_dim
128 with a 64-wide partial rope, index_ratio 4, budget 2048 -> 512 blocks -> 2051 selected
tokens, page_size 64); only the head counts, the hidden size and the layer count are scaled
down so a test fits on a shared GPU. Holds no tests itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info

EOS = 7
VOCAB = 512

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def hf_config(
    num_layers: int = 4,
    head_dim: int = 256,
    num_q: int = 4,
    num_kv: int = 2,
    index_head_dim: int = 128,
    index_heads: int = 4,
    budget: int = 2048,
    ratio: int = 4,
    hidden: int = 256,
    max_position: int = 1 << 16,
    rope_theta: float = 10000000.0,
    **text_overrides,
) -> SimpleNamespace:
    text = SimpleNamespace(
        num_hidden_layers=num_layers,
        hidden_size=hidden,
        vocab_size=VOCAB,
        head_dim=head_dim,
        num_attention_heads=num_q,
        num_key_value_heads=num_kv,
        layer_types=[
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(num_layers)
        ],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": rope_theta,
            "partial_rotary_factor": 0.25,
        },
        max_position_embeddings=max_position,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        output_gate_type="sigmoid",
        indexer_n_heads=index_heads,
        indexer_kv_heads=1,
        indexer_head_dim=index_head_dim,
        indexer_budget=budget,
        indexer_compress_ratio=ratio,
        hc_count=4,
        hc_lowrank=16,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4,
        eos_token_id=EOS,
    )
    for name, value in text_overrides.items():
        setattr(text, name, value)
    return SimpleNamespace(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=text,
        quantization_config=None,
    )


def toy_hf_config(num_layers: int = 4, **text_overrides) -> SimpleNamespace:
    """The small-geometry config the PLE/skeleton tests share (hidden 128, head_dim 64)."""
    return hf_config(
        num_layers=num_layers, head_dim=64, num_kv=1, index_head_dim=64, index_heads=2,
        budget=16, hidden=128, max_position=4096, rope_theta=10000.0, **text_overrides,
    )


def hash_constants(args):
    """Checkpoint-shape int64 hash tensors, derived like the dummy-weight path."""
    from freetoken.models.qwen4_exp.ple import derive_ngram_hash_constants

    multipliers, sizes, offsets = derive_ngram_hash_constants(
        vocab_size=VOCAB,
        ngram_size=args.ngram_size,
        num_ngram_heads=args.num_ngram_heads,
        ngram_vocab_size_base=args.ngram_vocab_size_base,
        ple_layer_index=0,
    )
    return [torch.tensor(v, dtype=torch.int64) for v in (multipliers, sizes, offsets)]


def parsed_config(**kwargs):
    from freetoken.models.qwen4_exp.config import parse_config

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    return parse_config(hf_config(**kwargs))


def fresh_ctx(page_size: int = 64, **fields):
    import freetoken.core as core
    from freetoken.core import Context, set_global_ctx

    core._GLOBAL_CTX = None  # test-only: each scenario builds its own ctx
    ctx = Context(page_size=page_size)
    for name, value in fields.items():
        setattr(ctx, name, value)
    set_global_ctx(ctx)
    return ctx


def fill_weights(op, seed: int, device: torch.device, scale: float = 0.05) -> None:
    gen = torch.Generator(device=device).manual_seed(seed)
    for tensor in op.state_dict().values():
        if tensor.is_floating_point():
            tensor.normal_(0.0, scale, generator=gen)
        else:
            tensor.zero_()


class Fixture:
    """QSA pool + page table + the sparse backend, with a first-fit page allocator."""

    def __init__(
        self,
        config,
        num_pages: int,
        max_running_req: int = 8,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        page_size: int = 64,
    ) -> None:
        from freetoken.attention.qsa_sparse import QSASparseAttnBackend
        from freetoken.kvcache import create_kvcache_pool

        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.page_size = page_size
        self.num_req_slots = max_running_req + 1
        self.pool = create_kvcache_pool(
            model_config=config,
            num_pages=num_pages + 1,  # + 1 for the dummy page, as create_kv_pool does
            page_size=page_size,
            dtype=dtype,
            device=self.device,
            num_req_slots=self.num_req_slots,
        )
        self.page_table = torch.zeros(
            (self.num_req_slots, num_pages * page_size), dtype=torch.int32, device=self.device
        )
        self.page_table[max_running_req].fill_(num_pages * page_size)  # dummy page
        self.ctx = fresh_ctx(
            page_size=page_size, page_table=self.page_table, kv_cache=self.pool
        )
        self.backend = QSASparseAttnBackend(config)
        self.ctx.attn_backend = self.backend
        self._free = list(range(num_pages))

    def layer(self, layer_id: int, seed: int = 1):
        from freetoken.models.qwen4_exp.attention import Qwen4ExpAttention
        from freetoken.utils.torch_utils import torch_dtype

        with torch.device(self.device), torch_dtype(self.dtype):
            attn = Qwen4ExpAttention(self.config, layer_id=layer_id)
        fill_weights(attn, seed, self.device)
        return attn

    def allocate(self, table_idx: int, cached_len: int, device_len: int) -> None:
        for page in range(-(-cached_len // self.page_size), -(-device_len // self.page_size)):
            base = self._free.pop(0) * self.page_size
            columns = slice(page * self.page_size, (page + 1) * self.page_size)
            self.page_table[table_idx, columns] = torch.arange(
                base, base + self.page_size, dtype=torch.int32, device=self.device
            )

    def req(self, table_idx: int, cached_len: int, device_len: int) -> SimpleNamespace:
        self.allocate(table_idx, cached_len, device_len)
        return SimpleNamespace(
            table_idx=table_idx,
            cached_len=cached_len,
            device_len=device_len,
            extend_len=device_len - cached_len,
        )

    def step(self, req: SimpleNamespace) -> None:
        self.allocate(req.table_idx, req.device_len, req.device_len + 1)
        req.cached_len, req.device_len, req.extend_len = req.device_len, req.device_len + 1, 1

    def batch(self, reqs, phase: str) -> SimpleNamespace:
        positions = torch.cat(
            [
                torch.arange(r.cached_len, r.device_len, dtype=torch.int32, device=self.device)
                for r in reqs
            ]
        )
        out_loc = torch.cat(
            [self.page_table[r.table_idx, r.cached_len : r.device_len] for r in reqs]
        ).contiguous()
        batch = SimpleNamespace(
            reqs=reqs,
            padded_reqs=reqs,
            phase=phase,
            size=len(reqs),
            padded_size=len(reqs),
            is_prefill=phase == "prefill",
            is_decode=phase == "decode",
            positions=positions,
            out_loc=out_loc,
            attn_metadata=None,
            active_table_idx=torch.tensor(
                [r.table_idx for r in reqs], dtype=torch.int32, device=self.device
            ),
        )
        self.backend.prepare_metadata(batch)
        return batch


def selection_spy(monkeypatch, backend) -> dict:
    """Record the expanded token selection of every ``_select`` call."""
    from freetoken.attention.qsa_sparse import QSASparseAttnBackend

    seen: dict[str, torch.Tensor] = {}
    original = QSASparseAttnBackend._select

    def spy(self, index, md, slot):
        indices = original(self, index, md, slot)
        seen["indices"] = indices.clone()
        return indices

    monkeypatch.setattr(QSASparseAttnBackend, "_select", spy)
    return seen
