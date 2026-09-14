"""V4.1 sparse attention with explicit compressed-KV source ownership."""

from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from .dsv4_sparse import DSV4SparseAttnBackend


class DSV41SparseAttnBackend(DSV4SparseAttnBackend):
    def __init__(self, config):
        self.config = config
        self.device = get_global_ctx().kv_cache.device
        self.window_size = config.dsv41_args.window_size
        self.capture = None
        self.capture_bs = []
        self.max_graph_bs = 0
        self._window_ar = torch.arange(self.window_size, device=self.device)
        self.begin_forward()

    def begin_forward(self):
        self.shared_indices = {}
        self.shared_candidates = {}

    def scatter_compressed(self, layer_id, tier, rows, kv):
        if tier == "attn":
            self.pool.store_compressed(kv, layer_id, rows)
        elif tier == "idx":
            self.pool.store_indexer(kv, layer_id, rows)
        else:
            raise ValueError(f"Unknown V4.1 compressed tier: {tier}")

    def attend(self, q, layer_id, topk_idxs, n_window, attn_sink, softmax_scale,
               cmp_counts=None, has_compression=True):
        pool = self.pool
        source = pool.kv_sources[layer_id]
        compressed = pool.cmp_pool[source] if source is not None else pool.window_pool[layer_id]
        packed = getattr(pool, "kv_quant", "none") == "fp8-fp4"
        if q.is_cuda:
            if packed:
                from freetoken.kernel.triton.dsv41.sparse_attn import sparse_attn_paged
            else:
                from freetoken.kernel.triton.dsv4.sparse_attn import sparse_attn_paged
            return sparse_attn_paged(q, pool.window_pool[layer_id], compressed, attn_sink,
                                     topk_idxs.int(), n_window, softmax_scale, cmp_counts)
        flat = q.reshape(-1, *q.shape[-2:])
        ids = topk_idxs.reshape(flat.shape[0], -1).long()
        window = pool.window_pool[layer_id][ids[:, :n_window].clamp_min(0)]
        cmps = compressed[ids[:, n_window:].clamp_min(0)]
        if packed:
            from freetoken.kernel.triton.dsv41.quant import unpack_fp4, unpack_fp8
            window = unpack_fp8(window, block_size=32, dtype=torch.bfloat16)
            cmps = (unpack_fp4(cmps, block_size=16, scale_format="e4m3", dtype=torch.bfloat16)
                    if source is not None else window[:, :0])
        kv = torch.cat((window, cmps), 1)
        logits = torch.einsum("qhd,qkd->qhk", flat.float(), kv.float()) * softmax_scale
        live = ids >= 0
        if cmp_counts is not None:
            columns = torch.arange(ids.shape[1], device=ids.device)
            live &= columns[None] < n_window + cmp_counts.reshape(-1, 1)
        logits.masked_fill_(~live[:, None, :], -torch.inf)
        sink = attn_sink.float().view(1, -1, 1).expand(flat.shape[0], -1, 1)
        prob = torch.cat((logits, sink), -1).softmax(-1)[..., :-1]
        return torch.einsum("qhk,qkd->qhd", prob, kv.float()).to(q.dtype).view_as(q)
