"""KV host offload for the QSA paged pool (Qwen3.8-Flash-Next, ``--kv-host-pages``).

The engine keeps TWO page counts: ``num_pages`` logical pages (what the scheduler, the page
table and the radix tree see) and a smaller GPU-resident physical pool. This module is the
bridge:

* the pinned host **mirror** holds one K/V page per logical page (the backing store);
* the GPU K/V buffer is an **LRU cache** of physical slots over the mirror, driven by
  flashlib's ``lru_ensure`` (device-side, fixed-shape, CUDA-graph safe -- the same machinery
  the MoE expert offload uses);
* every ``store_kv`` is **write-through**: a Triton kernel mirrors the token's K/V rows into
  the host bank over UVA, so a physical slot is always clean and eviction is a pure
  rebinding (no write-back, no dirty tracking);
* reads translate: the QSA backend selects pages in logical space (the compressed index slab
  is per logical page and always resident), ensures the selected pages are resident
  (``lru_ensure`` + one fused H2D multi-bank copy), rewrites the block table to physical
  slots, and calls the attend kernel unmodified.

The dummy page is an ordinary cacheable id (``num_logical_pages``); its mirror page stays
zeroed. Stale bindings of freed/reallocated pages are harmless: a new tenant's pages are
fully written (or position-masked) before they are read, and the mirror catches up through
the same write-through.
"""

from __future__ import annotations

import torch
from flashlib.kernels.slot_cache import lru_ensure

from freetoken.moe.host_banks import HostBank
from freetoken.utils import init_logger

logger = init_logger(__name__)


class KVHostOffloader:
    """GPU-slot LRU cache over the pinned host mirror of the logical KV page space."""

    def __init__(self, pool, num_logical_pages: int, device: torch.device) -> None:
        buf = pool._kv_buffer  # [2, L, P_gpu + 1, page_size, kv_heads, head_dim]
        assert buf.shape[0] == 2 and buf.is_contiguous()
        _, num_layers, num_slots, page_size, kv_heads, head_dim = buf.shape
        self.pool = pool
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_slots = num_slots  # buffer pages, dummy GPU page included
        self.num_logical = num_logical_pages  # scheduler-visible pages; dummy id == this
        self.dummy_page = num_logical_pages
        self.device = device
        # The selection touches whole pages: select_width tokens can span at most
        # select_width // page_size + 2 distinct pages (boundary straddle + open group).
        self.max_sel_pages = 0  # set by set_select_width (backend init)
        # Static decode buffers, created by init_graph_buffers (capture) or lazily.
        self._g: dict[str, torch.Tensor] = {}

        self.mirror = HostBank(
            (2, num_layers, num_logical_pages + 1, page_size, kv_heads, head_dim), buf.dtype
        )
        self.mirror.tensor.zero_()  # fresh/dummy pages read back as zeros, never stale bytes
        self.mirror.pin()

        # fp8 pool (kv_quant="fp8", PR#354): the pool stores e4m3 CODES + fp32 per-token
        # scales. The mirror must carry BOTH or rehydrated pages decode garbage (the read
        # path below restores codes+scales into the pool; a bf16-era mirror of raw k/v
        # would silently corrupt). The raw-k/v Triton mirror_store only runs on bf16 pools.
        self._scale_buf = getattr(pool, "_scale_buffer", None)  # [2, L, P*page, H] | None
        self.mirror_scales: HostBank | None = None
        if self._scale_buf is not None:
            scale_heads = self._scale_buf.shape[-1]
            self.mirror_scales = HostBank(
                (2, num_layers, (num_logical_pages + 1) * page_size, scale_heads),
                self._scale_buf.dtype,
            )
            self.mirror_scales.tensor.zero_()
            self.mirror_scales.pin()

        # LRU slot maps (lru_ensure's id space is logical pages, slot space is buffer pages).
        self.phys_of = torch.full((num_logical_pages + 1,), -1, dtype=torch.int32, device=device)
        self.logical_of = torch.full((num_slots,), -1, dtype=torch.int32, device=device)
        self.usage = torch.zeros((num_slots,), dtype=torch.int64, device=device)
        self.step = torch.zeros((), dtype=torch.int64, device=device)
        self.src_ids = torch.empty((num_slots,), dtype=torch.int32, device=device)
        self.dst_slots = torch.empty((num_slots,), dtype=torch.int32, device=device)
        self.num_copy = torch.zeros((1,), dtype=torch.int64, device=device)
        self._out_scratch = torch.empty((num_slots,), dtype=torch.int32, device=device)
        # Eager-prefill scratch (never captured): per-layer compacted selections (+ their
        # pinned host copy for the CPU group packer), the per-group tight query (pinned
        # staging + device buffer) and the translated block table. Lazy, grow-only.
        self._sel_all: torch.Tensor | None = None
        self._counts_dev: torch.Tensor | None = None
        self._sel_host = None
        self._counts_host = None
        self._query_host = None
        self._query_buf: torch.Tensor | None = None
        self._eff_buf: torch.Tensor | None = None
        self._trunc_eager = torch.zeros(1, dtype=torch.int32, device=device)

        from freetoken.kernel.pinned import device_ptr

        # Fused-copy descriptor: 2 * num_layers banks (K/V of each layer), one page row each.
        row_bytes = page_size * kv_heads * head_dim * buf.dtype.itemsize
        assert row_bytes % 16 == 0
        mirror = self.mirror.tensor
        dst_ptrs, src_ptrs, host_ptrs = [], [], []
        for kv in range(2):
            for layer in range(num_layers):
                assert buf[kv, layer].is_contiguous() and mirror[kv, layer].is_contiguous()
                dst_ptrs.append(buf[kv, layer].data_ptr())
                src_ptrs.append(device_ptr(mirror[kv, layer]))
        for layer in range(num_layers):  # K/V interleaved, the mirror-store kernel's layout
            host_ptrs.append(device_ptr(mirror[0, layer]))
            host_ptrs.append(device_ptr(mirror[1, layer]))
        self._dst_ptrs = torch.tensor(dst_ptrs, dtype=torch.int64, device=device)
        self._src_ptrs = torch.tensor(src_ptrs, dtype=torch.int64, device=device)
        self._feats = torch.full((2 * num_layers,), row_bytes, dtype=torch.int64, device=device)
        self._host_ptrs = torch.tensor(host_ptrs, dtype=torch.int64, device=device)
        # fp8: estende o descritor com os bancos de scales (mesma lista de páginas —
        # o fused copy roda uma linha por banco por página; feats heterogêneos ok).
        # E prepara os ponteiros do mirror_store quantizado (graph-safe, UVA).
        self._quant_src_ptrs = None
        self._quant_host_ptrs = None
        if self.mirror_scales is not None:
            sc = self._scale_buf  # [2, L, P*page, H] -> por página: [page, H]
            scale_row = page_size * sc.shape[-1] * sc.element_size()
            assert scale_row % 16 == 0
            qsrc, qhost = [], []
            for kv in range(2):
                for layer in range(num_layers):
                    pool_v = sc[kv, layer].view(-1, page_size, sc.shape[-1])
                    mir_v = self.mirror_scales.tensor[kv, layer].view(
                        -1, page_size, sc.shape[-1])
                    assert pool_v.is_contiguous() and mir_v.is_contiguous()
                    dst_ptrs.append(pool_v.data_ptr())
                    src_ptrs.append(device_ptr(mir_v))
            for layer in range(num_layers):
                # ordem por camada: (kcode, kscale, vcode, vscale) — casa com o kernel
                qsrc += [buf[0, layer].data_ptr(), sc[0, layer].data_ptr(),
                         buf[1, layer].data_ptr(), sc[1, layer].data_ptr()]
                qhost += [device_ptr(mirror[0, layer]),
                          device_ptr(self.mirror_scales.tensor[0, layer]),
                          device_ptr(mirror[1, layer]),
                          device_ptr(self.mirror_scales.tensor[1, layer])]
            self._quant_src_ptrs = torch.tensor(qsrc, dtype=torch.int64, device=device)
            self._quant_host_ptrs = torch.tensor(qhost, dtype=torch.int64, device=device)
            self._dst_ptrs = torch.tensor(dst_ptrs, dtype=torch.int64, device=device)
            self._src_ptrs = torch.tensor(src_ptrs, dtype=torch.int64, device=device)
            self._feats = torch.cat([
                self._feats,
                torch.full((2 * num_layers,), scale_row, dtype=torch.int64, device=device),
            ])
        self.mirror_bytes = self.mirror.nbytes + (
            self.mirror_scales.nbytes if self.mirror_scales is not None else 0)

    # ----- geometry ---------------------------------------------------------
    def set_select_width(self, select_width: int, block_topk: int = 0,
                         max_bs: int = 1) -> None:
        # The sparse selection is block_topk short runs of consecutive tokens SCATTERED
        # across the sequence -- each run can touch its own page (two if it straddles a
        # boundary). Contiguity (select_width // page_size) badly underestimates this.
        # Hard cap: the WHOLE decode batch's query (max_bs x (1 + sel)) must fit the pool,
        # or mid-ensure evictions corrupt the attend. Divide by the batch, not the row.
        worst = 2 * (block_topk or (select_width // 4)) + 8
        per_batch = max(16, (self.num_slots - 73) // max(1, max_bs))
        self.max_sel_pages = min(worst, per_batch)

    def prefill_fetch_rows(self, n_write_pages: int) -> int:
        """Query rows per prefill attend sub-chunk: the write pages plus this many rows' worth
        of selected pages must always fit the physical pool (lru_ensure's distinct bound)."""
        margin = 8
        budget = self.num_slots - margin - n_write_pages
        if budget < self.max_sel_pages:
            raise RuntimeError(
                f"KV host offload: {self.num_slots} physical pages cannot hold {n_write_pages} "
                f"write pages + one row's selection ({self.max_sel_pages}); the GPU pool is too small"
            )
        return max(1, budget // self.max_sel_pages)

    # ----- graph buffers ------------------------------------------------------
    def init_graph_buffers(self, max_bs: int, width_pages: int, select_width: int | None = None) -> None:
        """Static decode buffers (capture-safe addresses); sliced to the live bs per step."""
        if select_width is not None:
            self.set_select_width(select_width, max_bs=max_bs)
        assert self.max_sel_pages > 0, "set_select_width must run before init_graph_buffers"
        d = self.device
        need = max(self.num_slots, max_bs * (1 + self.max_sel_pages))
        if self._out_scratch.numel() < need:
            self._out_scratch = torch.empty((need,), dtype=torch.int32, device=d)
        self._g = {
            "write_pages": torch.zeros(max_bs, dtype=torch.int32, device=d),
            "out_loc_gpu": torch.zeros(max_bs, dtype=torch.int32, device=d),
            "sel_pages": torch.zeros((max_bs, self.max_sel_pages), dtype=torch.int32, device=d),
            "query": torch.zeros(max_bs * (1 + self.max_sel_pages), dtype=torch.int32, device=d),
            "eff_table": torch.zeros((max_bs, width_pages), dtype=torch.int32, device=d),
            "trunc": torch.zeros(1, dtype=torch.int32, device=d),
        }

    def invalidate_pages(self, logical_pages: torch.Tensor) -> None:
        """Marca páginas lógicas como frias (phys_of=-1), desfazendo bindings defasados.
        Necessário quando o espelho foi reescrito por fora (reidratação host->espelho):
        sem isso, um binding antigo apontaria pra um slot físico com conteúdo de outro
        inquilino e o fetch seria pulado. Só desfaz se o slot ainda aponta pra esta página."""
        lp = logical_pages.long().to(self.device)
        old = self.phys_of[lp]
        mask = old >= 0
        if not bool(mask.any()):
            return
        old_slots = old[mask].long()
        cur = self.logical_of[old_slots]
        still = cur == lp[mask].long()
        self.logical_of[old_slots[still]] = -1
        self.phys_of[lp[mask]] = -1

    def trunc_count(self) -> int:
        """Dropped selection pages accumulated since the last read (0 = no truncation)."""
        n = int(self._trunc_eager.item())
        if n:
            self._trunc_eager.zero_()
        t = self._g.get("trunc")
        if t is not None:
            m = int(t.item())
            if m:
                t.zero_()
            n += m
        return n

    def _graph_buffers(self, bs: int, width_pages: int) -> None:
        """Lazy (eager-only) static-buffer init; capture paths get them from init_graph_buffers."""
        g = self._g
        if not g or g["write_pages"].shape[0] < bs or g["eff_table"].shape[1] != width_pages:
            self.init_graph_buffers(max(bs, g["write_pages"].shape[0] if g else 0), width_pages)

    # ----- residency engine ---------------------------------------------------
    # lru_ensure's phase-1 register tile is O(K^2): the kernel is tuned for K<=512
    # (MoE's whole-layer materialize). Chunk bigger queries into calls inside that envelope.
    _ENSURE_CHUNK = 512

    def _ensure(self, query: torch.Tensor) -> None:
        """Bind every logical page in ``query`` to a physical slot and fill the misses."""
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        n = query.numel()
        # The chunks are disjoint slices, so distinct(query) <= n; keeping n <= num_slots
        # makes the kernel's |distinct| <= num_cached requirement hold by construction.
        assert n <= self.num_slots, (
            f"ensure query of {n} pages exceeds the {self.num_slots} physical slots"
        )
        if n > self._out_scratch.numel():
            # eager-only path (prefill); capture-time queries always fit the pre-sized buffer
            self._out_scratch = torch.empty((n,), dtype=torch.int32, device=self.device)
        for off in range(0, n, self._ENSURE_CHUNK):
            q = query[off : off + self._ENSURE_CHUNK]
            k = q.numel()
            lru_ensure(
                q,
                self.phys_of,
                self.logical_of,
                self.usage,
                self.step,
                self._out_scratch[:k],
                self.src_ids,
                self.dst_slots,
                self.num_copy,
            )
            fast_index_copy_multi_jit(
                self._dst_ptrs, self._src_ptrs, self._feats,
                self.dst_slots, self.src_ids, self.num_copy,
            )

    # ----- write path ---------------------------------------------------------
    def ensure_write_pages(self, out_loc: torch.Tensor, is_decode: bool) -> torch.Tensor:
        """Make every page ``out_loc`` writes to resident; returns the write-page query
        (the fetch path re-includes it so a write page can never be evicted mid-forward)."""
        from freetoken.kernel.triton.qsa.offload import write_pages

        if is_decode:
            n = out_loc.numel()
            wp = self._g["write_pages"][:n]
            write_pages(out_loc, wp, self.page_size)
            self._ensure(wp)
            return wp
        # The chunk's tokens are position-contiguous, so every 64th slot names a written
        # page exactly once -- no torch.unique (a GPU sync) needed.
        pages = (out_loc[:: self.page_size] // self.page_size).contiguous()
        self._ensure(pages)
        return pages

    def translate_slots(self, out_loc: torch.Tensor, is_decode: bool) -> torch.Tensor:
        """Logical token slots -> physical (post-ensure; the write pages are pinned-hot)."""
        from freetoken.kernel.triton.qsa.offload import translate_slots

        if is_decode:
            n = out_loc.numel()
            out = self._g["out_loc_gpu"][:n]
        else:
            out = torch.empty_like(out_loc)
        translate_slots(out_loc, self.phys_of, out, self.page_size)
        return out

    def mirror_store(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor,
                     layer_id: int, out_loc_gpu: torch.Tensor | None = None) -> None:
        if self.mirror_scales is None:
            from freetoken.kernel.triton.qsa.offload import mirror_store

            mirror_store(k, v, out_loc, self._host_ptrs, self.pool._dense(layer_id))
            return
        # fp8 pool: o pool JÁ quantizou (store_kv rodou antes). Espelhamos codes+scales
        # LIDOS DO POOL (a única fonte da verdade pós-quantização) — nunca re-quantizar.
        assert out_loc_gpu is not None, "fp8 mirror_store precisa dos slots físicos"
        self._mirror_store_fp8(out_loc, out_loc_gpu, self.pool._dense(layer_id))

    def _mirror_store_fp8(self, out_loc_logical: torch.Tensor, out_loc_phys: torch.Tensor,
                          dense_layer: int) -> None:
        """codes+scales do pool (slots físicos, pós-quantização) -> espelho (slots lógicos).
        Graph-safe: kernel Triton com stores UVA, nada de .cpu() no caminho."""
        from freetoken.kernel.triton.qsa.offload import mirror_store_quant

        H, D = self.mirror.tensor.shape[4], self.mirror.tensor.shape[5]
        scH = self._scale_buf.shape[-1]
        mirror_store_quant(
            self.pool._kv_buffer[0, dense_layer].view(-1, H * D),
            self._scale_buf[0, dense_layer],
            self.pool._kv_buffer[1, dense_layer].view(-1, H * D),
            self._scale_buf[1, dense_layer],
            out_loc_phys, out_loc_logical,
            self._quant_src_ptrs, self._quant_host_ptrs, dense_layer,
        )

    # ----- read path ------------------------------------------------------------
    def compact_all(self, indices: torch.Tensor, token_to_req: torch.Tensor,
                    block_table: torch.Tensor) -> torch.Tensor:
        """One launch for the whole layer: per query row, the distinct logical pages its
        selection touches ([rows, max_sel_pages], first-page padded) plus per-row distinct
        counts. No host sync."""
        from freetoken.kernel.triton.qsa.offload import compact_selected_pages

        rows = indices.shape[0]
        buf = self._sel_all
        if buf is None or buf.shape[0] < rows:
            n = max(rows, 4096)
            buf = self._sel_all = torch.empty((n, self.max_sel_pages), dtype=torch.int32, device=self.device)
            self._counts_dev = torch.empty((n,), dtype=torch.int32, device=self.device)
        sel = buf[:rows]
        compact_selected_pages(
            indices, token_to_req, block_table, sel, self.page_size, self.dummy_page,
            self._trunc_eager, counts=self._counts_dev[:rows],
        )
        return sel

    def iter_prefill_groups(self, sel_all: torch.Tensor, write_pages: torch.Tensor,
                            block_table: torch.Tensor):
        """Pack the layer's query rows into residency-feasible groups, then per group upload
        the tight query (write set + the group's REAL union), ensure, translate; yields
        ``((row_start, row_end), eff_table)`` and the caller attends between iterations.

        One D2H per layer (the compacted page lists); the CPU packer then works with exact
        sets, so the per-group distinct bound (|write| + |union| + slack <= num_slots) is
        VERIFIED, not estimated -- lru_ensure's capacity requirement holds by construction.
        Consecutive rows select mostly the same pages, so groups pack several rows each.
        Buffers are reused across groups; single-stream ordering (attend of group i is
        enqueued before group i+1's H2D/ensure/translate) makes that safe.
        """
        import numpy as np

        rows = sel_all.shape[0]
        host = self._sel_host
        if host is None or host.shape[0] < rows:
            n = max(rows, 4096)
            host = self._sel_host = torch.empty((n, self.max_sel_pages), dtype=torch.int32, pin_memory=True)
            self._counts_host = torch.empty((n,), dtype=torch.int32, pin_memory=True)
        sel_h, cnt_h = host[:rows], self._counts_host[:rows]
        sel_h.copy_(sel_all, non_blocking=True)
        cnt_h.copy_(self._counts_dev[:rows], non_blocking=True)
        torch.cuda.synchronize(self.device)  # the one sync per layer
        sel_np, cnt_np = sel_h.numpy(), cnt_h.numpy()
        write_np = write_pages.cpu().numpy().astype(np.int64)
        # |write| + |union of the group's pages| + slack <= physical slots
        cap = self.num_slots - write_np.size - 9
        n_ids = self.num_logical + 1

        gmask = np.zeros(n_ids, dtype=bool)
        gmask[write_np] = True
        gstart, gcost = 0, 0
        groups: list[tuple[int, int, np.ndarray]] = []
        for r in range(rows):
            pages = sel_np[r, : cnt_np[r]].astype(np.int64)
            extra = int((~gmask[pages]).sum())
            if gcost + extra > cap and r > gstart:
                groups.append((gstart, r, np.flatnonzero(gmask)))
                gmask[:] = False
                gmask[write_np] = True
                gstart, gcost = r, 0
                extra = int((~gmask[pages]).sum())
            gmask[pages] = True
            gcost += extra
        groups.append((gstart, rows, np.flatnonzero(gmask)))

        from freetoken.kernel.triton.qsa.offload import translate_table

        for gstart, gend, pages_np in groups:
            n = pages_np.size
            qh = self._query_host
            if qh is None or qh.numel() < n:
                qh = self._query_host = torch.empty((max(n, 2048),), dtype=torch.int32, pin_memory=True)
                self._query_buf = torch.empty_like(qh, device=self.device)
            qh[:n] = torch.from_numpy(pages_np.astype(np.int32))
            qd = self._query_buf[:n]
            qd.copy_(qh[:n], non_blocking=True)
            self._ensure(qd)
            eff = self._eff_buf
            if eff is None or eff.shape != block_table.shape:
                eff = self._eff_buf = torch.empty_like(block_table)
            translate_table(block_table, self.phys_of, eff)
            yield (gstart, gend), eff

    def fetch_for_attend(self, indices: torch.Tensor, md) -> torch.Tensor:
        """Decode (graph-safe): ensure the selected pages resident, return the PHYSICAL
        block table for the attend kernel. Fixed shapes throughout."""
        from freetoken.kernel.triton.qsa.offload import compact_selected_pages, translate_table

        rows = indices.shape[0]
        self._graph_buffers(rows, md.block_table.shape[1])
        g = self._g
        sel = g["sel_pages"][:rows]
        compact_selected_pages(indices, md.token_to_req, md.block_table, sel, self.page_size, self.dummy_page, g["trunc"])
        per_row = 1 + self.max_sel_pages
        query = g["query"][: rows * per_row].view(rows, per_row)
        query[:, 0].copy_(md.write_pages)
        query[:, 1:].copy_(sel)
        self._ensure(query.view(-1))
        eff = g["eff_table"][:rows]
        translate_table(md.block_table, self.phys_of, eff)
        return eff


__all__ = ["KVHostOffloader"]
