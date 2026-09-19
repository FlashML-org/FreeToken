"""Host-side cold-prefix tier (Nível 3 — Degrau 1: snapshots GDN; Degrau 3: KV de prefixo).

Dois artefatos em RAM pinned:
1. um banco de snapshots GDN (write na evicção do tree, read no match/admissão);
2. um índice de prefixos frios: token ids + slots KV host + slot GDN host.

Degrau 3: o KV vive num **slot space host separado** (banco pinned [2, L, S, page, H, D] de
códigos + scales, mesma geometria do pool GPU). Ids lógicos de página são reutilizados pelo
scheduler, então o conteúdo é COPIADO (D2H) na hora do evict do tree — nunca referenciado por
id lógico depois. A reidratação aloca páginas novas no pool e copia H2D de volta.

Ver IMPLEMENTACAO-NIVEL3.md para o design completo.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from freetoken.moe.host_banks import HostBank


class HostMatch:
    __slots__ = ("host_len", "kv_pages", "gdn_slot")

    def __init__(self, host_len: int, kv_pages: torch.Tensor | None, gdn_slot: int | None) -> None:
        self.host_len = host_len            # tokens cobertos (page-aligned)
        self.kv_pages = kv_pages            # int32 CPU: slots KV do tier (não ids lógicos!)
        self.gdn_slot = gdn_slot            # slot host do snapshot GDN (None = sem snapshot)


@dataclass
class _Entry:
    key_tokens: torch.Tensor                      # CPU, o prefixo (page-aligned)
    kv_slots: torch.Tensor | None = None          # int32 CPU: um slot host por página
    gdn_slot: int | None = None                   # snapshot GDN host do boundary


class HostPrefixTier:
    """Índice + storage RAM pra prefixos evictados do radix tree GPU.

    O índice é LRU por entrada (hit re-marca); quando a RAM do tier enche (KV) ou os slots
    GDN acabam, drop-oldest libera a entrada mais fria inteira (KV slots + slot GDN).
    As páginas KV são copiadas para slots host próprios na evicção — o conteúdo sobrevive à
    reutilização dos ids lógicos pelo scheduler.

    Geometria GDN: os tensores do LinearStatePool são [layers, slots, ...]; cada banco host
    é [host_slots, layers, ...] — a cópia de um slot é [layers, ...] <-> [layers, ...].
    Geometria KV: pool._kv_buffer [2, L, P, page, H, D] (códigos e4m3 no pool fp8) +
    pool._scale_buffer [2, L, P*page, H] fp32; os banks host espelham isso com S host slots.
    """

    def __init__(self, state_pool, kv_pool=None, *, gdn_slots_host: int = 64,
                 kv_budget_bytes: int = 0, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)
        self.pool = state_pool
        tensors = {"conv": state_pool.conv_states, "rec": state_pool.recurrent_states}
        tensors.update(state_pool.slot_states)
        self._order = list(tensors)
        self._banks: dict[str, tuple[torch.Tensor, HostBank]] = {}
        for name, t in tensors.items():
            # pool: [layers, slots, *rest] -> bank: [host_slots, layers, *rest]
            bank = HostBank((gdn_slots_host, t.shape[0], *t.shape[2:]), t.dtype)
            if self.device.type == "cuda" and torch.cuda.is_available():
                bank.pin()
            self._banks[name] = (t, bank)
        self.num_slots = gdn_slots_host
        self._free = list(range(gdn_slots_host))
        self._in_use: set[int] = set()
        self._index: OrderedDict[bytes, _Entry] = OrderedDict()
        # key: bytes dos token ids page-aligned -> _Entry

        # ---- KV page store (Degrau 3) ----
        self.kv_pool = kv_pool
        self.kv_page_bytes = 0
        self._kv_codes: HostBank | None = None
        self._kv_scales: HostBank | None = None
        self._kv_free: list[int] = []
        if kv_pool is not None and kv_budget_bytes > 0:
            codes = kv_pool._kv_buffer            # [2, L, P, page, H, D]
            scales = kv_pool._scale_buffer        # [2, L, P*page, H] | None (fp8)
            _, layers, _, page_size, heads, dim = codes.shape
            page_bytes = 2 * layers * page_size * heads * dim * codes.element_size()
            if scales is not None:
                page_bytes += 2 * layers * page_size * heads * scales.element_size()
            n = max(1, kv_budget_bytes // page_bytes)
            self._kv_codes = HostBank((2, layers, n, page_size, heads, dim), codes.dtype)
            if scales is not None:
                self._kv_scales = HostBank((2, layers, n, page_size, heads), scales.dtype)
            if self.device.type == "cuda" and torch.cuda.is_available():
                self._kv_codes.pin()
                if self._kv_scales is not None:
                    self._kv_scales.pin()
            self._kv_free = list(range(n))
            self.kv_page_bytes = page_bytes
        self.copy_stream = (
            torch.cuda.Stream(self.device) if self.device.type == "cuda" else None)

    # ---------------- GDN snapshot storage ----------------
    def write_gdn(self, gpu_slot: int) -> int:
        """D2H do snapshot do gpu_slot pra um slot host livre. Retorna o host_slot.
        Deve ser enfileirado ANTES de state_pool.free(gpu_slot): a ordem de stream do
        engine garante leitura antes de qualquer reescrita do slot (ver spec)."""
        if not self._free:
            raise RuntimeError("HostPrefixTier: sem slots host GDN livres (chame evict_host)")
        hs = self._free.pop()
        self._in_use.add(hs)
        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                for name in self._order:
                    t, bank = self._banks[name]
                    bank.tensor[hs].copy_(t[:, gpu_slot], non_blocking=True)
            self.copy_stream.synchronize()
        else:
            for name in self._order:
                t, bank = self._banks[name]
                bank.tensor[hs].copy_(t[:, gpu_slot])
        return hs

    def read_gdn(self, host_slot: int, dst_gpu_slot: int) -> None:
        """H2D host_slot -> dst_gpu_slot do pool (já alocado pelo caller)."""
        if host_slot not in self._in_use:
            raise KeyError(f"HostPrefixTier: slot host {host_slot} não está em uso")
        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                for name in self._order:
                    t, bank = self._banks[name]
                    t[:, dst_gpu_slot].copy_(bank.tensor[host_slot], non_blocking=True)
            self.copy_stream.synchronize()
        else:
            for name in self._order:
                t, bank = self._banks[name]
                t[:, dst_gpu_slot].copy_(bank.tensor[host_slot])

    def free_gdn(self, host_slot: int) -> None:
        self._in_use.discard(host_slot)
        self._free.append(host_slot)

    # ---------------- KV page storage (Degrau 3) ----------------
    @property
    def has_kv(self) -> bool:
        return self._kv_codes is not None

    def write_kv(self, kv_indices: torch.Tensor) -> torch.Tensor | None:
        """Copia as páginas (token slots do pool, page-aligned) para slots host livres.
        Retorna os slots host (int32 CPU), ou None se o tier não tem KV store.
        Deve rodar DURANTE o evict do tree, com os bytes ainda válidos no buffer GPU."""
        if not self.has_kv:
            return None
        page_size = self._kv_codes.tensor.shape[3]
        pages = (kv_indices[::page_size] // page_size).long().to(self.device)
        n = pages.numel()
        if n == 0:
            return None
        codes, scales = self.kv_pool._kv_buffer, self.kv_pool._scale_buffer
        # Defesa: um slot fora do pool indicaria um bug de lifetime no tree. Pular a escrita
        # (o prefixo simplesmente não vai pro tier) em vez de derrubar o scheduler.
        max_page = int(pages.max().item())
        if max_page >= codes.shape[2]:
            from freetoken.utils import init_logger
            init_logger(__name__).warning(
                f"HostPrefixTier.write_kv: page {max_page} fora do pool "
                f"({codes.shape[2]} páginas): n={n}, kv_slots min/max = "
                f"{int(kv_indices.min())}/{int(kv_indices.max())}, prefixo de {kv_indices.numel()} tokens -- "
                "escrita ignorada"
            )
            return None
        while len(self._kv_free) < n:
            self._drop_oldest()  # libera os slots da entrada mais fria (inclui GDN)
        slots = torch.tensor([self._kv_free.pop() for _ in range(n)], dtype=torch.int64)
        # NB: sem staging do prefixo inteiro — cada fatia calcula seus token-ids.

        def _go():
            # Chunked: NUNCA estagiar o prefixo inteiro (o evict acontece sob pressão de
            # memória — um staging GB-sized explode a VRAM e derruba o scheduler).
            CH = 128  # páginas por fatia (~100 MB de temp por fatia em fp8)
            for i in range(0, n, CH):
                p = pages[i : i + CH]
                s = slots[i : i + CH]
                staged = codes[:, :, p].contiguous()
                self._kv_codes.tensor[:, :, s] = staged.cpu() if staged.is_cuda else staged
                del staged
            if scales is not None:
                for i in range(0, n, CH):
                    p = pages[i : i + CH]
                    s = slots[i : i + CH]
                    tk = (p.unsqueeze(1) * page_size
                          + torch.arange(page_size, device=self.device)).flatten()
                    staged_s = scales[:, :, tk].view(
                        2, scales.shape[1], p.numel(), page_size, scales.shape[3]).contiguous()
                    self._kv_scales.tensor[:, :, s] = (
                        staged_s.cpu() if staged_s.is_cuda else staged_s)
                    del staged_s

        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                _go()
            self.copy_stream.synchronize()
        else:
            _go()
        return slots.to(torch.int32).cpu()

    def read_kv(self, host_slots: torch.Tensor, dst_pages: torch.Tensor) -> None:
        """H2D dos slots host para as páginas dst (ids de página do pool, device)."""
        assert self.has_kv
        slots = host_slots.long()
        pages = dst_pages.long().to(self.device)
        codes, scales = self.kv_pool._kv_buffer, self.kv_pool._scale_buffer
        page_size = self._kv_codes.tensor.shape[3]
        n = slots.numel()

        def _go():
            CH = 128
            for i in range(0, n, CH):
                s = slots[i : i + CH]
                p = pages[i : i + CH]
                staged = self._kv_codes.tensor[:, :, s].to(self.device, non_blocking=True)
                codes[:, :, p] = staged
                del staged
            if scales is not None:
                for i in range(0, n, CH):
                    s = slots[i : i + CH]
                    p = pages[i : i + CH]
                    src = self._kv_scales.tensor[:, :, s].reshape(
                        2, scales.shape[1], p.numel() * page_size, scales.shape[3])
                    tk = (p.unsqueeze(1) * page_size
                          + torch.arange(page_size, device=self.device)).flatten()
                    staged_s = src.to(self.device, non_blocking=True)
                    scales[:, :, tk] = staged_s
                    del staged_s

        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                _go()
            self.copy_stream.synchronize()
        else:
            _go()

    def read_kv_to_mirror(self, host_slots: torch.Tensor, mirror_codes,
                          mirror_scales, logical_pages: torch.Tensor) -> None:
        """H2H (pinned memcpy): slots KV do tier -> espelho do offloader nas páginas
        lógicas dadas. Usado quando o offloader está ativo: reidratar NÃO sobe bytes pra
        GPU — o fetch lazy do offloader cuida da residência na atenção."""
        assert self.has_kv
        s = host_slots.long()
        lp = logical_pages.long().cpu()
        page = self._kv_codes.tensor.shape[3]
        mirror_codes[:, :, lp] = self._kv_codes.tensor[:, :, s]
        if self._kv_scales is not None and mirror_scales is not None:
            t_s = self._kv_scales.tensor.view(2, -1, self._kv_codes.tensor.shape[2], page,
                                              self._kv_scales.tensor.shape[-1])
            m_s = mirror_scales.view(2, -1, mirror_codes.shape[2], page,
                                     self._kv_scales.tensor.shape[-1])
            m_s[:, :, lp] = t_s[:, :, s]

    def _drop_oldest(self) -> None:
        if not self._index:
            raise RuntimeError("HostPrefixTier: KV host store cheio e índice vazio")
        _, ent = self._index.popitem(last=False)  # o mais frio
        if ent.kv_slots is not None:
            self._kv_free.extend(ent.kv_slots.tolist())
            ent.kv_slots = None
        if ent.gdn_slot is not None:
            self.free_gdn(ent.gdn_slot)
            ent.gdn_slot = None

    # ---------------- índice de prefixos ----------------
    @staticmethod
    def _key_of(tokens: torch.Tensor) -> bytes:
        return tokens.to(torch.int64).cpu().numpy().tobytes()

    def put_prefix(self, key_tokens: torch.Tensor, kv_pages: torch.Tensor | None = None,
                   host_gdn_slot: int | None = None) -> None:
        """Registra (ou faz merge numa entrada existente) um prefixo frio. key_tokens deve
        ser page-aligned (o tree já alinha). kv_pages = slots host do KV store."""
        key = self._key_of(key_tokens)
        ent = self._index.get(key)
        if ent is None:
            ent = _Entry(key_tokens.cpu())
            self._index[key] = ent
        else:
            self._index.move_to_end(key)  # refresh LRU
        if kv_pages is not None:
            if ent.kv_slots is not None and ent.kv_slots is not kv_pages:
                self._kv_free.extend(ent.kv_slots.tolist())  # substituição: libera os velhos
            ent.kv_slots = kv_pages.to(torch.int32).cpu()
        if host_gdn_slot is not None:
            if ent.gdn_slot is not None and ent.gdn_slot != host_gdn_slot:
                self.free_gdn(ent.gdn_slot)  # a cópia nova substitui a anterior
            ent.gdn_slot = host_gdn_slot

    def lookup(self, input_ids: torch.Tensor, page_size: int = 64) -> HostMatch | None:
        """Maior prefixo page-aligned presente no índice (python/CPU, fora do hot path)."""
        toks = input_ids.to(torch.int64).cpu()
        n = (toks.numel() // page_size) * page_size
        while n > 0:
            key = self._key_of(toks[:n])
            hit = self._index.get(key)
            if hit is not None:
                self._index.move_to_end(key)
                return HostMatch(n, hit.kv_slots, hit.gdn_slot)
            n -= page_size
        return None

    def evict_host(self, key_tokens: torch.Tensor) -> None:
        """Remove uma entrada do índice e libera os slots host (KV + GDN)."""
        ent = self._index.pop(self._key_of(key_tokens), None)
        if ent is None:
            return
        if ent.kv_slots is not None:
            self._kv_free.extend(ent.kv_slots.tolist())
        if ent.gdn_slot is not None:
            self.free_gdn(ent.gdn_slot)


__all__ = ["HostPrefixTier", "HostMatch"]
