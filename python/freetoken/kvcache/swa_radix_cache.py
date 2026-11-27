"""SWA (sliding-window-attention) radix cache.

A SEPARATE class from ``RadixPrefixCache`` (like ``HybridRadixCache`` for GDN) that REUSES the
shared ``RadixTreeNode`` + walk/split. Strictly mirrors sglang ``SWARadixCache``.

The second "currency" is NOT a stored slot: ``node.value`` (full-pool page indices) is canonical
and radix-shared; the swa KV for those tokens lives in the smaller swa pool reached via the pool's
``full_to_swa`` mapping and is freed independently as tokens slide out of the window -> the node is
marked ``swa_tombstone`` (full KV survives). A matched prefix is reusable for SWA layers only up to
the deepest node with ``>= sliding_window`` contiguous live (non-tombstone) tokens back to the last
tombstone, or tombstone-free to root (sglang ``_match_prefix_helper``).

A model may keep more than one sliding window: ``windows`` lists them longest first, and tier ``k``
(1-based) holds per-token rows a resume needs for the last ``windows[k - 1]`` tokens. Tiers nest
(a token with tier k live also has every longer tier live), so a node carries one ``swa_level``,
the number of its live tiers. A resume point needs, for every tier, ``>= windows[k - 1]`` tokens of
level ``>= k`` before it. With one window this is the plain SWA tree above.

Two-tier lock (``full_ref >= swa_ref``): full pins node..root; swa pins only the trailing longest
window, bounded by ``swa_uuid``, at every level. Two-tier eviction: full pass = unlocked leaf LRU
(frees every pool); window pass = unlocked node LRU incl. internal -> drop the tier and the
shorter ones in place (full KV stays). A leaf below the top level resumes nowhere and is unlinked.

LRU: FreeToken heapq + timestamp, but ``match_prefix`` stamps the matched path with strictly
DECREASING timestamps toward root (sglang ``_match_post_processor``) so the heap evicts near-root
SWA nodes first -- the same victim order as sglang's maintained LRUList.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Sequence, Tuple

import torch

from freetoken.utils import align_down

from .base import BaseCacheHandle
from .radix_cache import RadixTreeNode, _get_key_fn


@dataclass(frozen=True)
class SWACacheHandle(BaseCacheHandle):
    """Lock handle for a matched SWA prefix: the windowed-safe matched node (lock target) + the
    reusable full-pool KV page indices. ``swa_uuid`` is the window-boundary returned by
    ``inc_lock``; ``CacheManager.lock`` records it on the handle (via object.__setattr__ -- the
    base handle is a frozen dataclass) and ``unlock``/``dec_lock`` consume it."""

    node: RadixTreeNode
    kv_indices: torch.Tensor
    swa_uuid: Optional[int] = None

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices

# A logical-clock event id is multiplied by this so a single match/insert event leaves room to
# encode per-node depth offsets below it (decreasing toward root) without colliding with the next
# event. Bounds the max radix path depth -- 2^24 nodes is far beyond any real prefix.
_EVENT_STRIDE = 1 << 24


class SWAMatch(NamedTuple):
    kv_indices: torch.Tensor   # reused full-pool KV page indices for [0:cached_len) (windowed-safe)
    cached_len: int            # truncated to the windowed-reuse boundary
    node: RadixTreeNode        # the matched windowed-safe node (lock target)


class SWAEvictResult(NamedTuple):
    kv_indices: torch.Tensor   # full-pool page indices removed from the tree (-> free_slots)
    window_indices: Tuple[torch.Tensor, ...]  # per tier: full-pool indices whose rows to free

    @property
    def swa_indices(self) -> torch.Tensor:
        return self.window_indices[0]


class SWARadixCache:
    def __init__(self, device: torch.device, page_size: int, sliding_window_size: int | Sequence[int]) -> None:
        windows = (sliding_window_size,) if isinstance(sliding_window_size, int) else tuple(sliding_window_size)
        assert windows and all(w > 0 for w in windows), "SWARadixCache requires positive sliding windows"
        assert all(a > b for a, b in zip(windows, windows[1:])), f"windows must be longest first: {windows}"
        self.device = device
        self.page_size = page_size
        self.windows = windows
        self.levels = len(windows)
        self.sliding_window_size = windows[0]
        self.key_fn = _get_key_fn(page_size)
        # Page-index dtype: matches the engine's int32 page_table / free_slots so an empty result
        # (returned by insert/evict/match) never promotes them to int64 when concatenated.
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.root = RadixTreeNode(self.key_fn)
        self.root.set_key_value(self.empty, self.empty)
        self.root.ref_count = 1  # root is always protected
        self.full_evictable = 0
        self.full_protected = 0
        # per tier: tokens of live, unlocked (evictable) / locked (protected) rows
        self.window_evictable = [0] * self.levels
        self.window_protected = [0] * self.levels
        self._clk = 0            # logical access clock (tie-free, for near-root-first order)
        self._uuid = 0           # monotonic swa window-boundary handle counter
        self._revives = 0        # observability: # of insert-side revives (a tree node adopts the request's rows)

    @property
    def swa_evictable(self) -> int:
        return self.window_evictable[0]

    @property
    def swa_protected(self) -> int:
        return self.window_protected[0]

    # ---------------------------------------------------------------- match / insert
    def match_prefix(self, input_ids: torch.Tensor) -> SWAMatch:
        """Match the token prefix for the full layers, then truncate the reusable length to the
        deepest node end where every tier has its window of live tokens behind it (or the path
        is live in that tier back to root)."""
        node = self.root
        value: List[torch.Tensor] = []
        # path connected to root without a dead token is always reusable -> start at +inf.
        runs = [float("inf")] * self.levels
        best_value_len = 0       # number of leading entries of ``value`` that are windowed-safe
        best_node = self.root
        prefix_pos, total = 0, len(input_ids)

        while prefix_pos < total:
            child = node.children.get(self.key_fn(input_ids[prefix_pos:]))
            if child is None:
                break
            match_len = align_down(child.get_match_len(input_ids[prefix_pos:]), self.page_size)
            if match_len == 0:
                break
            partial = match_len < child.length
            if partial:
                child = child.split_at(match_len)  # child := the matched prefix segment
            value.append(child.value)
            for k in range(self.levels):
                runs[k] = runs[k] + child.length if child.swa_level > k else 0
            if all(run >= w for run, w in zip(runs, self.windows)):
                best_value_len = len(value)
                best_node = child
            node = child
            prefix_pos += match_len
            if partial:
                break

        self._stamp_path(best_node)
        kv = torch.cat(value[:best_value_len]) if best_value_len else self.empty
        return SWAMatch(kv, int(kv.numel()), best_node)

    def insert(self, input_ids: torch.Tensor, kv_indices: torch.Tensor,
               swa_evicted_seqlen: int = 0, update_kv_after_len: int = 0,
               live: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
               ) -> Tuple[int, torch.Tensor]:
        """Insert the committed full KV prefix. ``kv_indices`` are the request's full-pool page
        indices (the window rows ride along via the full->row mappings, live where the request
        allocated them). Two frontiers drive the reconcile:

        - ``update_kv_after_len`` (= the request's old reused-prefix length): only nodes extending
          PAST it carry the request's freshly-(re)computed live rows. Nodes entirely within it
          are tree-shared slots and are left untouched.
        - ``swa_evicted_seqlen`` (= positions whose swa the request itself freed): below it the
          request's swa is the sentinel; at/above it it is live. A tree with window states takes
          ``live`` instead: per tier, the ``[lo, hi)`` spans where the request's rows are live.

        A matched node in the fresh region whose level is below the request's there is REVIVED
        -- not by rewriting the mapping but by adopting the request's slots into ``node.value``
        and raising its level (Branches 1/2, split where the request's level changes).
        Otherwise the tree keeps its slots and the request's dup is freed (Branch 3). New suffix
        tokens get the request's level; with ``swa_evicted_seqlen`` the last page is kept live
        (the free_swa ``-page_size`` margin guarantees it). Returns ``(matched_prefix_len,
        freed_full_indices)``: the full-pool slots the caller must return to EVERY pool
        (displaced old tree slots + non-adopted request dups; the frees are idempotent on the
        already-sentinel rows)."""
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, kv_indices = input_ids[:insert_len], kv_indices[:insert_len]
        frontier = live is None
        live = self._live(swa_evicted_seqlen, insert_len, live)
        freed: List[torch.Tensor] = []
        node = self.root
        total = 0
        while total < insert_len:
            child = node.children.get(self.key_fn(input_ids[total:]))
            if child is None:
                break
            match_len = align_down(child.get_match_len(input_ids[total:]), self.page_size)
            if match_len == 0:
                break
            partial = match_len < child.length
            if partial:
                child = child.split_at(match_len)
            if update_kv_after_len < total + match_len:
                for piece, start in reversed(self._split_at(child, total, live)):
                    seg_kv = kv_indices[start:start + piece.length]   # request's fresh slots for this span
                    level = self._level_at(start, live)
                    if level > piece.swa_level and piece.ref_count == 0:
                        # Branch 1/2: the request holds more live tiers here -> adopt its slots.
                        # Free the old tree slots (with their rows) for the request's.
                        assert piece.swa_ref_count == 0
                        freed.append(piece.value)
                        piece.set_key_value(piece._key, seg_kv.clone())
                        self._set_level(piece, level)
                        piece.timestamp = self._tick()
                        self._revives += 1
                    else:
                        # The tree slots are canonical; drop the request's dup. A full-locked
                        # reader still gathers the node's CURRENT slots through its own row, so a
                        # locked node is never revived -- it stays as is until the lock clears.
                        freed.append(seg_kv.clone())
            total += match_len
            node = child
            if partial:
                break
        # Suffix: tokens [total:insert_len) not yet in the tree, one node per request level.
        if total < insert_len:
            # The suffix nodes can share this copy without retaining the request buffer.
            suffix_ids = input_ids[total:].clone()
            suffix_kv = kv_indices[total:].clone()
            if frontier:
                # never a tombstone leaf: the clamp + the free_swa -page_size margin keep a live tail
                boundary = max(0, min(swa_evicted_seqlen, insert_len) - total)
                boundary = min(boundary, max(0, len(suffix_ids) - self.page_size))
                live = (((total + boundary, insert_len),),)
            for lo, hi, level in self._segments(total, insert_len, live):
                node = self._add_child(node, suffix_ids[lo - total:hi - total],
                                       suffix_kv[lo - total:hi - total], level=level)
        return total, (torch.cat(freed) if freed else self.empty)

    def has_locked_tombstone(self, input_ids: torch.Tensor, after_len: int, swa_evicted_seqlen: int = 0,
                             live: Optional[Sequence[Sequence[Tuple[int, int]]]] = None) -> bool:
        """True if the same insert would keep a node below the request's level because another
        request still full-locks it, freeing the caller's copy of it."""
        live = self._live(swa_evicted_seqlen, len(input_ids), live)
        node, total = self.root, 0
        while total < len(input_ids):
            child = node.children.get(self.key_fn(input_ids[total:]))
            if child is None:
                return False
            match_len = align_down(child.get_match_len(input_ids[total:]), self.page_size)
            if match_len == 0:
                return False
            if child.ref_count > 0 and after_len < total + match_len and any(
                    level > child.swa_level for _, _, level in self._segments(total, total + match_len, live)):
                return True
            if match_len < child.length:
                return False
            node, total = child, total + match_len
        return False

    def _live(self, swa_evicted_seqlen: int, end: int,
              live: Optional[Sequence[Sequence[Tuple[int, int]]]]) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        if live is None:
            assert self.levels == 1, "a tree with window states takes the request's live spans"
            return (((swa_evicted_seqlen, end),),)
        assert len(live) == self.levels, f"live spans for {len(live)} tiers, tree has {self.levels}"
        return tuple(tuple(tier) for tier in live)

    @staticmethod
    def _level_at(pos: int, live: Tuple[Tuple[Tuple[int, int], ...], ...]) -> int:
        """The request's level at ``pos``: its leading tiers live there (a shorter tier counts only inside the longer ones)."""
        level = 0
        for tier in live:
            if not any(lo <= pos < hi for lo, hi in tier):
                break
            level += 1
        return level

    def _segments(self, lo: int, hi: int, live) -> List[Tuple[int, int, int]]:
        """``[lo, hi)`` cut where the request's level changes: ``(start, end, level)`` runs."""
        cuts = sorted({lo, hi, *(e for tier in live for span in tier for e in span if lo < e < hi)})
        segs: List[Tuple[int, int, int]] = []
        for a, b in zip(cuts, cuts[1:]):
            level = self._level_at(a, live)
            if segs and segs[-1][2] == level:
                segs[-1] = (segs[-1][0], b, level)
            else:
                segs.append((a, b, level))
        return segs

    def _split_at(self, node: RadixTreeNode, start: int, live) -> List[Tuple[RadixTreeNode, int]]:
        """Split an unlocked ``node`` (covering ``[start, start + length)``) below the top level
        where the request's level changes, so each piece it can adopt has one request level.
        Returns ``(piece, piece_start)`` root-side first; the last piece is ``node`` itself."""
        pieces = []
        if node.ref_count == 0 and node.swa_level < self.levels:
            for lo, _, _ in self._segments(start, start + node.length, live)[1:]:
                pieces.append((node.split_at(lo - start), start))
                start = lo
        pieces.append((node, start))
        return pieces

    def _add_child(self, parent: RadixTreeNode, ids: torch.Tensor, kv: torch.Tensor,
                   *, level: int) -> RadixTreeNode:
        child = RadixTreeNode(self.key_fn, self._tick())
        child.set_key_value(ids, kv)
        child.set_parent(parent)
        child.swa_level = 0
        self.full_evictable += child.length
        self._set_level(child, level)
        return child

    def _set_level(self, node: RadixTreeNode, level: int, drops: Optional[List[List[torch.Tensor]]] = None) -> None:
        """Move ``node`` to ``level``, keeping the tier counters; the tiers it loses go to ``drops``."""
        counts = self.window_protected if node.swa_ref_count else self.window_evictable
        old = node.swa_level
        for k in range(min(old, level), max(old, level)):
            counts[k] += node.length if level > old else -node.length
            if level < old and drops is not None:
                drops[k].append(node.value)
        node.swa_level = level

    # ---------------------------------------------------------------- locking (dual)
    def inc_lock(self, node: RadixTreeNode) -> Optional[int]:
        """Pin the matched node's KV path (full ref node..root) and the trailing window of live
        swa tokens (swa ref). Stamps and returns ``swa_uuid_for_lock`` at the node where the
        cumulative swa lock first covers the window (None if the path is shorter than a window).
        Enforces full_ref >= swa_ref. Mirrors sglang ``inc_lock_ref``."""
        swa_uuid_for_lock: Optional[int] = None
        swa_locked = 0
        cur = node
        while not cur.is_root():
            if cur.ref_count == 0:
                self.full_evictable -= cur.length
                self.full_protected += cur.length
            cur.ref_count += 1
            if swa_locked < self.sliding_window_size and not cur.swa_tombstone:
                if cur.swa_ref_count == 0:
                    for k in range(cur.swa_level):
                        self.window_evictable[k] -= cur.length
                        self.window_protected[k] += cur.length
                cur.swa_ref_count += 1
                swa_locked += cur.length
                if swa_locked >= self.sliding_window_size:
                    if cur.swa_uuid is None:
                        self._uuid += 1
                        cur.swa_uuid = self._uuid
                    swa_uuid_for_lock = cur.swa_uuid
            cur = cur.parent
        return swa_uuid_for_lock

    def dec_lock(self, node: RadixTreeNode, swa_uuid_for_lock: Optional[int] = None,
                 skip_swa: bool = False) -> None:
        dec_swa = not skip_swa
        cur = node
        while not cur.is_root():
            cur.ref_count -= 1
            assert cur.ref_count >= 0
            if cur.ref_count == 0:
                self.full_evictable += cur.length
                self.full_protected -= cur.length
            if dec_swa and not cur.swa_tombstone and cur.swa_ref_count > 0:
                cur.swa_ref_count -= 1
                if cur.swa_ref_count == 0:
                    for k in range(cur.swa_level):
                        self.window_evictable[k] += cur.length
                        self.window_protected[k] -= cur.length
                if swa_uuid_for_lock is not None and cur.swa_uuid == swa_uuid_for_lock:
                    dec_swa = False  # window boundary reached (inclusive); stop swa decrement
            cur = cur.parent

    # ---------------------------------------------------------------- eviction (dual)
    def evict_full(self, num_tokens: int) -> SWAEvictResult:
        """Evict full KV by LRU over UNLOCKED LEAF nodes (an internal node's KV is a prefix
        dependency). Frees every pool for each evicted leaf; cascade-reclaims the leaves it
        exposes that resume nowhere."""
        leaves = [n for n in self._leaves() if n.ref_count == 0]
        heapq.heapify(leaves)
        kv: List[torch.Tensor] = []
        drops = self._no_drops()
        freed = 0
        while freed < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if node.ref_count != 0 or not node.is_leaf() or node.is_root():
                continue
            freed += self._remove_leaf(node, kv, drops)
            parent, casc = self._cascade_dead_leaves(self._unlink(node), kv, drops)
            freed += casc
            # Re-push the SURVIVING ancestor the cascade returned (the original parent may have
            # been unlinked/freed by the cascade -- re-pushing it would double-free / KeyError).
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                heapq.heappush(leaves, parent)
        return self._evicted(kv, drops)

    def evict_swa(self, num_tokens: int) -> SWAEvictResult:
        return self.evict_window(1, num_tokens)

    def evict_window(self, level: int, num_tokens: int) -> SWAEvictResult:
        """Free tier-``level`` rows by LRU over UNLOCKED nodes holding it -- internal nodes too.
        Internal (or full-locked) node -> drop the tier and the shorter ones in place (keep full
        KV + children). Free leaf -> free every pool, unlink, cascade."""
        cands = [n for n in self._all_nodes() if n.swa_level >= level and n.swa_ref_count == 0]
        heapq.heapify(cands)
        kv: List[torch.Tensor] = []
        drops = self._no_drops()
        freed = 0
        while freed < num_tokens and cands:
            node = heapq.heappop(cands)
            if node.swa_level < level or node.swa_ref_count != 0 or node.is_root():
                continue
            freed += node.length
            if node.is_leaf() and node.ref_count == 0:
                self._remove_leaf(node, kv, drops)
                self._cascade_dead_leaves(self._unlink(node), kv, drops)
            else:
                self._set_level(node, level - 1, drops)
        return self._evicted(kv, drops)

    def trim_head_swa(self, input_ids: torch.Tensor, keep_from: int) -> torch.Tensor:
        """Tombstone the path strictly below ``keep_from`` in every tier (full KV stays),
        returning the full indices whose rows the caller must free. Retention companion of the
        finish-time soft pin: only the trailing window ``[keep_from, P)`` needs to stay live for
        a next-turn cut near P, so the head is reclaimed eagerly instead of lingering as LRU
        litter. Locked (swa_ref>0), tombstoned and leaf nodes are left untouched; ``keep_from``
        must be page-aligned."""
        if keep_from <= 0:
            return self.empty
        self.match_prefix(input_ids[:keep_from])  # ensure a node boundary at keep_from (splits)
        freed: List[torch.Tensor] = []
        node, pos = self.root, 0
        while pos < keep_from:
            child = node.children.get(self.key_fn(input_ids[pos:]))
            if child is None or pos + child.length > keep_from:
                break
            if not child.swa_tombstone and child.swa_ref_count == 0 and not child.is_leaf():
                freed.append(child.value)
                self._set_level(child, 0)
            node, pos = child, pos + child.length
        return torch.cat(freed) if freed else self.empty

    @property
    def full_evictable_size(self) -> int:
        return self.full_evictable

    @property
    def swa_evictable_size(self) -> int:
        return self.window_evictable[0]

    @property
    def size_info(self):
        from .base import SizeInfo
        return SizeInfo(evictable_size=self.full_evictable, protected_size=self.full_protected)

    def check_integrity(self) -> None:
        for n in self._all_nodes():
            assert n.ref_count >= 0 and n.swa_ref_count >= 0
            assert n.ref_count >= n.swa_ref_count, "full_ref must be >= swa_ref"
            assert 0 <= n.swa_level <= self.levels
            if n.swa_tombstone:
                assert n.swa_ref_count == 0, "a tombstoned node cannot hold a swa lock"

    # ---------------------------------------------------------------- helpers
    def _no_drops(self) -> List[List[torch.Tensor]]:
        return [[] for _ in range(self.levels)]

    def _evicted(self, kv: List[torch.Tensor], drops: List[List[torch.Tensor]]) -> SWAEvictResult:
        return SWAEvictResult(torch.cat(kv) if kv else self.empty,
                              tuple(torch.cat(d) if d else self.empty for d in drops))

    def _remove_leaf(self, node: RadixTreeNode, kv: List[torch.Tensor], drops: List[List[torch.Tensor]]) -> int:
        """Account a leaf about to be unlinked: its full KV to ``kv``, its live rows to ``drops``."""
        kv.append(node.value)
        self.full_evictable -= node.length
        self._set_level(node, 0, drops)
        return node.length

    def _unlink(self, node: RadixTreeNode) -> RadixTreeNode:
        parent = node.parent
        del parent.children[self.key_fn(node._key)]
        return parent

    def _cascade_dead_leaves(
        self, parent: RadixTreeNode, kv_out: List[torch.Tensor], drops: List[List[torch.Tensor]]
    ) -> Tuple[RadixTreeNode, int]:
        """After a leaf is unlinked, eagerly reclaim the leaves it exposes upward that resume
        nowhere (below the top level: some tier is gone at their end), unlocked: free their full
        KV and rows and unlink. Keeps the 'a leaf always resumes' invariant. Returns ``(surviving_ancestor, freed_full_tokens)``
        -- the caller MUST use the returned ancestor (the passed-in ``parent`` may itself have
        been unlinked/freed by the cascade)."""
        freed = 0
        while (parent.swa_level < self.levels and parent.is_leaf()
               and parent.ref_count == 0 and not parent.is_root()):
            freed += self._remove_leaf(parent, kv_out, drops)
            parent = self._unlink(parent)
        return parent, freed

    def _stamp_path(self, node: RadixTreeNode) -> None:
        """Stamp the matched path with strictly DECREASING timestamps toward root (deepest newest)
        so the eviction heap (min on timestamp) reclaims near-root swa nodes first -- the same
        victim order as sglang's reset_node_and_parents_mru."""
        self._clk += 1
        base = self._clk * _EVENT_STRIDE
        off, cur = 0, node
        while not cur.is_root():
            cur.timestamp = base - off
            off += 1
            cur = cur.parent

    def _tick(self) -> int:
        self._clk += 1
        return self._clk * _EVENT_STRIDE

    def _leaves(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.is_leaf():
                if not n.is_root():
                    out.append(n)
            else:
                stack.extend(n.children.values())
        return out

    def _all_nodes(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if not n.is_root():
                out.append(n)
            stack.extend(n.children.values())
        return out

__all__ = ["SWARadixCache", "SWAMatch", "SWAEvictResult", "SWACacheHandle"]
