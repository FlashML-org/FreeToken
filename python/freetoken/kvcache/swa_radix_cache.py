"""SWA (sliding-window-attention) radix cache.

A SEPARATE class from ``RadixPrefixCache`` (like ``HybridRadixCache`` for GDN) that REUSES the
shared ``RadixTreeNode`` + walk/split. Strictly mirrors sglang ``SWARadixCache``.

The second "currency" is NOT a stored slot: ``node.value`` (full-pool page indices) is canonical
and radix-shared; the swa KV for those tokens lives in the smaller swa pool reached via the pool's
``full_to_swa`` mapping and is freed independently as tokens slide out of the window -> the node is
marked ``swa_tombstone`` (full KV survives). A matched prefix is reusable for SWA layers only up to
the deepest node with ``>= sliding_window`` contiguous live (non-tombstone) tokens back to the last
tombstone, or tombstone-free to root (sglang ``_match_prefix_helper``).

Two-tier lock (``full_ref >= swa_ref``): full pins node..root; swa pins only the trailing window,
bounded by ``swa_uuid``. Two-tier eviction: full pass = unlocked leaf LRU (frees both pools); swa
pass = unlocked node LRU incl. internal -> tombstone in place (free swa only).

LRU: FreeToken heapq + timestamp, but ``match_prefix`` stamps the matched path with strictly
DECREASING timestamps toward root (sglang ``_match_post_processor``) so the heap evicts near-root
SWA nodes first -- the same victim order as sglang's maintained LRUList.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

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
    swa_indices: torch.Tensor  # full-pool indices whose swa slots to free_swa (tombstoned + removed)


class SWARadixCache:
    def __init__(self, device: torch.device, page_size: int, sliding_window_size: int) -> None:
        assert sliding_window_size > 0, "SWARadixCache requires a positive sliding window"
        self.device = device
        self.page_size = page_size
        self.sliding_window_size = sliding_window_size
        self.key_fn = _get_key_fn(page_size)
        # Page-index dtype: matches the engine's int32 page_table / free_slots so an empty result
        # (returned by insert/evict/match) never promotes them to int64 when concatenated.
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.root = RadixTreeNode(self.key_fn)
        self.root.set_key_value(self.empty, self.empty)
        self.root.ref_count = 1  # root is always protected
        self.full_evictable = 0
        self.full_protected = 0
        self.swa_evictable = 0   # tokens of live (non-tombstone), unlocked swa
        self.swa_protected = 0
        self._clk = 0            # logical access clock (tie-free, for near-root-first order)
        self._uuid = 0           # monotonic swa window-boundary handle counter
        self._revives = 0        # observability: # of insert-side tombstone revives (Branches 1/2)

    # ---------------------------------------------------------------- match / insert
    def match_prefix(self, input_ids: torch.Tensor) -> SWAMatch:
        """Match the token prefix for the full layers, then truncate the reusable length to the
        windowed-reuse boundary for the swa layers: the deepest node whose run of contiguous live
        (non-tombstone) tokens back to the last tombstone is ``>= sliding_window_size`` (or the
        path is tombstone-free to root). Mirrors sglang ``_match_prefix_helper``."""
        node = self.root
        value: List[torch.Tensor] = []
        # path connected to root without a tombstone is always reusable -> start at +inf.
        match_since_tomb = float("inf")
        best_value_len = 0       # number of leading entries of ``value`` that are windowed-safe
        best_node = self.root
        prefix_pos, total = 0, len(input_ids)

        while prefix_pos < total:
            child = node.children.get(self.key_fn(input_ids[prefix_pos:]))
            if child is None:
                break
            if child.swa_tombstone:
                # commit the windowed-safe boundary at the PARENT (before this tombstone) if the
                # live run leading up to it already covers the window; then reset the run.
                if match_since_tomb >= self.sliding_window_size:
                    best_value_len = len(value)
                    best_node = node
                match_since_tomb = 0
            match_len = align_down(child.get_match_len(input_ids[prefix_pos:]), self.page_size)
            if match_len == 0:
                break
            if match_len < child.length:
                child = child.split_at(match_len)  # child := the matched prefix segment
                value.append(child.value)
                if not child.swa_tombstone:
                    match_since_tomb += child.length
                node = child
                prefix_pos += match_len
                break
            value.append(child.value)
            if not child.swa_tombstone:
                match_since_tomb += child.length
            node = child
            prefix_pos += match_len

        if match_since_tomb >= self.sliding_window_size:
            best_value_len = len(value)
            best_node = node

        self._stamp_path(best_node)
        kv = torch.cat(value[:best_value_len]) if best_value_len else self.empty
        return SWAMatch(kv, int(kv.numel()), best_node)

    def insert(self, input_ids: torch.Tensor, kv_indices: torch.Tensor,
               swa_evicted_seqlen: int = 0, update_kv_after_len: int = 0
               ) -> Tuple[int, torch.Tensor]:
        """Insert the committed full KV prefix. ``kv_indices`` are the request's full-pool page
        indices (the swa rides along via the full->swa mapping, live where the request allocated
        it). Two frontiers drive the sglang ``_insert_helper`` reconcile:

        - ``update_kv_after_len`` (= the request's old reused-prefix length): only nodes extending
          PAST it carry the request's freshly-(re)computed live-swa indices. Nodes entirely within
          it are tree-shared slots and are left untouched.
        - ``swa_evicted_seqlen`` (= positions whose swa the request itself freed during decode):
          below it the request's swa is the sentinel; at/above it it is live.

        A matched node that is ``swa_tombstone`` (swa freed by an earlier ``evict_swa``) and extends
        into the fresh region is REVIVED -- not by rewriting the mapping but by adopting the
        request's live-swa indices into ``node.value`` and clearing the tombstone (sglang Branches
        1/2/3). Tokens of the new suffix before ``swa_evicted_seqlen`` are inserted as a tombstone;
        a live (non-tombstone) leaf is always added (the free_swa ``-page_size`` margin guarantees a
        live tail). Returns ``(matched_prefix_len, freed_full_indices)``: the full-pool slots the
        caller must return to BOTH pools (displaced old tree slots + non-adopted request dups;
        ``free_swa`` is idempotent on the already-sentinel ones)."""
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, kv_indices = input_ids[:insert_len], kv_indices[:insert_len]
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
            seg_kv = kv_indices[total:total + match_len]   # request's fresh slots for this span
            if update_kv_after_len < total + match_len:
                if child.swa_tombstone:
                    assert child.swa_ref_count == 0, "a tombstoned node cannot hold a swa lock"
                    if child.ref_count > 0:
                        # A full-locked reader still gathers the node's CURRENT slots through its
                        # own row; freeing them on revive would hand live KV to the next alloc.
                        # Keep the tombstone + the tree's value, drop the request's dup (Branch 3
                        # shape) -- the window simply stays unrevived until the lock clears.
                        freed.append(seg_kv.clone())
                    elif swa_evicted_seqlen <= total:
                        # Branch 1: the node's swa is live in the request -> revive it whole. Free
                        # the old (sentinel-mapped) tree slots; adopt the request's live-swa slots.
                        freed.append(child.value)
                        child.set_key_value(child._key, seg_kv.clone())
                        child.swa_tombstone = False
                        child.timestamp = self._tick()
                        self.swa_evictable += child.length
                        self._revives += 1
                    elif swa_evicted_seqlen < total + match_len:
                        # Branch 2: the request's freed-swa frontier falls inside the node. Split;
                        # the head [:start] stays tombstone (full KV kept), revive the live tail.
                        start = swa_evicted_seqlen - total
                        child.split_at(start)            # head=[:start] tombstone; child=tail[start:]
                        freed.append(child.value)        # tail's old (sentinel) tree slots
                        freed.append(seg_kv[:start].clone())   # dup for the still-tombstone head
                        child.set_key_value(child._key, seg_kv[start:].clone())
                        child.swa_tombstone = False
                        child.timestamp = self._tick()
                        self.swa_evictable += child.length
                        self._revives += 1
                    else:
                        # Branch 3: node still wholly out-of-window -> keep tombstone, drop the dup.
                        freed.append(seg_kv.clone())
                else:
                    # Matched a live node -> tree slots are canonical; drop the request's dup.
                    freed.append(seg_kv.clone())
            total += match_len
            node = child
            if partial:
                break
        # Suffix: tokens [total:insert_len) not yet in the tree. Tombstone [total, swa_evicted_seqlen),
        # then a live (non-tombstone) leaf for the in-window remainder (never a tombstone leaf --
        # the clamp + the free_swa -page_size margin guarantee a live tail).
        if total < insert_len:
            # Both suffix nodes can share this copy without retaining the request buffer.
            suffix_ids = input_ids[total:].clone()
            suffix_kv = kv_indices[total:].clone()
            boundary = max(0, min(swa_evicted_seqlen, insert_len) - total)
            boundary = min(boundary, max(0, len(suffix_ids) - self.page_size))
            if boundary > 0:
                node = self._add_child(node, suffix_ids[:boundary], suffix_kv[:boundary],
                                       tombstone=True)
                suffix_ids, suffix_kv = suffix_ids[boundary:], suffix_kv[boundary:]
            if len(suffix_ids):
                self._add_child(node, suffix_ids, suffix_kv, tombstone=False)
        return total, (torch.cat(freed) if freed else self.empty)

    def _add_child(self, parent: RadixTreeNode, ids: torch.Tensor, kv: torch.Tensor,
                   *, tombstone: bool) -> RadixTreeNode:
        child = RadixTreeNode(self.key_fn, self._tick())
        child.set_key_value(ids, kv)
        child.set_parent(parent)
        child.swa_tombstone = tombstone
        self.full_evictable += child.length
        if not tombstone:
            self.swa_evictable += child.length
        return child

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
                    self.swa_evictable -= cur.length
                    self.swa_protected += cur.length
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
                    self.swa_evictable += cur.length
                    self.swa_protected -= cur.length
                if swa_uuid_for_lock is not None and cur.swa_uuid == swa_uuid_for_lock:
                    dec_swa = False  # window boundary reached (inclusive); stop swa decrement
            cur = cur.parent

    # ---------------------------------------------------------------- eviction (dual)
    def evict_full(self, num_tokens: int) -> SWAEvictResult:
        """Evict full KV by LRU over UNLOCKED LEAF nodes (an internal node's KV is a prefix
        dependency). Frees both pools for each evicted leaf; cascade-reclaims exposed swa-tombstone
        leaves' full KV. Mirrors sglang ``evict`` full pass."""
        leaves = [n for n in self._leaves() if n.ref_count == 0]
        heapq.heapify(leaves)
        kv, swa, freed = [], [], 0
        while freed < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if node.ref_count != 0 or not node.is_leaf() or node.is_root():
                continue
            freed += node.length
            kv.append(node.value)
            self.full_evictable -= node.length
            if not node.swa_tombstone:
                swa.append(node.value)          # its swa is still live -> free it too
                self.swa_evictable -= node.length
            parent, casc = self._cascade_swa_tombstone_leaves(self._unlink(node), kv)
            freed += casc
            # Re-push the SURVIVING ancestor the cascade returned (the original parent may have
            # been unlinked/freed by the cascade -- re-pushing it would double-free / KeyError).
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                heapq.heappush(leaves, parent)
        return SWAEvictResult(torch.cat(kv) if kv else self.empty,
                              torch.cat(swa) if swa else self.empty)

    def evict_swa(self, num_tokens: int) -> SWAEvictResult:
        """Evict swa KV by LRU over UNLOCKED, non-tombstone swa-bearing nodes -- internal nodes
        too. Internal (or full-locked) node -> TOMBSTONE in place (free swa, keep full KV +
        children). Free leaf -> free both pools, unlink, cascade. Mirrors sglang ``evict`` swa
        pass + the leaf-with-full-lock edge."""
        cands = [n for n in self._swa_nodes() if n.swa_ref_count == 0]
        heapq.heapify(cands)
        kv, swa, freed = [], [], 0
        while freed < num_tokens and cands:
            node = heapq.heappop(cands)
            if node.swa_tombstone or node.swa_ref_count != 0 or node.is_root():
                continue
            if node.is_leaf() and node.ref_count == 0:
                kv.append(node.value)
                swa.append(node.value)
                self.full_evictable -= node.length
                self.swa_evictable -= node.length
                freed += node.length
                node.swa_tombstone = True
                self._cascade_swa_tombstone_leaves(self._unlink(node), kv)
            else:
                swa.append(node.value)          # tombstone internal / full-locked leaf in place
                self.swa_evictable -= node.length
                freed += node.length
                node.swa_tombstone = True
        return SWAEvictResult(torch.cat(kv) if kv else self.empty,
                              torch.cat(swa) if swa else self.empty)

    def trim_head_swa(self, input_ids: torch.Tensor, keep_from: int) -> torch.Tensor:
        """Tombstone the swa currency of the path strictly below ``keep_from`` (full KV stays),
        returning the freed swa-bearing full indices for the caller to release. Retention
        companion of the finish-time soft pin: only the trailing window ``[keep_from, P)``
        needs to stay swa-live for a next-turn cut near P, so the head is reclaimed eagerly
        instead of lingering as LRU litter. Locked (swa_ref>0), tombstoned and leaf nodes are
        left untouched; ``keep_from`` must be page-aligned."""
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
                self.swa_evictable -= child.length
                child.swa_tombstone = True
            node, pos = child, pos + child.length
        return torch.cat(freed) if freed else self.empty

    @property
    def full_evictable_size(self) -> int:
        return self.full_evictable

    @property
    def swa_evictable_size(self) -> int:
        return self.swa_evictable

    @property
    def size_info(self):
        from .base import SizeInfo
        return SizeInfo(evictable_size=self.full_evictable, protected_size=self.full_protected)

    def check_integrity(self) -> None:
        for n in self._all_nodes():
            assert n.ref_count >= 0 and n.swa_ref_count >= 0
            assert n.ref_count >= n.swa_ref_count, "full_ref must be >= swa_ref"
            if n.swa_tombstone:
                assert n.swa_ref_count == 0, "a tombstoned node cannot hold a swa lock"

    # ---------------------------------------------------------------- helpers
    def _unlink(self, node: RadixTreeNode) -> RadixTreeNode:
        parent = node.parent
        del parent.children[self.key_fn(node._key)]
        return parent

    def _cascade_swa_tombstone_leaves(
        self, parent: RadixTreeNode, kv_out: List[torch.Tensor]
    ) -> Tuple[RadixTreeNode, int]:
        """After a leaf is unlinked, eagerly reclaim the swa-tombstone leaves it exposes upward
        (full KV still present, no children, unlocked): free their full KV and unlink. Keeps the
        'a live leaf always carries live swa' invariant (sglang _iteratively_delete_tombstone_leaf).
        Returns ``(surviving_ancestor, freed_full_tokens)`` -- the caller MUST use the returned
        ancestor (the passed-in ``parent`` may itself have been unlinked/freed by the cascade)."""
        freed = 0
        while (parent.swa_tombstone and parent.is_leaf()
               and parent.ref_count == 0 and not parent.is_root()):
            kv_out.append(parent.value)
            self.full_evictable -= parent.length
            freed += parent.length
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

    def _swa_nodes(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if not n.is_root() and not n.swa_tombstone:
                out.append(n)
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
