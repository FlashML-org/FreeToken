"""Sliding-window radix cache whose reuse points also carry a recurrent-state snapshot.

For models whose sliding-window layers run next to per-request recurrent state (Inkling 2's
short convolutions). A prefix is resumable at boundary ``b`` only when the tree holds

1. the full KV of ``[0, b)`` (``node.value``),
2. the window KV of ``[b - W, b)`` (live, non-tombstone nodes, as in ``SWARadixCache``),
3. a snapshot of the state at ``b`` (``node.mamba_value``, a LinearStatePool slot).

``SWARadixCache`` keeps (1) and (2); this class adds (3) as a third currency with its own
lock and LRU, like ``HybridRadixCache``. Window KV is worth keeping only in front of a
snapshot, so ``trim_swa`` tombstones the rest of a committed path, a snapshot whose window
was tombstoned is freed, and a leaf without a snapshot (unreachable as a reuse point) is
removed with its KV.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

from freetoken.utils import align_down

from .radix_cache import RadixTreeNode
from .swa_radix_cache import SWACacheHandle, SWARadixCache


@dataclass(frozen=True)
class SWASnapshotCacheHandle(SWACacheHandle):
    """``SWACacheHandle`` plus whether ``CacheManager.lock`` also pinned the node's snapshot (set via
    object.__setattr__ like ``swa_uuid``), so unlock releases exactly what lock took."""

    mamba_locked: bool = False


class SWASnapshotMatch(NamedTuple):
    kv_indices: torch.Tensor     # full-pool KV of [0, cached_len)
    cached_len: int              # the deepest resumable boundary (snapshot + live window)
    node: RadixTreeNode          # the node ending there (lock target)
    mamba_value: Optional[int]   # snapshot slot to restore from (None = cold start)


class SWASnapshotEvictResult(NamedTuple):
    kv_indices: torch.Tensor     # full-pool slots to return to free_slots
    swa_indices: torch.Tensor    # full-pool slots whose window slots to free_swa
    mamba_slots: List[int]       # snapshot slots to return to the LinearStatePool


class SWASnapshotRadixCache(SWARadixCache):
    def __init__(self, device: torch.device, page_size: int, sliding_window_size: int) -> None:
        super().__init__(device, page_size, sliding_window_size)
        self.mamba_evictable = 0     # live, unlocked snapshots
        self.mamba_protected = 0

    # ---------------------------------------------------------------- match / insert
    def match_prefix(self, input_ids: torch.Tensor) -> SWASnapshotMatch:
        """Walk the token prefix, then cut to the deepest node that ends in a snapshot and whose
        trailing window is live: the run of live tokens back to the last tombstone covers ``W``,
        or the path has no tombstone at all."""
        node, pos, total = self.root, 0, len(input_ids)
        value: List[torch.Tensor] = []
        run = float("inf")
        best_len, best_node = 0, self.root
        while pos < total:
            child = node.children.get(self.key_fn(input_ids[pos:]))
            if child is None:
                break
            match_len = align_down(child.get_match_len(input_ids[pos:]), self.page_size)
            if match_len == 0:
                break
            partial = match_len < child.length
            if partial:
                child = child.split_at(match_len)
            value.append(child.value)
            run = 0 if child.swa_tombstone else run + child.length
            pos += match_len
            node = child
            if child.mamba_value is not None and run >= self.sliding_window_size:
                best_len, best_node = len(value), child
            if partial:
                break
        self._stamp_path(best_node)
        kv = torch.cat(value[:best_len]) if best_len else self.empty
        return SWASnapshotMatch(kv, int(kv.numel()), best_node, best_node.mamba_value)

    def walk_path(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, torch.Tensor]:
        """The node ending exactly at ``len(input_ids)`` (splitting one if needed) and the KV of
        the path to it. The path must be in the tree (the caller just inserted it)."""
        walked = self._walk_exact(input_ids)
        assert walked is not None, "walk_path: the path is not in the tree"
        node, _, kv = walked
        return node, kv

    def attach_snapshot(self, input_ids: torch.Tensor, slot: int) -> bool:
        """Donate ``slot`` (the state after ``input_ids``) to the node ending there. Returns True
        when the tree did not take it -- the path is not in the tree, a snapshot is already
        there, the boundary is the root, or its window is not live -- and the caller keeps
        (frees) the slot."""
        if len(input_ids) == 0 or align_down(len(input_ids), self.page_size) != len(input_ids):
            return True
        walked = self._walk_exact(input_ids)
        if walked is None:
            return True
        node, run, _ = walked
        if node.is_root() or node.mamba_value is not None or run < self.sliding_window_size:
            return True
        node.mamba_value = slot
        node.timestamp = self._tick()
        if node.mamba_ref_count == 0:
            self.mamba_evictable += 1
        return False

    # ---------------------------------------------------------------- snapshot lock
    def inc_mamba(self, node: RadixTreeNode) -> bool:
        """Pin the node's snapshot (a pending restore reads it). False if it has none."""
        if node.mamba_value is None:
            return False
        if node.mamba_ref_count == 0:
            self.mamba_evictable -= 1
            self.mamba_protected += 1
        node.mamba_ref_count += 1
        return True

    def dec_mamba(self, node: RadixTreeNode) -> None:
        assert node.mamba_value is not None and node.mamba_ref_count > 0
        node.mamba_ref_count -= 1
        if node.mamba_ref_count == 0:
            self.mamba_evictable += 1
            self.mamba_protected -= 1

    # ---------------------------------------------------------------- eviction (three currencies)
    def evict_full(self, num_tokens: int) -> SWASnapshotEvictResult:
        """Free full KV by LRU over unlocked leaves; each takes its window KV and snapshot along."""
        out = self._new_result()
        leaves = [n for n in self._leaves() if n.ref_count == 0]
        heapq.heapify(leaves)
        freed = 0
        while freed < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if node.ref_count != 0 or not node.is_leaf() or node.is_root():
                continue
            parent, n = self._remove_leaf(node, out)
            freed += n
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                heapq.heappush(leaves, parent)
        return self._result(out)

    def evict_swa(self, num_tokens: int) -> SWASnapshotEvictResult:
        """Free window KV by LRU over unlocked live nodes: an unlocked leaf goes whole, any other
        node is tombstoned in place; snapshots left without a live window are freed after."""
        out = self._new_result()
        cands = [n for n in self._swa_nodes() if n.swa_ref_count == 0]
        heapq.heapify(cands)
        freed = 0
        while freed < num_tokens and cands:
            node = heapq.heappop(cands)
            if node.swa_tombstone or node.swa_ref_count != 0 or node.is_root():
                continue
            if node.is_leaf() and node.ref_count == 0:
                freed += node.length
                self._remove_leaf(node, out)
            else:
                freed += node.length
                self._tombstone(node, out)
        self._drop_dead_snapshots(out)
        return self._result(out)

    def evict_mamba(self, num: int) -> SWASnapshotEvictResult:
        """Free snapshot slots by LRU over unlocked snapshots; a node left as an unlocked leaf
        without a snapshot is removed with its KV."""
        out = self._new_result()
        cands = [n for n in self._snapshot_nodes() if n.mamba_ref_count == 0]
        heapq.heapify(cands)
        freed = 0
        while freed < num and cands:
            node = heapq.heappop(cands)
            if node.mamba_value is None or node.mamba_ref_count != 0 or node.is_root():
                continue
            self._free_snapshot(node, out)
            freed += 1
            if node.is_leaf() and node.ref_count == 0:
                self._remove_leaf(node, out)
        return self._result(out)

    def trim_swa(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Tombstone the window KV on the path of ``input_ids`` that no snapshot reads: a token is
        kept live only within ``W`` before a snapshot boundary in its node's subtree. Locked
        (swa_ref > 0) nodes are left alone, and a leaf keeps a live tail. Returns the freed
        window's full indices."""
        min_end = self._min_snapshot_ends()
        W = self.sliding_window_size
        out = self._new_result()
        path: List[Tuple[RadixTreeNode, int]] = []
        node, pos = self.root, 0
        while pos < len(input_ids):
            child = node.children.get(self.key_fn(input_ids[pos:]))
            if child is None or pos + child.length > len(input_ids):
                break
            path.append((child, pos))
            node, pos = child, pos + child.length
        for child, start in path:
            if child.swa_tombstone or child.swa_ref_count > 0:
                continue
            keep_from = min_end.get(id(child), float("inf")) - W
            end = start + child.length
            if keep_from <= start or (child.is_leaf() and keep_from >= end):
                continue
            if keep_from < end:
                cut = align_down(int(keep_from) - start, self.page_size)
                if cut == 0:
                    continue
                child = child.split_at(cut)
            self._tombstone(child, out)
        return torch.cat(out[1]) if out[1] else self.empty

    @property
    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable

    def check_integrity(self) -> None:
        super().check_integrity()
        evictable = protected = 0
        for n in self._snapshot_nodes():
            assert n.mamba_ref_count >= 0
            if n.mamba_ref_count == 0:
                evictable += 1
            else:
                protected += 1
        assert (evictable, protected) == (self.mamba_evictable, self.mamba_protected), (
            f"snapshot counters ({self.mamba_evictable}, {self.mamba_protected}) != tree ({evictable}, {protected})"
        )
        for n in self._all_nodes():
            if n.mamba_value is None:
                assert n.mamba_ref_count == 0, "a lock on a node without a snapshot"

    # ---------------------------------------------------------------- helpers
    def _walk_exact(self, input_ids: torch.Tensor) -> Optional[Tuple[RadixTreeNode, float, torch.Tensor]]:
        """(node ending at len(input_ids), live-window run there, path KV), or None if the tree
        does not hold the whole path."""
        node, pos, total = self.root, 0, len(input_ids)
        run = float("inf")
        value: List[torch.Tensor] = []
        while pos < total:
            child = node.children.get(self.key_fn(input_ids[pos:]))
            if child is None:
                return None
            match_len = align_down(child.get_match_len(input_ids[pos:]), self.page_size)
            if match_len == 0 or (match_len < child.length and pos + match_len != total):
                return None
            if match_len < child.length:
                child = child.split_at(match_len)
            value.append(child.value)
            run = 0 if child.swa_tombstone else run + child.length
            node, pos = child, pos + child.length
        return node, run, (torch.cat(value) if value else self.empty)

    def _new_result(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
        return [], [], []

    def _result(self, out) -> SWASnapshotEvictResult:
        kv, swa, slots = out
        return SWASnapshotEvictResult(
            torch.cat(kv) if kv else self.empty, torch.cat(swa) if swa else self.empty, slots
        )

    def _free_snapshot(self, node: RadixTreeNode, out) -> None:
        if node.mamba_value is None:
            return
        assert node.mamba_ref_count == 0, "freeing a locked snapshot"
        out[2].append(node.mamba_value)
        node.mamba_value = None
        self.mamba_evictable -= 1

    def _tombstone(self, node: RadixTreeNode, out) -> None:
        assert node.swa_ref_count == 0 and not node.swa_tombstone
        out[1].append(node.value)
        self.swa_evictable -= node.length
        node.swa_tombstone = True

    def _remove_leaf(self, node: RadixTreeNode, out) -> Tuple[RadixTreeNode, int]:
        """Remove an unlocked leaf (full KV, live window, snapshot), then every unlocked leaf it
        exposes that has no snapshot. Returns (surviving ancestor, freed full tokens)."""
        freed = 0
        while True:
            out[0].append(node.value)
            self.full_evictable -= node.length
            freed += node.length
            if not node.swa_tombstone:
                out[1].append(node.value)
                self.swa_evictable -= node.length
            self._free_snapshot(node, out)
            parent = self._unlink(node)
            node._parent = None
            if parent.is_root() or not parent.is_leaf() or parent.ref_count != 0 or parent.mamba_value is not None:
                return parent, freed
            node = parent

    def _drop_dead_snapshots(self, out) -> None:
        """Free the unlocked snapshots whose window is no longer live."""
        W = self.sliding_window_size
        dead: List[RadixTreeNode] = []
        stack: List[Tuple[RadixTreeNode, float]] = [(self.root, float("inf"))]
        while stack:
            node, run = stack.pop()
            for child in node.children.values():
                r = 0 if child.swa_tombstone else run + child.length
                if child.mamba_value is not None and r < W and child.mamba_ref_count == 0:
                    dead.append(child)
                stack.append((child, r))
        for node in dead:
            if node.mamba_value is None or node.is_root():
                continue
            self._free_snapshot(node, out)
            if node.is_leaf() and node.ref_count == 0:
                self._remove_leaf(node, out)

    def _min_snapshot_ends(self) -> Dict[int, int]:
        """For every node, the shallowest snapshot boundary (token position) in its subtree."""
        result: Dict[int, int] = {}
        order: List[Tuple[RadixTreeNode, int]] = []
        stack: List[Tuple[RadixTreeNode, int]] = [(self.root, 0)]
        while stack:
            node, end = stack.pop()
            order.append((node, end))
            for child in node.children.values():
                stack.append((child, end + child.length))
        for node, end in reversed(order):
            best = end if node.mamba_value is not None and not node.is_root() else None
            for child in node.children.values():
                c = result.get(id(child))
                if c is not None and (best is None or c < best):
                    best = c
            if best is not None:
                result[id(node)] = best
        return result

    def _snapshot_nodes(self) -> List[RadixTreeNode]:
        return [n for n in self._all_nodes() if n.mamba_value is not None]


__all__ = [
    "SWASnapshotRadixCache",
    "SWASnapshotMatch",
    "SWASnapshotEvictResult",
    "SWASnapshotCacheHandle",
]
