"""Hybrid (full-attn KV + GDN linear-state) radix cache.

A SEPARATE class from ``RadixPrefixCache`` (Option C) that REUSES the shared ``RadixTreeNode``
and walk/split logic, so the production KV radix is untouched (zero risk to non-hybrid models).
It adds a second "currency": an optional GDN state snapshot (``node.mamba_value`` = a
LinearStatePool slot id) attached at chunk/page-aligned boundary nodes, with its own LRU
eviction. Mirrors sglang ``MambaRadixCache`` (donate-not-copy, dual eviction, internal-node
tombstone, ``full_ref >= mamba_ref``) on FreeToken's tree.

Currency seam: the secondary value + its eviction is the slot a future SWA component plugs
into. This class is pool-agnostic -- it stores/returns slot ids and KV page indices; the
caller (CacheManager / scheduler) does the actual LinearStatePool / KV-pool free.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import torch

from freetoken.utils import align_down

from .base import BaseCacheHandle
from .radix_cache import RadixTreeNode, _get_key_fn


@dataclass(frozen=True)
class HybridCacheHandle(BaseCacheHandle):
    """Lock handle for a matched hybrid prefix: the matched node (lock target) + the reusable
    KV page indices. ``cached_len`` is already truncated to the deepest live-snapshot boundary.
    Plugs into PrefillAdder (reads ``.cached_len`` / ``.get_matched_indices()``) like the plain
    RadixCacheHandle; the restore slot rides on ``MatchResult.mamba_value``."""

    node: RadixTreeNode
    kv_indices: torch.Tensor

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices


class HybridMatch(NamedTuple):
    kv_indices: torch.Tensor      # reused KV page indices for [0:cached_len)
    cached_len: int               # truncated to the deepest LIVE-snapshot boundary
    mamba_value: Optional[int]    # GDN snapshot slot to restore from (None = cold start)
    node: RadixTreeNode           # the matched node (lock target)
    # Same boundary, host-tier copy. Only the device one is preferred: a resume needs one
    # checkpoint, not two, and the bank slot stays attached for whoever asks next.
    mamba_host: Optional[int] = None


class EvictResult(NamedTuple):
    kv_indices: torch.Tensor      # KV page indices to free
    mamba_slots: List[int]        # GDN state slots to free
    # Bank slots freed alongside (KV eviction kills a host checkpoint too: resuming needs the
    # KV of the prefix the checkpoint describes).
    host_slots: List[int] = []


class HybridRadixCache:
    def __init__(self, device: torch.device, page_size: int) -> None:
        from freetoken.kernel.fla.chunk import CHUNK_SIZE
        # Snapshots land on ×CHUNK_SIZE boundaries; require them to be page-aligned so the KV
        # node boundary and the GDN-state boundary coincide (page_size in {1,2,4,8,16,32,64}).
        assert CHUNK_SIZE % page_size == 0, (
            f"hybrid_radix needs CHUNK_SIZE({CHUNK_SIZE}) % page_size({page_size}) == 0"
        )
        self.device = device
        self.page_size = page_size
        self.key_fn = _get_key_fn(page_size)
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.root = RadixTreeNode(self.key_fn)
        self.root.set_key_value(self.empty, self.empty)
        self.root.ref_count = 1  # root is always protected
        self.full_evictable = 0
        self.full_protected = 0
        self.mamba_evictable = 0     # number of live, unlocked snapshots
        self.mamba_protected = 0
        # Host-tier snapshots: counted separately so every existing device-slot conservation
        # check keeps meaning exactly what it did before.
        self.mamba_host_evictable = 0
        self.mamba_host_protected = 0

    # ---------------------------------------------------------------- match / insert
    def match_prefix(self, input_ids: torch.Tensor) -> HybridMatch:
        """Match the token prefix, then truncate the reusable length to the deepest node on
        the path that still owns a checkpoint -- on the device OR in the host bank (a
        continuation can only resume the GDN recurrence from a checkpointed boundary)."""
        node, _ = self._walk(input_ids)
        # walk up to the deepest node whose END boundary has a live snapshot
        cur, end_len = node, self._path_len(node)
        while not cur.is_root():
            if cur.mamba_value is not None or cur.mamba_host is not None:
                return HybridMatch(
                    self._collect_kv(cur), end_len, cur.mamba_value, cur, cur.mamba_host)
            end_len -= cur.length
            cur = cur.parent
        return HybridMatch(self.empty, 0, None, self.root)

    def insert(self, input_ids: torch.Tensor, kv_indices: torch.Tensor,
               mamba_value: int, freed_host: List[int] | None = None) -> Tuple[int, bool]:
        """Insert the committed KV prefix and DONATE ``mamba_value`` at the (page-aligned) end
        boundary node. Returns (matched_prefix_len, mamba_exist). If the boundary node already
        owns a live snapshot, returns mamba_exist=True and does not attach (caller frees the
        donated slot -- dedup)."""
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, kv_indices = input_ids[:insert_len], kv_indices[:insert_len]
        node, prefix_len = self._walk(input_ids)
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.key_fn)
            # A view would keep the request's full output-budget token buffer alive.
            new_node.set_key_value(
                input_ids[prefix_len:].clone(), kv_indices[prefix_len:].clone()
            )
            new_node.set_parent(node)
            self.full_evictable += new_node.length
            node = new_node
        if node.is_root():
            return prefix_len, True   # root can't hold a snapshot; report exist so caller frees it
        if node.mamba_value is not None:
            return prefix_len, True                 # dedup: caller frees its donated slot
        if (freed_host is not None and node.mamba_host is not None
                and node.mamba_host_ref_count == 0):
            # The device copy supersedes the bank one at the same boundary. A locked bank slot
            # stays put: some request is copying from it right now.
            self._detach_host(node, freed_host)
        node.mamba_value = mamba_value              # fills a fresh node or a tombstone
        if node.mamba_ref_count == 0:
            self.mamba_evictable += 1
        return prefix_len, False

    def attach_host(self, input_ids: torch.Tensor, host_idx: int) -> bool:
        """Checkpoint the node ending exactly at ``len(input_ids)`` with a host-tier snapshot.
        Used for the boundaries an intermediate prefill chunk froze: its KV only becomes
        tree-owned at the final commit, which is when this gets called. False means the caller
        keeps the bank slot (boundary not in the tree, or the node is already checkpointed)."""
        node, prefix_len = self._walk(input_ids)
        # A checkpoint describes the state AT this length; attaching it shallower than the walk
        # reached would resume a request with the state of different tokens.
        if prefix_len != len(input_ids) or node.is_root():
            return False
        if node.mamba_value is not None or node.mamba_host is not None:
            return False
        node.mamba_host = host_idx
        if node.mamba_host_ref_count == 0:
            self.mamba_host_evictable += 1
        return True

    def evict_host(self, num: int) -> List[int]:
        """Drop host-tier checkpoints by LRU (node timestamp) to make room in the bank. Unlike
        the device currency this frees no VRAM and deletes no node -- the KV stays, the boundary
        just stops being resumable. Returns the bank slots the caller must return to the pool."""
        cands = [n for n in self._host_nodes() if n.mamba_host_ref_count == 0]
        heapq.heapify(cands)
        dropped: List[int] = []
        while len(dropped) < num and cands:
            node = heapq.heappop(cands)
            if node.mamba_host is None or node.mamba_host_ref_count != 0:
                continue
            self._detach_host(node, dropped)
        return dropped

    # ---------------------------------------------------------------- locking (dual)
    def inc_lock(self, node: RadixTreeNode) -> None:
        """Protect a matched node's snapshot (mamba ref on the node) and its KV path
        (full ref node..root). Enforces full_ref >= mamba_ref: using a snapshot at N pins the
        whole root..N KV chain."""
        if node.mamba_value is not None:
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1
                self.mamba_protected += 1
            node.mamba_ref_count += 1
        if node.mamba_host is not None:
            # Locked separately from the device copy: a request restoring from the bank must not
            # have its source slot recycled mid-copy.
            if node.mamba_host_ref_count == 0:
                self.mamba_host_evictable -= 1
                self.mamba_host_protected += 1
            node.mamba_host_ref_count += 1
        cur = node
        while not cur.is_root():
            if cur.ref_count == 0:
                self.full_evictable -= cur.length
                self.full_protected += cur.length
            cur.ref_count += 1
            cur = cur.parent

    def dec_lock(self, node: RadixTreeNode) -> None:
        if node.mamba_value is not None and node.mamba_ref_count > 0:
            node.mamba_ref_count -= 1
            if node.mamba_ref_count == 0:
                self.mamba_evictable += 1
                self.mamba_protected -= 1
        if node.mamba_host is not None and node.mamba_host_ref_count > 0:
            node.mamba_host_ref_count -= 1
            if node.mamba_host_ref_count == 0:
                self.mamba_host_evictable += 1
                self.mamba_host_protected -= 1
        cur = node
        while not cur.is_root():
            cur.ref_count -= 1
            assert cur.ref_count >= 0
            if cur.ref_count == 0:
                self.full_evictable += cur.length
                self.full_protected -= cur.length
            cur = cur.parent

    # ---------------------------------------------------------------- eviction (dual)
    def evict_full(self, num_tokens: int) -> EvictResult:
        """Evict KV tokens by LRU over UNLOCKED LEAF nodes (an internal node's KV is a prefix
        dependency for all descendants). Frees each evicted node's snapshot too -- a checkpoint
        is only good for the prefix whose KV it describes, so the bank copy dies with the KV."""
        leaves = [n for n in self._leaves() if n.ref_count == 0]
        heapq.heapify(leaves)
        kv, mamba, host, freed = [], [], [], 0
        while freed < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if node.ref_count != 0 or not node.is_leaf() or node.is_root():
                continue
            freed += node.length
            kv.append(node.value)
            self.full_evictable -= node.length
            self._free_node_mamba(node, mamba)
            self._detach_host(node, host)
            parent, casc = self._cascade_tombstone_leaves(self._unlink(node), kv)
            freed += casc
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                heapq.heappush(leaves, parent)
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba, host)

    def evict_mamba(self, num: int) -> EvictResult:
        """Evict GDN snapshots by LRU over UNLOCKED snapshot-bearing nodes -- internal nodes
        too. Internal node -> TOMBSTONE (free the slot, keep KV + children). Leaf node -> free
        both KV and slot and unlink, then cascade-delete any KV-only tombstone leaves it exposes
        upward (so a leaf always carries a live snapshot -- mirrors sglang)."""
        cands = [n for n in self._snapshot_nodes() if n.mamba_ref_count == 0]
        heapq.heapify(cands)
        kv, mamba, freed = [], [], 0
        while freed < num and cands:
            node = heapq.heappop(cands)
            if node.mamba_value is None or node.mamba_ref_count != 0 or node.is_root():
                continue
            if node.is_leaf() and node.ref_count == 0 and node.mamba_host is None:
                # A leaf that keeps a bank checkpoint still has a reason to exist, so it is
                # tombstoned (KV kept) rather than deleted -- that checkpoint is the resume point.
                kv.append(node.value)
                self.full_evictable -= node.length
                self._free_node_mamba(node, mamba)
                freed += 1
                self._cascade_tombstone_leaves(self._unlink(node), kv)
            else:
                self._free_node_mamba(node, mamba)  # tombstone internal (or locked-KV) node
                freed += 1
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba)

    @property
    def full_evictable_size(self) -> int:
        return self.full_evictable

    @property
    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable

    @property
    def size_info(self):
        """KV-page currency, for code that reads a BasePrefixCache size_info (metrics/usage).
        The GDN-snapshot currency is reported via mamba_evictable_size."""
        from .base import SizeInfo
        return SizeInfo(evictable_size=self.full_evictable, protected_size=self.full_protected)

    def check_integrity(self) -> None:
        # Structural: every snapshot-bearing node holds a real slot id; ref counts non-negative.
        # (KV/page conservation is checked by CacheManager.check_integrity.)
        for n in self._snapshot_nodes():
            assert n.mamba_value is not None and n.mamba_ref_count >= 0 and n.ref_count >= 0
        host_nodes = self._host_nodes()
        for n in host_nodes:
            assert n.mamba_host is not None and n.mamba_host_ref_count >= 0
        assert self.mamba_host_evictable + self.mamba_host_protected == len(host_nodes), (
            f"bank leak: evictable({self.mamba_host_evictable}) + "
            f"protected({self.mamba_host_protected}) != nodes({len(host_nodes)})"
        )

    # ---------------------------------------------------------------- helpers
    def _free_node_mamba(self, node: RadixTreeNode, out: List[int]) -> None:
        if node.mamba_value is not None:
            out.append(node.mamba_value)
            node.mamba_value = None
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1

    def _detach_host(self, node: RadixTreeNode, out: List[int]) -> None:
        if node.mamba_host is None:
            return
        # A locked bank slot is some request's restore source; every caller here only reaches
        # unlocked nodes, so hitting this means a currency got un-locked without its KV lock.
        assert node.mamba_host_ref_count == 0, "recycling a locked mamba host slot"
        out.append(node.mamba_host)
        node.mamba_host = None
        self.mamba_host_evictable -= 1

    def _unlink(self, node: RadixTreeNode) -> RadixTreeNode:
        parent = node.parent
        del parent.children[self.key_fn(node._key)]
        return parent

    def _cascade_tombstone_leaves(self, parent: RadixTreeNode, kv_out: List[torch.Tensor]):
        """After a leaf is unlinked, eagerly reclaim the KV-only tombstone leaves it exposes
        upward (mamba_value None, no children, unlocked): free their KV and unlink, walking up.
        Keeps the 'a leaf always carries a live snapshot' invariant (sglang
        _iteratively_delete_tombstone_leaf). Returns (highest surviving ancestor, freed_tokens)."""
        freed = 0
        while (parent.mamba_value is None and parent.mamba_host is None and parent.is_leaf()
               and parent.ref_count == 0 and not parent.is_root()):
            kv_out.append(parent.value)
            self.full_evictable -= parent.length
            freed += parent.length
            parent = self._unlink(parent)
        return parent, freed

    def _path_len(self, node: RadixTreeNode) -> int:
        n, total = node, 0
        while not n.is_root():
            total += n.length
            n = n.parent
        return total

    def _collect_kv(self, node: RadixTreeNode) -> torch.Tensor:
        vals: List[torch.Tensor] = []
        n = node
        while not n.is_root():
            vals.append(n.value)
            n = n.parent
        vals.reverse()
        return torch.cat(vals) if vals else self.empty

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

    def _snapshot_nodes(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.mamba_value is not None and not n.is_root():
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _host_nodes(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.mamba_host is not None and not n.is_root():
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len, total = 0, len(input_ids)
        node = self.root
        tic = time.monotonic_ns()
        while prefix_len < total:
            child = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child is None:
                return node, prefix_len
            node = child
            match_len = align_down(node.get_match_len(input_ids[prefix_len:]), self.page_size)
            prefix_len += match_len
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len
            node.timestamp = tic
        return node, prefix_len


__all__ = ["HybridRadixCache", "HybridMatch", "EvictResult", "HybridCacheHandle"]
