"""What is NOT specific to one of the three caches.

Two jobs no single-class module can do:

1. SHARED MACHINERY, class-free -- ``RadixTreeNode.split_at`` identity semantics (the matched
   node stays put and becomes the SUFFIX, so already-issued deep handles survive), ``key_fn``
   scalar-vs-tuple at page_size 1 vs >1, ``get_match_len``, and page alignment via
   ``align_down``.  ``RadixPrefixCache``, ``SWARadixCache`` and ``HybridRadixCache`` are siblings,
   not a hierarchy; this is the code all three genuinely share, so it is tested once here instead
   of three times over.

2. HARNESS SELF-TEST -- wrap a cache in a deliberately broken proxy and prove the invariant
   battery and the reference model actually fire.  A model that cannot fail is decoration.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import pytest
import torch

from freetoken.kvcache.base import MatchResult
from freetoken.kvcache.radix_cache import RadixCacheHandle, RadixTreeNode, _get_key_fn
from freetoken.utils import align_down

from .adapters import node_path_slots
from .driver import CacheSpec, Session, page_blocks
from .model import HarnessFailure

KINDS = ("plain", "swa", "hybrid")

#: sliding windows matching ``ALL_SPECS`` so ids read the same across the suite.
_WINDOW = {1: 4, 4: 8, 128: 128}


def _spec(kind: str, page_size: int) -> CacheSpec:
    return CacheSpec(kind, page_size, window=_WINDOW[page_size] if kind == "swa" else 0)


def _blocks(page_size: int, n: int = 6) -> List[Tuple[int, ...]]:
    """Whole pages that deliberately share leading tokens (see ``page_blocks``)."""
    return page_blocks(page_size, n)


def _t(xs: Sequence[int], dtype=torch.int64) -> torch.Tensor:
    return torch.tensor(list(xs), dtype=dtype)


# =============================================================================================
# 1. SHARED TREE MACHINERY, tested once and class-free
# =============================================================================================
def _bare_node(key_fn, key: Sequence[int], value: Sequence[int], parent=None) -> RadixTreeNode:
    n = RadixTreeNode(key_fn)
    n.set_key_value(_t(key), _t(value, torch.int32))
    if parent is not None:
        n.set_parent(parent)
    return n


def test_split_at_keeps_the_original_object_as_the_suffix() -> None:
    """``split_at`` is an IDENTITY contract, not just a structural one.

    The caller holds handles (lock targets, ``MatchResult`` nodes, children) that were issued
    before the split.  The implementation therefore creates a NEW node for the root-side prefix
    and mutates the original object into the suffix in place, so every previously issued handle
    -- including nodes deeper in the tree -- still resolves to the same slots.  Everything a
    sibling class stores on a node has its own split rule, and all of them live here.
    """
    key_fn = _get_key_fn(2)
    root = _bare_node(key_fn, [], [])
    node = _bare_node(key_fn, [1, 2, 3, 4, 5, 6], [11, 12, 13, 14, 15, 16], root)
    deep = _bare_node(key_fn, [7, 8], [17, 18], node)

    node.ref_count = 2
    node.swa_ref_count = 1
    node.swa_level = 0
    node.swa_uuid = 77
    node.mamba_value = 99
    node.mamba_ref_count = 1
    stamp = node.timestamp
    path_before = node_path_slots(deep)

    prefix = node.split_at(4)

    # identity: the matched node object IS the suffix; the prefix is the new object
    assert prefix is not node
    assert node._key.tolist() == [5, 6] and node.value.tolist() == [15, 16]
    assert prefix._key.tolist() == [1, 2, 3, 4] and prefix.value.tolist() == [11, 12, 13, 14]
    assert node.length == 2 and prefix.length == 4

    # already-issued deep handles still resolve to exactly the same slots
    assert deep.parent is node
    assert node_path_slots(deep) == path_before == [11, 12, 13, 14, 15, 16, 17, 18]

    # re-linking: root -> prefix -> suffix, each registered under its own FIRST page key
    assert prefix.parent is root and node.parent is prefix
    assert root.children[key_fn(prefix._key)] is prefix
    assert prefix.children[key_fn(node._key)] is node
    assert len(root.children) == 1 and len(prefix.children) == 1

    # per-field split rules (each one a different class's invariant)
    assert prefix.ref_count == 2 and node.ref_count == 2          # full lock covers both halves
    assert prefix.swa_ref_count == 1 and node.swa_ref_count == 1  # window covers both halves
    assert prefix.swa_tombstone and node.swa_tombstone            # a tombstone covers all tokens
    assert prefix.swa_uuid == 77 and node.swa_uuid is None        # boundary MIGRATES root-side
    assert prefix.mamba_value is None and node.mamba_value == 99  # a snapshot cannot be split
    assert prefix.mamba_ref_count == 0 and node.mamba_ref_count == 1
    assert prefix.timestamp == stamp                              # LRU age is inherited

    # the split point must land strictly inside the node
    with pytest.raises(AssertionError):
        node.split_at(0)
    with pytest.raises(AssertionError):
        node.split_at(node.length)


def test_key_fn_buckets_a_whole_page_not_a_token() -> None:
    """``_get_key_fn`` is the reason the reference model is keyed by page.

    A child is registered under its FIRST PAGE only, and the lookup slices the query down to that
    page -- so a longer query finds the same bucket, and two distinct pages that share leading
    tokens are distinct buckets.  A token-keyed view of the tree would predict sharing that never
    happens.
    """
    k1, k4 = _get_key_fn(1), _get_key_fn(4)

    assert k1(_t([5, 9, 9])) == 5 and isinstance(k1(_t([5])), int)   # scalar at page_size 1
    assert k1(_t([1, 7])) == k1(_t([1, 9]))                          # tail ignored
    assert k4(_t([1, 7, 7, 2])) == (1, 7, 7, 2)                      # tuple at page_size > 1
    assert k4(_t([1, 7, 7, 2, 3, 3, 3, 3])) == k4(_t([1, 7, 7, 2]))  # tail ignored
    assert k4(_t([1, 7, 7, 2])) != k4(_t([1, 7, 7, 3]))              # same lead, different page

    # ... and the consequence in every one of the three trees: two pages sharing three leading
    # tokens share NO prefix at all.
    left = (1, 3, 1, 5) + (3, 2, 2, 9)
    right = (1, 3, 1, 6) + (0, 0, 0, 0)
    for kind in KINDS:
        s = Session(_spec(kind, 4))
        s.do_insert(left)
        got, _ = s.do_insert(right)
        s.check()
        assert got.matched_len == 0, f"{kind} shared a prefix between two distinct pages"
        assert len(s.ad.nodes()) == 2

    plain = Session(_spec("plain", 4))
    plain.do_insert(left)
    plain.do_insert(right)
    m, _ = plain.do_match(left[:4])
    plain.check()
    assert m.cached_len == 4
    assert m.indices == list(plain.ad.structure()[left[:4]]["slots"])[:4]


@pytest.mark.parametrize("page_size", (1, 4, 128))
def test_get_match_len_and_align_down_bound_reuse_to_whole_pages(page_size: int) -> None:
    """``get_match_len`` counts equal TOKENS; the tree only ever reuses whole PAGES.

    Every walk in all three classes is ``align_down(child.get_match_len(...), page_size)``, so a
    divergence inside a page discards that whole page.
    """
    key_fn = _get_key_fn(page_size)
    node = _bare_node(key_fn, list(range(100, 108)), list(range(200, 208)))
    assert node.get_match_len(_t(range(100, 108))) == 8               # identical
    assert node.get_match_len(_t(list(range(100, 106)) + [0, 0])) == 6  # first diff at 6
    assert node.get_match_len(_t([100, 101, 102])) == 3               # query shorter than key
    assert node.get_match_len(_t([0, 101, 102])) == 0                 # diff at 0

    assert align_down(6, page_size) == (6 // page_size) * page_size
    assert align_down(0, page_size) == 0

    # the same arithmetic, observed through a real cache: reuse is truncated to the page below
    P = page_size
    b = _blocks(P)
    ids = b[0] + b[1]
    ragged = ids[: 2 * P - 1]                       # one token short of the second page
    for kind in ("plain", "swa"):                   # (hybrid truncation is snapshot-driven)
        s = Session(_spec(kind, P))
        s.do_insert(ids)
        got, _ = s.do_match(ragged)
        s.check()
        assert got.cached_len == align_down(len(ragged), P) == P
        assert len(got.indices) == P


# =============================================================================================
# 2. HARNESS SELF-TEST: prove the model would catch a wrong implementation
# =============================================================================================
class _Proxy:
    """Transparent wrapper around a real cache; subclasses break exactly one thing."""

    def __init__(self, inner) -> None:
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):          # only reached for attrs the proxy does not define
        return getattr(self._inner, name)


class _ForgetfulInsert(_Proxy):
    """insert_prefix links the new node but forgets to grow ``evictable_size``."""

    def insert_prefix(self, ids, indices):
        before = self._inner.evictable_size
        res = self._inner.insert_prefix(ids, indices)
        self._inner.evictable_size = before
        return res


class _LeakyEvict(_Proxy):
    """evict() drops one extra leaf from the tree without returning its slots -- the classic
    'cascade forgot to collect the node's KV' bug."""

    def evict(self, size):
        out = self._inner.evict(size)
        for leaf in self._inner._collect_leave_nodes_for_evict():
            parent = leaf.parent
            del parent.children[self._inner.key_fn(leaf._key)]
            self._inner.evictable_size -= leaf.length
            break
        return out


class _NewestFirstEvict(_Proxy):
    """evict() picks the most recently used leaf instead of the least -- an LRU inversion that
    keeps every counter and every slot perfectly conserved."""

    def evict(self, size):
        leaves = self._inner._collect_leave_nodes_for_evict()
        victim = max(leaves, key=lambda n: n.timestamp)
        parent = victim.parent
        del parent.children[self._inner.key_fn(victim._key)]
        self._inner.evictable_size -= victim.length
        return victim.value


class _ShortMatch(_Proxy):
    """match_prefix stops one node early -- a silent loss of prefix reuse."""

    def match_prefix(self, ids):
        res = self._inner.match_prefix(ids)
        node = res.cuda_handle.node
        if node.is_root():
            return res
        return MatchResult(RadixCacheHandle(res.cuda_handle.cached_len - node.length, node.parent))


_BROKEN = [
    (_ForgetfulInsert, "insert", "counters"),
    (_LeakyEvict, "evict", "conservation.kv"),
    (_NewestFirstEvict, "evict", "model.evict"),
    (_ShortMatch, "match", "match.cached_len"),
]


@pytest.mark.parametrize("factory,trigger,tag", _BROKEN,
                         ids=[f.__name__.lstrip("_") for f, _, _ in _BROKEN])
def test_harness_catches_a_broken_implementation(factory, trigger: str, tag: str) -> None:
    """The battery is only worth its runtime if it can fail.

    Each proxy breaks one thing a plausible regression breaks -- a counter, a freed slot, the LRU
    order, the matched length -- and each must be caught by the check whose name says so, not by
    a downstream avalanche.
    """
    P = 4
    b = _blocks(P)
    keys = [b[0], b[1], b[2]]
    s = Session(_spec("plain", P))
    for k in keys:                       # three one-page leaves with distinct LRU stamps
        s.do_insert(k)
        s.check()
    assert len(s.ad.nodes()) == 3

    s.ad.cache = factory(s.ad.cache)     # the implementation goes bad from here on
    run = {"insert": lambda: s.do_insert(b[3]),
           "evict": lambda: s.do_evict_full(P),
           "match": lambda: s.do_match(keys[0])}[trigger]

    with pytest.raises(HarnessFailure) as excinfo:
        run()                            # either the op's own comparison against the model ...
        s.check()                        # ... or the battery that follows every op
    assert excinfo.value.tag == tag, (
        f"the broken cache was caught by {excinfo.value.tag!r}, expected {tag!r}: "
        f"{excinfo.value.detail}")
